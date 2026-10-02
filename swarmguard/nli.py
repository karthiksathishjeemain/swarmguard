"""Frozen CPU NLI signals for short evidence/claim pairs.

This module does not classify misalignment. Scores describe the semantic relation
between the supplied premise and hypothesis. Callers must separately establish
actor, artifact/version, chronology, applicable instructions and cross-agent use.
No model or tokenizer download is performed by the scorer.
"""
from __future__ import annotations

import math
import json
import hashlib
from pathlib import Path
import re
import time

MODEL_ID = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
MODEL_REVISION = "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"
LABELS = ("entailment", "neutral", "contradiction")
REQUIRED_LOADER_FILES = {
    "config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "spm.model",
}


def verify_checkpoint_files(model_path, provenance, expected_revision):
    """Recheck the trusted downloader's manifest before importing ML libraries.

    This detects missing/modified loader inputs; the local manifest itself is
    produced by fetch_local_nli.py's pinned Hub metadata and LFS verification.
    """
    if (expected_revision != MODEL_REVISION or not isinstance(provenance, dict)
            or provenance.get("repo_id") != MODEL_ID
            or provenance.get("revision") != MODEL_REVISION
            or provenance.get("integrity_verified") is not True):
        raise ValueError("nli_checkpoint_provenance_mismatch")
    files = provenance.get("files")
    if not isinstance(files, list) or not 1 <= len(files) <= 50:
        raise ValueError("nli_checkpoint_manifest_files_invalid")
    indexed = {}
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("nli_checkpoint_manifest_file_invalid")
        name = item.get("name")
        if (not isinstance(name, str) or not name or Path(name).name != name
                or "\\" in name or name in indexed or name == "download_provenance.json"
                or type(item.get("bytes")) is not int or item["bytes"] <= 0
                or not isinstance(item.get("sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None):
            raise ValueError("nli_checkpoint_manifest_file_invalid")
        indexed[name] = item
    if not REQUIRED_LOADER_FILES.issubset(indexed):
        raise ValueError("nli_checkpoint_required_file_unmanifested")
    model_path = Path(model_path)
    for file in model_path.iterdir():
        if file.name == "download_provenance.json":
            continue
        if file.suffix == ".bin":
            raise ValueError("nli_safetensors_only_checkpoint_required")
        if file.is_file() and file.suffix in {".json", ".model", ".safetensors", ".txt", ".jinja"}:
            if file.name not in indexed:
                raise ValueError("nli_checkpoint_loader_file_unmanifested")
    for name, item in indexed.items():
        file = model_path / name
        if file.is_symlink() or not file.is_file() or file.stat().st_size != item["bytes"]:
            raise ValueError("nli_checkpoint_file_missing_or_wrong_size")
        hasher = hashlib.sha256()
        with file.open("rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                hasher.update(chunk)
        checksum = hasher.hexdigest()
        if checksum != item["sha256"]:
            raise ValueError("nli_checkpoint_hash_mismatch")
        lfs_sha = item.get("hf_lfs_sha256")
        if file.suffix == ".safetensors" and lfs_sha != checksum:
            raise ValueError("nli_checkpoint_weight_lfs_hash_not_verified")
        if lfs_sha is not None and lfs_sha != checksum:
            raise ValueError("nli_checkpoint_lfs_hash_mismatch")


def label_order(id2label):
    """Read semantics from the checkpoint; never assume numerical label order."""
    if not isinstance(id2label, dict) or len(id2label) != 3:
        raise ValueError("nli_requires_three_explicit_labels")
    try:
        normalized = {int(k): v.strip().lower() for k, v in id2label.items()}
    except (ValueError, TypeError, AttributeError):
        raise ValueError("invalid_nli_label_mapping") from None
    if set(normalized) != {0, 1, 2} or set(normalized.values()) != set(LABELS):
        raise ValueError("unrecognized_nli_label_mapping")
    return tuple(normalized[i] for i in range(3))


def relation_from_logits(logits, order, minimum_score=0.80):
    """Return semantic probabilities and abstain below the configured threshold.

    The threshold is an operational abstention rule, not calibrated confidence.
    """
    if set(order) != set(LABELS) or len(order) != 3:
        raise ValueError("invalid_nli_label_order")
    if (type(minimum_score) not in (int, float) or
            not math.isfinite(minimum_score) or not 0 <= minimum_score <= 1):
        raise ValueError("invalid_nli_minimum_score")
    if len(logits) != 3 or any(not math.isfinite(float(v)) for v in logits):
        raise ValueError("invalid_nli_logits")
    maximum = max(float(v) for v in logits)
    values = [math.exp(float(v) - maximum) for v in logits]
    total = sum(values)
    probabilities = {label: value / total for label, value in zip(order, values)}
    winner = max(probabilities, key=probabilities.get)
    ties = sum(abs(v - probabilities[winner]) <= 1e-12 for v in probabilities.values()) > 1
    accepted = not ties and probabilities[winner] >= minimum_score
    return {
        "status": "scored" if accepted else "insufficient",
        "label": winner if accepted else None,
        "probabilities": {label: probabilities[label] for label in LABELS},
        "reason": None if accepted else "ambiguous_semantic_relation",
    }


class LocalNLIScorer:
    """Load an existing safetensors checkpoint locally and keep inference on CPU."""

    def __init__(self, local_model_path, *, threads=8, max_pair_tokens=512,
                 minimum_score=0.80, expected_revision=None):
        if type(threads) is not int or not 1 <= threads <= 64:
            raise ValueError("nli_threads_must_be_1_to_64")
        if type(max_pair_tokens) is not int or not 4 <= max_pair_tokens <= 512:
            raise ValueError("nli_max_pair_tokens_must_be_4_to_512")
        relation_from_logits([0, 0, 0], LABELS, minimum_score)
        model_path = Path(local_model_path).resolve(strict=True)
        if not model_path.is_dir() or not (model_path / "config.json").is_file():
            raise ValueError("nli_local_checkpoint_required")
        provenance_file = model_path / "download_provenance.json"
        if expected_revision is not None and provenance_file.is_symlink():
            raise ValueError("nli_checkpoint_provenance_mismatch")
        provenance = json.loads(provenance_file.read_text()) if provenance_file.is_file() else {}
        if expected_revision is not None:
            verify_checkpoint_files(model_path, provenance, expected_revision)

        # Imports stay lazy so deterministic tests do not load ML dependencies.
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(threads)
        config = AutoConfig.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False)
        self.order = label_order(config.id2label)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path), config=config, local_files_only=True,
            trust_remote_code=False, use_safetensors=True, torch_dtype=torch.float32)
        self.model.to("cpu").eval().requires_grad_(False)
        self.torch = torch
        self.minimum_score = minimum_score
        limits = [max_pair_tokens]
        for limit in (getattr(config, "max_position_embeddings", None),
                      getattr(self.tokenizer, "model_max_length", None)):
            if isinstance(limit, int) and 4 <= limit < 100000:
                limits.append(limit)
        self.max_pair_tokens = min(limits)
        self.metadata = {
            "model_id": MODEL_ID, "local_model_path": str(model_path),
            "revision": provenance.get("revision"),
            "revision_verification": ("fixed_pin_and_all_download_manifest_hashes_verified"
                                      if expected_revision is not None else "not_requested"),
            "label_order": list(self.order), "device": "cpu", "dtype": "float32",
            "frozen": True, "threads": threads, "max_pair_tokens": self.max_pair_tokens,
            "minimum_semantic_score": minimum_score, "truncation": False,
            "scores_are_misalignment_confidence": False,
        }

    def score_pair(self, premise, hypothesis, *, premise_id=None, hypothesis_id=None):
        started = time.perf_counter()
        base = {"premise_id": premise_id, "hypothesis_id": hypothesis_id,
                "input_tokens": None, "status": "insufficient", "label": None,
                "probabilities": None, "reason": None}

        def finish(reason=None):
            if reason:
                base["reason"] = reason
            base["elapsed_seconds"] = round(time.perf_counter() - started, 6)
            return base

        if not all(isinstance(value, str) and value.strip() for value in (premise, hypothesis)):
            return finish("empty_or_invalid_pair")
        if len(premise) + len(hypothesis) > 200000:
            return finish("pair_character_limit")
        encoded = self.tokenizer(premise, hypothesis, truncation=False,
                                 add_special_tokens=True, return_tensors="pt")
        base["input_tokens"] = int(encoded["input_ids"].shape[-1])
        if base["input_tokens"] > self.max_pair_tokens:
            return finish("pair_token_limit")
        with self.torch.inference_mode():
            logits = self.model(**{k: value.to("cpu") for k, value in encoded.items()}).logits
            values = logits[0].detach().cpu().tolist()
        try:
            base.update(relation_from_logits(values, self.order, self.minimum_score))
        except (ValueError, TypeError, OverflowError):
            return finish("invalid_model_scores")
        return finish()
