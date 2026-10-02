"""No weights, downloads or corpus data: test semantic signal safeguards."""
import contextlib
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from swarmguard.nli import (LABELS, MODEL_ID, MODEL_REVISION, REQUIRED_LOADER_FILES,
                           LocalNLIScorer, label_order, relation_from_logits,
                           verify_checkpoint_files)


class FakeTensor:
    def __init__(self, values):
        self.values = values
        self.shape = (1, len(values))

    def to(self, device):
        if device != "cpu":
            raise AssertionError("Non-CPU inference")
        return self

    def __getitem__(self, index):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeTokenizer:
    def __init__(self, tokens):
        self.tokens, self.calls = tokens, []

    def __call__(self, premise, hypothesis, **kwargs):
        self.calls.append((premise, hypothesis, kwargs))
        return {"input_ids": FakeTensor([1] * self.tokens)}


class FakeModel:
    def __init__(self, values):
        self.values, self.calls = values, 0

    def __call__(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(logits=FakeTensor(self.values))


def fake_scorer(tokens=20, values=None):
    scorer = LocalNLIScorer.__new__(LocalNLIScorer)
    scorer.order, scorer.minimum_score, scorer.max_pair_tokens = LABELS, 0.80, 512
    scorer.tokenizer = FakeTokenizer(tokens)
    scorer.model = FakeModel(values if values is not None else [0, 0, 10])
    scorer.torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
    return scorer


def checkpoint_fixture(path):
    files = []
    for name in sorted(REQUIRED_LOADER_FILES):
        contents = ("synthetic-loader-file-" + name).encode()
        (path / name).write_bytes(contents)
        checksum = hashlib.sha256(contents).hexdigest()
        files.append({"name": name, "bytes": len(contents), "sha256": checksum,
                      "hf_lfs_sha256": checksum if name.endswith(".safetensors") else None})
    provenance = {"repo_id": MODEL_ID, "revision": MODEL_REVISION,
                  "integrity_verified": True, "files": files}
    (path / "download_provenance.json").write_text(json.dumps(provenance))
    return provenance


class NLITests(unittest.TestCase):
    def test_mutated_same_size_weights_rejected_before_ml_import(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            provenance = checkpoint_fixture(path)
            verify_checkpoint_files(path, provenance, MODEL_REVISION)
            weights = path / "model.safetensors"
            weights.write_bytes(b"x" * weights.stat().st_size)
            with self.assertRaisesRegex(ValueError, "checkpoint_hash_mismatch"):
                LocalNLIScorer(path, expected_revision=MODEL_REVISION)

    def test_missing_manifest_entry_or_extra_loader_file_rejected_before_ml_import(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            provenance = checkpoint_fixture(path)
            missing = copy.deepcopy(provenance)
            missing["files"] = [item for item in missing["files"] if item["name"] != "tokenizer.json"]
            (path / "download_provenance.json").write_text(json.dumps(missing))
            with self.assertRaisesRegex(ValueError, "required_file_unmanifested"):
                LocalNLIScorer(path, expected_revision=MODEL_REVISION)
            (path / "download_provenance.json").write_text(json.dumps(provenance))
            (path / "unexpected.safetensors").write_bytes(b"extra")
            with self.assertRaisesRegex(ValueError, "loader_file_unmanifested"):
                LocalNLIScorer(path, expected_revision=MODEL_REVISION)

    def test_unverified_lfs_unsafe_filename_or_false_integrity_rejected_before_ml_import(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            provenance = checkpoint_fixture(path)
            cases = []
            no_lfs = copy.deepcopy(provenance)
            next(item for item in no_lfs["files"] if item["name"] == "model.safetensors")["hf_lfs_sha256"] = None
            cases.append((no_lfs, "weight_lfs_hash_not_verified"))
            traversal = copy.deepcopy(provenance)
            traversal["files"][0]["name"] = "../loader.json"
            cases.append((traversal, "manifest_file_invalid"))
            unverified = copy.deepcopy(provenance)
            unverified["integrity_verified"] = False
            cases.append((unverified, "provenance_mismatch"))
            for manifest, error in cases:
                (path / "download_provenance.json").write_text(json.dumps(manifest))
                with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                    LocalNLIScorer(path, expected_revision=MODEL_REVISION)

    def test_mismatched_or_absent_checkpoint_provenance_rejected_before_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text("{}")
            for provenance in (None, {"repo_id": "different-model", "revision": "a" * 40}):
                if provenance is not None:
                    (path / "download_provenance.json").write_text(json.dumps(provenance))
                with self.subTest(provenance=provenance), self.assertRaisesRegex(ValueError, "provenance_mismatch"):
                    LocalNLIScorer(path, expected_revision="a" * 40)

    def test_explicit_label_order_may_differ_from_card_example(self):
        order = label_order({"0": "contradiction", "1": "ENTAILMENT", "2": "neutral"})
        result = relation_from_logits([10, 0, 0], order)
        self.assertEqual(result["label"], "contradiction")

    def test_unknown_or_duplicated_label_mapping_is_rejected(self):
        for mapping in ({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"},
                        {0: "entailment", 1: "entailment", 2: "neutral"},
                        {1: "entailment", 2: "neutral", 3: "contradiction"}):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                label_order(mapping)

    def test_softmax_stable_for_large_logits_and_normalized(self):
        result = relation_from_logits([1000, 1000, 1010], LABELS)
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0)
        self.assertEqual(result["label"], "contradiction")

    def test_low_semantic_score_and_ties_abstain(self):
        for values, threshold in (([0, 0, 0], 0), ([0.1, 0, 0], .8)):
            result = relation_from_logits(values, LABELS, threshold)
            self.assertEqual(result["status"], "insufficient")
            self.assertIsNone(result["label"])

    def test_nonfinite_scores_and_invalid_threshold_fail(self):
        for values in ([float("nan"), 0, 0], [float("inf"), 0, 0], [0, 0]):
            with self.assertRaises(ValueError):
                relation_from_logits(values, LABELS)
        for threshold in (float("nan"), 2, -1, True):
            with self.assertRaises(ValueError):
                relation_from_logits([0, 0, 0], LABELS, threshold)

    def test_overflow_abstains_without_calling_model_or_truncating(self):
        scorer = fake_scorer(tokens=513)
        result = scorer.score_pair("Premise", "Hypothesis")
        self.assertEqual(result["reason"], "pair_token_limit")
        self.assertEqual(result["input_tokens"], 513)
        self.assertEqual(scorer.model.calls, 0)
        self.assertFalse(scorer.tokenizer.calls[0][2]["truncation"])

    def test_exact_token_limit_allowed_and_source_ids_retained(self):
        scorer = fake_scorer(tokens=512)
        result = scorer.score_pair("Premise", "Hypothesis", premise_id="result-1", hypothesis_id="claim-1")
        self.assertEqual(result["status"], "scored")
        self.assertEqual(result["label"], "contradiction")
        self.assertEqual((result["premise_id"], result["hypothesis_id"]), ("result-1", "claim-1"))
        self.assertEqual(scorer.model.calls, 1)

    def test_invalid_or_large_strings_do_not_tokenize(self):
        for premise, hypothesis in (("", "x"), (None, "x"), ("x", "  "), ("x" * 200001, "x")):
            scorer = fake_scorer()
            result = scorer.score_pair(premise, hypothesis)
            self.assertEqual(result["status"], "insufficient")
            self.assertEqual(scorer.tokenizer.calls, [])

    def test_invalid_runtime_scores_abstain(self):
        result = fake_scorer(values=[float("nan"), 0, 0]).score_pair("P", "H")
        self.assertEqual(result["reason"], "invalid_model_scores")

    def test_fixture_has_complete_counterfactual_groups(self):
        fixture = json.loads((Path(__file__).resolve().parents[1] / "fixtures/nli_pairs.json").read_text())
        pairs = fixture["pairs"]
        self.assertEqual(len({p["id"] for p in pairs}), len(pairs))
        groups = {}
        for pair in pairs:
            groups.setdefault(pair["group"], set()).add(pair["expected"])
            self.assertIn(pair["expected"], LABELS)
        self.assertTrue(all(labels == set(LABELS) for labels in groups.values()))


if __name__ == "__main__":
    unittest.main()
