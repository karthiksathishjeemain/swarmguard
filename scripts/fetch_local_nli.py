#!/usr/bin/env python3
"""Fetch and verify the pinned public NLI checkpoint directly on Azure scratch.

Only explicit model/tokenizer/data-card files are permitted. No repository Python
files are downloaded or executed. Existing cached files can be verified without
downloading weights using --verify-only. Hub requests are unauthenticated because
this checkpoint is public; no token contents are read or printed by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarmguard.nli import MODEL_ID, MODEL_REVISION

SCRATCH_MODELS = Path("/mnt/ai-swarm-scratch/models")
FILES = (
    "config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "spm.model", "README.md",
)


def _lfs_sha(info):
    value = getattr(info, "lfs", None)
    return value.get("sha256") if isinstance(value, dict) else getattr(value, "sha256", None)


def verify_files(path, siblings):
    """Verify every expected size and every supplied Hub LFS SHA-256."""
    remote = {item.rfilename: item for item in siblings}
    verified = []
    for name in FILES:
        info = remote.get(name)
        if info is None or type(info.size) is not int or info.size <= 0:
            raise ValueError("incomplete_pinned_hub_metadata")
        file = path / name
        if not file.is_file() or file.is_symlink() or file.stat().st_size != info.size:
            raise ValueError("checkpoint_file_missing_or_wrong_size")
        digest = hashlib.sha256()
        with file.open("rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        checksum = digest.hexdigest()
        expected_lfs_sha = _lfs_sha(info)
        if name == "model.safetensors" and not expected_lfs_sha:
            raise ValueError("weight_lfs_checksum_missing")
        if expected_lfs_sha is not None and checksum != expected_lfs_sha:
            raise ValueError("checkpoint_lfs_checksum_mismatch")
        verified.append({"name": name, "bytes": info.size, "sha256": checksum,
                         "hf_lfs_sha256": expected_lfs_sha})
    return verified


def save_manifest(path, manifest):
    staged = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path, prefix=".nli-provenance-",
                                         suffix=".tmp", delete=False) as out:
            staged = Path(out.name)
            os.fchmod(out.fileno(), 0o600)
            json.dump(manifest, out, indent=2, sort_keys=True)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        staged.replace(path / "download_provenance.json")
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=SCRATCH_MODELS / ("deberta-nli-" + MODEL_REVISION[:8]))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if (sys.platform != "linux" or not Path("/home/azureuser/ai-swarm").is_dir()
            or not Path("/mnt/ai-swarm-scratch").is_mount()):
        parser.error("Model fetching and verification must run on the authorized Azure VM")
    destination = args.output.resolve()
    if destination == SCRATCH_MODELS or not destination.is_relative_to(SCRATCH_MODELS):
        parser.error("The checkpoint directory must be inside Azure scratch/models")
    started = time.perf_counter()
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi(token=False).model_info(MODEL_ID, revision=MODEL_REVISION, files_metadata=True)
    if info.sha != MODEL_REVISION:
        raise ValueError("hub_revision_mismatch")
    if not args.verify_only:
        snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=destination,
                          allow_patterns=list(FILES), token=False, max_workers=2)
    verified = verify_files(destination, info.siblings)
    manifest = {"repo_id": MODEL_ID, "revision": MODEL_REVISION,
                "seconds": round(time.perf_counter() - started, 6),
                "files": verified, "integrity_verified": True,
                "hf_lfs_files_verified": sum(item["hf_lfs_sha256"] is not None for item in verified),
                "repository_code_downloaded_or_executed": False}
    save_manifest(destination, manifest)
    print(json.dumps({"status": "verified", "model_id": MODEL_ID, "revision": MODEL_REVISION,
                      "files": len(verified), "bytes": sum(item["bytes"] for item in verified),
                      "hf_lfs_files_verified": manifest["hf_lfs_files_verified"],
                      "verify_only": args.verify_only, "seconds": manifest["seconds"],
                      "repository_code_downloaded_or_executed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, TypeError, KeyError, OSError, RuntimeError):
        print(json.dumps({"status": "checkpoint_integrity_not_verified"}))
        raise SystemExit(2)
