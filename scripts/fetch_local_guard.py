#!/usr/bin/env python3
"""Download an allowlisted pinned local guard/judge checkpoint on Azure."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

MODEL = "AI45Research/AgentDoG1.5-Unified-Qwen3.5-4B"
SCRATCH = Path("/mnt/ai-swarm-scratch")
ROOT = Path("/home/azureuser/ai-swarm")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("guard", "judge", "judge9b"), default="guard")
    parser.add_argument("--revision", help="Full commit SHA; omit only to resolve and record a new pin")
    args = parser.parse_args()
    if sys.platform != "linux" or not SCRATCH.is_mount():
        parser.error("Run on the Azure VM with its scratch disk mounted")
    if args.revision and not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("revision must be a full lowercase commit SHA")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()
    model = MODEL if args.model == "guard" else "Qwen/Qwen3.5-4B"
    directory_name = "agentdog-unified-4b" if args.model == "guard" else "qwen35-4b-judge"
    manifest_name = "agentdog-local-model.json" if args.model == "guard" else "qwen-local-model.json"
    if args.model == "judge9b":
        model, directory_name, manifest_name = "Qwen/Qwen3.5-9B", "qwen35-9b-judge", "qwen9b-local-model.json"
    info = HfApi().model_info(model, revision=args.revision, files_metadata=True)
    revision = info.sha
    destination = SCRATCH / "models" / directory_name / revision
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    allow_names = {"config.json", "generation_config.json", "tokenizer.json",
                   "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
                   "vocab.json", "merges.txt", "chat_template.jinja", "tokenizer.model",
                   "model.safetensors.index.json"}
    files = [f for f in info.siblings if "/" not in f.rfilename and
             (f.rfilename in allow_names or f.rfilename.endswith(".safetensors"))]
    required = sum(f.size or 0 for f in files)
    size_limit = (24 if args.model == "judge9b" else 12) * 1024**3
    if not files or required > size_limit or shutil.disk_usage(SCRATCH).free < required + 10 * 1024**3:
        raise RuntimeError("UnexpectedDownloadSizeOrInsufficientSpace")
    print(json.dumps({"stage": "download_plan", "repo": model, "revision": revision,
                      "files": len(files), "bytes": required}), flush=True)
    manifest, start = [], time.monotonic()
    for item in files:
        path = Path(hf_hub_download(model, item.rfilename, revision=revision,
                                  local_dir=destination))
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        actual_hash = digest.hexdigest()
        expected_hash = getattr(item.lfs, "sha256", None) if item.lfs else None
        if item.size and path.stat().st_size != item.size:
            raise RuntimeError("DownloadSizeMismatch")
        if expected_hash and actual_hash != expected_hash:
            raise RuntimeError("DownloadHashMismatch")
        manifest.append({"name": item.rfilename, "bytes": path.stat().st_size,
                         "sha256": actual_hash, "lfs_hash_verified": bool(expected_hash)})
        print(json.dumps({"stage": "file_verified", "name": item.rfilename,
                          "bytes": path.stat().st_size}), flush=True)
    result = {"repo": model, "revision": revision, "directory": str(destination),
              "files": manifest, "wall_seconds": round(time.monotonic() - start, 3),
              "free_scratch_gib": round(shutil.disk_usage(SCRATCH).free / 1024**3, 3)}
    manifest_path = ROOT / "manifests" / manifest_name
    manifest_path.parent.mkdir(exist_ok=True, mode=0o700)
    manifest_path.write_text(json.dumps(result, indent=2) + "\n")
    manifest_path.chmod(0o600)
    print(json.dumps({k: v for k, v in result.items() if k != "files"}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_class": type(exc).__name__}), flush=True)
        raise SystemExit(2)
