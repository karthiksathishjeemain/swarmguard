#!/usr/bin/env python3
"""Capture reproducibility and resource metadata on Azure without reading secrets."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
from datetime import datetime, timezone

ROOT = Path("/home/azureuser/ai-swarm")
SCRATCH = Path("/mnt/ai-swarm-scratch")
PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-revision", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.code_revision):
        parser.error("Use the full local Git commit")
    if sys.platform != "linux" or not SCRATCH.is_mount():
        parser.error("Run on Azure with scratch mounted")
    import psutil
    sources = []
    for glob in ("swarmguard/*.py", "scripts/*.py", "config/*.toml", "requirements*.txt", "fixtures/*.json"):
        for file in sorted(PROJECT.glob(glob)):
            sources.append({"path": str(file.relative_to(PROJECT)), "sha256": hashlib.sha256(file.read_bytes()).hexdigest()})
    packages = {}
    for package in ("torch", "transformers", "huggingface_hub", "tokenizers", "safetensors", "numpy", "psutil", "duckdb", "pyarrow", "polars"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    active = 0
    for process in psutil.process_iter(["pid", "cmdline"]):
        if process.pid == os.getpid():
            continue
        command = process.info["cmdline"] or []
        if command and "python" in Path(command[0]).name and any("/project/scripts/run_local_" in item or "/project/scripts/run_swarm_review.py" in item for item in command):
            active += 1
    report = {"captured_at": datetime.now(timezone.utc).isoformat(), "code_revision": args.code_revision,
              "python": platform.python_version(), "packages": packages, "sources": sources,
              "cpus": len(os.sched_getaffinity(0)), "available_ram_gib": round(psutil.virtual_memory().available / 1024**3, 3),
              "root_free_gib": round(psutil.disk_usage(ROOT).free / 1024**3, 3),
              "scratch_free_gib": round(psutil.disk_usage(SCRATCH).free / 1024**3, 3),
              "active_inference_processes": active, "external_model_api_calls": 0,
              "dataset_revision": "838b4150303ca8228e8edb432d8b8ccae353d258"}
    path = ROOT / "reports/local-environment.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    path.chmod(0o600)
    print(json.dumps({key: value for key, value in report.items() if key not in {"sources", "packages"}}, sort_keys=True))


if __name__ == "__main__":
    main()
