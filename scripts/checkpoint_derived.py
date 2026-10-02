#!/usr/bin/env python3
"""Save or restore VM-only derived Parquet on the persistent Azure root disk.

The local NVMe scratch mount can disappear when Azure deallocates or moves the
VM. This never transfers data off the VM and does not include model weights.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import psutil


ROOT = Path("/home/azureuser/ai-swarm")
SCRATCH = Path("/mnt/ai-swarm-scratch")
SOURCE = SCRATCH / "derived"
BACKUP = ROOT / "checkpoint/derived"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save() -> dict:
    files = sorted(SOURCE.glob("*.parquet"))
    if not files:
        raise RuntimeError("NoDerivedParquetFiles")
    required = sum(path.stat().st_size for path in files)
    free = psutil.disk_usage(ROOT).free
    if free < required + 5 * 1024**3:
        raise RuntimeError("InsufficientPersistentSpace")
    parent = BACKUP.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = parent / "derived.tmp"
    previous = parent / "derived.previous"
    if temporary.exists() or previous.exists():
        raise RuntimeError("UnfinishedCheckpointPresent")
    temporary.mkdir()
    manifest = []
    start = time.monotonic()
    for source in files:
        destination = temporary / source.name
        shutil.copy2(source, destination)
        digest = sha256(destination)
        if digest != sha256(source):
            raise RuntimeError(f"CopyHashMismatch:{source.name}")
        manifest.append({"name": source.name, "bytes": destination.stat().st_size,
                         "sha256": digest})
    (temporary / "manifest.json").write_text(
        json.dumps({"files": manifest}, indent=2, sort_keys=True) + "\n")
    if BACKUP.exists():
        os.replace(BACKUP, previous)
    os.replace(temporary, BACKUP)
    if previous.exists():
        shutil.rmtree(previous)
    return {"mode": "save", "files": len(manifest), "bytes": required,
            "wall_seconds": round(time.monotonic() - start, 3),
            "persistent_free_bytes": psutil.disk_usage(ROOT).free,
            "raw_text_transferred_off_vm": False}


def restore() -> dict:
    manifest = json.loads((BACKUP / "manifest.json").read_text())
    SOURCE.mkdir(parents=True, exist_ok=True)
    count = 0
    total = 0
    for item in manifest["files"]:
        source = BACKUP / item["name"]
        if source.stat().st_size != item["bytes"] or sha256(source) != item["sha256"]:
            raise RuntimeError(f"CheckpointHashMismatch:{item['name']}")
        target = SOURCE / item["name"]
        if target.exists():
            if target.stat().st_size != item["bytes"] or sha256(target) != item["sha256"]:
                raise RuntimeError(f"ExistingScratchArtifactDiffers:{item['name']}")
        else:
            staged = SOURCE / (item["name"] + ".restore")
            shutil.copy2(source, staged)
            if sha256(staged) != item["sha256"]:
                raise RuntimeError(f"RestoreHashMismatch:{item['name']}")
            os.replace(staged, target)
            count += 1
        total += item["bytes"]
    return {"mode": "restore", "files_in_manifest": len(manifest["files"]),
            "files_restored": count, "bytes_in_manifest": total,
            "scratch_free_bytes": psutil.disk_usage(SCRATCH).free,
            "raw_text_transferred_off_vm": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("save", "restore"))
    args = parser.parse_args()
    if sys.platform != "linux" or not SCRATCH.is_mount():
        parser.error("Run on the Azure VM with scratch mounted")
    print(json.dumps(save() if args.mode == "save" else restore(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
