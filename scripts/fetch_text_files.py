#!/usr/bin/env python3
"""Download explicitly named, pinned AI Village text tables to VM scratch.

The script uses the VM's existing Hugging Face login. It will never fetch media
archives or print tokens, raw records, request headers, or signed URLs.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import psutil


REPO = "aidigestorg/ai-village"
TABLES = (
    "villages.jsonl.gz",
    "village_goals.jsonl.gz",
    "agents.jsonl.gz",
    "agent_goals.jsonl.gz",
    "chat_rooms.jsonl.gz",
    "chat_messages.jsonl.gz",
    "events.jsonl.gz",
    "computer_use_sessions.jsonl.gz",
    "computer_use_turns.jsonl.gz",
    "agent_memories.jsonl.gz",
    "summaries.jsonl.gz",
    "claude_code_messages.jsonl.gz",
)
SCRATCH = Path("/mnt/ai-swarm-scratch")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", choices=TABLES)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--destination", type=Path, default=SCRATCH / "raw")
    parser.add_argument("--manifest", type=Path,
                        default=Path("/home/azureuser/ai-swarm/manifests/text-files.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if sys.platform != "linux" or not SCRATCH.is_mount():
        parser.error("Run only on the Azure VM with scratch mounted")
    if len(args.revision) != 40:
        parser.error("Revision must be a 40-character commit SHA")
    if not args.destination.resolve().is_relative_to(SCRATCH):
        parser.error("Destination must be on VM scratch")
    if not args.manifest.resolve().is_relative_to("/home/azureuser/ai-swarm"):
        parser.error("Manifest must be on persistent VM storage")
    if len(args.files) != len(set(args.files)):
        parser.error("Do not repeat file names")
    os.environ["HF_HUB_CACHE"] = str(SCRATCH / "hf-hub-cache")
    os.environ["HF_XET_CACHE"] = str(SCRATCH / "hf-xet-cache")

    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError

    try:
        info = HfApi().dataset_info(REPO, revision=args.revision, files_metadata=True)
    except HfHubHTTPError as exc:
        print(json.dumps({"error": "metadata_access", "http_status": exc.response.status_code}))
        return 2
    if info.sha != args.revision:
        print(json.dumps({"error": "revision_mismatch"}))
        return 2
    siblings = {item.rfilename: item for item in info.siblings}
    sizes = {name: item.size for name, item in siblings.items()}
    if any(sizes.get(name) is None for name in args.files):
        print(json.dumps({"error": "missing_size_metadata"}))
        return 2
    disk = psutil.disk_usage(SCRATCH)
    reserve = max(50 * 1024**3, disk.total // 5)
    # Assume every selected file may need a full transfer, even if a local
    # same-size copy exists; only the upstream digest can validate that copy.
    needed = sum(sizes[name] for name in args.files)
    plan = {"repo": REPO, "revision": args.revision, "files": [
        {"name": name, "expected_bytes": sizes[name],
         "upstream_sha256": getattr(siblings[name].lfs, "sha256", None)}
        for name in args.files
    ], "scratch_free_bytes": disk.free, "required_reserve_bytes": reserve,
            "planned_transfer_bytes_upper_bound": needed}
    if disk.free < needed + reserve:
        print(json.dumps({"error": "insufficient_scratch", **plan}, sort_keys=True))
        return 2
    if args.dry_run:
        print(json.dumps({"event": "dry_run", **plan}, sort_keys=True))
        return 0

    args.destination.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(args.manifest.read_text()) if args.manifest.exists() else None
    if previous and (previous.get("repo") != REPO
                     or previous.get("revision") != args.revision):
        print(json.dumps({"error": "manifest_revision_mismatch"}))
        return 2
    entries = {item["name"]: item for item in previous.get("files", [])} if previous else {}
    for name in args.files:
        started = time.monotonic()
        try:
            path = Path(hf_hub_download(
                repo_id=REPO, filename=name, repo_type="dataset",
                revision=args.revision, local_dir=args.destination,
            ))
        except HfHubHTTPError as exc:
            print(json.dumps({"error": "file_access", "file": name,
                              "http_status": exc.response.status_code}, sort_keys=True))
            return 2
        except Exception as exc:
            print(json.dumps({"error": "file_download", "file": name,
                              "type": type(exc).__name__}, sort_keys=True))
            return 2
        actual = path.stat().st_size
        if actual != sizes[name]:
            print(json.dumps({"error": "size_mismatch", "file": name,
                              "expected_bytes": sizes[name], "actual_bytes": actual}, sort_keys=True))
            return 2
        digest = sha256_file(path)
        expected_digest = getattr(siblings[name].lfs, "sha256", None)
        if expected_digest and digest != expected_digest:
            print(json.dumps({"error": "hash_mismatch", "file": name}, sort_keys=True))
            return 2
        entry = {"name": name, "bytes": actual, "sha256": digest,
                 "elapsed_seconds": round(time.monotonic() - started, 3),
                 "scratch_free_bytes": psutil.disk_usage(SCRATCH).free}
        entries[name] = entry
        manifest = {"repo": REPO, "revision": args.revision,
                    "files": [entries[key] for key in sorted(entries)]}
        staged = args.manifest.with_suffix(".tmp")
        staged.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        staged.replace(args.manifest)
        print(json.dumps({"event": "file_ready", **entry}, sort_keys=True), flush=True)
        if psutil.disk_usage(SCRATCH).free < reserve:
            print(json.dumps({"error": "scratch_reserve_reached"}))
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
