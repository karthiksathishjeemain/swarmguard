#!/usr/bin/env python3
"""Fetch only the pinned AI Village schema documents on the Azure VM.

Uses Hugging Face's existing VM-side login; never accepts or prints credentials.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


REPO = "aidigestorg/ai-village"
FILES = ("SCHEMA.md", "CHANGELOG.md", "manifest.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="Pinned dataset commit SHA")
    parser.add_argument("--destination", default="/home/azureuser/ai-swarm/source-docs")
    args = parser.parse_args()
    if sys.platform != "linux" or not Path("/mnt/ai-swarm-scratch").is_mount():
        parser.error("Run only on the Azure VM with its scratch disk mounted")
    if len(args.revision) != 40:
        parser.error("Revision must be a 40-character commit SHA")
    destination = Path(args.destination)
    destination.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_CACHE"] = "/mnt/ai-swarm-scratch/hf-hub-cache"

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError

    report = {"repo": REPO, "revision": args.revision, "files": []}
    for name in FILES:
        try:
            path = Path(hf_hub_download(
                repo_id=REPO,
                filename=name,
                repo_type="dataset",
                revision=args.revision,
                local_dir=destination,
            ))
        except HfHubHTTPError as exc:
            # Do not print exception text: signed URLs or request headers may appear.
            report["error"] = {"file": name, "http_status": exc.response.status_code}
            print(json.dumps(report, sort_keys=True))
            return 2
        except Exception as exc:
            report["error"] = {"file": name, "type": type(exc).__name__}
            print(json.dumps(report, sort_keys=True))
            return 2
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        report["files"].append({
            "name": name, "bytes": path.stat().st_size, "sha256": digest,
        })
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
