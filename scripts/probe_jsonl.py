#!/usr/bin/env python3
"""Inspect bounded AI Village JSONL samples on the VM without printing values.

This streams a compressed file and reports field names, value types, and resource
use. It is a schema probe, not an exhaustive count or a semantic analysis.
"""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import re
import resource
import sys
import time

import psutil


SAFE_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")


def kind(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def clean_key(key: object) -> str:
    if isinstance(key, str) and SAFE_KEY.fullmatch(key):
        return key
    return "<nonstandard-key>"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--max-lines", type=int, default=10000)
    args = parser.parse_args()
    if sys.platform != "linux" or not Path("/mnt/ai-swarm-scratch").is_mount():
        parser.error("Run only on the Azure VM with its scratch disk mounted")
    if not args.file.is_file() or not args.file.name.endswith(".jsonl.gz"):
        parser.error("File must be a VM-side .jsonl.gz file")
    if not args.file.resolve().is_relative_to("/mnt/ai-swarm-scratch"):
        parser.error("Dataset file must reside on VM scratch")
    if args.max_lines < 1:
        parser.error("--max-lines must be positive")

    fields: Counter[str] = Counter()
    types: Counter[str] = Counter()
    nested: Counter[str] = Counter()
    malformed = 0
    lines = 0
    uncompressed_bytes = 0
    started = time.monotonic()
    cpu_started = time.process_time()
    with gzip.open(args.file, "rb") as source:
        for line in source:
            if lines >= args.max_lines:
                break
            lines += 1
            uncompressed_bytes += len(line)
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                malformed += 1
                continue
            if not isinstance(record, dict):
                malformed += 1
                continue
            for raw_key, value in record.items():
                key = clean_key(raw_key)
                fields[key] += 1
                types[f"{key}:{kind(value)}"] += 1
                if isinstance(value, dict):
                    for child in value:
                        nested[f"{key}.{clean_key(child)}"] += 1
    elapsed = time.monotonic() - started
    print(json.dumps({
        "file": args.file.name,
        "compressed_file_bytes": args.file.stat().st_size,
        "sample_lines": lines,
        "malformed_lines": malformed,
        "sample_uncompressed_bytes": uncompressed_bytes,
        "fields_present": dict(fields.most_common(120)),
        "field_types": dict(types.most_common(180)),
        "nested_keys": dict(nested.most_common(120)),
        "wall_seconds": round(elapsed, 3),
        "process_cpu_seconds": round(time.process_time() - cpu_started, 3),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "system_available_bytes": psutil.virtual_memory().available,
        "scratch_free_bytes": psutil.disk_usage("/mnt/ai-swarm-scratch").free,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
