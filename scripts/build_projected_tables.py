#!/usr/bin/env python3
"""Project pinned AI Village JSONL into compact, VM-only Parquet tables.

The raw corpus never leaves Azure. Model-response payloads and screenshots are
excluded; text required for evidence review stays on disposable VM scratch.
"""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

import orjson
import psutil
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path("/home/azureuser/ai-swarm")
SCRATCH = Path("/mnt/ai-swarm-scratch")
RAW = SCRATCH / "raw"
DERIVED = SCRATCH / "derived"
SOURCE_MANIFEST = ROOT / "manifests/text-files.json"
EXPORT_MANIFEST = ROOT / "source-docs/manifest.json"

SCHEMAS = {
    "chat_messages": pa.schema([
        ("id", pa.string()), ("agent_id", pa.string()),
        ("speaker_type", pa.string()), ("room_id", pa.string()),
        ("created_at", pa.string()), ("content", pa.string()),
        ("content_hash", pa.string()), ("content_chars", pa.int32()),
    ]),
    "computer_use_sessions": pa.schema([
        ("id", pa.string()), ("agent_id", pa.string()),
        ("created_at", pa.string()), ("session_goal", pa.string()),
    ]),
    "computer_use_turns": pa.schema([
        ("id", pa.string()), ("session_id", pa.string()),
        ("created_at", pa.string()), ("action_type", pa.string()),
        ("command_excerpt", pa.string()), ("command_chars", pa.int32()),
        ("typed_text_excerpt", pa.string()), ("typed_text_chars", pa.int32()),
        ("message_content", pa.string()), ("message_hash", pa.string()),
        ("output_excerpt", pa.string()), ("output_chars", pa.int32()),
        ("error_excerpt", pa.string()), ("error_chars", pa.int32()),
        ("screenshot_redacted", pa.bool_()),
    ]),
}


def text_hash(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return hashlib.blake2b(value.strip().encode("utf-8"), digest_size=16).hexdigest()


def clipped(value: object, limit: int) -> tuple[str | None, int]:
    if not isinstance(value, str):
        return None, -1
    length = len(value)
    if length <= limit:
        return value, length
    head = limit * 3 // 4
    return value[:head] + "\n[TRUNCATED]\n" + value[-(limit - head):], length


def project(table: str, row: dict, limit: int) -> dict:
    if table == "chat_messages":
        content = row.get("content")
        return {"id": row.get("id"), "agent_id": row.get("agent_speaker_id"),
                "speaker_type": row.get("speaker_type"), "room_id": row.get("room_id"),
                "created_at": row.get("created_at"), "content": content,
                "content_hash": text_hash(content),
                "content_chars": len(content) if isinstance(content, str) else -1}
    if table == "computer_use_sessions":
        return {"id": row.get("id"), "agent_id": row.get("agent_id"),
                "created_at": row.get("created_at"),
                "session_goal": row.get("session_goal")}
    action = row.get("agent_action")
    action = action if isinstance(action, dict) else {}
    command, command_chars = clipped(action.get("command"), limit)
    typed_text, typed_text_chars = clipped(action.get("text"), limit)
    output, output_chars = clipped(row.get("output"), limit)
    error, error_chars = clipped(row.get("error"), limit)
    message = action.get("content") if action.get("action") == "send_message_back_to_chat" else None
    action_type = action.get("action")
    if not isinstance(action_type, str):
        action_type = "bash" if command is not None else "none"
    return {"id": row.get("id"), "session_id": row.get("session_id"),
            "created_at": row.get("created_at"), "action_type": action_type,
            "command_excerpt": command, "command_chars": command_chars,
            "typed_text_excerpt": typed_text, "typed_text_chars": typed_text_chars,
            "message_content": message, "message_hash": text_hash(message),
            "output_excerpt": output, "output_chars": output_chars,
            "error_excerpt": error, "error_chars": error_chars,
            "screenshot_redacted": row.get("screenshot_is_redacted") is True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", required=True, choices=tuple(SCHEMAS))
    parser.add_argument("--batch-rows", type=int, default=20000)
    parser.add_argument("--field-limit", type=int, default=4000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if sys.platform != "linux" or not SCRATCH.is_mount():
        parser.error("Run only on the Azure VM with scratch mounted")
    if args.batch_rows < 1000 or args.field_limit < 100:
        parser.error("Batch rows must be >=1000 and field limit >=100")
    source = RAW / f"{args.table}.jsonl.gz"
    if not source.is_file():
        parser.error("The selected pinned source file is missing")
    source_manifest = json.loads(SOURCE_MANIFEST.read_text())
    source_entry = next((f for f in source_manifest["files"] if f["name"] == source.name), None)
    if source_entry is None or source.stat().st_size != source_entry["bytes"]:
        parser.error("The source file is not in the verified download manifest")
    export = json.loads(EXPORT_MANIFEST.read_text())
    if source_manifest["revision"] != "838b4150303ca8228e8edb432d8b8ccae353d258":
        parser.error("Unexpected dataset revision")
    DERIVED.mkdir(parents=True, exist_ok=True)
    output = DERIVED / f"{args.table}.parquet"
    staged = DERIVED / f"{args.table}.parquet.tmp"
    if output.exists() and not args.force:
        print(json.dumps({"event": "already_exists", "table": args.table,
                          "bytes": output.stat().st_size}))
        return 0
    staged.unlink(missing_ok=True)
    report_path = ROOT / "reports" / f"projection-{args.table}.json"
    log_path = ROOT / "logs" / f"projection-{args.table}.jsonl"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    started = time.monotonic()
    cpu_started = sum(process.cpu_times()[:2])
    stop = threading.Event()
    samples: list[tuple[int, int, float, int]] = []
    row_count = 0
    uncompressed_bytes = 0
    reserve = max(50 * 1024**3, psutil.disk_usage(SCRATCH).total // 5)

    def observe() -> None:
        psutil.cpu_percent(interval=None)
        with log_path.open("a") as log:
            while not stop.wait(5):
                usage = psutil.disk_usage(SCRATCH)
                sample = (process.memory_info().rss, psutil.virtual_memory().available,
                          psutil.cpu_percent(interval=None), usage.free)
                samples.append(sample)
                log.write(json.dumps({"elapsed_seconds": round(time.monotonic() - started, 2),
                                      "rows": row_count, "rss_bytes": sample[0],
                                      "system_available_bytes": sample[1],
                                      "system_cpu_percent": sample[2],
                                      "scratch_free_bytes": sample[3]}) + "\n")
                log.flush()

    watcher = threading.Thread(target=observe, daemon=True)
    watcher.start()
    batch: list[dict] = []
    try:
        with pq.ParquetWriter(staged, SCHEMAS[args.table], compression="zstd",
                              compression_level=3) as writer:
            with gzip.open(source, "rb") as stream:
                for line in stream:
                    uncompressed_bytes += len(line)
                    batch.append(project(args.table, orjson.loads(line), args.field_limit))
                    if len(batch) >= args.batch_rows:
                        writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMAS[args.table]))
                        row_count += len(batch)
                        batch.clear()
                        if psutil.disk_usage(SCRATCH).free < reserve:
                            raise RuntimeError("ScratchReserveReached")
                if batch:
                    writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMAS[args.table]))
                    row_count += len(batch)
                    batch.clear()
        if row_count != export["rowCounts"][args.table]:
            raise RuntimeError("ExportRowCountMismatch")
        os.replace(staged, output)
    finally:
        stop.set()
        watcher.join(timeout=1)
    cpu_ended = sum(process.cpu_times()[:2])
    report = {"table": args.table, "dataset_revision": source_manifest["revision"],
              "source_sha256": source_entry["sha256"], "rows": row_count,
              "uncompressed_source_bytes": uncompressed_bytes,
              "parquet_bytes": output.stat().st_size,
              "wall_seconds": round(time.monotonic() - started, 3),
              "process_cpu_seconds": round(cpu_ended - cpu_started, 3),
              "peak_process_rss_bytes": max((x[0] for x in samples),
                                            default=process.memory_info().rss),
              "minimum_system_available_bytes": min((x[1] for x in samples),
                                                     default=psutil.virtual_memory().available),
              "scratch_free_bytes": psutil.disk_usage(SCRATCH).free}
    temporary_report = report_path.with_suffix(".tmp")
    temporary_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary_report.replace(report_path)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
