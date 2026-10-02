#!/usr/bin/env python3
"""Build bounded action windows for potential public completion claims on the VM.

Lexical screening is deliberately broad. It retrieves cases for Laya and human
review; it is not a finding that any statement is false or misaligned.
"""

import argparse
from collections import deque
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import threading
import time

import duckdb
import psutil
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path("/home/azureuser/ai-swarm")
SCRATCH = Path("/mnt/ai-swarm-scratch")
DERIVED = SCRATCH / "derived"
FIRST_PERSON = re.compile(r"\b(?:i|we|i.ve|we.ve)\b", re.IGNORECASE)
COMPLETION = re.compile(
    r"\b(?:done|completed|finished|deployed|published|sent|created|fixed|"
    r"pushed|merged|verified|confirmed|submitted|updated|implemented|"
    r"launched|resolved|successfully)\b", re.IGNORECASE)

SCHEMA = pa.schema([
    ("send_turn_id", pa.string()), ("chat_id", pa.string()),
    ("agent_id", pa.string()), ("session_id", pa.string()),
    ("room_id", pa.string()), ("send_at", pa.string()),
    ("message_content", pa.string()), ("session_goal", pa.string()),
    ("selection_group", pa.string()),
    ("evidence_json", pa.string()), ("preceding_action_count", pa.int32()),
    ("prior_session_actions", pa.int32()), ("window_truncated", pa.bool_()),
    ("recent_bash_count", pa.int32()), ("recent_output_count", pa.int32()),
    ("recent_error_count", pa.int32()),
    ("last_action_gap_seconds", pa.float64()),
])


def select_sends(controls: int) -> dict[str, dict]:
    file = pq.ParquetFile(DERIVED / "sent_message_links.parquet")
    chosen: dict[str, dict] = {}
    reservoir: list[dict] = []
    rng = random.Random(20261002)
    seen_controls = 0
    for batch in file.iter_batches(batch_size=8192):
        for row in batch.to_pylist():
            content = row.get("message_content") or ""
            if FIRST_PERSON.search(content) and COMPLETION.search(content):
                row["selection_group"] = "lexical_candidate"
                chosen[row["send_turn_id"]] = row
            else:
                seen_controls += 1
                if len(reservoir) < controls:
                    reservoir.append(row)
                else:
                    index = rng.randrange(seen_controls)
                    if index < controls:
                        reservoir[index] = row
    for row in reservoir:
        row["selection_group"] = "random_control"
        chosen[row["send_turn_id"]] = row
    return chosen


def sort_turns() -> tuple[Path, float]:
    sorted_file = DERIVED / "turns_by_session.parquet"
    if sorted_file.is_file():
        return sorted_file, 0.0
    staged = DERIVED / "turns_by_session.parquet.tmp"
    staged.unlink(missing_ok=True)
    temp_dir = SCRATCH / "duckdb-temp"
    temp_dir.mkdir(exist_ok=True)
    conn = duckdb.connect()
    conn.execute("SET threads = 16")
    conn.execute("SET memory_limit = '16GB'")
    conn.execute(f"SET temp_directory = '{temp_dir}'")
    start = time.monotonic()
    source = DERIVED / "computer_use_turns.parquet"
    conn.execute(f"""
        COPY (
          SELECT id, session_id, created_at, action_type,
                 command_excerpt, typed_text_excerpt,
                 output_excerpt, error_excerpt,
                 output_chars, error_chars
          FROM read_parquet('{source}')
          ORDER BY session_id, created_at, id
        ) TO '{staged}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    count = conn.execute(f"SELECT count(*) FROM read_parquet('{staged}')").fetchone()[0]
    conn.close()
    if count != 2510487:
        raise RuntimeError("SortedTurnCountMismatch")
    os.replace(staged, sorted_file)
    return sorted_file, time.monotonic() - start


def excerpt(value: object, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    if len(value) <= limit:
        return value
    head = limit * 3 // 4
    return value[:head] + "\n[TRUNCATED AGAIN]\n" + value[-(limit - head):]


def evidence_item(row: dict) -> dict:
    return {"turn_id": row["id"], "at": row["created_at"],
            "action": row["action_type"],
            "command": excerpt(row.get("command_excerpt"), 900),
            "typed_text": excerpt(row.get("typed_text_excerpt"), 400),
            "output": excerpt(row.get("output_excerpt"), 1200),
            "output_chars": row.get("output_chars"),
            "error": excerpt(row.get("error_excerpt"), 1000),
            "error_chars": row.get("error_chars")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-actions", type=int, default=20)
    parser.add_argument("--controls", type=int, default=500)
    parser.add_argument("--batch-rows", type=int, default=1000)
    args = parser.parse_args()
    if sys.platform != "linux" or not SCRATCH.is_mount():
        parser.error("Run only on the Azure VM with scratch mounted")
    if args.window_actions < 1 or args.window_actions > 100 or args.controls < 0:
        parser.error("Choose 1–100 window actions and nonnegative controls")
    if args.batch_rows < 100:
        parser.error("Batch rows must be at least 100")
    if not (DERIVED / "sent_message_links.parquet").is_file():
        parser.error("Run link_sent_messages.py first")
    selected = select_sends(args.controls)
    sorted_file, sort_seconds = sort_turns()
    output = DERIVED / "evidence_windows.parquet"
    staged = DERIVED / "evidence_windows.parquet.tmp"
    staged.unlink(missing_ok=True)
    started = time.monotonic()
    process = psutil.Process()
    cpu_started = sum(process.cpu_times()[:2])
    stop = threading.Event()
    samples: list[tuple[int, int, int]] = []
    scanned_rows = 0
    emitted_rows = 0
    lexical_rows = 0
    controls_rows = 0

    def observe() -> None:
        with (ROOT / "logs/evidence-windows.jsonl").open("a") as log:
            while not stop.wait(5):
                sample = (process.memory_info().rss,
                          psutil.virtual_memory().available,
                          psutil.disk_usage(SCRATCH).free)
                samples.append(sample)
                log.write(json.dumps({"elapsed_seconds": round(time.monotonic()-started, 2),
                                      "scanned_turns": scanned_rows,
                                      "emitted_windows": emitted_rows,
                                      "rss_bytes": sample[0],
                                      "system_available_bytes": sample[1],
                                      "scratch_free_bytes": sample[2]}) + "\n")
                log.flush()

    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    watcher = threading.Thread(target=observe, daemon=True)
    watcher.start()
    recent: deque[dict] = deque(maxlen=args.window_actions)
    current_session = None
    prior_actions = 0
    batch: list[dict] = []
    try:
        with pq.ParquetWriter(staged, SCHEMA, compression="zstd",
                              compression_level=3) as writer:
            for record_batch in pq.ParquetFile(sorted_file).iter_batches(batch_size=8192):
                for row in record_batch.to_pylist():
                    scanned_rows += 1
                    sid = row["session_id"]
                    if sid != current_session:
                        current_session = sid
                        recent.clear()
                        prior_actions = 0
                    send = selected.get(row["id"])
                    if send:
                        prior = list(recent)
                        last_at = prior[-1]["at"] if prior else None
                        gap = (datetime.fromisoformat(row["created_at"])
                               - datetime.fromisoformat(last_at)).total_seconds() if last_at else None
                        item = {
                            "send_turn_id": row["id"], "chat_id": send.get("chat_id"),
                            "agent_id": send["agent_id"], "session_id": sid,
                            "room_id": send.get("room_id"), "send_at": row["created_at"],
                            "message_content": send.get("message_content"),
                            "session_goal": send.get("session_goal"),
                            "selection_group": send["selection_group"],
                            "evidence_json": json.dumps(prior, ensure_ascii=False),
                            "preceding_action_count": len(prior),
                            "prior_session_actions": prior_actions,
                            "window_truncated": prior_actions > len(prior),
                            "recent_bash_count": sum(x["action"] == "bash" for x in prior),
                            "recent_output_count": sum(x["output"] is not None for x in prior),
                            "recent_error_count": sum(x["error"] is not None for x in prior),
                            "last_action_gap_seconds": gap,
                        }
                        batch.append(item)
                        emitted_rows += 1
                        lexical_rows += send["selection_group"] == "lexical_candidate"
                        controls_rows += send["selection_group"] == "random_control"
                        if len(batch) >= args.batch_rows:
                            writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
                            batch.clear()
                    recent.append(evidence_item(row))
                    prior_actions += 1
            if batch:
                writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
        if emitted_rows != len(selected) or scanned_rows != 2510487:
            raise RuntimeError("EvidenceCoverageMismatch")
        os.replace(staged, output)
    finally:
        stop.set()
        watcher.join(timeout=1)
    cpu_ended = sum(process.cpu_times()[:2])
    report = {"scanned_turns": scanned_rows, "evidence_windows": emitted_rows,
              "lexical_candidates": lexical_rows, "random_controls": controls_rows,
              "window_actions": args.window_actions,
              "sort_seconds": round(sort_seconds, 3),
              "sorted_parquet_bytes": sorted_file.stat().st_size,
              "evidence_parquet_bytes": output.stat().st_size,
              "window_wall_seconds": round(time.monotonic() - started, 3),
              "window_process_cpu_seconds": round(cpu_ended - cpu_started, 3),
              "peak_process_rss_bytes": max((x[0] for x in samples),
                                            default=process.memory_info().rss),
              "minimum_system_available_bytes": min((x[1] for x in samples),
                                                     default=psutil.virtual_memory().available),
              "scratch_free_bytes": psutil.disk_usage(SCRATCH).free}
    report_path = ROOT / "reports/evidence-windows.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(report_path)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
