#!/usr/bin/env python3
"""Link chat-send turns to their public chat rows by content, agent, and time.

All joins and materialized text remain on the Azure VM. The report exposes
aggregate coverage only; an unmatched send is retained for later inspection.
"""

import json
import os
from pathlib import Path
import sys
import threading
import time

import duckdb
import psutil


ROOT = Path("/home/azureuser/ai-swarm")
SCRATCH = Path("/mnt/ai-swarm-scratch")
DERIVED = SCRATCH / "derived"


def main() -> int:
    if sys.platform != "linux" or not SCRATCH.is_mount():
        print(json.dumps({"error": "AzureScratchRequired"}))
        return 2
    inputs = {name: DERIVED / f"{name}.parquet" for name in
              ("chat_messages", "computer_use_sessions", "computer_use_turns")}
    if any(not path.is_file() for path in inputs.values()):
        print(json.dumps({"error": "ProjectionMissing"}))
        return 2
    output = DERIVED / "sent_message_links.parquet"
    staged = DERIVED / "sent_message_links.parquet.tmp"
    staged.unlink(missing_ok=True)
    started = time.monotonic()
    process = psutil.Process()
    cpu_started = sum(process.cpu_times()[:2])
    samples: list[tuple[int, int, float, int]] = []
    stop = threading.Event()

    def observe() -> None:
        psutil.cpu_percent(interval=None)
        while not stop.wait(2):
            disk = psutil.disk_usage(SCRATCH)
            samples.append((process.memory_info().rss, psutil.virtual_memory().available,
                            psutil.cpu_percent(interval=None), disk.free))

    watcher = threading.Thread(target=observe, daemon=True)
    watcher.start()
    try:
        conn = duckdb.connect()
        conn.execute("SET threads = 16")
        conn.execute("SET memory_limit = '16GB'")
        temp = SCRATCH / "duckdb-temp"
        temp.mkdir(exist_ok=True)
        conn.execute(f"SET temp_directory = '{temp}'")
        turns = str(inputs["computer_use_turns"])
        sessions = str(inputs["computer_use_sessions"])
        chat = str(inputs["chat_messages"])
        conn.execute(f"""
            COPY (
              WITH sends AS (
                SELECT t.id AS send_turn_id, t.session_id, s.agent_id,
                       t.created_at AS send_at, t.message_content,
                       t.message_hash, s.session_goal
                FROM read_parquet('{turns}') t
                JOIN read_parquet('{sessions}') s ON t.session_id = s.id
                WHERE t.action_type = 'send_message_back_to_chat'
              )
              SELECT s.*, c.id AS chat_id, c.room_id,
                     c.created_at AS chat_at,
                     abs(date_diff('millisecond', CAST(s.send_at AS TIMESTAMP),
                                   CAST(c.created_at AS TIMESTAMP))) AS match_delta_ms
              FROM sends s
              LEFT JOIN read_parquet('{chat}') c
                ON s.message_hash = c.content_hash AND s.agent_id = c.agent_id
               AND abs(date_diff('second', CAST(s.send_at AS TIMESTAMP),
                                 CAST(c.created_at AS TIMESTAMP))) <= 300
              QUALIFY row_number() OVER (
                PARTITION BY s.send_turn_id
                ORDER BY c.id IS NULL, match_delta_ms NULLS LAST, c.id
              ) = 1
            ) TO '{staged}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        counts = conn.execute(f"""
            SELECT count(*) AS sends,
                   count(chat_id) AS matched,
                   count(*) - count(chat_id) AS unmatched,
                   max(match_delta_ms) AS max_match_delta_ms
            FROM read_parquet('{staged}')
        """).fetchone()
        conn.close()
        if counts[0] != 100354:
            raise RuntimeError("UnexpectedSendActionCount")
        os.replace(staged, output)
    finally:
        stop.set()
        watcher.join(timeout=1)
    cpu_ended = sum(process.cpu_times()[:2])
    report = {"send_actions": counts[0], "matched_chat_rows": counts[1],
              "unmatched_send_actions": counts[2], "max_match_delta_ms": counts[3],
              "matched_fraction": round(counts[1] / counts[0], 6),
              "output_bytes": output.stat().st_size,
              "wall_seconds": round(time.monotonic() - started, 3),
              "process_cpu_seconds": round(cpu_ended - cpu_started, 3),
              "peak_process_rss_bytes": max((s[0] for s in samples),
                                            default=process.memory_info().rss),
              "minimum_system_available_bytes": min((s[1] for s in samples),
                                                     default=psutil.virtual_memory().available),
              "scratch_free_bytes": psutil.disk_usage(SCRATCH).free}
    report_path = ROOT / "reports/sent-message-links.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(report_path)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
