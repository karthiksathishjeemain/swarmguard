#!/usr/bin/env python3
"""Stream AI Village turns and memories on the VM; emit aggregate diagnostics only."""

from collections import Counter
import gzip
import json
from pathlib import Path
import re
import resource
import statistics
import sys
import threading
import time

import psutil


SCRATCH = Path("/mnt/ai-swarm-scratch")
RAW = SCRATCH / "raw"
ROOT = Path("/home/azureuser/ai-swarm")
ACTION_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")


def rows(name: str):
    with gzip.open(RAW / f"{name}.jsonl.gz", "rb") as source:
        for line in source:
            yield json.loads(line)


def bounds_update(bounds: list[str | None], stamp: object) -> None:
    if isinstance(stamp, str) and stamp:
        if bounds[0] is None or stamp < bounds[0]:
            bounds[0] = stamp
        if bounds[1] is None or stamp > bounds[1]:
            bounds[1] = stamp


def quantiles(values: list[int]) -> dict:
    if not values:
        return {}
    values.sort()
    return {"min": values[0], "p50": values[len(values) // 2],
            "p90": values[int((len(values) - 1) * 0.9)],
            "p99": values[int((len(values) - 1) * 0.99)], "max": values[-1]}


def main() -> int:
    if sys.platform != "linux" or not SCRATCH.is_mount():
        print(json.dumps({"error": "AzureScratchRequired"}))
        return 2
    required = ("agents", "computer_use_sessions", "computer_use_turns", "agent_memories")
    if any(not (RAW / f"{name}.jsonl.gz").is_file() for name in required):
        print(json.dumps({"error": "RequiredTableMissing"}))
        return 2
    manifest = json.loads((ROOT / "source-docs/manifest.json").read_text())
    agents = {row["id"] for row in rows("agents")}
    sessions = {row["id"] for row in rows("computer_use_sessions")}
    started = time.monotonic()
    process = psutil.Process()
    cpu_started = sum(process.cpu_times()[:2])
    samples: list[tuple[int, int, float]] = []
    stop = threading.Event()

    def observe() -> None:
        psutil.cpu_percent(interval=None)
        while not stop.wait(0.5):
            samples.append((process.memory_info().rss,
                            psutil.virtual_memory().available,
                            psutil.cpu_percent(interval=None)))

    watcher = threading.Thread(target=observe, daemon=True)
    watcher.start()
    turn_count = 0
    turn_date = [None, None]
    turn_months: Counter = Counter()
    action_types: Counter = Counter()
    agent_messages_shapes: Counter = Counter()
    turns_per_session: Counter = Counter()
    missing_session_rows = 0
    null_action_rows = 0
    command_rows = 0
    typed_text_rows = 0
    output_rows = 0
    error_rows = 0
    redacted_screenshot_rows = 0
    system_note_rows = 0
    memory_count = 0
    memory_date = [None, None]
    memory_months: Counter = Counter()
    memory_lengths: list[int] = []
    memory_agents: set[str] = set()
    memory_missing_agent_rows = 0
    try:
        for row in rows("computer_use_turns"):
            turn_count += 1
            sid = row.get("session_id")
            turns_per_session[sid] += 1
            missing_session_rows += sid not in sessions
            action = row.get("agent_action")
            if isinstance(action, dict):
                label = action.get("action")
                if not isinstance(label, str) or not ACTION_NAME.fullmatch(label):
                    label = "<unspecified-or-nonstandard>"
                action_types[label] += 1
                command_rows += isinstance(action.get("command"), str)
                typed_text_rows += isinstance(action.get("text"), str)
            else:
                null_action_rows += 1
            messages = row.get("agent_messages")
            agent_messages_shapes[type(messages).__name__] += 1
            output_rows += isinstance(row.get("output"), str)
            error_rows += isinstance(row.get("error"), str)
            redacted_screenshot_rows += row.get("screenshot_is_redacted") is True
            system_note_rows += isinstance(row.get("system"), str)
            stamp = row.get("created_at")
            bounds_update(turn_date, stamp)
            if isinstance(stamp, str):
                turn_months[stamp[:7]] += 1
        for row in rows("agent_memories"):
            memory_count += 1
            actor = row.get("agent_id")
            if isinstance(actor, str):
                memory_agents.add(actor)
            memory_missing_agent_rows += actor not in agents
            content = row.get("content")
            memory_lengths.append(len(content) if isinstance(content, str) else 0)
            stamp = row.get("created_at")
            bounds_update(memory_date, stamp)
            if isinstance(stamp, str):
                memory_months[stamp[:7]] += 1
    finally:
        stop.set()
        watcher.join(timeout=1)
    cpu_ended = sum(process.cpu_times()[:2])
    report = {
        "source_exported_at": manifest.get("exportedAt"),
        "counts": {"computer_use_turns": turn_count, "agent_memories": memory_count},
        "manifest_count_match": {
            "computer_use_turns": turn_count == manifest["rowCounts"]["computer_use_turns"],
            "agent_memories": memory_count == manifest["rowCounts"]["agent_memories"],
        },
        "turns": {"date_range": turn_date, "monthly_counts": dict(sorted(turn_months.items())),
                  "action_types": dict(action_types.most_common()),
                  "agent_messages_shapes": dict(agent_messages_shapes),
                  "missing_session_ref_rows": missing_session_rows,
                  "sessions_with_turns": len(turns_per_session),
                  "sessions_without_turns": len(sessions - turns_per_session.keys()),
                  "turns_per_session": quantiles(list(turns_per_session.values())),
                  "null_action_rows": null_action_rows, "command_rows": command_rows,
                  "typed_text_rows": typed_text_rows, "output_rows": output_rows,
                  "error_rows": error_rows,
                  "redacted_screenshot_rows": redacted_screenshot_rows,
                  "system_note_rows": system_note_rows},
        "memories": {"date_range": memory_date,
                     "monthly_counts": dict(sorted(memory_months.items())),
                     "active_agent_count": len(memory_agents),
                     "missing_agent_ref_rows": memory_missing_agent_rows,
                     "content_length_chars": quantiles(memory_lengths)},
        "resources": {
            "wall_seconds": round(time.monotonic() - started, 3),
            "process_cpu_seconds": round(cpu_ended - cpu_started, 3),
            "peak_process_rss_bytes": max((s[0] for s in samples),
                                          default=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "minimum_system_available_bytes": min((s[1] for s in samples),
                                                   default=psutil.virtual_memory().available),
            "mean_system_cpu_percent": round(statistics.mean(s[2] for s in samples), 2)
            if samples else None,
            "scratch_free_bytes": psutil.disk_usage(SCRATCH).free,
        },
    }
    output = ROOT / "reports/turns-memories-audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.with_suffix(".tmp")
    staged.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    staged.replace(output)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
