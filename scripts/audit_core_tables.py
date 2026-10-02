#!/usr/bin/env python3
"""Stream core AI Village tables on the VM and report aggregate join coverage.

Reads no screenshots and emits no IDs, message text, goals, or model outputs.
The output is suitable for sharing as an aggregate reconnaissance artifact.
"""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import resource
import statistics
import sys
import threading
import time

import psutil


SCRATCH = Path("/mnt/ai-swarm-scratch")
RAW = SCRATCH / "raw"
MANIFEST = Path("/home/azureuser/ai-swarm/source-docs/manifest.json")


def rows(table: str):
    with gzip.open(RAW / f"{table}.jsonl.gz", "rb") as source:
        for line in source:
            yield json.loads(line)


def minmax(bounds: list[str | None], value: object) -> None:
    if not isinstance(value, str) or not value:
        return
    if bounds[0] is None or value < bounds[0]:
        bounds[0] = value
    if bounds[1] is None or value > bounds[1]:
        bounds[1] = value


def quantiles(values: list[int]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    return {"min": ordered[0], "p50": ordered[len(ordered) // 2],
            "p90": ordered[int((len(ordered) - 1) * 0.9)],
            "p99": ordered[int((len(ordered) - 1) * 0.99)], "max": ordered[-1]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("/home/azureuser/ai-swarm/reports/core-audit.json"))
    args = parser.parse_args()
    if sys.platform != "linux" or not SCRATCH.is_mount():
        parser.error("Run only on the Azure VM with scratch mounted")
    if not args.output.resolve().is_relative_to("/home/azureuser/ai-swarm"):
        parser.error("Output must be on persistent VM storage")
    required = ("agents", "chat_rooms", "village_goals", "agent_goals",
                "chat_messages", "events", "computer_use_sessions", "summaries")
    missing = [name for name in required if not (RAW / f"{name}.jsonl.gz").is_file()]
    if missing or not MANIFEST.is_file():
        parser.error("Required tables or pinned export manifest are missing")
    expected = json.loads(MANIFEST.read_text())["rowCounts"]
    started = time.monotonic()
    process = psutil.Process()
    cpu_started = sum(process.cpu_times()[:2])
    io_started = process.io_counters()
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
    try:
        agents = {row["id"] for row in rows("agents")}
        rooms = {row["id"] for row in rows("chat_rooms")}
        village_goals = list(rows("village_goals"))
        agent_goals = list(rows("agent_goals"))

        chat_ids: set[str] = set()
        chat_agents: set[str] = set()
        chat_speaker_types: Counter = Counter()
        chat_months: Counter = Counter()
        chat_lengths: list[int] = []
        chat_missing_room = 0
        chat_duplicate_ids = 0
        chat_agent_not_in_roster = 0
        chat_date = [None, None]
        for row in rows("chat_messages"):
            identifier = row.get("id")
            if identifier in chat_ids:
                chat_duplicate_ids += 1
            else:
                chat_ids.add(identifier)
            chat_speaker_types[row.get("speaker_type")] += 1
            actor = row.get("agent_speaker_id")
            if actor:
                chat_agents.add(actor)
                chat_agent_not_in_roster += actor not in agents
            chat_missing_room += row.get("room_id") not in rooms
            content = row.get("content")
            chat_lengths.append(len(content) if isinstance(content, str) else 0)
            stamp = row.get("created_at")
            minmax(chat_date, stamp)
            if isinstance(stamp, str):
                chat_months[stamp[:7]] += 1

        sessions: set[str] = set()
        session_agents: set[str] = set()
        session_agent_not_in_roster = 0
        session_duplicate_ids = 0
        session_goal_empty = 0
        session_date = [None, None]
        for row in rows("computer_use_sessions"):
            identifier = row.get("id")
            if identifier in sessions:
                session_duplicate_ids += 1
            else:
                sessions.add(identifier)
            actor = row.get("agent_id")
            if actor:
                session_agents.add(actor)
            session_agent_not_in_roster += actor not in agents
            session_goal_empty += not bool(row.get("session_goal"))
            minmax(session_date, row.get("created_at"))

        event_ids: set[str] = set()
        event_indices: set[int] = set()
        event_action_types: Counter = Counter()
        event_months: Counter = Counter()
        event_message_ids: set[str] = set()
        event_session_ids: set[str] = set()
        event_actor_ids: set[str] = set()
        event_duplicate_ids = 0
        event_duplicate_indices = 0
        event_nonmonotonic_rows = 0
        event_actor_not_in_roster = 0
        event_message_refs = 0
        event_session_refs = 0
        previous_index = None
        event_date = [None, None]
        for row in rows("events"):
            identifier = row.get("id")
            if identifier in event_ids:
                event_duplicate_ids += 1
            else:
                event_ids.add(identifier)
            index = row.get("event_index")
            if index in event_indices:
                event_duplicate_indices += 1
            else:
                event_indices.add(index)
            if isinstance(index, int):
                event_nonmonotonic_rows += previous_index is not None and index < previous_index
                previous_index = index
            data = row.get("data")
            data = data if isinstance(data, dict) else {}
            action_type = data.get("actionType")
            event_action_types[action_type] += 1
            message_id = data.get("messageId")
            if isinstance(message_id, str):
                event_message_refs += 1
                event_message_ids.add(message_id)
            session_id = data.get("computerUseSessionId")
            if isinstance(session_id, str):
                event_session_refs += 1
                event_session_ids.add(session_id)
            actor = data.get("speakerId") if action_type == "AGENT_TALK" else data.get("agentId")
            if isinstance(actor, str):
                event_actor_ids.add(actor)
                event_actor_not_in_roster += actor not in agents
            stamp = row.get("created_at")
            minmax(event_date, stamp)
            if isinstance(stamp, str):
                event_months[stamp[:7]] += 1

        summary_types: Counter = Counter()
        summary_date = [None, None]
        summary_count = 0
        for row in rows("summaries"):
            summary_count += 1
            summary_types[row.get("type")] += 1
            minmax(summary_date, row.get("created_at"))

        counts = {"agents": len(agents), "chat_rooms": len(rooms),
                  "village_goals": len(village_goals), "agent_goals": len(agent_goals),
                  "chat_messages": len(chat_ids) + chat_duplicate_ids,
                  "computer_use_sessions": len(sessions) + session_duplicate_ids,
                  "events": len(event_ids) + event_duplicate_ids,
                  "summaries": summary_count}
        report = {
            "source_exported_at": json.loads(MANIFEST.read_text()).get("exportedAt"),
            "counts": counts,
            "manifest_count_match": {name: count == expected.get(name)
                                     for name, count in counts.items()},
            "date_ranges": {"chat_messages": chat_date,
                            "computer_use_sessions": session_date,
                            "events": event_date, "summaries": summary_date},
            "chat": {"speaker_types": dict(chat_speaker_types),
                     "active_agent_count": len(chat_agents),
                     "content_length_chars": quantiles(chat_lengths),
                     "missing_room_ref_count": chat_missing_room,
                     "missing_agent_ref_count": chat_agent_not_in_roster,
                     "duplicate_id_count": chat_duplicate_ids,
                     "monthly_counts": dict(sorted(chat_months.items()))},
            "sessions": {"active_agent_count": len(session_agents),
                         "missing_agent_ref_count": session_agent_not_in_roster,
                         "empty_goal_count": session_goal_empty,
                         "duplicate_id_count": session_duplicate_ids},
            "events": {"action_type_counts": dict(event_action_types.most_common()),
                       "active_actor_count": len(event_actor_ids),
                       "missing_actor_ref_count": event_actor_not_in_roster,
                       "duplicate_id_count": event_duplicate_ids,
                       "duplicate_event_index_count": event_duplicate_indices,
                       "nonmonotonic_row_count": event_nonmonotonic_rows,
                       "monthly_counts": dict(sorted(event_months.items()))},
            "links": {"event_message_ref_rows": event_message_refs,
                      "event_message_unique_ids": len(event_message_ids),
                      "event_message_ids_missing_from_chat": len(event_message_ids - chat_ids),
                      "chat_ids_missing_from_events": len(chat_ids - event_message_ids),
                      "event_session_ref_rows": event_session_refs,
                      "event_session_unique_ids": len(event_session_ids),
                      "event_session_ids_missing_from_sessions": len(event_session_ids - sessions)},
            "summary_types": dict(summary_types),
        }
    finally:
        stop.set()
        watcher.join(timeout=1)
    cpu_ended = sum(process.cpu_times()[:2])
    io_ended = process.io_counters()
    report["resources"] = {
        "wall_seconds": round(time.monotonic() - started, 3),
        "process_cpu_seconds": round(cpu_ended - cpu_started, 3),
        "peak_process_rss_bytes": max((s[0] for s in samples),
                                      default=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "minimum_system_available_bytes": min((s[1] for s in samples),
                                               default=psutil.virtual_memory().available),
        "mean_system_cpu_percent": round(statistics.mean(s[2] for s in samples), 2)
        if samples else None,
        "process_read_bytes": io_ended.read_bytes - io_started.read_bytes,
        "scratch_free_bytes": psutil.disk_usage(SCRATCH).free,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staged = args.output.with_suffix(".tmp")
    staged.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    staged.replace(args.output)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
