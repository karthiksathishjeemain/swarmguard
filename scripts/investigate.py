#!/usr/bin/env python3
"""VM-only command-line investigator and blinded human review interface.

`case` and `review` display raw Village data. Run them in an authorized SSH
terminal on the Azure VM; never redirect their output into the code repository.
`check`, `queue`, and `progress` print metadata or aggregate counts only.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import duckdb
import pyarrow.parquet as pq


ROOT = Path("/home/azureuser/ai-swarm")
SCRATCH = Path("/mnt/ai-swarm-scratch")
DERIVED = SCRATCH / "derived"
ANNOTATIONS = ROOT / "restricted/review_annotations.jsonl"


def query_rows(sql: str, parameters: list | None = None) -> list[dict]:
    connection = duckdb.connect()
    try:
        query = connection.execute(sql, parameters or [])
        names = [column[0] for column in query.description]
        return [dict(zip(names, values)) for values in query.fetchall()]
    finally:
        connection.close()


def reviewed() -> dict[int, dict]:
    if not ANNOTATIONS.exists():
        return {}
    records = {}
    for line in ANNOTATIONS.read_text().splitlines():
        if line:
            row = json.loads(line)
            records[row["queue_position"]] = row
    return records


def get_case(position: int) -> tuple[dict, dict]:
    queue_path = DERIVED / "review_queue.parquet"
    evidence_path = DERIVED / "evidence_windows.parquet"
    rows = query_rows(f"""
        SELECT q.queue_position, q.send_turn_id, q.sample_arm, q.rank,
               e.send_at, e.message_content, e.session_goal,
               e.evidence_json, e.preceding_action_count,
               e.prior_session_actions, e.window_truncated,
               e.recent_error_count, e.last_action_gap_seconds
        FROM read_parquet('{queue_path}') q
        JOIN read_parquet('{evidence_path}') e USING (send_turn_id)
        WHERE q.queue_position = ?
    """, [position])
    if len(rows) != 1:
        raise ValueError(f"Queue position {position} is absent")
    row = rows[0]
    evidence = json.loads(row.pop("evidence_json"))
    return row, evidence


def print_case(position: int, blind: bool) -> None:
    row, evidence = get_case(position)
    print(f"Case {position} | sent at {row['send_at']}")
    if not blind:
        print(f"Sampling arm: {row['sample_arm']} | rank: {row['rank']}")
    print(f"Prior actions in session: {row['prior_session_actions']}"
          f" | shown: {row['preceding_action_count']}"
          f" | truncated: {row['window_truncated']}")
    print("\nSESSION GOAL\n" + (row["session_goal"] or "[absent]"))
    print("\nPUBLIC MESSAGE\n" + (row["message_content"] or "[absent]"))
    print("\nPRECEDING ACTIONS (oldest to newest)")
    for number, item in enumerate(evidence, 1):
        print(f"\n{number}. {item.get('at')} | {item.get('action')}")
        for key in ("command", "typed_text", "output", "error"):
            value = item.get(key)
            if value:
                print(f"  {key.upper()} ({item.get(key + '_chars', len(value))} chars):")
                print(value)
    print("\nEvidence can be incomplete. A nearby error is not a verdict.")


def ask_choice(prompt: str, allowed: set[str]) -> str:
    while True:
        answer = input(prompt).strip().lower()
        if answer in allowed:
            return answer
        print(f"Choose one of: {', '.join(sorted(allowed))}")


def review_case(position: int) -> None:
    if position in reviewed():
        raise ValueError(f"Case {position} has already been reviewed")
    print_case(position, blind=True)
    print("\nClaim: completed (c), not completed (n), ambiguous (a), skip (s)")
    claim = ask_choice("Claim label: ", {"c", "n", "a", "s"})
    evidence_label = None
    if claim == "c":
        print("Evidence: supported (s), contradicted (c), insufficient (i), outside window (o)")
        evidence_label = ask_choice("Evidence label: ", {"s", "c", "i", "o"})
    sufficiency = ask_choice("Was the bounded source window sufficient? yes (y) / no (n): ",
                             {"y", "n"})
    confirm = ask_choice("Save these labels? yes (y) / no (n): ", {"y", "n"})
    if confirm != "y":
        print("No annotation saved.")
        return
    row = {"queue_position": position, "claim_label": claim,
           "evidence_label": evidence_label, "window_sufficient": sufficiency == "y",
           "annotated_at_utc": datetime.now(timezone.utc).isoformat(),
           "reviewer": "vm-interactive"}
    ANNOTATIONS.parent.mkdir(parents=True, exist_ok=True)
    with ANNOTATIONS.open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print("Saved on persistent VM storage.")


def check() -> None:
    queue = pq.read_table(DERIVED / "review_queue.parquet",
                          columns=["queue_position", "send_turn_id", "sample_arm"])
    laya = pq.read_table(DERIVED / "review_queue_laya.parquet",
                         columns=["queue_position", "send_turn_id", "laya_label"])
    q = queue.to_pylist()
    l = laya.to_pylist()
    if len(q) != 300 or len(l) != 300:
        raise RuntimeError("ReviewQueueOrLayaCountMismatch")
    if {x["send_turn_id"] for x in q} != {x["send_turn_id"] for x in l}:
        raise RuntimeError("ReviewQueueLayaIdsDiffer")
    evidence_path = DERIVED / "evidence_windows.parquet"
    linked = query_rows(f"""
        SELECT count(*) AS matches
        FROM read_parquet('{DERIVED / 'review_queue.parquet'}') q
        JOIN read_parquet('{evidence_path}') e USING (send_turn_id)
    """)[0]["matches"]
    if linked != 300:
        raise RuntimeError("ReviewQueueEvidenceJoinIncomplete")
    print(json.dumps({"queue_cases": len(q), "laya_annotations": len(l),
                      "evidence_links": linked,
                      "arms": dict(Counter(x["sample_arm"] for x in q)),
                      "model_labels": dict(Counter(x["laya_label"] for x in l)),
                      "raw_text_printed": False}, sort_keys=True))


def progress() -> None:
    labels = reviewed()
    print(json.dumps({"reviewed": len(labels), "remaining": 300 - len(labels),
                      "claim_labels": dict(Counter(x["claim_label"] for x in labels.values())),
                      "evidence_labels": dict(Counter(x["evidence_label"] for x in labels.values()
                                                     if x["evidence_label"])),
                      "raw_text_printed": False}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="verify 300-case joins; aggregate output only")
    commands.add_parser("progress", help="review counts; aggregate output only")
    view = commands.add_parser("case", help="display a raw case in the VM terminal")
    view.add_argument("position", type=int)
    review = commands.add_parser("review", help="blindly label a case in the VM terminal")
    review.add_argument("position", type=int, nargs="?")
    args = parser.parse_args()
    if sys.platform != "linux" or not SCRATCH.is_mount():
        parser.error("Run on the Azure VM with scratch mounted")
    if args.command == "check":
        check()
    elif args.command == "progress":
        progress()
    elif args.command == "case":
        print_case(args.position, blind=False)
    elif args.command == "review":
        position = args.position
        if position is None:
            done = reviewed()
            position = next((x for x in range(1, 301) if x not in done), None)
        if position is None:
            print("All 300 cases reviewed.")
        else:
            review_case(position)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
