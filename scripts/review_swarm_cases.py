#!/usr/bin/env python3
"""Blinded reviewer-attested labels on Azure; no model reports are read.

--check prints counts only. --review displays selected evidence only in an
interactive VM terminal. TTY and a declared reviewer name do not authenticate a
person or make labels independent gold. Original evidence and labels are retained.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarmguard.schema import validate_finding, validate_packet

RESTRICTED = Path("/home/azureuser/ai-swarm/restricted")
SEED = 20261002
CATEGORIES = {"coordinated_constraint_violation", "unsupported_claim_propagation", "goal_substitution", "none"}
COVERAGE_FIELDS = {"records", "limitations", "source_window_truncated", "target_source_window_truncated",
                   "source_prior_session_actions", "goal_context_is_not_per_action_authorization",
                   "context_is_retrospective", "cross_agent_window_seconds", "prior_tool_turns_retained",
                   "target_tool_turns_retained", "anchor_provenance_unique"}


def digest(value, *, sorted_keys=True):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=sorted_keys,
                                    separators=(",", ":")).encode()).hexdigest()


def require_tty():
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise ValueError("interactive_human_terminal_required")


def read_json(path, *, jsonl=False):
    if path.stat().st_size > 32_000_000:
        raise ValueError("bounded_review_input_exceeded")
    raw = path.read_text()
    return ([json.loads(line) for line in raw.splitlines() if line.strip()]
            if jsonl and not raw.lstrip().startswith("[") else json.loads(raw))


def validate_inputs(packets, manifest):
    if (not isinstance(packets, list) or not 1 <= len(packets) <= 100
            or not all(validate_packet(packet).valid for packet in packets)
            or len({packet["episode_id"] for packet in packets}) != len(packets)
            or not isinstance(manifest, dict)):
        raise ValueError("invalid_review_packets")
    for packet in packets:
        coverage = manifest.get(packet["episode_id"], {})
        if not isinstance(coverage, dict) or coverage.get("packet_sha256") != digest(packet, sorted_keys=False):
            raise ValueError("missing_or_stale_coverage_manifest")


def review_order(packets):
    ordered = sorted(packets, key=lambda packet: packet["episode_id"])
    random.Random(SEED).shuffle(ordered)
    return ordered


def make_finding(packet, answers):
    """Copy exact text only for IDs explicitly selected by the reviewer."""
    by_id = {record["id"]: record for record in packet["records"]}
    def citations(field):
        ids = answers[field]
        if not isinstance(ids, list) or len(set(ids)) != len(ids) or any(record_id not in by_id for record_id in ids):
            raise ValueError("invalid_review_citation_ids")
        return [{"record_id": record_id, "quote": by_id[record_id]["text"]} for record_id in ids]
    verdict, category = answers["verdict"], answers["category"]
    if (verdict not in {"incident", "no_incident", "insufficient"} or category not in CATEGORIES
            or (verdict != "incident" and category != "none")):
        raise ValueError("inconsistent_review_verdict_category")
    for field in ("rationale", "benign_alternative"):
        if not isinstance(answers[field], str) or not 1 <= len(answers[field].strip()) <= 2000:
            raise ValueError("review_explanation_required_and_bounded")
    finding = {"verdict": verdict, "category": category, "implicated_agents": answers["implicated_agents"],
               "summary": answers["rationale"], "governing_rule": citations("governing_ids"),
               "supporting": citations("supporting_ids"), "counterevidence": citations("counterevidence_ids"),
               "benign_alternative": answers["benign_alternative"], "missing_context": answers["missing_context"],
               "first_violation_record_id": answers["first_violation_record_id"],
               "graph_edge_ids": answers["graph_edge_ids"]}
    if verdict == "no_incident" and not finding["supporting"]:
        raise ValueError("no_incident_requires_selected_evidence")
    if not validate_finding(packet, finding).valid:
        raise ValueError("review_finding_failed_structural_validation")
    return finding


def _rows(stream):
    if os.fstat(stream.fileno()).st_size > 32_000_000:
        raise ValueError("review_log_size_limit")
    stream.seek(0)
    return [json.loads(line) for line in stream if line.strip()]


def read_labels(path):
    if not path.exists():
        return []
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "r") as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        return _rows(stream)


def append_review(path, packet, coverage, finding, *, reviewer, position, order_hash):
    require_tty()
    if not validate_finding(packet, finding).valid:
        raise ValueError("invalid_annotation")
    row = {"episode_id": packet["episode_id"], "packet_sha256": digest(packet), "coverage_sha256": digest(coverage),
           "annotated_at_utc": datetime.now(timezone.utc).isoformat(), "reviewer_declared": reviewer,
           "review_source": "interactive_reviewer_attestation", "independent_gold": False,
           "order_seed": SEED, "queue_position": position, "review_order_sha256": order_hash,
           "packet_snapshot": packet, "coverage_snapshot": coverage, "finding": finding}
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "r+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.fchmod(stream.fileno(), 0o600)
        if any(old.get("episode_id") == row["episode_id"] and old.get("packet_sha256") == row["packet_sha256"]
               for old in _rows(stream)):
            raise ValueError("duplicate_annotation_rejected")
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def counts(packets, labels):
    current = {(packet["episode_id"], digest(packet)): packet for packet in packets}
    selected = [row for row in labels if (row.get("episode_id"), row.get("packet_sha256")) in current
                and row.get("review_source") == "interactive_reviewer_attestation"]
    unique = {(row["episode_id"], row["packet_sha256"]) for row in selected}
    if len(unique) != len(selected):
        raise ValueError("duplicate_annotation_log")
    if any(not validate_finding(current[(row["episode_id"], row["packet_sha256"])], row.get("finding")).valid
           for row in selected):
        raise ValueError("invalid_stored_review_finding")
    return {"cases": len(packets), "reviewer_attested_cases": len(selected), "remaining": len(packets) - len(selected),
            "verdict_counts": dict(Counter(row["finding"]["verdict"] for row in selected)),
            "unmatched_or_other_rows": len(labels) - len(selected), "independent_gold_cases": 0,
            "dataset_text_in_output": False, "note": "Reviewer declarations are not independently authenticated or adjudicated."}


def collect_answers():
    def ids(prompt):
        return [value.strip() for value in input(prompt + " (comma-separated; blank for none): ").split(",") if value.strip()]
    verdict = input("Verdict (incident/no_incident/insufficient): ").strip()
    category = input("Category (" + "/".join(sorted(CATEGORIES)) + "): ").strip() if verdict == "incident" else "none"
    answers = {"verdict": verdict, "category": category, "supporting_ids": ids("Supporting record IDs"),
               "governing_ids": ids("Governing instruction/goal IDs"), "counterevidence_ids": ids("Counterevidence IDs"),
               "implicated_agents": ids("Implicated agent IDs") if verdict == "incident" else [],
               "graph_edge_ids": ids("Direct/explicit-reference edge IDs") if verdict == "incident" else [],
               "first_violation_record_id": input("First violation record ID: ").strip() if verdict == "incident" else None,
               "rationale": input("Short rationale: ").strip(), "benign_alternative": input("Benign alternative considered: ").strip()}
    missing = input("Missing context (blank if none identified): ").strip()
    return {**answers, "missing_context": [missing] if missing else []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--review", action="store_true")
    for name in ("packets", "manifest", "labels"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reviewer", default="anonymous-declared", help="Optional declared pseudonym, not verified identity")
    args = parser.parse_args()
    if sys.platform != "linux" or not Path("/mnt/ai-swarm-scratch").is_mount():
        raise ValueError("azure_vm_required")
    if args.review:
        require_tty()  # Refuse redirected/automated input before any corpus file is read.
    if not all(path.resolve().is_relative_to(RESTRICTED) for path in (args.packets, args.manifest, args.labels)):
        raise ValueError("restricted_paths_required")
    if len({path.resolve() for path in (args.packets, args.manifest, args.labels)}) != 3:
        raise ValueError("review_inputs_and_labels_must_be_distinct")
    if not 1 <= len(args.reviewer.strip()) <= 120:
        raise ValueError("invalid_reviewer_pseudonym")
    packets, manifest = read_json(args.packets, jsonl=True), read_json(args.manifest)
    validate_inputs(packets, manifest)
    labels = read_labels(args.labels)
    if args.check:
        print(json.dumps(counts(packets, labels), sort_keys=True))
        return 0
    ordered = review_order(packets)
    order_hash = digest([packet["episode_id"] for packet in ordered])
    done = {(row.get("episode_id"), row.get("packet_sha256")) for row in labels}
    for position, packet in enumerate(ordered, 1):
        if (packet["episode_id"], digest(packet)) in done:
            continue
        coverage = manifest[packet["episode_id"]]
        print("\nCASE", position, "of", len(ordered), "— evidence may be incomplete; references can reject claims.")
        print(json.dumps({"packet": packet, "coverage": {key: value for key, value in coverage.items() if key in COVERAGE_FIELDS}},
                         ensure_ascii=False, indent=2))
        try:
            finding = make_finding(packet, collect_answers())
        except ValueError:
            print("Review not saved: identifiers, category, or incident evidence requirements did not validate.")
            continue
        if input("Save this reviewer-attested finding? Type yes: ").strip().lower() == "yes":
            append_review(args.labels, packet, coverage, finding, reviewer=args.reviewer,
                          position=position, order_hash=order_hash)
            print("Saved privately; this is a reviewer attestation, not independent gold.")
        if input("Continue to another case? Type yes: ").strip().lower() != "yes":
            break
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, TypeError, OSError, EOFError, KeyboardInterrupt):
        print(json.dumps({"status": "review_stopped_or_invalid_input", "error_details_include_dataset_text": False}))
        raise SystemExit(2)
