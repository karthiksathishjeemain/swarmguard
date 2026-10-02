#!/usr/bin/env python3
"""Compare bound model findings with reviewer attestations; never call them gold.

Only aggregates are emitted. Abstentions are separate from negatives. Selective
recall excludes abstained positive references, whose count is reported explicitly.
This selected review set does not establish deployment-wide detection quality.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarmguard.local_guard import digest, save_private
from swarmguard.local_judge import bind_judge_result
from swarmguard.schema import validate_finding, validate_packet

RESTRICTED = Path("/home/azureuser/ai-swarm/restricted")


def ratio(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None}


def evaluate(packets, labels, report):
    if (type(packets) is not list or not 1 <= len(packets) <= 100
            or any(not validate_packet(packet).valid for packet in packets)
            or len({packet["episode_id"] for packet in packets}) != len(packets)
            or type(labels) is not list or len(labels) > 1000):
        raise ValueError("invalid_scoring_inputs")
    indexed = {packet["episode_id"]: packet for packet in packets}
    references, seen = {}, set()
    excluded_insufficient = excluded_other = 0
    for row in labels:
        if type(row) is not dict:
            raise ValueError("invalid_reference_row")
        if row.get("review_source") != "interactive_reviewer_attestation":
            excluded_other += 1
            continue
        episode_id = row.get("episode_id")
        if type(episode_id) is not str or episode_id not in indexed:
            raise ValueError("reference_episode_not_in_packets")
        if episode_id in seen:
            raise ValueError("duplicate_reference_annotation")
        seen.add(episode_id)
        packet, snapshot = indexed[episode_id], row.get("packet_snapshot")
        if (not validate_packet(snapshot).valid or digest(packet) != row.get("packet_sha256")
                or digest(snapshot) != row.get("packet_sha256")):
            raise ValueError("stale_reference_packet_hash")
        if (not validate_finding(packet, row.get("finding")).valid or row.get("independent_gold") is not False
                or type(row.get("reviewer_declared")) is not str or not row["reviewer_declared"].strip()):
            raise ValueError("invalid_reviewer_attested_finding")
        try:
            at = datetime.fromisoformat(row["annotated_at_utc"].replace("Z", "+00:00"))
            if at.tzinfo is None:
                raise ValueError()
        except (KeyError, AttributeError, TypeError, ValueError):
            raise ValueError("invalid_reference_timestamp") from None
        verdict = row["finding"]["verdict"]
        if verdict == "insufficient":
            excluded_insufficient += 1
        else:
            references[episode_id] = verdict == "incident"
    summary = {"status": "pending_reference_review" if not references else "reviewer_reference_comparison",
               "packets": len(packets), "reviewer_attested_references": len(references),
               "excluded_insufficient_references": excluded_insufficient, "excluded_other_source_rows": excluded_other,
               "independent_gold_cases": 0, "confusion_vs_reviewer_attested": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
               "model_abstained_or_unavailable": 0, "reference_positive_abstained": 0,
               "reference_positive_cases": sum(references.values()), "model_binding_issue_counts": {},
               "coverage": ratio(0, len(references)), "precision": ratio(0, 0), "selective_recall": ratio(0, 0),
               "dataset_text_or_ids_in_output": False,
               "note": "Comparison is against reviewer attestations, not independent gold. Selective recall excludes abstentions; selected-case results do not establish population performance."}
    if not references:
        return summary
    if report is None:
        report = {"results": [], "model": {}}
    if type(report) is not dict or type(report.get("results")) is not list:
        raise ValueError("invalid_judge_report")
    model_rows, binding_errors = {}, {}
    for row in report["results"]:
        if type(row) is not dict or type(row.get("episode_id")) is not str or row["episode_id"] in model_rows:
            raise ValueError("invalid_or_duplicate_model_episode")
        if row["episode_id"] not in indexed:
            raise ValueError("model_episode_outside_packet_scope")
        model_rows[row["episode_id"]] = row
        binding_errors[row["episode_id"]] = bind_judge_result(indexed[row["episode_id"]], row, report.get("model"))
    issues = Counter()
    for episode_id, positive_reference in references.items():
        row = model_rows.get(episode_id)
        errors = binding_errors[episode_id] if row is not None else bind_judge_result(indexed[episode_id], None, report.get("model"))
        verdict = None if errors else row["finding"]["verdict"]
        if errors or verdict not in {"incident_candidate", "no_finding"}:
            summary["model_abstained_or_unavailable"] += 1
            summary["reference_positive_abstained"] += int(positive_reference)
            issues.update(errors or ["model_returned_insufficient"])
            continue
        predicted = verdict == "incident_candidate"
        key = "tp" if predicted and positive_reference else "fp" if predicted else "fn" if positive_reference else "tn"
        summary["confusion_vs_reviewer_attested"][key] += 1
    confusion = summary["confusion_vs_reviewer_attested"]
    summary.update(model_binding_issue_counts=dict(issues), coverage=ratio(sum(confusion.values()), len(references)),
                   precision=ratio(confusion["tp"], confusion["tp"] + confusion["fp"]),
                   selective_recall=ratio(confusion["tp"], confusion["tp"] + confusion["fn"]))
    return summary


def read_json(path, *, rows=False, missing_ok=False):
    if missing_ok and not path.exists():
        return [] if rows else None
    if path.stat().st_size > 32_000_000:
        raise ValueError("bounded_scoring_input_exceeded")
    raw = path.read_text()
    return ([json.loads(line) for line in raw.splitlines() if line.strip()]
            if rows and not raw.lstrip().startswith("[") else json.loads(raw))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("packets", "labels", "judge-report", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    paths = (args.packets, args.labels, args.judge_report, args.output)
    if sys.platform != "linux" or not Path("/mnt/ai-swarm-scratch").is_mount():
        raise ValueError("azure_vm_required")
    if not all(path.resolve().is_relative_to(RESTRICTED) for path in paths) or len({path.resolve() for path in paths}) != 4:
        raise ValueError("distinct_restricted_paths_required")
    packets, labels = read_json(args.packets, rows=True), read_json(args.labels, rows=True, missing_ok=True)
    report = read_json(args.judge_report, missing_ok=True) if labels else None
    summary = evaluate(packets, labels, report)
    summary["scored_at_utc"] = datetime.now(timezone.utc).isoformat()
    save_private(args.output, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, TypeError, OSError):
        print(json.dumps({"status": "invalid_reference_or_model_inputs", "dataset_text_or_ids_in_output": False}))
        raise SystemExit(2)
