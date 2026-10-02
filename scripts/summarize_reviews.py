#!/usr/bin/env python3
"""Aggregate VM-only human annotations without exporting corpus records."""

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys

import pyarrow.parquet as pq


ROOT = Path("/home/azureuser/ai-swarm")
DERIVED = Path("/mnt/ai-swarm-scratch/derived")
ANNOTATIONS = ROOT / "restricted/review_annotations.jsonl"


def fraction(numerator: int, denominator: int) -> dict | None:
    if denominator == 0:
        return None
    point = numerator / denominator
    z = 1.96
    scale = 1 + z * z / denominator
    center = (point + z * z / (2 * denominator)) / scale
    half = z * math.sqrt(point * (1 - point) / denominator
                         + z * z / (4 * denominator * denominator)) / scale
    return {"numerator": numerator, "denominator": denominator,
            "fraction": round(point, 4),
            "wilson_95_low": round(max(0, center - half), 4),
            "wilson_95_high": round(min(1, center + half), 4)}


def main() -> int:
    if sys.platform != "linux" or not Path("/mnt/ai-swarm-scratch").is_mount():
        print(json.dumps({"error": "AzureScratchRequired"}))
        return 2
    queue = {r["queue_position"]: r for r in
             pq.read_table(DERIVED / "review_queue.parquet").to_pylist()}
    laya = {r["queue_position"]: r for r in
            pq.read_table(DERIVED / "review_queue_laya.parquet").to_pylist()}
    annotations = {}
    if ANNOTATIONS.is_file():
        for line in ANNOTATIONS.read_text().splitlines():
            if line:
                row = json.loads(line)
                annotations[row["queue_position"]] = row
    if len(queue) != 300 or len(laya) != 300:
        raise RuntimeError("QueueIncomplete")
    by_arm = defaultdict(list)
    for position, annotation in annotations.items():
        if position not in queue or position not in laya:
            raise RuntimeError("AnnotationOutsideQueue")
        by_arm[queue[position]["sample_arm"]].append((annotation, laya[position]))
    details = {}
    for arm in ("top_ranked", "lexical_random", "random_control"):
        rows = by_arm[arm]
        determinate = [(a, m) for a, m in rows if a["claim_label"] in {"c", "n"}]
        positive = [(a, m) for a, m in determinate if m["laya_label"] == "completed"]
        actual_completed = [(a, m) for a, m in determinate if a["claim_label"] == "c"]
        contradictions = sum(a["evidence_label"] == "c" for a, _ in rows)
        details[arm] = {
            "reviewed": len(rows),
            "claim_labels": dict(Counter(a["claim_label"] for a, _ in rows)),
            "evidence_labels": dict(Counter(a["evidence_label"] for a, _ in rows
                                            if a["evidence_label"])),
            "laya_labels": dict(Counter(m["laya_label"] for _, m in rows)),
            "laya_positive_precision_on_reviewed_determinate": fraction(
                sum(a["claim_label"] == "c" for a, _ in positive), len(positive)),
            "laya_positive_recall_on_reviewed_determinate": fraction(
                sum(m["laya_label"] == "completed" for _, m in actual_completed),
                len(actual_completed)),
            "verified_contradictions_per_100_reviewed":
                round(100 * contradictions / len(rows), 2) if rows else None,
        }
    report = {
        "reviewed_total": len(annotations), "queue_total": 300,
        "by_sampling_arm": details,
        "limits": [
            "These are within-arm reviewed-case estimates, not corpus recall or a population misalignment rate.",
            "Laya abstains when option orders disagree; ambiguous human labels are excluded from precision/recall denominators.",
            "Wilson intervals assume independent cases and may be too narrow for clustered sessions or agents.",
            "A contradicted claim is not by itself evidence of intentional misalignment.",
        ],
        "raw_text_in_report": False,
    }
    output = ROOT / "reports/review-metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.with_suffix(".tmp")
    staged.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    staged.replace(output)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
