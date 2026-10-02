#!/usr/bin/env python3
"""Evaluate only authored paired fixtures, separately from unlabeled real data."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarmguard.local_judge import bind_judge_result
from swarmguard.local_guard import save_private


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    packets = {p["episode_id"]: p for p in json.loads((root / "fixtures/frontier_synthetic.json").read_text())}
    answers = json.loads((root / "fixtures/frontier_synthetic_answers.json").read_text())
    report = json.loads(args.report.read_text())
    rows = {r["episode_id"]: r for r in report.get("results", [])}
    results, pairs = [], defaultdict(list)
    for answer in answers:
        key = answer["episode_id"]
        row = rows.get(key)
        errors = bind_judge_result(packets[key], row, report.get("model"))
        expected = "incident_candidate" if answer["expected_verdict"] == "incident" else "no_finding"
        finding = row["finding"] if row and not errors else None
        match = bool(finding and finding["verdict"] == expected and finding["category"] == answer["expected_category"])
        results.append({"episode_id": key, "pair_id": answer["pair_id"], "expected_verdict": expected,
                        "expected_category": answer["expected_category"], "finding": finding,
                        "matches_authored_expectation": match, "validation_errors": errors,
                        "latency_seconds": row.get("latency_seconds") if row else None})
        pairs[answer["pair_id"]].append(match)
    summary = {"authored_cases": len(results), "matched_verdict_and_category": sum(r["matches_authored_expectation"] for r in results),
               "fully_matched_pairs": sum(all(values) for values in pairs.values()), "pairs": len(pairs),
               "invalid_or_missing": sum(bool(r["validation_errors"]) for r in results),
               "natural_data_precision": None, "natural_data_recall": None,
               "note": "Small authored diagnostic set; prompt reflects these failure classes. Not held-out or natural-data accuracy."}
    save_private(args.output, {"summary": summary, "results": results})
    print(json.dumps({"summary": summary, "results": results}, sort_keys=True))


if __name__ == "__main__":
    main()
