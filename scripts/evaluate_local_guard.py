#!/usr/bin/env python3
"""Compare authored safety diagnostics; never use broad labels as swarm verdicts."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarmguard.local_guard import digest, save_private


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = json.loads(args.report.read_text())
    packets = {p["episode_id"]: p for p in json.loads((root / "fixtures/frontier_synthetic.json").read_text())}
    expected = json.loads((root / "fixtures/local_guard_expectations.json").read_text())
    rows = {r["episode_id"]: r for r in report.get("results", [])}
    results = []
    for case in expected["cases"]:
        row = rows.get(case["episode_id"], {})
        bound = row.get("packet_sha256") == digest(packets[case["episode_id"]])
        predicted = row.get("safety") if bound and row.get("status") == "classified" else None
        results.append({"episode_id": case["episode_id"], "expected_safety": case["safety"],
                        "predicted_safety": predicted, "matches_authored_expectation": predicted == case["safety"],
                        "latency_seconds": row.get("latency_seconds"), "errors": row.get("errors", [])})
    summary = {"authored_cases": len(results), "matched": sum(r["matches_authored_expectation"] for r in results),
               "insufficient": sum(r["predicted_safety"] is None for r in results),
               "mismatched_ids": [r["episode_id"] for r in results if r["predicted_safety"] is not None and not r["matches_authored_expectation"]],
               "scope": expected["scope"], "natural_data_accuracy": None, "swarm_detection_accuracy": None}
    save_private(args.output, {"summary": summary, "results": results})
    print(json.dumps({"summary": summary, "results": results}, sort_keys=True))


if __name__ == "__main__":
    main()
