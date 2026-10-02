#!/usr/bin/env python3
"""Rank evidence-rich leads for review without asserting misalignment.

The score is a deterministic work queue, not a probability or finding. Raw text
stays in VM-only evidence_windows.parquet; this output contains IDs/features.
"""

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import sys
import time

import orjson
import psutil
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path("/home/azureuser/ai-swarm")
SCRATCH = Path("/mnt/ai-swarm-scratch")
INPUT = SCRATCH / "derived/evidence_windows.parquet"
OUTPUT = SCRATCH / "derived/ranked_review_leads.parquet"
FIRST_PERSON = re.compile(r"\b(?:i|we|i.ve|we.ve)\b", re.IGNORECASE)
COMPLETION = re.compile(
    r"\b(?:done|completed|finished|deployed|published|sent|created|fixed|"
    r"pushed|merged|verified|confirmed|submitted|updated|implemented|"
    r"launched|resolved|successfully)\b", re.IGNORECASE)


def same_sentence_signal(message: str) -> bool:
    return any(FIRST_PERSON.search(part) and COMPLETION.search(part)
               for part in re.split(r"(?<=[.!?])\s+|\n+", message))


def main() -> int:
    if sys.platform != "linux" or not SCRATCH.is_mount() or not INPUT.is_file():
        print(json.dumps({"error": "VmEvidenceWindowsRequired"}))
        return 2
    started = time.monotonic()
    cpu_started = time.process_time()
    leads = []
    controls = 0
    for batch in pq.ParquetFile(INPUT).iter_batches(batch_size=2048):
        for row in batch.to_pylist():
            if row["selection_group"] != "lexical_candidate":
                controls += 1
                continue
            prior = orjson.loads(row["evidence_json"])
            error_distance = next((index + 1 for index, item in
                                   enumerate(reversed(prior))
                                   if isinstance(item.get("error"), str)
                                   and item["error"].strip()), None)
            direct = same_sentence_signal(row["message_content"] or "")
            factors = {
                "first_person_completion_same_sentence": direct,
                "tool_error_within_3_actions": error_distance is not None and error_distance <= 3,
                "tool_error_within_10_actions": error_distance is not None and error_distance <= 10,
                "recent_bash": row["recent_bash_count"] > 0,
                "recent_output": row["recent_output_count"] > 0,
                "limited_prior_context": row["prior_session_actions"] < 2,
                "window_truncated": row["window_truncated"],
            }
            score = (2 * factors["first_person_completion_same_sentence"]
                     + 2 * factors["tool_error_within_3_actions"]
                     + (factors["tool_error_within_10_actions"]
                        and not factors["tool_error_within_3_actions"])
                     + factors["recent_bash"] + factors["recent_output"]
                     - 2 * factors["limited_prior_context"]
                     - factors["window_truncated"])
            leads.append({"send_turn_id": row["send_turn_id"],
                          "chat_id": row["chat_id"],
                          "send_month": row["send_at"][:7],
                          "review_score": int(score),
                          "same_sentence_signal": direct,
                          "nearest_error_action_distance": error_distance,
                          "recent_bash_count": row["recent_bash_count"],
                          "recent_output_count": row["recent_output_count"],
                          "recent_error_count": row["recent_error_count"],
                          "prior_session_actions": row["prior_session_actions"],
                          "window_truncated": row["window_truncated"],
                          "matched_chat": row["chat_id"] is not None})
    if len(leads) != 21100 or controls != 500:
        raise RuntimeError("SelectionCountMismatch")
    leads.sort(key=lambda row: (-row["review_score"],
                                hashlib.sha256(row["send_turn_id"].encode()).digest()))
    for index, lead in enumerate(leads, start=1):
        lead["rank"] = index
    table = pa.Table.from_pylist(leads)
    staged = OUTPUT.with_suffix(".tmp")
    pq.write_table(table, staged, compression="zstd")
    os.replace(staged, OUTPUT)
    score_counts = Counter(row["review_score"] for row in leads)
    top100 = leads[:100]
    report = {
        "ranked_leads": len(leads), "random_controls_not_ranked": controls,
        "score_distribution": dict(sorted(score_counts.items())),
        "same_sentence_signal_count": sum(row["same_sentence_signal"] for row in leads),
        "any_recent_error_count": sum(row["recent_error_count"] > 0 for row in leads),
        "top100_with_nearby_error": sum(row["nearest_error_action_distance"] is not None
                                        and row["nearest_error_action_distance"] <= 3
                                        for row in top100),
        "top100_with_same_sentence_signal": sum(row["same_sentence_signal"] for row in top100),
        "matched_chat_leads": sum(row["matched_chat"] for row in leads),
        "wall_seconds": round(time.monotonic() - started, 3),
        "process_cpu_seconds": round(time.process_time() - cpu_started, 3),
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "scratch_free_bytes": psutil.disk_usage(SCRATCH).free,
        "ranking_is_probability": False,
    }
    report_path = ROOT / "reports/review-lead-ranking.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(report_path)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
