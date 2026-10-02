#!/usr/bin/env python3
"""Freeze a blinded 300-case review queue on the Azure VM.

The queue has 100 highest-ranked lexical leads, 100 other lexical leads
selected by a stable hash, and 100 stable-hash controls. It contains IDs and
sampling metadata only; raw text remains in evidence_windows.parquet.
"""

import hashlib
import json
import os
from pathlib import Path
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq
import psutil


ROOT = Path("/home/azureuser/ai-swarm")
SCRATCH = Path("/mnt/ai-swarm-scratch")
DERIVED = SCRATCH / "derived"
SEED = "ai-swarm-review-v1-20261002"


def order_key(turn_id: str, suffix: str = "") -> bytes:
    return hashlib.sha256(f"{SEED}:{suffix}:{turn_id}".encode()).digest()


def main() -> int:
    if sys.platform != "linux" or not SCRATCH.is_mount():
        print(json.dumps({"error": "AzureScratchRequired"}))
        return 2
    start = time.monotonic()
    ranked = pq.read_table(DERIVED / "ranked_review_leads.parquet",
                           columns=["send_turn_id", "rank"]).to_pylist()
    if len(ranked) != 21100:
        raise RuntimeError("RankedLeadCountMismatch")
    top = sorted(ranked, key=lambda row: row["rank"])[:100]
    top_ids = {row["send_turn_id"] for row in top}
    lexical = sorted((row for row in ranked if row["send_turn_id"] not in top_ids),
                     key=lambda row: order_key(row["send_turn_id"], "lexical"))[:100]
    controls = []
    for batch in pq.ParquetFile(DERIVED / "evidence_windows.parquet").iter_batches(
            batch_size=4096, columns=["send_turn_id", "selection_group"]):
        controls.extend({"send_turn_id": row["send_turn_id"], "rank": None}
                        for row in batch.to_pylist()
                        if row["selection_group"] == "random_control")
    if len(controls) != 500:
        raise RuntimeError("ControlCountMismatch")
    controls = sorted(controls, key=lambda row: order_key(row["send_turn_id"],
                                                           "control"))[:100]
    selected = ([dict(row, sample_arm="top_ranked") for row in top]
                + [dict(row, sample_arm="lexical_random") for row in lexical]
                + [dict(row, sample_arm="random_control") for row in controls])
    if len(selected) != 300 or len({r["send_turn_id"] for r in selected}) != 300:
        raise RuntimeError("ReviewQueueCountMismatch")
    selected.sort(key=lambda row: order_key(row["send_turn_id"], "blind-order"))
    for index, row in enumerate(selected, 1):
        row["queue_position"] = index
    schema = pa.schema([("queue_position", pa.int32()), ("send_turn_id", pa.string()),
                        ("sample_arm", pa.string()), ("rank", pa.int32())])
    output = DERIVED / "review_queue.parquet"
    staged = output.with_suffix(".tmp")
    pq.write_table(pa.Table.from_pylist(selected, schema=schema), staged,
                   compression="zstd")
    os.replace(staged, output)
    report = {"sample_size": 300, "top_ranked": 100,
              "lexical_random": 100, "random_control": 100,
              "seed_sha256": hashlib.sha256(SEED.encode()).hexdigest(),
              "queue_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "wall_seconds": round(time.monotonic() - start, 3),
              "scratch_free_bytes": psutil.disk_usage(SCRATCH).free,
              "raw_text_in_queue": False}
    path = ROOT / "reports/review-queue.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    staged_report = path.with_suffix(".tmp")
    staged_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    staged_report.replace(path)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
