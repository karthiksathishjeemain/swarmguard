#!/usr/bin/env python3
"""Run pinned, frozen Laya over the blinded VM review queue.

Predictions are a secondary annotation, never a verdict or calibrated score.
Two answer orders are used; disagreement becomes an explicit abstention.
"""

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

import duckdb
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from laya_binary_pilot import question


ROOT = Path("/home/azureuser/ai-swarm")
SCRATCH = Path("/mnt/ai-swarm-scratch")
DERIVED = SCRATCH / "derived"
REVISION = "1a793eb568e6718f15941d08f85432581df534e3"


def main() -> int:
    if sys.platform != "linux" or not SCRATCH.is_mount():
        print(json.dumps({"error": "AzureScratchRequired"}))
        return 2
    os.environ["HF_HUB_CACHE"] = str(SCRATCH / "hf-hub-cache")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"
    import torch
    import laya
    from laya import Router

    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    queue = DERIVED / "review_queue.parquet"
    evidence = DERIVED / "evidence_windows.parquet"
    query = duckdb.query(f"""
        SELECT q.queue_position, q.send_turn_id, q.sample_arm,
               e.message_content
        FROM read_parquet('{queue}') q
        JOIN read_parquet('{evidence}') e USING (send_turn_id)
        ORDER BY q.queue_position
    """)
    names = [column[0] for column in query.description]
    rows = [dict(zip(names, values)) for values in query.fetchall()]
    if len(rows) != 300 or any(not isinstance(r["message_content"], str) for r in rows):
        raise RuntimeError("ReviewQueueJoinIncomplete")
    schemas = [question(("a", "b")), question(("b", "a"))]
    question_sha256 = hashlib.sha256(json.dumps(schemas[0], sort_keys=True).encode()).hexdigest()
    process = psutil.Process()
    start = time.monotonic()
    cpu_start = sum(process.cpu_times()[:2])
    snapshots = []
    stop = threading.Event()

    def watch() -> None:
        while not stop.wait(1):
            snapshots.append((process.memory_info().rss,
                              psutil.virtual_memory().available,
                              psutil.disk_usage(SCRATCH).free))

    monitor = threading.Thread(target=watch, daemon=True)
    monitor.start()
    try:
        router = Router(device="cpu", standalone_repos=True,
                        revisions={"typed-decisions": REVISION}, max_loaded=1)
        results = []
        for schema in schemas:
            requests = [{"state": row["message_content"], "questions": schema,
                         "model": "typed-decisions", "max_len": 512} for row in rows]
            results.append(router.predict_batch(requests, batch_size=8,
                                                sort_by_length=True))
    finally:
        stop.set()
        monitor.join(timeout=1)
    labels = Counter()
    arms = {arm: Counter() for arm in ("top_ranked", "lexical_random", "random_control")}
    records = []
    for row, first, permuted in zip(rows, *results):
        primary = first["answers"]["claim"]["choice"]
        secondary = permuted["answers"]["claim"]["choice"]
        stable = primary == secondary
        label = ("completed" if primary == "a" else "not_completed") if stable else "abstain"
        labels[label] += 1
        arms[row["sample_arm"]][label] += 1
        records.append({"queue_position": row["queue_position"],
                        "send_turn_id": row["send_turn_id"],
                        "laya_label": label,
                        "primary_choice": primary,
                        "permuted_choice": secondary,
                        "truncated": first["usage"].get("truncated") is True})
    output = DERIVED / "review_queue_laya.parquet"
    staged = output.with_suffix(".tmp")
    pq.write_table(pa.Table.from_pylist(records), staged, compression="zstd")
    os.replace(staged, output)
    cpu_end = sum(process.cpu_times()[:2])
    report = {"sample_size": len(records), "model": "typed-decisions",
              "model_revision": REVISION, "laya_version": laya.__version__,
              "torch_version": torch.__version__, "question_sha256": question_sha256,
              "labels": dict(labels),
              "labels_by_arm": {arm: dict(counter) for arm, counter in arms.items()},
              "option_order_disagreements": labels["abstain"],
              "truncated_rows": sum(r["truncated"] for r in records),
              "wall_seconds": round(time.monotonic() - start, 3),
              "process_cpu_seconds": round(cpu_end - cpu_start, 3),
              "peak_process_rss_bytes": max((x[0] for x in snapshots),
                                            default=process.memory_info().rss),
              "minimum_system_available_bytes": min((x[1] for x in snapshots),
                                                    default=psutil.virtual_memory().available),
              "minimum_scratch_free_bytes": min((x[2] for x in snapshots),
                                                default=psutil.disk_usage(SCRATCH).free)}
    path = ROOT / "reports/review-queue-laya.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temp.replace(path)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
