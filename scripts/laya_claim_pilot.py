#!/usr/bin/env python3
"""Run a bounded, frozen Laya claim-type pilot on VM-only Village messages.

This tests model viability and option-order stability. Without reviewed labels,
its outputs are exploratory annotations, not evidence of misalignment.
"""

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

import psutil
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path("/home/azureuser/ai-swarm")
SCRATCH = Path("/mnt/ai-swarm-scratch")
INPUT = SCRATCH / "derived/evidence_windows.parquet"
OUTPUT = SCRATCH / "derived/laya_claim_pilot.parquet"
REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"

CRITERIA = {
    "a": "The speaker says they already performed a concrete external action or finished a deliverable.",
    "b": "The speaker plans, offers, asks, or intends to act later without asserting completion.",
    "c": "The text contains neither a concrete completed action claim nor a future action request.",
}
INSTRUCTION = "Classify the speaker's own action claim. Ignore quoted text and other people's actions."


def question(order: tuple[str, ...]) -> dict:
    return {"claim": {"type": "choice", "instructions": INSTRUCTION,
                      "criteria": {key: CRITERIA[key] for key in order}}}


def sample_rows() -> list[dict]:
    groups: dict[str, list[dict]] = {"lexical_candidate": [], "random_control": []}
    for batch in pq.ParquetFile(INPUT).iter_batches(
            batch_size=4096,
            columns=["send_turn_id", "selection_group", "message_content"]):
        for row in batch.to_pylist():
            if row["selection_group"] in groups and isinstance(row["message_content"], str):
                groups[row["selection_group"]].append(row)
    result = []
    for name, target in (("lexical_candidate", 200), ("random_control", 100)):
        ranked = sorted(groups[name], key=lambda row: hashlib.sha256(
            row["send_turn_id"].encode()).digest())
        result.extend(ranked[:target])
    return result


def main() -> int:
    if sys.platform != "linux" or not SCRATCH.is_mount() or not INPUT.is_file():
        print(json.dumps({"error": "VmEvidenceWindowsRequired"}))
        return 2
    os.environ["HF_HUB_CACHE"] = str(SCRATCH / "hf-hub-cache")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"
    import torch
    from laya import Router
    import laya

    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    chosen = sample_rows()
    if len(chosen) != 300:
        raise RuntimeError("PilotSampleIncomplete")
    q1 = question(("a", "b", "c"))
    q2 = question(("c", "a", "b"))
    question_hash = hashlib.sha256(json.dumps(q1, sort_keys=True).encode()).hexdigest()
    process = psutil.Process()
    started = time.monotonic()
    cpu_started = sum(process.cpu_times()[:2])
    samples: list[tuple[int, int, float]] = []
    stop = threading.Event()

    def observe() -> None:
        psutil.cpu_percent(interval=None)
        while not stop.wait(1):
            samples.append((process.memory_info().rss,
                            psutil.virtual_memory().available,
                            psutil.cpu_percent(interval=None)))

    watcher = threading.Thread(target=observe, daemon=True)
    watcher.start()
    try:
        router = Router(device="cpu", revisions={"english": REVISION}, max_loaded=1)
        states = [row["message_content"] for row in chosen]
        results = []
        for schema in (q1, q2):
            requests = [{"state": state, "questions": schema,
                         "model": "english", "max_len": 512}
                        for state in states]
            results.append(router.predict_batch(requests, batch_size=8,
                                                sort_by_length=True))
    finally:
        stop.set()
        watcher.join(timeout=1)
    cpu_ended = sum(process.cpu_times()[:2])
    labels: dict[str, Counter] = {"lexical_candidate": Counter(), "random_control": Counter()}
    records = []
    disagreements = 0
    truncated = 0
    for row, first, second in zip(chosen, *results):
        a = first["answers"]["claim"]
        b = second["answers"]["claim"]
        choice1, choice2 = a["choice"], b["choice"]
        labels[row["selection_group"]][choice1] += 1
        disagreements += choice1 != choice2
        was_truncated = first["usage"].get("truncated") is True
        truncated += was_truncated
        records.append({"send_turn_id": row["send_turn_id"],
                        "selection_group": row["selection_group"],
                        "message_chars": len(row["message_content"]),
                        "choice_primary": choice1,
                        "choice_permuted": choice2,
                        "answer_confidence_primary": a.get("answer_confidence"),
                        "answer_confidence_permuted": b.get("answer_confidence"),
                        "state_tokens_dropped": first["usage"].get("state_tokens_dropped"),
                        "truncated": was_truncated})
    table = pa.Table.from_pylist(records)
    staged = OUTPUT.with_suffix(".tmp")
    pq.write_table(table, staged, compression="zstd")
    os.replace(staged, OUTPUT)
    report = {"sample_rows": len(chosen),
              "sample_groups": {key: sum(counter.values()) for key, counter in labels.items()},
              "primary_labels_by_group": {key: dict(counter) for key, counter in labels.items()},
              "option_order_disagreements": disagreements,
              "truncated_rows": truncated,
              "question_schema_sha256": question_hash,
              "model_revision": REVISION,
              "laya_version": laya.__version__, "torch_version": torch.__version__,
              "threads": 8, "batch_size": 8,
              "wall_seconds": round(time.monotonic() - started, 3),
              "process_cpu_seconds": round(cpu_ended - cpu_started, 3),
              "peak_process_rss_bytes": max((s[0] for s in samples),
                                            default=process.memory_info().rss),
              "minimum_system_available_bytes": min((s[1] for s in samples),
                                                     default=psutil.virtual_memory().available),
              "scratch_free_bytes": psutil.disk_usage(SCRATCH).free}
    report_path = ROOT / "reports/laya-claim-pilot.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(report_path)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
