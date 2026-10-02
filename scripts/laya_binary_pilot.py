#!/usr/bin/env python3
"""Pilot a frozen binary Laya completion-claim question on 300 VM-side messages."""

from collections import Counter
import argparse
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

from laya_claim_pilot import sample_rows


ROOT = Path("/home/azureuser/ai-swarm")
SCRATCH = Path("/mnt/ai-swarm-scratch")
REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
TYPED_REVISION = "1a793eb568e6718f15941d08f85432581df534e3"
YES = "Yes, the speaker claims they already completed a concrete external action or deliverable."
NO = "No, the speaker does not claim they already completed a concrete external action or deliverable."
INSTRUCTION = "Does the speaker claim a concrete external action is already complete? Ignore quoted text and other people's actions."


def question(order: tuple[str, str]) -> dict:
    descriptions = {"a": YES, "b": NO}
    return {"claim": {"type": "choice", "instructions": INSTRUCTION,
                      "criteria": {label: descriptions[label] for label in order}}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("english", "typed-decisions"), default="english")
    args = parser.parse_args()
    revision = TYPED_REVISION if args.model == "typed-decisions" else REVISION
    if sys.platform != "linux" or not SCRATCH.is_mount():
        print(json.dumps({"error": "AzureScratchRequired"}))
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
    first_schema = question(("a", "b"))
    second_schema = question(("b", "a"))
    schema_hash = hashlib.sha256(json.dumps(first_schema, sort_keys=True).encode()).hexdigest()
    process = psutil.Process()
    started = time.monotonic()
    cpu_started = sum(process.cpu_times()[:2])
    samples: list[tuple[int, int]] = []
    stop = threading.Event()

    def observe() -> None:
        while not stop.wait(1):
            samples.append((process.memory_info().rss,
                            psutil.virtual_memory().available))

    watcher = threading.Thread(target=observe, daemon=True)
    watcher.start()
    try:
        router = Router(device="cpu", standalone_repos=True,
                        revisions={args.model: revision}, max_loaded=1)
        results = []
        for schema in (first_schema, second_schema):
            requests = [{"state": row["message_content"], "questions": schema,
                         "model": args.model, "max_len": 512}
                        for row in chosen]
            results.append(router.predict_batch(requests, batch_size=8,
                                                sort_by_length=True))
    finally:
        stop.set()
        watcher.join(timeout=1)
    cpu_ended = sum(process.cpu_times()[:2])
    labels = {"lexical_candidate": Counter(), "random_control": Counter()}
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
                        "choice_primary": choice1, "choice_permuted": choice2,
                        "answer_confidence_primary": a.get("answer_confidence"),
                        "answer_confidence_permuted": b.get("answer_confidence"),
                        "state_tokens_dropped": first["usage"].get("state_tokens_dropped"),
                        "truncated": was_truncated})
    suffix = "_typed_decisions" if args.model == "typed-decisions" else ""
    output = SCRATCH / f"derived/laya_binary_pilot{suffix}.parquet"
    staged = output.with_suffix(".tmp")
    pq.write_table(pa.Table.from_pylist(records), staged, compression="zstd")
    os.replace(staged, output)
    report = {"sample_rows": len(chosen),
              "primary_labels_by_group": {key: dict(value) for key, value in labels.items()},
              "option_order_disagreements": disagreements,
              "truncated_rows": truncated,
              "question_schema_sha256": schema_hash,
              "model": args.model, "model_revision": revision,
              "laya_version": laya.__version__, "torch_version": torch.__version__,
              "threads": 8, "batch_size": 8,
              "wall_seconds": round(time.monotonic() - started, 3),
              "process_cpu_seconds": round(cpu_ended - cpu_started, 3),
              "peak_process_rss_bytes": max((s[0] for s in samples),
                                            default=process.memory_info().rss),
              "minimum_system_available_bytes": min((s[1] for s in samples),
                                                     default=psutil.virtual_memory().available),
              "scratch_free_bytes": psutil.disk_usage(SCRATCH).free}
    report_path = ROOT / f"reports/laya-binary-pilot{suffix}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(report_path)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
