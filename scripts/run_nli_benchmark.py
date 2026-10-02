#!/usr/bin/env python3
"""Benchmark frozen local NLI on authored pairs on the Azure VM, with no API calls."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarmguard.nli import LABELS, LocalNLIScorer, MODEL_ID


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--revision", required=True, help="Pinned 40-character Hugging Face commit SHA")
    parser.add_argument("--fixtures", type=Path, default=Path(__file__).resolve().parents[1] / "fixtures/nli_pairs.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--minimum-score", type=float, default=0.80)
    args = parser.parse_args()
    if not Path("/home/azureuser/ai-swarm").is_dir() or platform.system() != "Linux":
        parser.error("Model benchmarks must run on the Azure VM, not the local Mac.")
    if len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision):
        parser.error("--revision must be a full lowercase commit SHA")
    if not 1 <= args.repeat <= 10:
        parser.error("--repeat must be between 1 and 10")
    raw = args.fixtures.read_bytes()
    fixture = json.loads(raw)
    pairs = fixture["pairs"]
    if not pairs or len(pairs) > 100 or len({p["id"] for p in pairs}) != len(pairs):
        parser.error("Expected 1 to 100 unique author-written diagnostic pairs")
    for pair in pairs:
        if pair["expected"] not in LABELS or not pair["premise"] or not pair["hypothesis"]:
            parser.error("Invalid diagnostic pair")

    import psutil
    proc = psutil.Process()
    started = time.perf_counter()
    initial_cpu = proc.cpu_times()
    stop = threading.Event()
    samples = []

    def monitor():
        while not stop.is_set():
            samples.append({"rss_bytes": proc.memory_info().rss,
                            "system_cpu_percent": psutil.cpu_percent(interval=None),
                            "available_memory_bytes": psutil.virtual_memory().available})
            stop.wait(.1)

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        load_started = time.perf_counter()
        scorer = LocalNLIScorer(args.model_path, threads=args.threads,
                               minimum_score=args.minimum_score, expected_revision=args.revision)
        load_seconds = time.perf_counter() - load_started
        warmup_started = time.perf_counter()
        scorer.score_pair(pairs[0]["premise"], pairs[0]["hypothesis"])
        warmup_seconds = time.perf_counter() - warmup_started
        inference_started = time.perf_counter()
        results = []
        for repetition in range(args.repeat):
            for pair in pairs:
                result = scorer.score_pair(pair["premise"], pair["hypothesis"])
                results.append({"id": pair["id"], "group": pair["group"],
                                "repetition": repetition, "expected": pair["expected"], **result})
        inference_seconds = time.perf_counter() - inference_started
    finally:
        stop.set()
        watcher.join()
    elapsed = time.perf_counter() - started
    final_cpu = proc.cpu_times()
    cpu_seconds = (final_cpu.user + final_cpu.system - initial_cpu.user - initial_cpu.system)
    scored = [r for r in results if r["status"] == "scored"]
    correct = sum(r["label"] == r["expected"] for r in scored)
    durations = [r["elapsed_seconds"] for r in results]
    disk = shutil.disk_usage(args.model_path)
    metadata = dict(scorer.metadata, revision=args.revision,
                    config_sha256=hashlib.sha256((args.model_path / "config.json").read_bytes()).hexdigest())
    aggregate = {
        "model_id": MODEL_ID, "revision": args.revision, "pairs": len(pairs),
        "evaluations": len(results), "scored": len(scored), "abstained": len(results) - len(scored),
        "correct_scored": correct,
        "diagnostic_accuracy_with_abstention_as_incorrect": correct / len(results),
        "diagnostic_accuracy_on_scored": correct / len(scored) if scored else None,
        "predictions": dict(Counter(r["label"] or "insufficient" for r in results)),
        "mismatched_or_abstained_fixture_ids": sorted({r["id"] for r in results if r["label"] != r["expected"]}),
        "load_seconds": round(load_seconds, 4), "warmup_seconds": round(warmup_seconds, 4),
        "inference_seconds": round(inference_seconds, 4), "total_seconds": round(elapsed, 4),
        "median_pair_seconds": round(statistics.median(durations), 6),
        "maximum_pair_seconds": max(durations), "cpu_seconds": round(cpu_seconds, 4),
        "mean_process_cpu_percent_one_core_100": round(cpu_seconds / elapsed * 100, 2),
        "peak_rss_bytes": max([proc.memory_info().rss] + [s["rss_bytes"] for s in samples]),
        "minimum_available_memory_bytes": min([psutil.virtual_memory().available] + [s["available_memory_bytes"] for s in samples]),
        "system_cpu_percent_sample_mean": statistics.mean(s["system_cpu_percent"] for s in samples) if samples else None,
        "model_files_bytes": sum(p.stat().st_size for p in args.model_path.rglob("*") if p.is_file() and ".cache" not in p.parts),
        "disk_free_bytes": disk.free, "threads": args.threads, "external_inference_calls": 0,
        "note": "Authored diagnostic pairs only. These semantic scores and this small test set do not establish swarm misalignment accuracy. Actor/artifact/chronology checks remain necessary.",
    }
    report = {"created_at": datetime.now(timezone.utc).isoformat(),
              "fixtures_sha256": hashlib.sha256(raw).hexdigest(), "model": metadata,
              "aggregate": aggregate, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as out:
        os.fchmod(out.fileno(), 0o600)
        json.dump(report, out, indent=2, sort_keys=True)
        out.write("\n")
    temporary.replace(args.output)
    print(json.dumps(aggregate, sort_keys=True))


if __name__ == "__main__":
    main()
