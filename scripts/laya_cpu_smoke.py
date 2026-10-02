#!/usr/bin/env python3
"""Bounded, synthetic Laya CPU smoke benchmark for the Azure VM.

Run only on the VM. This deliberately reads no AI Village records and makes no
claim about task accuracy; it measures setup cost and repeat-call throughput.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time

import psutil


STATE = (
    "Hi, we were billed twice for March. Please refund the duplicate today "
    "or we will cancel our plan."
)
QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "other": "everything else",
        },
    }
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="Pinned Laya model commit SHA")
    parser.add_argument("--cache", default="/mnt/ai-swarm-scratch/hf-hub-cache")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if sys.platform != "linux" or not Path("/mnt/ai-swarm-scratch").is_mount():
        parser.error("This benchmark requires the mounted Azure VM scratch disk")
    if args.threads < 1 or args.repeat < 1 or len(args.revision) != 40:
        parser.error("Use a 40-character revision, positive threads, and positive repeats")
    cache = Path(args.cache)
    if not cache.is_relative_to("/mnt/ai-swarm-scratch"):
        parser.error("Cache must remain on the VM scratch mount")
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_CACHE"] = str(cache)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)

    import torch
    from laya import Router
    import laya

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    process = psutil.Process()
    samples: list[tuple[int, int, float]] = []
    stop = threading.Event()

    def observe() -> None:
        psutil.cpu_percent(interval=None)
        while not stop.wait(0.25):
            samples.append((
                process.memory_info().rss,
                psutil.virtual_memory().available,
                psutil.cpu_percent(interval=None),
            ))

    watcher = threading.Thread(target=observe, daemon=True)
    watcher.start()
    start_wall = time.monotonic()
    start_cpu = sum(process.cpu_times()[:2])
    start_io = process.io_counters()
    router = Router(device="cpu", revisions={"english": args.revision}, max_loaded=1)
    answers = []
    first_end = None
    try:
        for index in range(args.repeat):
            result = router.predict(STATE, QUESTIONS, model="english", max_len=512)
            answers.append(result["answers"]["department"]["choice"])
            if index == 0:
                first_end = time.monotonic()
    finally:
        stop.set()
        watcher.join(timeout=1)

    end_wall = time.monotonic()
    end_cpu = sum(process.cpu_times()[:2])
    end_io = process.io_counters()
    usage = psutil.disk_usage("/mnt/ai-swarm-scratch")
    print(json.dumps({
        "event": "laya_cpu_smoke",
        "model_revision": args.revision,
        "laya_version": laya.__version__,
        "torch_version": torch.__version__,
        "threads": args.threads,
        "repeat": args.repeat,
        "answers": answers,
        "first_call_seconds": round(first_end - start_wall, 3) if first_end else None,
        "warm_calls_seconds": round(end_wall - first_end, 3) if first_end else None,
        "total_wall_seconds": round(end_wall - start_wall, 3),
        "process_cpu_seconds": round(end_cpu - start_cpu, 3),
        "peak_process_rss_bytes": max((s[0] for s in samples), default=process.memory_info().rss),
        "minimum_system_available_bytes": min((s[1] for s in samples), default=psutil.virtual_memory().available),
        "mean_system_cpu_percent": round(sum(s[2] for s in samples) / len(samples), 2) if samples else None,
        "process_read_bytes": end_io.read_bytes - start_io.read_bytes,
        "process_write_bytes": end_io.write_bytes - start_io.write_bytes,
        "scratch_free_bytes": usage.free,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
