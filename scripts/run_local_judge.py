#!/usr/bin/env python3
"""Bounded, CPU-only Qwen contextual challenger on Azure; no model APIs."""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import resource
import signal
import sys
import threading
import time
import uuid

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from swarmguard.local_guard import GuardConfig, save_private
from swarmguard.local_judge import (JUDGE_MANIFEST, JUDGE_MODEL_REVISION, LocalJudge,
                                   judge_checkpoint_metadata, summarize_judge, get_model_profile)
from swarmguard.schema import validate_packet

ROOT = Path("/home/azureuser/ai-swarm")
SCRATCH = Path("/mnt/ai-swarm-scratch")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-profile", choices=("4b", "9b"), default="4b")
    p.add_argument("--model-path", type=Path)
    p.add_argument("--revision")
    p.add_argument("--model-manifest", type=Path)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--packets", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-episodes", type=int, default=8)
    p.add_argument("--max-input-tokens", type=int, default=16384)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-episode-seconds", type=int, default=180)
    p.add_argument("--max-runtime-seconds", type=int, default=2400)
    p.add_argument("--max-rss-gib", type=int)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--thinking", action="store_true", help="Enable native reasoning; discard its text before saving output")
    args = p.parse_args()
    profile = get_model_profile(args.model_profile)
    args.model_path = args.model_path or profile["model_path"]
    args.revision = args.revision or profile["revision"]
    args.model_manifest = args.model_manifest or profile["manifest"]
    args.max_rss_gib = args.max_rss_gib if args.max_rss_gib is not None else profile["max_rss_gib"]
    if sys.platform != "linux" or not SCRATCH.is_mount():
        raise ValueError("judge_requires_azure_vm")
    if not args.output.resolve().is_relative_to(ROOT / "restricted"):
        raise ValueError("judge_output_requires_restricted_directory")
    if args.synthetic:
        if args.packets:
            raise ValueError("synthetic_uses_fixed_authored_fixtures")
        packets_path = PROJECT / "fixtures/frontier_synthetic.json"
    else:
        if not args.packets or not args.packets.resolve().is_relative_to(ROOT / "restricted"):
            raise ValueError("real_packets_require_restricted_directory")
        packets_path = args.packets
    for path in (args.model_path, args.model_manifest):
        if not any(path.resolve().is_relative_to(root) for root in (ROOT, SCRATCH)):
            raise ValueError("model_paths_must_remain_on_vm")
    if args.output.resolve() in {packets_path.resolve(), args.model_manifest.resolve()}:
        raise ValueError("output_must_not_replace_input")
    config = GuardConfig(args.max_input_tokens, args.max_new_tokens, args.threads,
                         args.max_episodes, args.max_episode_seconds, "bfloat16")
    config.validate()
    if not 60 <= args.max_runtime_seconds <= 7200 or not 8 <= args.max_rss_gib <= 128:
        raise ValueError("invalid_resource_bound")
    if packets_path.stat().st_size > 4_000_000:
        raise ValueError("packet_file_size_limit")
    raw = packets_path.read_text()
    packets = json.loads(raw) if raw.lstrip().startswith("[") else [json.loads(v) for v in raw.splitlines() if v.strip()]
    if type(packets) is not list or not 1 <= len(packets) <= 20:
        raise ValueError("invalid_packet_count")
    packets = packets[:config.max_episodes]
    if any(not validate_packet(v).valid for v in packets) or len({v["episode_id"] for v in packets}) != len(packets):
        raise ValueError("invalid_or_duplicate_packets")
    import psutil
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_fd = os.open(args.output.with_suffix(".lock"), os.O_WRONLY | os.O_CREAT, 0o600)
    with os.fdopen(lock_fd, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        results = []
        report = {"schema_version": 1, "run_id": uuid.uuid4().hex, "status": "running",
                  "config": {**asdict(config), "thinking_enabled": args.thinking,
                             "model_profile": args.model_profile, "max_rss_gib": args.max_rss_gib}, "results": results,
                  "limitations": ["Valid JSON and source IDs do not establish semantic correctness.",
                                  "Candidates require human review; model intent judgments are not made.",
                                  "No hidden reasoning or free-form rationale is retained."]}
        # Mark the current run before hashing/loading so failure cannot expose a
        # prior completed report as the current execution.
        save_private(args.output, report)
        process = psutil.Process()
        stats = {"samples": 0, "peak_rss_gib": 0.0, "peak_process_cpu_percent": 0.0,
                 "minimum_available_ram_gib": psutil.virtual_memory().available / 1024**3,
                 "minimum_output_disk_free_gib": psutil.disk_usage(args.output.parent).free / 1024**3}
        stop = threading.Event()
        def interrupt(_signum, _frame):
            raise TimeoutError("resource_bound")
        signal.signal(signal.SIGALRM, interrupt)
        signal.alarm(args.max_runtime_seconds)
        def observe():
            process.cpu_percent()
            while not stop.is_set():
                rss = process.memory_info().rss / 1024**3
                stats["samples"] += 1
                stats["peak_rss_gib"] = max(stats["peak_rss_gib"], rss)
                stats["peak_process_cpu_percent"] = max(stats["peak_process_cpu_percent"], process.cpu_percent())
                stats["minimum_available_ram_gib"] = min(stats["minimum_available_ram_gib"], psutil.virtual_memory().available / 1024**3)
                stats["minimum_output_disk_free_gib"] = min(stats["minimum_output_disk_free_gib"], psutil.disk_usage(args.output.parent).free / 1024**3)
                if rss > args.max_rss_gib:
                    os.kill(os.getpid(), signal.SIGALRM)
                    return
                stop.wait(1)
        monitor = threading.Thread(target=observe, daemon=True)
        monitor.start()
        start, cpu_start = time.monotonic(), time.process_time()
        status = "completed"
        try:
            report["model"] = judge_checkpoint_metadata(args.model_path, args.revision, args.model_manifest,
                                                        model_profile=args.model_profile)
            report["model"]["thinking_enabled"] = args.thinking
            log_fd = os.open(args.output.with_suffix(".loader.log"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(log_fd, "w") as log, redirect_stdout(log), redirect_stderr(log):
                judge = LocalJudge(args.model_path, config, thinking=args.thinking, model_profile=args.model_profile)
                report["model"].update({"versions": judge.versions,
                                       "load_seconds": round(judge.load_seconds, 4),
                                       "load_cpu_seconds": round(judge.load_cpu_seconds, 4)})
                for packet in packets:
                    results.append(judge.run(packet))
                    report["summary"] = summarize_judge(results)
                    save_private(args.output, report)
        except TimeoutError:
            status = "resource_bound_exceeded"
        except Exception as exc:
            status = "execution_error"
            report["error_class"] = type(exc).__name__
        finally:
            signal.alarm(0)
            stop.set()
            monitor.join(timeout=2)
        report["status"] = status
        report["summary"] = {**summarize_judge(results), "status": status,
                             "total_wall_seconds": round(time.monotonic() - start, 4),
                             "total_cpu_seconds": round(time.process_time() - cpu_start, 4),
                             "resource_samples": {k: round(v, 4) for k, v in stats.items()},
                             "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 4)}
        save_private(args.output, report)
        print(json.dumps(report["summary"], sort_keys=True))
        return 0 if status == "completed" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "invalid_input_or_setup", "error_class": type(exc).__name__,
                          "dataset_or_model_text_in_stdout": False, "external_api_calls": 0}))
        raise SystemExit(2)
