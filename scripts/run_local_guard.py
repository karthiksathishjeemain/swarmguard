#!/usr/bin/env python3
"""Bounded VM-only AgentDoG CPU pilot. Output stdout contains aggregates only."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarmguard.local_guard import (GuardConfig, LocalGuard, MODEL_MANIFEST, checkpoint_metadata,
                                   save_private, summarize)
from swarmguard.schema import validate_packet

VM_ROOT = Path("/home/azureuser/ai-swarm")
SCRATCH = Path("/mnt/ai-swarm-scratch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--model-manifest", type=Path, default=MODEL_MANIFEST)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-episodes", type=int, default=8)
    parser.add_argument("--max-episode-seconds", type=int, default=180)
    parser.add_argument("--max-runtime-seconds", type=int, default=2400)
    parser.add_argument("--max-rss-gib", type=int, default=48)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    args = parser.parse_args()
    config = GuardConfig(args.max_input_tokens, args.max_new_tokens, args.threads,
                         args.max_episodes, args.max_episode_seconds, args.dtype)
    config.validate()
    if not 60 <= args.max_runtime_seconds <= 7200 or not 8 <= args.max_rss_gib <= 128:
        raise ValueError("invalid_resource_bound")
    if sys.platform != "linux" or not SCRATCH.is_mount():
        raise ValueError("local_inference_requires_azure_vm")
    for path in (args.packets, args.output, args.model_path, args.model_manifest):
        if not any(path.resolve().is_relative_to(root) for root in (VM_ROOT, SCRATCH)):
            raise ValueError("path_must_remain_on_vm")
    if args.output.resolve() == args.packets.resolve():
        raise ValueError("output_must_not_replace_input")
    if args.packets.stat().st_size > 4_000_000:
        raise ValueError("packet_file_too_large")
    raw = args.packets.read_text()
    packets = (json.loads(raw) if raw.lstrip().startswith("[") else
               [json.loads(line) for line in raw.splitlines() if line.strip()])
    if not isinstance(packets, list) or not 1 <= len(packets) <= 20:
        raise ValueError("invalid_packet_count")
    # An explicit cap permits a first-case runtime smoke without rewriting input.
    packets = packets[:config.max_episodes]
    if any(not validate_packet(packet).valid for packet in packets):
        raise ValueError("invalid_packet_schema")
    if len({p["episode_id"] for p in packets}) != len(packets):
        raise ValueError("duplicate_episode_ids")

    import psutil
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_fd = os.open(args.output.with_suffix(".lock"), os.O_WRONLY | os.O_CREAT, 0o600)
    with os.fdopen(lock_fd, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stop = threading.Event()
        process = psutil.Process()
        resources = {"samples": 0, "peak_rss_gib": 0.0, "peak_process_cpu_percent": 0.0,
                     "minimum_available_ram_gib": psutil.virtual_memory().available / 1024**3,
                     "minimum_output_disk_free_gib": psutil.disk_usage(args.output.parent).free / 1024**3}
        bound_hit = threading.Event()

        def timeout_handler(_signum, _frame):
            raise TimeoutError("resource_bound")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(args.max_runtime_seconds)

        def observe():
            process.cpu_percent()
            while not stop.is_set():
                rss = process.memory_info().rss / 1024**3
                resources["samples"] += 1
                resources["peak_rss_gib"] = max(resources["peak_rss_gib"], rss)
                resources["peak_process_cpu_percent"] = max(resources["peak_process_cpu_percent"], process.cpu_percent())
                resources["minimum_available_ram_gib"] = min(resources["minimum_available_ram_gib"], psutil.virtual_memory().available / 1024**3)
                resources["minimum_output_disk_free_gib"] = min(resources["minimum_output_disk_free_gib"], psutil.disk_usage(args.output.parent).free / 1024**3)
                if rss > args.max_rss_gib:
                    bound_hit.set()
                    os.kill(os.getpid(), signal.SIGALRM)
                    return
                stop.wait(1)

        started, cpu_started = time.monotonic(), time.process_time()
        observer = threading.Thread(target=observe, daemon=True)
        observer.start()
        results = []
        report = {"schema_version": 1, "config": asdict(config), "results": results,
                  "status": "running", "limitations": [
                      "AgentDoG safety labels are not alignment or swarm incident verdicts.",
                      "Retrieved records may omit original profiles, tool definitions and context.",
                      "This adapted prompt requires separate task validation.",
                  ]}
        status = "completed"
        try:
            report["model"] = checkpoint_metadata(args.model_path, args.revision, args.model_manifest)
            # Loader diagnostics remain private. Prompts, generations and reasoning
            # are never printed by our adapter, including into this log.
            log_fd = os.open(args.output.with_suffix(".loader.log"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(log_fd, "w") as loader_log, redirect_stdout(loader_log), redirect_stderr(loader_log):
                guard = LocalGuard(args.model_path, config)
                report["model"].update({"versions": guard.versions,
                                       "load_seconds": round(guard.load_seconds, 4),
                                       "load_cpu_seconds": round(guard.load_cpu_seconds, 4)})
                for packet in packets:
                    result = guard.run(packet)
                    results.append(result)
                    report["summary"] = summarize(results)
                    save_private(args.output, report)
                    if bound_hit.is_set():
                        raise TimeoutError("resource_bound")
        except TimeoutError:
            status = "resource_bound_exceeded"
        except Exception as exc:
            # Exception strings may contain source data; retain only the class.
            status = "execution_error"
            report["error_class"] = type(exc).__name__
        finally:
            signal.alarm(0)
            stop.set()
            observer.join(timeout=2)
        report["status"] = status
        report["summary"] = {
            **summarize(results), "status": status,
            "total_wall_seconds": round(time.monotonic() - started, 4),
            "total_cpu_seconds": round(time.process_time() - cpu_started, 4),
            "resource_samples": {k: round(v, 4) for k, v in resources.items()},
            "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 4),
        }
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
