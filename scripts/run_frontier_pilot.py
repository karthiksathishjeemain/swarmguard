#!/usr/bin/env python3
"""Plan offline by default; --execute requires VM, credentials and transfer approval."""
import argparse
import fcntl
import json
from pathlib import Path
import resource
import sys
import threading
import time
import tomllib
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarmguard.runner import VM_ROOT, plan, read_key, run_packet, save_private


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packets", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config/frontier.toml")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--approved-transfer", action="store_true",
                   help="Use only after the data owner approves these packets going to OpenAI")
    args = p.parse_args()
    config = tomllib.loads(args.config.read_text())
    if config["model"]["name"] != "gpt-6-astra":
        p.error("Changing model requires updating and verifying pricing first")
    if not 1 <= config["limits"]["max_episodes"] <= 20:
        p.error("Pilot accepts 1-20 episodes")
    if not 1 <= config["model"]["max_output_tokens"] <= 12000:
        p.error("Invalid output token cap")
    if not 0 < config["limits"]["budget_usd"] <= 100:
        p.error("Invalid local budget cap")
    if not 1024 <= config["limits"]["max_request_bytes"] <= 128000:
        p.error("Pilot request cap must be at most 128000 bytes")
    if args.packets.stat().st_size > 4_000_000:
        p.error("Packet file exceeds the bounded pilot size")
    raw = args.packets.read_text()
    packets = (json.loads(raw) if raw.lstrip().startswith("[") else
               [json.loads(line) for line in raw.splitlines() if line.strip()])
    if not 1 <= len(packets) <= config["limits"]["max_episodes"]:
        p.error("Wrong episode count")
    if len({v["episode_id"] for v in packets}) != len(packets):
        p.error("Duplicate episode identifiers")
    summary = plan(packets, config)
    if not args.execute:
        print(json.dumps(summary, sort_keys=True))
        return 0
    if not args.approved_transfer:
        p.error("Explicit owner approval is required for external API data transfer")
    if summary["packets_with_admissible_cross_agent_reference"] == 0:
        p.error("No packet has an admissible cross-agent reference; improve retrieval before paid inference")
    if (sys.platform != "linux" or not Path("/mnt/ai-swarm-scratch").is_mount()
            or not args.packets.resolve().is_relative_to(VM_ROOT / "restricted")):
        p.error("Paid pilot must run on Azure with packets under restricted/")
    key = read_key()
    directory = VM_ROOT / "restricted/frontier-run"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    started, cpu_started = time.monotonic(), time.process_time()
    counts = {}
    monitor_stop = threading.Event()
    process = psutil.Process()
    resources = {"sampled_peak_rss_mib": 0.0, "minimum_free_disk_gib": float("inf"),
                 "peak_process_cpu_percent": 0.0, "minimum_available_ram_gib": float("inf")}

    def observe():
        while not monitor_stop.is_set():
            resources["sampled_peak_rss_mib"] = max(resources["sampled_peak_rss_mib"], process.memory_info().rss / 1024**2)
            resources["minimum_free_disk_gib"] = min(resources["minimum_free_disk_gib"], psutil.disk_usage(VM_ROOT).free / 1024**3)
            resources["peak_process_cpu_percent"] = max(resources["peak_process_cpu_percent"], process.cpu_percent())
            resources["minimum_available_ram_gib"] = min(resources["minimum_available_ram_gib"], psutil.virtual_memory().available / 1024**3)
            monitor_stop.wait(1)

    with (directory / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger_path = directory / "ledger.json"
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"requests": {}}
        status = "completed"
        monitor = threading.Thread(target=observe, daemon=True)
        monitor.start()
        try:
            for packet in packets:
                result = run_packet(packet, config, directory, ledger, key)
                verdict = result["review_status"]
                counts[verdict] = counts.get(verdict, 0) + 1
        except Exception as exc:
            # Only our controlled error codes are emitted; never API/data exceptions.
            known = str(exc) if type(exc) is RuntimeError else "execution_error"
            status = known if known.replace("_", "").isalnum() else "execution_error"
        finally:
            monitor_stop.set()
            monitor.join(timeout=2)
        summary = {"status": status, "completed_episodes": sum(counts.values()),
                   "review_status_counts": counts, "human_adjudicated": 0,
                   "duration_seconds": round(time.monotonic() - started, 3),
                   "cpu_seconds": round(time.process_time() - cpu_started, 3),
                   "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 3),
                   "reservation_usd": round(sum(v["reservation_usd"] for v in ledger["requests"].values()), 6),
                   "requests_in_ledger": len(ledger["requests"]), "dataset_text_in_output": False,
                   "resource_samples": {k: round(v, 3) for k, v in resources.items()}}
        save_private(VM_ROOT / "reports/frontier-run.json", summary)
        print(json.dumps(summary, sort_keys=True))
        return 0 if status == "completed" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, RuntimeError):
        print(json.dumps({"status": "invalid_input_or_access_setup", "api_response_or_dataset_text_emitted": False}))
        raise SystemExit(2)
