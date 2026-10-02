#!/usr/bin/env python3
"""Run the bounded local pilot or render saved results; keep outputs on Azure."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid
import tomllib

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from swarmguard.decisions import route_episode, summarize_routes
from swarmguard.local_guard import MODEL_REVISION, save_private
from swarmguard.schema import validate_packet
from render_case_dashboard import read_json, render_dashboard, write_private, validate_real_paths

ROOT = Path("/home/azureuser/ai-swarm")
DEFAULT_MODEL = Path("/mnt/ai-swarm-scratch/models/agentdog-unified-4b") / MODEL_REVISION


def main():
    defaults = tomllib.loads((PROJECT / "config/local.toml").read_text())["runtime"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--packets", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--guard-report", type=Path, help="Reuse an existing report; every packet hash is checked")
    parser.add_argument("--max-episodes", type=int, default=defaults["max_episodes"])
    parser.add_argument("--threads", type=int, default=defaults["threads"])
    parser.add_argument("--max-input-tokens", type=int, default=defaults["max_input_tokens"])
    parser.add_argument("--max-new-tokens", type=int, default=defaults["max_new_tokens"])
    parser.add_argument("--max-episode-seconds", type=int, default=defaults["max_episode_seconds"])
    args = parser.parse_args()
    if sys.platform != "linux" or not Path("/mnt/ai-swarm-scratch").is_mount():
        parser.error("The local pipeline runs on Azure; use render_case_dashboard.py --synthetic for a Mac preview")
    if not args.output_dir.resolve().is_relative_to(ROOT / "restricted"):
        parser.error("outputs must remain in the VM restricted directory")
    if not 1 <= args.max_episodes <= 20:
        parser.error("max-episodes must be 1 to 20")
    if args.synthetic:
        if args.packets or args.manifest:
            parser.error("synthetic mode uses only authored repository fixtures")
        packets_path = PROJECT / "fixtures/frontier_synthetic.json"
        expectations = {r["episode_id"]: r for r in read_json(PROJECT / "fixtures/frontier_synthetic_answers.json")}
        manifest = {}
    else:
        if not args.packets or not args.manifest:
            parser.error("real data requires packets and retrieval manifest")
        validate_real_paths([args.packets, args.manifest], args.output_dir / "dashboard.html")
        packets_path = args.packets
        manifest = read_json(args.manifest)
        expectations = None
    if args.guard_report:
        validate_real_paths([args.guard_report], args.output_dir / "dashboard.html")
    packets = read_json(packets_path, jsonl=True)[:args.max_episodes]
    if (not packets or any(not validate_packet(p).valid for p in packets) or
            len({p["episode_id"] for p in packets}) != len(packets)):
        raise ValueError("InvalidOrDuplicatePackets")
    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    guard_path = args.guard_report or args.output_dir / ("guard-run-" + uuid.uuid4().hex + ".json")
    start = time.monotonic()
    if not args.guard_report:
        command = [sys.executable, str(PROJECT / "scripts/run_local_guard.py"),
                   "--model-path", str(args.model_path), "--revision", MODEL_REVISION,
                   "--packets", str(packets_path), "--output", str(guard_path),
                   "--max-episodes", str(args.max_episodes), "--threads", str(args.threads),
                   "--max-input-tokens", str(args.max_input_tokens),
                   "--max-new-tokens", str(args.max_new_tokens),
                   "--max-episode-seconds", str(args.max_episode_seconds)]
        completed = subprocess.run(command, check=False, timeout=2500, capture_output=True, text=True)
        if completed.returncode not in (0, 2) or not guard_path.is_file():
            raise RuntimeError("LocalGuardExecutionFailed")
    report = read_json(guard_path)
    signals = {r["episode_id"]: r for r in report.get("results", [])}
    if len(signals) != len(report.get("results", [])):
        raise ValueError("DuplicateModelEpisode")
    routes = [route_episode(p, signals.get(p["episode_id"]), report.get("model", {}),
                            manifest.get(p["episode_id"]), synthetic=args.synthetic) for p in packets]
    summary = {**summarize_routes(routes), "mode": "synthetic" if args.synthetic else "real_retrieved",
               "guard_status": report.get("status"), "pipeline_seconds": round(time.monotonic() - start, 3),
               "guard_measurements": report.get("summary", {}),
               "nli_status": "requires_verified_result_claim_mapping",
               "dashboard": str(args.output_dir / "dashboard.html")}
    summary["guard_report"] = str(guard_path)
    save_private(args.output_dir / "decisions.json", {"summary": summary, "results": routes})
    write_private(args.output_dir / "dashboard.html", render_dashboard(
        packets, report, manifest, synthetic=args.synthetic, expectations=expectations))
    print(json.dumps(summary, sort_keys=True))
    return 0 if report.get("status") == "completed" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_class": type(exc).__name__,
                          "dataset_text_in_stdout": False}))
        raise SystemExit(2)
