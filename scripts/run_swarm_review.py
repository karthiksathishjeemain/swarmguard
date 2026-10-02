#!/usr/bin/env python3
"""Bounded contextual review and private dashboard, entirely on Azure.

--judge-report validates and renders saved results without inference. New runs
require an empty output directory and unique worker report paths. Source IDs and
JSON validation establish structural consistency, not accuracy or intent.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import tomllib
import uuid

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from swarmguard.local_guard import GuardConfig, digest, save_private
from swarmguard.local_judge import (JUDGE_MODEL_ID, JUDGE_MODEL_REVISION,
                                   JUDGE_VALIDATOR_VERSION, bind_judge_result, summarize_judge,
                                   get_model_profile, profile_for_checkpoint)
from swarmguard.schema import validate_packet
from render_case_dashboard import read_json, render_dashboard, write_private, validate_real_paths

ROOT = Path("/home/azureuser/ai-swarm")
SCRATCH = Path("/mnt/ai-swarm-scratch")
DEFAULT_MODEL = SCRATCH / "models/qwen35-4b-judge" / JUDGE_MODEL_REVISION
ROW_FIELDS = {
    "episode_id", "packet_sha256", "packet_hash_algorithm", "model_id", "model_revision", "model_profile",
    "prompt_version", "prompt_sha256", "input_tokens", "output_tokens", "status", "finding",
    "errors", "input_truncated", "generation_incomplete", "generated_reasoning_retained",
    "human_validated", "swarm_incident_established", "latency_seconds", "cpu_seconds", "thinking_enabled", "validator_version",
}


def checked_number(value):
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def check_capacity(workers, threads, available_ram_gib, *, rss_gib=48):
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("workers_must_be_1_to_4")
    if type(threads) is not int or not 1 <= threads <= 32 or workers * threads > 32:
        raise ValueError("worker_thread_budget_exceeds_32")
    if rss_gib not in {48, 96}:
        raise ValueError("unsupported_worker_rss_profile")
    required = workers * rss_gib + 16
    if checked_number(available_ram_gib) is None or available_ram_gib < required:
        raise ValueError("insufficient_available_ram_for_workers")
    return required


def prepare_workers(directory, packets, workers, runtime_config, model_path, *, thinking=False, model_profile="4b"):
    """Create exclusive shards and report paths; never inspect old runs."""
    run_dir = directory / ("workers-" + uuid.uuid4().hex)
    run_dir.mkdir(mode=0o700, exist_ok=False)
    profile = get_model_profile(model_profile)
    jobs = []
    for index in range(min(workers, len(packets))):
        shard = packets[index::workers]
        if not shard:
            continue
        packet_path = run_dir / f"worker-{index}-packets.json"
        report_path = run_dir / f"worker-{index}-report.json"
        log_path = run_dir / f"worker-{index}.log"
        # Preserve input object order for prompt reproducibility. Model binding
        # uses its separately declared sorted-key packet hash convention.
        write_private(packet_path, json.dumps(shard, ensure_ascii=False, separators=(",", ":")))
        runtime = min(7200, 300 + len(shard) * runtime_config.max_episode_seconds)
        command = [sys.executable, str(PROJECT / "scripts/run_local_judge.py"),
                   "--model-profile", model_profile, "--model-path", str(model_path), "--revision", profile["revision"],
                   "--packets", str(packet_path), "--output", str(report_path),
                   "--max-episodes", str(len(shard)), "--threads", str(runtime_config.threads),
                   "--max-input-tokens", str(runtime_config.max_input_tokens),
                   "--max-new-tokens", str(runtime_config.max_new_tokens),
                   "--max-episode-seconds", str(runtime_config.max_episode_seconds),
                   "--max-runtime-seconds", str(runtime), "--max-rss-gib", str(profile["max_rss_gib"])]
        if thinking:
            command.append("--thinking")
        jobs.append({"worker_index": index, "packets": shard, "report_path": report_path,
                     "log_path": log_path, "command": command, "timeout_seconds": runtime + 60})
    return jobs


def execute_worker(job, execute=subprocess.run):
    """Require a fresh report and controlled child status; logs remain private."""
    if job["report_path"].exists():
        return {"worker_index": job["worker_index"], "status": "preexisting_report_rejected",
                "report": None, "wall_seconds": 0.0}
    started = time.monotonic()
    outcome = {"worker_index": job["worker_index"], "status": "execution_failed", "report": None}
    log_fd = os.open(job["log_path"], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(log_fd, "w") as log:
            try:
                child = execute(job["command"], check=False, stdout=log, stderr=subprocess.STDOUT,
                                timeout=job["timeout_seconds"], text=True)
            except subprocess.TimeoutExpired:
                outcome["status"] = "worker_timeout"
                return {**outcome, "wall_seconds": round(time.monotonic() - started, 4)}
        if child.returncode not in (0, 2):
            outcome["status"] = "unexpected_child_exit"
        elif not job["report_path"].is_file():
            outcome["status"] = "worker_report_missing"
        else:
            report = read_json(job["report_path"])
            if (type(report) is not dict or not re.fullmatch(r"[0-9a-f]{32}", str(report.get("run_id", "")))
                    or report.get("status") not in {"completed", "execution_error", "resource_bound_exceeded"}):
                outcome["status"] = "worker_report_invalid_or_unfinished"
            elif (child.returncode == 0) != (report["status"] == "completed"):
                outcome["status"] = "child_report_status_mismatch"
            else:
                outcome["report"] = report
                outcome["status"] = "completed" if child.returncode == 0 else "partial_worker"
    except (ValueError, TypeError, OSError):
        outcome["status"] = "worker_report_unreadable"
    return {**outcome, "wall_seconds": round(time.monotonic() - started, 4)}


def model_fingerprint(model):
    if type(model) is not dict:
        return None
    if (profile_for_checkpoint(model.get("model_id"), model.get("revision")) is None
            or model.get("revision_verification") != "fixed_pin_and_all_download_manifest_hashes_verified"
            or not re.fullmatch(r"[0-9a-f]{64}", str(model.get("download_manifest_sha256", "")))
            or not isinstance(model.get("files"), list) or not model["files"]):
        return None
    return digest({**{key: model[key] for key in
                      ("model_id", "revision", "revision_verification", "download_manifest_sha256", "files")},
                   "thinking_enabled": model.get("thinking_enabled", False)})


def sanitize_rows(packets, report, *, exact_scope):
    """Drop arbitrary fields and suppress findings failing source binding."""
    if type(report) is not dict or type(report.get("results")) is not list:
        return [], ["invalid_report_results"]
    expected = {p["episode_id"]: p for p in packets}
    indexed = {}
    for row in report["results"]:
        if type(row) is not dict or type(row.get("episode_id")) is not str:
            return [], ["invalid_result_row"]
        key = row["episode_id"]
        if key in indexed:
            return [], ["duplicate_result_episode"]
        indexed[key] = row
    issues = []
    if exact_scope and set(indexed) - set(expected):
        return [], ["worker_report_has_unassigned_episodes"]
    if set(expected) - set(indexed):
        issues.append("report_missing_expected_episodes")
    results = []
    for packet in packets:
        row = indexed.get(packet["episode_id"])
        if row is None:
            continue
        clean = {key: row[key] for key in ROW_FIELDS if key in row}
        errors = bind_judge_result(packet, row, report.get("model"))
        if errors:
            native_errors = row.get("errors", [])
            native_errors = [v for v in native_errors if type(v) is str and re.fullmatch(r"[a-z][a-z0-9_]{0,100}", v)] if type(native_errors) is list else []
            clean.update(status="insufficient", finding=None, errors=sorted(set(errors + native_errors)))
            # A native model abstention is an expected outcome; stale binding is
            # an invalid artifact and must prevent successful run completion.
            if any(error != "judge_abstained_or_incomplete" for error in errors):
                issues.append("invalid_or_stale_result_binding")
        clean.update(human_validated=False, swarm_incident_established=False,
                     generated_reasoning_retained=False)
        for name in ("input_tokens", "output_tokens", "latency_seconds", "cpu_seconds"):
            clean[name] = checked_number(clean.get(name)) or 0
        clean["input_truncated"] = clean.get("input_truncated") is True
        clean["generation_incomplete"] = clean.get("generation_incomplete") is True
        results.append(clean)
    return results, sorted(set(issues))


def collect_workers(jobs, outcomes):
    """Only compatible fresh workers contribute bound diagnostic findings."""
    outcomes_by_index = {outcome["worker_index"]: outcome for outcome in outcomes}
    aggregate_model, fingerprint = {}, None
    results, records, issues = [], [], []
    for job in jobs:
        outcome = outcomes_by_index.get(job["worker_index"], {"status": "worker_outcome_missing", "report": None})
        record = {"worker_index": job["worker_index"], "status": outcome["status"],
                  "assigned_episodes": len(job["packets"]), "wall_seconds": outcome.get("wall_seconds"),
                  "report_path": str(job["report_path"]), "reported_cpu_seconds": None,
                  "reported_peak_rss_gib": None}
        report = outcome.get("report")
        if outcome["status"] != "completed":
            issues.append("one_or_more_workers_incomplete")
        if report is not None:
            model = report.get("model", {})
            new_fingerprint = model_fingerprint(model)
            if new_fingerprint is None:
                issues.append("worker_model_manifest_unverified")
            elif fingerprint is not None and fingerprint != new_fingerprint:
                issues.append("worker_checkpoint_fingerprints_disagree")
                record["status"] = "incompatible_worker_model"
            else:
                aggregate_model, fingerprint = model, new_fingerprint
                rows, row_issues = sanitize_rows(job["packets"], report, exact_scope=True)
                results.extend(rows)
                issues.extend(row_issues)
                metrics = report.get("summary", {})
                if type(metrics) is dict:
                    record["reported_cpu_seconds"] = checked_number(metrics.get("total_cpu_seconds"))
                    record["reported_peak_rss_gib"] = checked_number(metrics.get("peak_rss_gib"))
        records.append(record)
    return {"model": aggregate_model, "results": results, "workers": records,
            "issues": sorted(set(issues)), "status": "partial" if issues else "completed"}


def route_contextual(packet, row, model, coverage=None, *, synthetic=False):
    result = {"episode_id": packet["episode_id"], "packet_sha256": digest(packet),
              "route": "insufficient", "model_verdict": None, "category": "none",
              "reason_code": None, "reasons": [], "coverage_gaps": [],
              "human_validated": False, "swarm_incident_established": False}
    errors = bind_judge_result(packet, row, model)
    if errors:
        result["reasons"] = errors
        return result
    finding = row["finding"]
    result.update(model_verdict=finding["verdict"], category=finding["category"], reason_code=finding["reason_code"])
    if not synthetic:
        builder_hash = hashlib.sha256(json.dumps(packet, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        if type(coverage) is not dict or coverage.get("packet_sha256") != builder_hash:
            result["reasons"] = ["missing_or_stale_retrieval_manifest"]
            return result
        result["coverage_gaps"].append("retrieved_window_is_not_complete_history")
        if coverage.get("source_window_truncated") or coverage.get("target_source_window_truncated"):
            result["coverage_gaps"].append("source_action_window_truncated")
        record_meta = coverage.get("records", {})
        if type(record_meta) is not dict:
            result["reasons"] = ["invalid_retrieval_record_metadata"]
            return result
        if any(type(record_meta.get(r["id"])) is dict and any(record_meta[r["id"]].get(flag) for flag in
               ("builder_truncated", "packet_bound_truncated", "existing_truncation_marker")) for r in packet["records"]):
            result["coverage_gaps"].append("record_text_excerpted_or_truncated")
    if finding["verdict"] == "incident_candidate":
        result["route"] = "needs_review" if synthetic else "needs_context_review"
        result["reasons"] = ["model_candidate_requires_semantic_and_source_review", "candidate_is_not_established_incident"]
    elif finding["verdict"] == "no_finding":
        result["route"] = "no_finding" if synthetic else "insufficient"
        result["reasons"] = ["no_finding_in_authored_packet" if synthetic else "no_finding_in_incomplete_history"]
    else:
        result["reasons"] = ["model_reported_insufficient_context"]
    return result


def validate_retrieval_manifest(packets, manifest):
    """Reject stale or ill-typed coverage before creating outputs or spending CPU."""
    if type(manifest) is not dict:
        raise ValueError("invalid_retrieval_manifest")
    for packet in packets:
        coverage = manifest.get(packet["episode_id"])
        expected_hash = hashlib.sha256(json.dumps(packet, ensure_ascii=False,
                                                 separators=(",", ":")).encode()).hexdigest()
        if type(coverage) is not dict or coverage.get("packet_sha256") != expected_hash:
            raise ValueError("missing_or_stale_retrieval_manifest")
        for flag in ("source_window_truncated", "target_source_window_truncated"):
            if flag in coverage and type(coverage[flag]) is not bool:
                raise ValueError("invalid_retrieval_coverage_flag")
        records = coverage.get("records")
        if (type(records) is not dict or any(type(key) is not str or type(value) is not dict
                                            for key, value in records.items())
                or not {record["id"] for record in packet["records"]}.issubset(records)):
            raise ValueError("missing_or_invalid_retrieval_record_metadata")
        for metadata in records.values():
            for flag in ("builder_truncated", "packet_bound_truncated", "existing_truncation_marker"):
                if flag in metadata and type(metadata[flag]) is not bool:
                    raise ValueError("invalid_retrieval_record_flag")


def validate_distinct_paths(args):
    inputs = [path.resolve() for path in (args.packets, args.manifest, args.judge_report, args.guard_report) if path]
    if len(set(inputs)) != len(inputs):
        raise ValueError("review_input_paths_must_be_distinct")
    output = args.output_dir.resolve()
    destinations = {output, *(output / name for name in ("judge.json", "decisions.json", "dashboard.html", ".review-run"))}
    if set(inputs) & destinations:
        raise ValueError("review_outputs_must_not_replace_inputs")


CONTEXTUAL_FIELDS = {
    "model_profile", "workers", "threads", "max_input_tokens", "max_new_tokens",
    "max_episode_seconds", "max_episodes", "thinking",
}


def validate_contextual_settings(settings):
    if type(settings) is not dict or set(settings) != CONTEXTUAL_FIELDS:
        raise ValueError("invalid_contextual_configuration_fields")
    get_model_profile(settings["model_profile"])
    if type(settings["thinking"]) is not bool:
        raise ValueError("contextual_thinking_must_be_boolean")
    workers, threads = settings["workers"], settings["threads"]
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("contextual_workers_must_be_1_to_4")
    GuardConfig(settings["max_input_tokens"], settings["max_new_tokens"], threads,
                settings["max_episodes"], settings["max_episode_seconds"], "bfloat16").validate()
    if workers * threads > 32:
        raise ValueError("worker_thread_budget_exceeds_32")


def contextual_defaults(config_path=None):
    path = config_path or PROJECT / "config/local.toml"
    settings = tomllib.loads(path.read_text()).get("contextual")
    validate_contextual_settings(settings)
    return settings


def parse_args(argv=None, *, config_path=None):
    defaults = contextual_defaults(config_path)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--packets", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-profile", choices=("4b", "9b"), default=defaults["model_profile"])
    p.add_argument("--model-path", type=Path)
    p.add_argument("--judge-report", type=Path, help="Validate/render saved results; no inference")
    p.add_argument("--guard-report", type=Path, help="Optional separate AgentDoG baseline")
    p.add_argument("--workers", type=int, default=defaults["workers"])
    p.add_argument("--threads", type=int, default=defaults["threads"])
    p.add_argument("--max-episodes", type=int, default=defaults["max_episodes"])
    p.add_argument("--max-input-tokens", type=int, default=defaults["max_input_tokens"])
    p.add_argument("--max-new-tokens", type=int, default=defaults["max_new_tokens"])
    p.add_argument("--max-episode-seconds", type=int, default=defaults["max_episode_seconds"])
    p.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=defaults["thinking"],
                   help="Enable native reasoning, with no reasoning text retained")
    args = p.parse_args(argv)
    validate_contextual_settings({name: getattr(args, name) for name in CONTEXTUAL_FIELDS})
    return args


def main(argv=None):
    args = parse_args(argv)
    validate_distinct_paths(args)
    profile = get_model_profile(args.model_profile)
    args.model_path = args.model_path or profile["model_path"]
    if sys.platform != "linux" or not SCRATCH.is_mount():
        raise ValueError("swarm_review_requires_azure_vm")
    if not args.output_dir.resolve().is_relative_to(ROOT / "restricted"):
        raise ValueError("review_output_requires_restricted_directory")
    config = GuardConfig(args.max_input_tokens, args.max_new_tokens, args.threads,
                         args.max_episodes, args.max_episode_seconds, "bfloat16")
    config.validate()
    if not 1 <= args.workers <= 4 or args.workers * args.threads > 32:
        raise ValueError("worker_thread_budget_exceeds_32")
    if args.synthetic:
        if args.packets or args.manifest:
            raise ValueError("synthetic_mode_uses_fixed_authored_fixtures")
        packets = read_json(PROJECT / "fixtures/frontier_synthetic.json")
        manifest = {}
        expectations = {r["episode_id"]: r for r in read_json(PROJECT / "fixtures/frontier_synthetic_answers.json")}
    else:
        if not args.packets or not args.manifest:
            raise ValueError("real_review_requires_packets_and_manifest")
        validate_real_paths([args.packets, args.manifest], args.output_dir / "dashboard.html")
        packets = read_json(args.packets, jsonl=True)
        manifest = read_json(args.manifest)
        if type(manifest) is not dict:
            raise ValueError("invalid_retrieval_manifest")
        expectations = None
    if type(packets) is not list or not 1 <= len(packets) <= 20:
        raise ValueError("invalid_review_packet_count")
    packets = packets[:args.max_episodes]
    if any(not validate_packet(v).valid for v in packets) or len({v["episode_id"] for v in packets}) != len(packets):
        raise ValueError("invalid_or_duplicate_review_packets")
    if not args.synthetic:
        validate_retrieval_manifest(packets, manifest)
    for report_path in (args.judge_report, args.guard_report):
        if report_path:
            validate_real_paths([report_path], args.output_dir / "dashboard.html")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("output_directory_must_be_new_or_empty")
    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    marker_fd = os.open(args.output_dir / ".review-run", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(marker_fd)
    start, cpu_start = time.monotonic(), time.process_time()
    run_id = uuid.uuid4().hex
    if args.judge_report:
        original = read_json(args.judge_report)
        rows, issues = sanitize_rows(packets, original, exact_scope=False)
        if type(original) is not dict:
            raise ValueError("invalid_reused_report")
        model = original.get("model", {})
        if model_fingerprint(model) is None:
            issues.append("reused_model_manifest_unverified")
            rows = []
        if original.get("status") != "completed":
            issues.append("reused_report_not_completed")
        result = {"model": model, "results": rows, "workers": [], "issues": sorted(set(issues)),
                  "status": "partial" if issues else "completed"}
        execution_mode = "reuse_without_inference"
    else:
        import psutil
        workers = min(args.workers, len(packets))
        check_capacity(workers, args.threads, psutil.virtual_memory().available / 1024**3,
                       rss_gib=profile["max_rss_gib"])
        jobs = prepare_workers(args.output_dir, packets, workers, config, args.model_path,
                               thinking=args.thinking, model_profile=args.model_profile)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            outcomes = list(pool.map(execute_worker, jobs))
        result = collect_workers(jobs, outcomes)
        execution_mode = "new_local_inference"
    indexed = {r["episode_id"]: r for r in result["results"]}
    routes = [route_contextual(packet, indexed.get(packet["episode_id"]), result["model"],
                              manifest.get(packet["episode_id"]), synthetic=args.synthetic) for packet in packets]
    summary = {
        "status": result["status"], "mode": "synthetic" if args.synthetic else "real_retrieved",
        "execution_mode": execution_mode, "episodes": len(packets),
        "model_results": len(result["results"]), "route_counts": dict(Counter(r["route"] for r in routes)),
        "judge_measurements": summarize_judge(result["results"]),
        "pipeline_wall_seconds": round(time.monotonic() - start, 4),
        "orchestrator_cpu_seconds": round(time.process_time() - cpu_start, 4),
        "reported_worker_cpu_seconds_sum": round(sum(w["reported_cpu_seconds"] or 0 for w in result["workers"]), 4),
        "worker_cpu_measurements_complete": all(w["reported_cpu_seconds"] is not None for w in result["workers"]) if result["workers"] else False,
        "worker_peak_rss_gib": [w["reported_peak_rss_gib"] for w in result["workers"]],
        "peak_process_tree_rss_measured": False,
        "human_validated_incidents": 0, "external_api_calls": 0, "dataset_text_in_stdout": False,
        "issues": result["issues"], "dashboard": str(args.output_dir / "dashboard.html"),
    }
    judge_report = {"schema_version": 1, "run_id": run_id, "status": result["status"],
                    "postvalidation_version": JUDGE_VALIDATOR_VERSION,
                    "model": result["model"], "results": result["results"], "summary": summary,
                    "workers": result["workers"], "source_report": str(args.judge_report) if args.judge_report else None}
    guard_report = read_json(args.guard_report) if args.guard_report else None
    save_private(args.output_dir / "judge.json", judge_report)
    save_private(args.output_dir / "decisions.json", {"run_id": run_id, "summary": summary, "results": routes})
    write_private(args.output_dir / "dashboard.html", render_dashboard(
        packets, guard_report, manifest, synthetic=args.synthetic, expectations=expectations,
        judge_diagnostics=judge_report))
    print(json.dumps(summary, sort_keys=True))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_class": type(exc).__name__,
                          "dataset_text_in_stdout": False, "external_api_calls": 0}))
        raise SystemExit(2)
