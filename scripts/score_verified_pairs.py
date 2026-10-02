#!/usr/bin/env python3
"""Gate supplied pair mappings before local NLI; only aggregates leave the VM CLI.

Inputs accept JSON arrays, JSONL, a single object, or the authored demo wrapper
with `packets` and `pairs` arrays. Real pair descriptors must be reviewed or
instrumented separately; this command does not invent metadata or select claims.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarmguard.nli import LocalNLIScorer, MODEL_ID, MODEL_REVISION
from swarmguard.pair_gate import gate_pair

VM_ROOT = Path("/home/azureuser/ai-swarm")


def load_rows(path, kind):
    path = Path(path)
    if path.stat().st_size > 8_000_000:
        raise ValueError("bounded_input_size_exceeded")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if isinstance(value, dict):
        value = value[kind] if kind in value else [value]
    if not isinstance(value, list) or not 1 <= len(value) <= 200:
        raise ValueError("bounded_input_count_exceeded")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError("input_rows_must_be_objects")
    return value, hashlib.sha256(raw).hexdigest()


def score_descriptors(packets, descriptors, model_path, *, threads=8,
                      scorer_factory=LocalNLIScorer):
    """Pure orchestration entry point; malformed gate inputs never load weights."""
    if not 1 <= len(descriptors) <= 100:
        raise ValueError("pair_count_must_be_1_to_100")
    packet_map = {}
    for packet in packets:
        episode_id = packet.get("episode_id")
        if not isinstance(episode_id, str) or episode_id in packet_map:
            raise ValueError("missing_or_duplicate_episode_id")
        packet_map[episode_id] = packet
    pair_ids = [descriptor.get("pair_id") for descriptor in descriptors]
    if any(not isinstance(pair_id, str) or not pair_id.strip() for pair_id in pair_ids):
        raise ValueError("pair_ids_must_be_nonempty_strings")
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("duplicate_pair_ids")
    results, eligible = [], []
    for descriptor in descriptors:
        packet = packet_map.get(descriptor.get("episode_id"))
        if packet is None:
            gate = {"status": "insufficient", "errors": ["packet_not_found"], "warnings": [], "pair": None}
            record = {"pair_id": descriptor["pair_id"], "episode_id": descriptor.get("episode_id"),
                      "gate": gate, "score": None}
        else:
            gated = gate_pair(packet, descriptor)
            record = {"pair_id": descriptor["pair_id"], "episode_id": descriptor.get("episode_id"),
                      "gate": gated.to_dict(), "score": None}
            if gated.eligible:
                eligible.append((record, gated.pair))
        results.append(record)

    model_metadata = None
    load_seconds = 0.0
    if eligible:
        load_started = time.perf_counter()
        scorer = scorer_factory(model_path, threads=threads, expected_revision=MODEL_REVISION)
        load_seconds = time.perf_counter() - load_started
        model_metadata = scorer.metadata
        for record, pair in eligible:
            record["score"] = scorer.score_pair(
                pair["premise"], pair["hypothesis"],
                premise_id=pair["premise_id"], hypothesis_id=pair["hypothesis_id"])
    score_counts = Counter()
    for record in results:
        if record["score"] is not None:
            score_counts[record["score"].get("label") or "insufficient"] += 1
    aggregate = {
        "pairs_submitted": len(results), "pairs_eligible": len(eligible),
        "pairs_rejected_before_inference": len(results) - len(eligible),
        "model_loaded": bool(eligible), "model_load_seconds": round(load_seconds, 6),
        "semantic_labels": dict(score_counts), "external_inference_calls": 0,
        "incident_verdicts_produced": 0, "scores_are_misalignment_confidence": False,
    }
    return {"aggregate": aggregate, "model": model_metadata, "results": results}


def save_private(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staged = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=".pair-scores-", suffix=".tmp", delete=False) as out:
            staged = Path(out.name)
            os.fchmod(out.fileno(), 0o600)
            json.dump(value, out, indent=2, sort_keys=True, allow_nan=False)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        staged.replace(path)
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if sys.platform != "linux" or not VM_ROOT.is_dir() or not Path("/mnt/ai-swarm-scratch").is_mount():
        parser.error("Model inference must run on the authorized Azure VM")
    if not args.output.resolve().is_relative_to(VM_ROOT / "restricted"):
        parser.error("Private pair reports must be written under /home/azureuser/ai-swarm/restricted")
    if not 1 <= args.threads <= 64:
        parser.error("--threads must be between 1 and 64")
    os.umask(0o077)
    packets, packets_digest = load_rows(args.packets, "packets")
    descriptors, pairs_digest = load_rows(args.pairs, "pairs")
    started, cpu_started = time.perf_counter(), time.process_time()
    report = score_descriptors(packets, descriptors, args.model_path, threads=args.threads)
    import psutil
    elapsed = time.perf_counter() - started
    cpu_seconds = time.process_time() - cpu_started
    report["created_at"] = datetime.now(timezone.utc).isoformat()
    report["inputs"] = {"packets_sha256": packets_digest, "descriptors_sha256": pairs_digest}
    report["configured_model"] = {"id": MODEL_ID, "revision": MODEL_REVISION, "threads": args.threads}
    report["aggregate"].update({
        "status": "completed", "duration_seconds": round(elapsed, 6),
        "cpu_seconds": round(cpu_seconds, 6),
        "mean_process_cpu_percent_one_core_100": round(cpu_seconds / elapsed * 100, 3),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "available_memory_bytes_after_run": psutil.virtual_memory().available,
        "scratch_free_bytes_after_run": psutil.disk_usage("/mnt/ai-swarm-scratch").free,
        "dataset_text_in_stdout": False,
        "note": "Semantic pair signals only. Verification attestations are not authenticated by this script; no incident verdict is produced.",
    })
    save_private(args.output, report)
    print(json.dumps(report["aggregate"], sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, TypeError, OSError, RuntimeError):
        # Exceptions can include data or paths. Never echo them to shared output.
        print(json.dumps({"status": "invalid_inputs_or_local_inference_failure",
                          "dataset_text_in_stdout": False, "external_inference_calls": 0}))
        raise SystemExit(2)
