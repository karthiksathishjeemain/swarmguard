#!/usr/bin/env python3
"""Export only allowlisted diagnostics bound to the repository's authored packets."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarmguard.local_guard import MODEL_ID, MODEL_REVISION, PROMPT_VERSION, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--kind", choices=("guard", "judge"), default="guard")
    args = parser.parse_args()
    fixtures = json.loads((Path(__file__).resolve().parents[1] / "fixtures/frontier_synthetic.json").read_text())
    packets = {p["episode_id"]: p for p in fixtures}
    report = json.loads(args.report.read_text())
    model = report.get("model", {})
    model_id, revision, prompt_version = MODEL_ID, MODEL_REVISION, PROMPT_VERSION
    allowed_prompts = {prompt_version}
    if args.kind == "judge":
        from swarmguard.local_judge import profile_for_checkpoint, JUDGE_PROMPT_VERSION, JUDGE_THINKING_PROMPT_VERSION
        profile = profile_for_checkpoint(model.get("model_id"), model.get("revision"))
        if profile is None:
            raise ValueError("UnrecognizedJudgeModel")
        model_id, revision = profile["model_id"], profile["revision"]
        allowed_prompts = {JUDGE_PROMPT_VERSION, JUDGE_THINKING_PROMPT_VERSION}
    if model.get("model_id") != model_id or model.get("revision") != revision:
        raise ValueError("UnrecognizedModel")
    rows = []
    fields = ("episode_id", "packet_sha256", "prompt_sha256", "prompt_version", "input_tokens", "output_tokens",
              "generated_reasoning_retained", "model_id", "latency_seconds", "cpu_seconds", "status", "safety",
              "diagnostics", "errors", "input_truncated", "generation_incomplete", "packet_hash_algorithm",
              "model_revision", "finding", "human_validated", "swarm_incident_established", "thinking_enabled", "validator_version")
    for row in report.get("results", []):
        packet = packets.get(row.get("episode_id"))
        if not packet or row.get("packet_sha256") != digest(packet) or row.get("prompt_version") not in allowed_prompts:
            raise ValueError("ReportNotBoundToAuthoredFixture")
        rows.append({key: row[key] for key in fields if key in row})
    print(json.dumps({"status": report["status"],
        "model": {key: model[key] for key in ("model_id", "revision", "revision_verification", "versions", "load_seconds", "thinking_enabled") if key in model},
        "results": rows, "export_scope": "Only authored fixture results; no corpus, prompts or generated reasoning"}, sort_keys=True))


if __name__ == "__main__":
    main()
