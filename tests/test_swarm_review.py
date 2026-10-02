"""Fresh-worker, missing-output and review-boundary tests with no model calls."""
from __future__ import annotations
import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_swarm_review as review
from swarmguard.local_guard import GuardConfig
from test_local_judge import PACKETS, bound, candidate, negative


def report_for(packet, finding):
    row, model = bound(packet, finding)
    row.update(input_tokens=100, output_tokens=80, latency_seconds=2.0, cpu_seconds=8.0)
    model.update(download_manifest_sha256="a" * 64, files=[
        {"name": "model.safetensors", "bytes": 1, "sha256": "b" * 64}])
    return {"run_id": "f" * 32, "status": "completed", "model": model, "results": [row],
            "summary": {"total_cpu_seconds": 9.0, "peak_rss_gib": 10.0}}


def coverage_for(packet):
    return {"packet_sha256": hashlib.sha256(json.dumps(packet, ensure_ascii=False,
            separators=(",", ":")).encode()).hexdigest(), "source_window_truncated": False,
            "records": {record["id"]: {"builder_truncated": False} for record in packet["records"]}}


class SwarmReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.packet = copy.deepcopy(PACKETS[1])
        self.report = report_for(self.packet, candidate())

    def jobs(self, packets=None, workers=1, thinking=False):
        return review.prepare_workers(self.directory, packets or [self.packet], workers,
                                      GuardConfig(), self.directory / "unused-checkpoint", thinking=thinking)

    def test_contextual_defaults_select_9b_without_changing_guard_runtime(self):
        args = review.parse_args(["--synthetic", "--output-dir", str(self.directory / "review")])
        self.assertEqual((args.model_profile, args.workers, args.threads), ("9b", 4, 8))
        self.assertEqual((args.max_input_tokens, args.max_new_tokens, args.max_episode_seconds, args.max_episodes),
                         (16384, 512, 180, 8))
        self.assertFalse(args.thinking)
        saved = tomllib.loads((ROOT / "config/local.toml").read_text())
        self.assertEqual(saved["runtime"]["max_input_tokens"], 8192)
        self.assertEqual(saved["runtime"]["max_rss_gib"], 48)

    def test_explicit_cli_overrides_configured_contextual_defaults(self):
        args = review.parse_args([
            "--synthetic", "--output-dir", str(self.directory / "review"),
            "--model-profile", "4b", "--workers", "1", "--threads", "4",
            "--max-input-tokens", "8192", "--max-new-tokens", "128",
            "--max-episode-seconds", "60", "--max-episodes", "3", "--thinking"])
        self.assertEqual((args.model_profile, args.workers, args.threads), ("4b", 1, 4))
        self.assertEqual((args.max_input_tokens, args.max_new_tokens, args.max_episode_seconds, args.max_episodes),
                         (8192, 128, 60, 3))
        self.assertTrue(args.thinking)
        source = (ROOT / "config/local.toml").read_text().replace("thinking = false", "thinking = true")
        configured = self.directory / "configured.toml"
        configured.write_text(source)
        args = review.parse_args(["--output-dir", str(self.directory / "review"), "--no-thinking"],
                                 config_path=configured)
        self.assertFalse(args.thinking)

    def test_invalid_configuration_or_effective_cpu_budget_is_rejected(self):
        for change in ({"model_profile": "unknown"}, {"workers": True},
                       {"thinking": "false"}, {"max_input_tokens": 65536}, {"workers": 4, "threads": 16}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                review.validate_contextual_settings({**review.contextual_defaults(), **change})
        with self.assertRaisesRegex(ValueError, "worker_thread_budget_exceeds_32"):
            review.parse_args(["--output-dir", str(self.directory / "review"), "--threads", "16"])

    def test_shards_are_disjoint_private_and_fresh(self):
        jobs = self.jobs(PACKETS, 2, thinking=True)
        all_ids = [p["episode_id"] for job in jobs for p in job["packets"]]
        self.assertEqual(len(all_ids), 8)
        self.assertEqual(len(set(all_ids)), 8)
        self.assertTrue(all(not job["report_path"].exists() for job in jobs))
        for job in jobs:
            packet_path = Path(job["command"][job["command"].index("--packets") + 1])
            self.assertEqual(packet_path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(json.loads(packet_path.read_text()), job["packets"])
            self.assertIn("--thinking", job["command"])
        other = self.jobs(PACKETS, 2)
        self.assertNotEqual(jobs[0]["report_path"], other[0]["report_path"])

    def test_capacity_reserves_memory_and_bounds_parallel_threads(self):
        self.assertEqual(review.check_capacity(2, 8, 112), 112)
        for args in ((2, 8, 111), (4, 9, 500), (5, 4, 500), (2, 8, float("nan"))):
            with self.subTest(args=args), self.assertRaises(ValueError):
                review.check_capacity(*args)

    def test_9b_reserves_96_gib_per_worker_and_passes_fixed_profile(self):
        profile = review.get_model_profile("9b")
        self.assertEqual(review.check_capacity(2, 8, 208, rss_gib=96), 208)
        with self.assertRaisesRegex(ValueError, "insufficient_available_ram"):
            review.check_capacity(2, 8, 207, rss_gib=96)
        jobs = review.prepare_workers(self.directory, PACKETS[:2], 2, GuardConfig(),
                                      profile["model_path"], model_profile="9b")
        for job in jobs:
            command = job["command"]
            self.assertEqual(command[command.index("--model-profile") + 1], "9b")
            self.assertEqual(command[command.index("--revision") + 1], profile["revision"])
            self.assertEqual(command[command.index("--max-rss-gib") + 1], "96")

    def test_fingerprint_accepts_only_exact_known_9b_pin(self):
        profile = review.get_model_profile("9b")
        model = copy.deepcopy(self.report["model"])
        model.update(model_id=profile["model_id"], revision=profile["revision"], model_profile="9b")
        self.assertIsNotNone(review.model_fingerprint(model))
        model["revision"] = "0" * 40
        self.assertIsNone(review.model_fingerprint(model))

    def test_preexisting_success_never_becomes_a_new_run(self):
        job = self.jobs()[0]
        job["report_path"].write_text(json.dumps(self.report))
        execute = Mock(side_effect=AssertionError("must not execute"))
        result = review.execute_worker(job, execute)
        self.assertEqual(result["status"], "preexisting_report_rejected")
        self.assertIsNone(result["report"])
        execute.assert_not_called()
        self.assertEqual(json.loads(job["report_path"].read_text()), self.report)

    def test_child_failure_with_no_report_cannot_reuse_any_other_file(self):
        job = self.jobs()[0]
        (self.directory / "old-success.json").write_text(json.dumps(self.report))
        result = review.execute_worker(job, lambda *a, **kw: SimpleNamespace(returncode=2))
        self.assertEqual(result["status"], "worker_report_missing")
        self.assertIsNone(result["report"])

    def test_child_status_must_match_new_report(self):
        job = self.jobs()[0]
        def execute(*args, **kwargs):
            job["report_path"].write_text(json.dumps(self.report))
            return SimpleNamespace(returncode=2)
        result = review.execute_worker(job, execute)
        self.assertEqual(result["status"], "child_report_status_mismatch")
        self.assertIsNone(result["report"])

    def test_worker_timeout_has_no_success_result(self):
        job = self.jobs()[0]
        execute = Mock(side_effect=subprocess.TimeoutExpired(["mock"], 1))
        result = review.execute_worker(job, execute)
        self.assertEqual(result["status"], "worker_timeout")
        self.assertIsNone(result["report"])

    def test_missing_worker_keeps_partial_status_without_fabricating_rows(self):
        jobs = self.jobs(PACKETS[:2], 2)
        first = report_for(PACKETS[0], negative())
        outcomes = [{"worker_index": 0, "status": "completed", "report": first, "wall_seconds": 3}]
        collected = review.collect_workers(jobs, outcomes)
        self.assertEqual(collected["status"], "partial")
        self.assertEqual(len(collected["results"]), 1)
        self.assertIn("one_or_more_workers_incomplete", collected["issues"])
        self.assertEqual(collected["workers"][0]["reported_peak_rss_gib"], 10.0)
        self.assertIsNone(collected["workers"][1]["reported_peak_rss_gib"])

    def test_incompatible_checkpoint_worker_excluded(self):
        jobs = self.jobs(PACKETS[:2], 2)
        first, second = report_for(PACKETS[0], negative()), copy.deepcopy(self.report)
        second["model"]["download_manifest_sha256"] = "c" * 64
        collected = review.collect_workers(jobs, [
            {"worker_index": 0, "status": "completed", "report": first},
            {"worker_index": 1, "status": "completed", "report": second}])
        self.assertEqual(len(collected["results"]), 1)
        self.assertIn("worker_checkpoint_fingerprints_disagree", collected["issues"])

    def test_unassigned_or_duplicate_episode_reports_rejected(self):
        extra = copy.deepcopy(self.report)
        extra["results"].append(report_for(PACKETS[0], negative())["results"][0])
        self.assertEqual(review.sanitize_rows([self.packet], extra, exact_scope=True)[0], [])
        duplicate = copy.deepcopy(self.report)
        duplicate["results"] *= 2
        self.assertIn("duplicate_result_episode", review.sanitize_rows([self.packet], duplicate, exact_scope=True)[1])

    def test_stale_bindings_and_raw_fields_are_suppressed(self):
        broken = copy.deepcopy(self.report)
        broken["results"][0].update(packet_sha256="wrong", raw_output="PRIVATE_CHAIN", rationale="PRIVATE_CHAIN")
        rows, issues = review.sanitize_rows([self.packet], broken, exact_scope=True)
        self.assertIsNone(rows[0]["finding"])
        self.assertNotIn("PRIVATE_CHAIN", json.dumps(rows))
        self.assertIn("invalid_or_stale_result_binding", issues)

    def test_native_abstention_is_preserved_as_insufficient_not_run_failure(self):
        report = copy.deepcopy(self.report)
        report["results"][0].update(status="insufficient", finding=None, errors=["invalid_json_output"])
        rows, issues = review.sanitize_rows([self.packet], report, exact_scope=True)
        self.assertEqual(rows[0]["status"], "insufficient")
        self.assertEqual(issues, [])

    def test_real_candidate_requires_matching_builder_manifest_and_review(self):
        row, model = self.report["results"][0], self.report["model"]
        self.assertEqual(review.route_contextual(self.packet, row, model)["route"], "insufficient")
        coverage = {"packet_sha256": hashlib.sha256(json.dumps(
            self.packet, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()}
        routed = review.route_contextual(self.packet, row, model, coverage)
        self.assertEqual(routed["route"], "needs_context_review")
        self.assertFalse(routed["swarm_incident_established"])
        self.assertFalse(routed["human_validated"])
        self.assertIn("retrieved_window_is_not_complete_history", routed["coverage_gaps"])

    def test_real_no_finding_does_not_establish_alignment(self):
        packet = PACKETS[0]
        report = report_for(packet, negative())
        coverage = {"packet_sha256": hashlib.sha256(json.dumps(
            packet, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()}
        row = review.route_contextual(packet, report["results"][0], report["model"], coverage)
        self.assertEqual(row["route"], "insufficient")
        self.assertEqual(row["model_verdict"], "no_finding")
        self.assertEqual(review.route_contextual(packet, report["results"][0], report["model"], synthetic=True)["route"], "no_finding")

    def test_manifest_preflight_requires_bound_hash_complete_metadata_and_boolean_flags(self):
        valid = {self.packet["episode_id"]: coverage_for(self.packet)}
        review.validate_retrieval_manifest([self.packet], valid)
        invalid_cases = []
        for mutation in (
            lambda coverage: coverage.update(packet_sha256="0" * 64),
            lambda coverage: coverage.update(records=[]),
            lambda coverage: coverage["records"].pop(self.packet["records"][0]["id"]),
            lambda coverage: coverage.update(source_window_truncated="false"),
            lambda coverage: coverage["records"][self.packet["records"][0]["id"]].update(builder_truncated="false"),
        ):
            altered = copy.deepcopy(valid)
            mutation(altered[self.packet["episode_id"]])
            invalid_cases.append(altered)
        invalid_cases.append({})
        for manifest in invalid_cases:
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                review.validate_retrieval_manifest([self.packet], manifest)

    def test_stale_manifest_main_fails_before_directory_or_workers(self):
        local_root = self.directory.resolve() / "vm"
        restricted = local_root / "restricted"
        restricted.mkdir(parents=True)
        packets_path, manifest_path = restricted / "packets.json", restricted / "manifest.json"
        packets_path.write_text(json.dumps([self.packet]))
        stale = coverage_for(self.packet)
        stale["packet_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps({self.packet["episode_id"]: stale}))
        output = restricted / "must-not-be-created"
        args = ["--packets", str(packets_path), "--manifest", str(manifest_path), "--output-dir", str(output)]
        with patch.object(review, "ROOT", local_root), patch.object(review, "SCRATCH", SimpleNamespace(is_mount=lambda: True)), \
                patch.object(review.sys, "platform", "linux"), patch.object(review, "validate_real_paths"), \
                patch.object(review, "prepare_workers") as prepare, patch.object(review, "execute_worker") as execute:
            with self.assertRaisesRegex(ValueError, "stale_retrieval_manifest"):
                review.main(args)
            prepare.assert_not_called()
            execute.assert_not_called()
        self.assertFalse(output.exists())

    def test_review_paths_cannot_alias_inputs_or_output_artifacts(self):
        paths = SimpleNamespace(packets=self.directory / "packets.json", manifest=self.directory / "manifest.json",
                                judge_report=None, guard_report=None, output_dir=self.directory / "output")
        review.validate_distinct_paths(paths)
        paths.manifest = paths.packets
        with self.assertRaisesRegex(ValueError, "input_paths_must_be_distinct"):
            review.validate_distinct_paths(paths)
        paths.manifest = self.directory / "manifest.json"
        paths.judge_report = paths.output_dir / "judge.json"
        with self.assertRaisesRegex(ValueError, "outputs_must_not_replace_inputs"):
            review.validate_distinct_paths(paths)

    def test_valid_real_manifest_allows_reuse_without_inference(self):
        local_root = self.directory.resolve() / "vm"
        restricted = local_root / "restricted"
        restricted.mkdir(parents=True)
        packets_path, manifest_path, saved = (restricted / name for name in ("packets.json", "manifest.json", "saved.json"))
        packets_path.write_text(json.dumps([self.packet]))
        manifest_path.write_text(json.dumps({self.packet["episode_id"]: coverage_for(self.packet)}))
        saved.write_text(json.dumps(self.report))
        output = restricted / "revalidated"
        args = ["--packets", str(packets_path), "--manifest", str(manifest_path), "--judge-report", str(saved),
                "--output-dir", str(output)]
        with patch.object(review, "ROOT", local_root), patch.object(review, "SCRATCH", SimpleNamespace(is_mount=lambda: True)), \
                patch.object(review.sys, "platform", "linux"), patch.object(review, "validate_real_paths"), \
                patch.object(review, "prepare_workers", side_effect=AssertionError("reuse must not infer")), \
                redirect_stdout(io.StringIO()) as printed:
            self.assertEqual(review.main(args), 0)
        self.assertEqual(json.loads(printed.getvalue())["execution_mode"], "reuse_without_inference")
        self.assertTrue((output / "dashboard.html").is_file())

    def test_reuse_main_renders_without_starting_workers_and_refuses_overwrite(self):
        local_root = self.directory.resolve() / "vm"
        restricted = local_root / "restricted"
        restricted.mkdir(parents=True)
        saved = restricted / "previous.json"
        saved.write_text(json.dumps(report_for(PACKETS[0], negative())))
        output = restricted / "new-view"
        args = ["--synthetic", "--max-episodes", "1", "--judge-report", str(saved), "--output-dir", str(output)]
        with patch.object(review, "ROOT", local_root), patch.object(review, "SCRATCH", SimpleNamespace(is_mount=lambda: True)), \
                patch.object(review.sys, "platform", "linux"), patch.object(review, "validate_real_paths"), \
                patch.object(review, "prepare_workers", side_effect=AssertionError("no inference on reuse")), \
                redirect_stdout(io.StringIO()) as printed:
            self.assertEqual(review.main(args), 0)
            with self.assertRaisesRegex(ValueError, "output_directory_must_be_new_or_empty"):
                review.main(args)
        summary = json.loads(printed.getvalue())
        self.assertEqual(summary["execution_mode"], "reuse_without_inference")
        self.assertFalse(summary["peak_process_tree_rss_measured"])
        self.assertTrue((output / "dashboard.html").is_file())
        self.assertEqual((output / "judge.json").stat().st_mode & 0o777, 0o600)
        self.assertNotIn("Prepare the approved", printed.getvalue())


if __name__ == "__main__":
    unittest.main()
