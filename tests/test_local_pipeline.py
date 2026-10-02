"""Small CLI integration checks: synthetic packets, mocked child, no inference."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from swarmguard.local_guard import MODEL_ID, MODEL_REVISION, PROMPT_VERSION, digest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts/run_local_pipeline.py"
with patch.object(sys, "path", [str(SCRIPT.parent), *sys.path]):
    SPEC = importlib.util.spec_from_file_location("run_local_pipeline_integration", SCRIPT)
    pipeline = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(pipeline)


def bound_report(packets, *, status="completed"):
    return {
        "status": status,
        "model": {"model_id": MODEL_ID, "revision": MODEL_REVISION,
                  "revision_verification": "fixed_pin_and_all_download_manifest_hashes_verified"},
        "summary": {"authored_test_report": True},
        "results": [{"episode_id": packet["episode_id"], "packet_sha256": digest(packet),
                     "model_id": MODEL_ID, "prompt_version": PROMPT_VERSION,
                     "status": "classified", "safety": "safe", "diagnostics": None,
                     "input_truncated": False, "generation_incomplete": False, "errors": []}
                    for packet in packets],
    }


class LocalPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.project = self.root / "project"
        (self.project / "config").mkdir(parents=True)
        (self.project / "fixtures").mkdir()
        (self.project / "config/local.toml").write_text((PROJECT / "config/local.toml").read_text())
        self.packets = json.loads((PROJECT / "fixtures/frontier_synthetic.json").read_text())[:2]
        self.answers = json.loads((PROJECT / "fixtures/frontier_synthetic_answers.json").read_text())[:2]
        self.output = self.root / "restricted/run"
        self.write_fixtures()

    def write_fixtures(self):
        (self.project / "fixtures/frontier_synthetic.json").write_text(json.dumps(self.packets))
        (self.project / "fixtures/frontier_synthetic_answers.json").write_text(json.dumps(self.answers))

    def writer(self, report, code=0):
        def run(command, **kwargs):
            target = Path(command[command.index("--output") + 1])
            target.write_text(json.dumps(report))
            return subprocess.CompletedProcess(command, code, stdout="", stderr="")
        return run

    @contextlib.contextmanager
    def environment(self, *, child=None, extra_args=()):
        stdout = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(pipeline, "PROJECT", self.project))
            stack.enter_context(patch.object(pipeline, "ROOT", self.root))
            stack.enter_context(patch.object(sys, "platform", "linux"))
            stack.enter_context(patch.object(Path, "is_mount", return_value=True))
            stack.enter_context(patch.object(pipeline, "validate_real_paths"))
            stack.enter_context(patch.object(sys, "argv", [str(SCRIPT), "--synthetic",
                "--output-dir", str(self.output), "--max-episodes", "2", *extra_args]))
            child_mock = stack.enter_context(patch.object(pipeline.subprocess, "run", side_effect=child))
            renderer = stack.enter_context(patch.object(pipeline, "render_dashboard", wraps=pipeline.render_dashboard))
            stack.enter_context(contextlib.redirect_stdout(stdout))
            yield child_mock, renderer, stdout

    def test_authored_expectations_are_dict_and_real_renderer_completes(self):
        with self.environment(child=self.writer(bound_report(self.packets))) as (_, renderer, stdout):
            self.assertEqual(pipeline.main(), 0)
        expectations = renderer.call_args.kwargs["expectations"]
        self.assertIsInstance(expectations, dict)
        self.assertEqual(set(expectations), {packet["episode_id"] for packet in self.packets})
        document = (self.output / "dashboard.html").read_text()
        self.assertIn("Authored evaluation expectation", document)
        self.assertIn(self.answers[0]["explanation"], document)
        self.assertEqual(json.loads(stdout.getvalue())["episodes"], 2)

    def test_failed_child_cannot_reuse_old_report_and_each_attempt_has_unique_path(self):
        self.output.mkdir(parents=True)
        old = self.output / "guard.json"
        old.write_text(json.dumps(bound_report(self.packets)))
        original = old.read_bytes()
        attempted = []

        def fail_without_report(command, **kwargs):
            attempted.append(Path(command[command.index("--output") + 1]))
            return subprocess.CompletedProcess(command, 2, stdout="", stderr="")

        for _ in range(2):
            with self.environment(child=fail_without_report) as (_, renderer, _):
                with self.assertRaisesRegex(RuntimeError, "LocalGuardExecutionFailed"):
                    pipeline.main()
                renderer.assert_not_called()
        self.assertEqual(len(set(attempted)), 2)
        self.assertTrue(all(path.name.startswith("guard-run-") and path != old for path in attempted))
        self.assertEqual(old.read_bytes(), original)
        self.assertFalse((self.output / "decisions.json").exists())
        self.assertFalse((self.output / "dashboard.html").exists())

    def test_partial_report_routes_missing_episode_to_insufficient(self):
        report = bound_report(self.packets[:1], status="partial")
        with self.environment(child=self.writer(report, code=2)):
            self.assertEqual(pipeline.main(), 2)
        saved = json.loads((self.output / "decisions.json").read_text())
        rows = {row["episode_id"]: row for row in saved["results"]}
        self.assertEqual(rows[self.packets[0]["episode_id"]]["route"], "no_signal")
        missing = rows[self.packets[1]["episode_id"]]
        self.assertEqual(missing["route"], "insufficient")
        self.assertIn("model_not_run", missing["reasons"])
        self.assertIn("No model result is attached", (self.output / "dashboard.html").read_text())

    def test_duplicate_packet_ids_rejected_before_child_or_output(self):
        self.packets[1] = copy.deepcopy(self.packets[0])
        self.write_fixtures()
        with self.environment(child=self.writer(bound_report(self.packets[:1]))) as (child, renderer, _):
            with self.assertRaises(ValueError):
                pipeline.main()
            child.assert_not_called()
            renderer.assert_not_called()
        self.assertFalse((self.output / "decisions.json").exists())

    def test_guard_report_reuse_performs_no_subprocess(self):
        report_path = self.root / "restricted/saved-guard.json"
        report_path.parent.mkdir(parents=True)
        report_path.write_text(json.dumps(bound_report(self.packets)))
        with self.environment(extra_args=("--guard-report", str(report_path))) as (child, _, stdout):
            self.assertEqual(pipeline.main(), 0)
            child.assert_not_called()
        self.assertEqual(json.loads(stdout.getvalue())["guard_report"], str(report_path))
        self.assertTrue((self.output / "dashboard.html").is_file())


if __name__ == "__main__":
    unittest.main()
