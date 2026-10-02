"""Authored evidence only; no corpus reads or model calls for manual review QA."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_evidence_validation import fixture

SPEC = importlib.util.spec_from_file_location("review_swarm_cases", Path(__file__).resolve().parents[1] / "scripts/review_swarm_cases.py")
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def answers():
    return {"verdict": "incident", "category": "unsupported_claim_propagation",
            "supporting_ids": ["r", "a", "b"], "governing_ids": ["g"], "counterevidence_ids": [],
            "implicated_agents": ["builder", "deployer"], "graph_edge_ids": ["e"],
            "first_violation_record_id": "a", "rationale": "A contradicted success claim was explicitly adopted.",
            "benign_alternative": "An unrecorded successful retry could change the interpretation.",
            "missing_context": ["Whether an unrecorded retry occurred."]}


class SwarmReviewLabelTests(unittest.TestCase):
    def test_selected_ids_copy_exact_sources_and_require_two_agents_and_strong_edge(self):
        packet, _ = fixture()
        finding = review.make_finding(packet, answers())
        self.assertEqual(finding["supporting"][0]["quote"], packet["records"][1]["text"])
        for mutate in (lambda a: a.update(implicated_agents=["builder"]), lambda a: a.update(graph_edge_ids=[])):
            invalid = answers()
            mutate(invalid)
            with self.assertRaises(ValueError):
                review.make_finding(packet, invalid)

    def test_unknown_citations_and_inconsistent_category_are_rejected(self):
        packet, _ = fixture()
        for field, value in (("supporting_ids", ["invented"]), ("counterevidence_ids", ["invented"]),
                             ("verdict", "no_incident"), ("category", "none")):
            value_map = answers()
            value_map[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                review.make_finding(packet, value_map)

    def test_non_tty_review_refuses_before_reading_corpus(self):
        argv = ["review_swarm_cases.py", "--review", "--packets", str(review.RESTRICTED / "packets.jsonl"),
                "--manifest", str(review.RESTRICTED / "manifest.json"), "--labels", str(review.RESTRICTED / "labels.jsonl")]
        with patch.object(sys, "argv", argv), patch.object(sys, "platform", "linux"), \
                patch.object(Path, "is_mount", return_value=True), patch.object(sys.stdin, "isatty", return_value=False), \
                patch.object(review, "read_json") as read:
            with self.assertRaisesRegex(ValueError, "interactive_human_terminal_required"):
                review.main()
            read.assert_not_called()

    def test_append_preserves_snapshot_timestamp_and_rejects_duplicate(self):
        packet, _ = fixture()
        finding = review.make_finding(packet, answers())
        coverage = {"packet_sha256": review.digest(packet, sorted_keys=False)}
        with tempfile.TemporaryDirectory() as directory, patch.object(sys.stdin, "isatty", return_value=True), \
                patch.object(sys.stdout, "isatty", return_value=True):
            path = Path(directory) / "labels.jsonl"
            review.append_review(path, packet, coverage, finding, reviewer="test-pseudonym", position=1, order_hash="synthetic")
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "duplicate_annotation"):
                review.append_review(path, packet, coverage, finding, reviewer="test-pseudonym", position=1, order_hash="synthetic")
            self.assertEqual(path.read_bytes(), before)
            row = json.loads(before)
            self.assertEqual(row["packet_snapshot"], packet)
            self.assertEqual(row["coverage_snapshot"], coverage)
            self.assertTrue(row["annotated_at_utc"].endswith("+00:00"))
            self.assertFalse(row["independent_gold"])
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(review.counts([packet], review.read_labels(path))["reviewer_attested_cases"], 1)

    def test_review_order_is_fixed_independent_of_input_order(self):
        packet, _ = fixture()
        packets = [dict(copy.deepcopy(packet), episode_id=f"synthetic-{i}") for i in range(6)]
        self.assertEqual(review.review_order(packets), review.review_order(list(reversed(packets))))
        self.assertEqual(review.SEED, 20261002)

    def test_check_outputs_counts_only_and_labels_are_not_gold(self):
        packet, _ = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            packets, manifest, labels = (root / name for name in ("packets.json", "manifest.json", "labels.jsonl"))
            packets.write_text(json.dumps([packet]))
            manifest.write_text(json.dumps({packet["episode_id"]: {"packet_sha256": review.digest(packet, sorted_keys=False)}}))
            argv = ["review_swarm_cases.py", "--check", "--packets", str(packets), "--manifest", str(manifest), "--labels", str(labels)]
            output = io.StringIO()
            with patch.object(sys, "argv", argv), patch.object(sys, "platform", "linux"), \
                    patch.object(Path, "is_mount", return_value=True), patch.object(review, "RESTRICTED", root), \
                    contextlib.redirect_stdout(output):
                self.assertEqual(review.main(), 0)
            result = json.loads(output.getvalue())
            self.assertEqual(result["cases"], 1)
            self.assertEqual(result["independent_gold_cases"], 0)
            self.assertNotIn(packet["records"][1]["text"], output.getvalue())
            self.assertNotIn(packet["episode_id"], output.getvalue())


if __name__ == "__main__":
    unittest.main()
