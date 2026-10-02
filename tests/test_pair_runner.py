"""NLI orchestration tests; fake scorer proves gating occurs before model load."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from test_pair_gate import fixture
from swarmguard.nli import MODEL_REVISION

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/score_verified_pairs.py"
SPEC = importlib.util.spec_from_file_location("score_verified_pairs", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class FakeFactory:
    def __init__(self):
        self.loads = []
        self.calls = []
        self.metadata = {"synthetic_fake": True}

    def __call__(self, path, **kwargs):
        self.loads.append((path, kwargs))
        return self

    def score_pair(self, premise, hypothesis, **kwargs):
        self.calls.append((premise, hypothesis, kwargs))
        return {"status": "scored", "label": "contradiction", "probabilities": {
            "entailment": .01, "neutral": .01, "contradiction": .98}, **kwargs}


class PairRunnerTests(unittest.TestCase):
    def test_all_rejected_pairs_never_load_model(self):
        packet, descriptor = fixture()
        descriptor["verification"]["method"] = "model_asserted"
        factory = FakeFactory()
        result = runner.score_descriptors([packet], [descriptor], "absent-model", scorer_factory=factory)
        self.assertEqual(factory.loads, [])
        self.assertEqual(factory.calls, [])
        self.assertFalse(result["aggregate"]["model_loaded"])
        self.assertEqual(result["aggregate"]["pairs_rejected_before_inference"], 1)

    def test_scoring_uses_only_gated_quotes_and_pinned_revision(self):
        packet, descriptor = fixture()
        factory = FakeFactory()
        result = runner.score_descriptors([packet], [descriptor], "pinned-local-model", threads=2,
                                          scorer_factory=factory)
        self.assertEqual(factory.loads, [("pinned-local-model", {"threads": 2, "expected_revision": MODEL_REVISION})])
        self.assertEqual(factory.calls[0][:2], (descriptor["result"]["quote"], descriptor["claim"]["quote"]))
        self.assertEqual(factory.calls[0][2], {"premise_id": "result-1", "hypothesis_id": "claim-1"})
        self.assertNotIn(descriptor["result"]["quote"], json.dumps(result))
        self.assertNotIn(descriptor["claim"]["quote"], json.dumps(result))
        self.assertEqual(result["aggregate"]["incident_verdicts_produced"], 0)

    def test_unknown_episode_rejects_before_model_load(self):
        packet, descriptor = fixture()
        descriptor["episode_id"] = "missing"
        factory = FakeFactory()
        result = runner.score_descriptors([packet], [descriptor], "absent", scorer_factory=factory)
        self.assertEqual(result["results"][0]["gate"]["errors"], ["packet_not_found"])
        self.assertEqual(factory.loads, [])

    def test_duplicate_pairs_or_packets_fail_before_load(self):
        packet, descriptor = fixture()
        for packets, pairs in (([packet, packet], [descriptor]), ([packet], [descriptor, descriptor])):
            factory = FakeFactory()
            with self.assertRaises(ValueError):
                runner.score_descriptors(packets, pairs, "absent", scorer_factory=factory)
            self.assertEqual(factory.loads, [])

    def test_demo_wrapper_and_jsonl_load(self):
        packet, descriptor = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text(json.dumps({"packets": [packet], "pairs": [descriptor]}))
            self.assertEqual(runner.load_rows(path, "packets")[0], [packet])
            self.assertEqual(runner.load_rows(path, "pairs")[0], [descriptor])
            path.write_text(json.dumps(packet) + "\n" + json.dumps(packet) + "\n")
            self.assertEqual(runner.load_rows(path, "packets")[0], [packet, packet])

    def test_output_mode_is_private_and_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private/report.json"
            runner.save_private(path, {"aggregate": {"pairs": 1}})
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(list(path.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
