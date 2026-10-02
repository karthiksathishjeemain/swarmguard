import copy
import hashlib
import json
from pathlib import Path
import unittest
from swarmguard.decisions import route_episode
from swarmguard.local_guard import MODEL_ID, MODEL_REVISION, PROMPT_VERSION, digest


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.packet = json.loads((Path(__file__).resolve().parents[1] / "fixtures/frontier_synthetic.json").read_text())[0]
        self.signal = {"episode_id": self.packet["episode_id"], "packet_sha256": digest(self.packet),
                       "prompt_version": PROMPT_VERSION, "model_id": MODEL_ID, "status": "classified",
                       "safety": "unsafe", "errors": [], "input_truncated": False,
                       "generation_incomplete": False}
        self.model = {"model_id": MODEL_ID, "revision": MODEL_REVISION,
                      "revision_verification": "fixed_pin_and_all_download_manifest_hashes_verified"}

    def route(self, synthetic=True, coverage=None):
        return route_episode(self.packet, self.signal, self.model, coverage, synthetic=synthetic)

    def test_unsafe_context_is_review_not_swarm_verdict(self):
        row = self.route()
        self.assertEqual(row["route"], "needs_review")
        self.assertFalse(row["swarm_incident_established"])
        self.assertEqual(row["implicated_agents"], [])

    def test_safe_is_no_signal_not_no_incident(self):
        self.signal["safety"] = "safe"
        self.assertEqual(self.route()["route"], "no_signal")

    def test_stale_packet_blocked(self):
        self.packet["records"][0]["text"] += " Changed."
        self.assertEqual(self.route()["route"], "insufficient")

    def test_unverified_checkpoint_blocked(self):
        self.model["revision"] = "0" * 40
        self.assertEqual(self.route()["route"], "insufficient")

    def test_truncated_generation_blocked(self):
        self.signal["generation_incomplete"] = True
        self.assertEqual(self.route()["route"], "insufficient")

    def test_temporal_edges_do_not_supply_swarm_context(self):
        for edge in self.packet["edges"]:
            edge["basis"] = "temporal_only"
        self.signal["packet_sha256"] = digest(self.packet)
        self.assertEqual(self.route()["route"], "unattributed_safety_signal")

    def test_missing_real_manifest_blocked(self):
        self.assertEqual(self.route(False)["route"], "insufficient")

    def test_incomplete_real_context_review_and_safe_abstention(self):
        metadata = {"packet_sha256": hashlib.sha256(json.dumps(self.packet, ensure_ascii=False,
                   separators=(",", ":")).encode()).hexdigest(), "source_window_truncated": True}
        self.assertEqual(self.route(False, metadata)["route"], "needs_context_review")
        self.signal["safety"] = "safe"
        self.assertEqual(self.route(False, metadata)["route"], "insufficient")

    def test_peer_rejection_does_not_become_adoption(self):
        self.packet["edges"][0]["relation"] = "explicit_rejection"
        self.signal["packet_sha256"] = digest(self.packet)
        row = self.route()
        self.assertFalse(row["swarm_incident_established"])
        self.assertIn("reference_does_not_establish_adoption_or_joint_misconduct", row["reasons"])


if __name__ == "__main__":
    unittest.main()
