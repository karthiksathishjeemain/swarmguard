"""Supplied synthetic provenance only; no models, corpus or network calls."""
import copy
import unittest

from swarmguard.pair_gate import gate_pair, packet_sha256


def fixture():
    packet = {
        "episode_id": "synthetic-gated-pair",
        "records": [
            {"id": "result-1", "kind": "result", "agent_id": "alice",
             "at": "2026-01-01T10:00:00Z", "text": "Alice's tests for report r7 failed."},
            {"id": "claim-1", "kind": "chat", "agent_id": "alice",
             "at": "2026-01-01T10:05:00Z", "text": "My tests for report r7 passed."},
        ], "edges": [],
    }

    def mapped(record):
        return {"record_id": record["id"], "quote": record["text"],
                "subject_actor_id": "alice", "artifact_id": "report", "artifact_version": "r7"}

    result, claim = (mapped(record) for record in packet["records"])
    descriptor = {
        "schema_version": 1, "pair_id": "diagnostic-1", "episode_id": packet["episode_id"],
        "packet_sha256": packet_sha256(packet),
        "verification": {"method": "reviewer_verified", "source_reference": "synthetic-fixture-author",
                         "verified_at": "2026-01-01T10:06:00Z"},
        "result": result, "claim": claim,
        "history": {"complete_through_claim": True, "latest_observation_record_id": "result-1",
                    "observations": [copy.deepcopy(result)], "retries": [], "unresolved_retry": False},
    }
    return packet, descriptor


def rehash(packet, descriptor):
    descriptor["packet_sha256"] = packet_sha256(packet)


class PairGateTests(unittest.TestCase):
    def assert_insufficient(self, packet, descriptor, reason):
        result = gate_pair(packet, descriptor)
        self.assertFalse(result.eligible)
        self.assertEqual(result.status, "insufficient")
        self.assertIsNone(result.pair)
        self.assertIn(reason, " ".join(result.errors))

    def test_consistent_attested_pair_only_grants_nli_eligibility(self):
        packet, descriptor = fixture()
        result = gate_pair(packet, descriptor)
        self.assertTrue(result.eligible)
        self.assertEqual(result.status, "eligible_for_nli")
        self.assertNotIn("verdict", result.to_dict())
        self.assertNotIn("premise", result.to_dict()["pair"])
        self.assertEqual(result.to_dict(include_text=True)["pair"]["premise"], descriptor["result"]["quote"])
        self.assertEqual(result.pair["hypothesis_id"], "claim-1")
        self.assertTrue(result.warnings)

    def test_instrumented_mapping_is_supported(self):
        packet, descriptor = fixture()
        descriptor["verification"]["method"] = "instrumented"
        self.assertTrue(gate_pair(packet, descriptor).eligible)

    def test_actor_mismatch_blocks_even_if_nli_would_be_confident(self):
        packet, descriptor = fixture()
        packet["records"][0]["agent_id"] = "bob"
        descriptor["result"]["subject_actor_id"] = "bob"
        descriptor["history"]["observations"][0]["subject_actor_id"] = "bob"
        rehash(packet, descriptor)
        self.assert_insufficient(packet, descriptor, "actor_mismatch")
        # A 0.996 contradiction from NLI cannot override this provenance failure.
        descriptor["nli_contradiction"] = 0.996
        self.assert_insufficient(packet, descriptor, "unexpected_field")

    def test_actor_metadata_cannot_override_source_actor(self):
        packet, descriptor = fixture()
        descriptor["result"]["subject_actor_id"] = "bob"
        self.assert_insufficient(packet, descriptor, "source_actor_subject_mismatch")

    def test_wrong_or_unknown_artifact_version_blocks(self):
        for field, value in (("artifact_version", "r6"), ("artifact_id", "other-report"),
                             ("artifact_version", ""), ("artifact_version", None)):
            packet, descriptor = fixture()
            descriptor["result"][field] = value
            with self.subTest(field=field, value=value):
                self.assertFalse(gate_pair(packet, descriptor).eligible)

    def test_matching_unknown_or_mutable_versions_still_block(self):
        for version in ("unknown", "UNSPECIFIED", "?", "latest", "current"):
            packet, descriptor = fixture()
            for mapping in (descriptor["result"], descriptor["claim"], descriptor["history"]["observations"][0]):
                mapping["artifact_version"] = version
            with self.subTest(version=version):
                self.assertFalse(gate_pair(packet, descriptor).eligible)

    def test_future_result_and_offsets_are_compared_chronologically(self):
        packet, descriptor = fixture()
        packet["records"][0]["at"] = "2026-01-01T11:06:00+01:00"
        rehash(packet, descriptor)
        self.assert_insufficient(packet, descriptor, "result_occurs_after_claim")
        packet["records"][0]["at"] = "2026-01-01T11:00:00+01:00"
        rehash(packet, descriptor)
        self.assertTrue(gate_pair(packet, descriptor).eligible)

    def test_naive_or_date_only_timestamps_are_insufficient(self):
        for value in ("2026-01-01T10:00:00", "2026-01-01"):
            packet, descriptor = fixture()
            packet["records"][0]["at"] = value
            rehash(packet, descriptor)
            with self.subTest(value=value):
                self.assertFalse(gate_pair(packet, descriptor).eligible)

    def test_packet_edits_invalidate_hash(self):
        packet, descriptor = fixture()
        packet["records"][0]["text"] += " Additional context."
        self.assert_insufficient(packet, descriptor, "packet_hash_mismatch")

    def test_hash_is_stable_for_object_key_order_but_not_record_order(self):
        packet, _ = fixture()
        self.assertEqual(packet_sha256(packet), packet_sha256(dict(reversed(list(packet.items())))))
        altered = copy.deepcopy(packet)
        altered["records"].reverse()
        self.assertNotEqual(packet_sha256(packet), packet_sha256(altered))

    def test_unverified_mapping_and_missing_source_are_insufficient(self):
        for method in ("model_asserted", "unverified", "unknown", None):
            packet, descriptor = fixture()
            descriptor["verification"]["method"] = method
            with self.subTest(method=method):
                self.assertFalse(gate_pair(packet, descriptor).eligible)
        packet, descriptor = fixture()
        descriptor["verification"]["source_reference"] = ""
        self.assert_insufficient(packet, descriptor, "empty_value")

    def test_exact_citations_and_source_kinds_required(self):
        packet, descriptor = fixture()
        descriptor["result"]["quote"] = "fabricated quotation"
        self.assert_insufficient(packet, descriptor, "quote_not_exact_substring")
        packet, descriptor = fixture()
        packet["records"][0]["kind"] = "chat"
        rehash(packet, descriptor)
        self.assert_insufficient(packet, descriptor, "incorrect_record_kind")

    def test_unknown_record_and_incomplete_history_block(self):
        packet, descriptor = fixture()
        descriptor["result"]["record_id"] = "missing"
        self.assert_insufficient(packet, descriptor, "unknown_record")
        for field, value in (("complete_through_claim", False), ("unresolved_retry", True),
                             ("observations", []), ("complete_through_claim", None)):
            packet, descriptor = fixture()
            descriptor["history"][field] = value
            with self.subTest(field=field, value=value):
                self.assertFalse(gate_pair(packet, descriptor).eligible)

    def add_retry(self, packet, descriptor, resolved=True):
        action = {"id": "retry", "kind": "action", "agent_id": "alice",
                  "at": "2026-01-01T10:02:00Z", "text": "Rerun tests for report r7."}
        packet["records"].append(action)
        mapping = {"record_id": "retry", "quote": action["text"], "subject_actor_id": "alice",
                   "artifact_id": "report", "artifact_version": "r7"}
        descriptor["history"]["retries"].append({"action": mapping,
                                                "resolved_by_record_id": "result-2" if resolved else None})
        if resolved:
            record = {"id": "result-2", "kind": "result", "agent_id": "alice",
                      "at": "2026-01-01T10:03:00Z", "text": "Alice's retry tests for report r7 passed."}
            packet["records"].append(record)
            newer = dict(mapping, record_id="result-2", quote=record["text"])
            descriptor["history"]["observations"].append(newer)
        rehash(packet, descriptor)

    def test_unresolved_retry_blocks_without_semantic_guessing(self):
        packet, descriptor = fixture()
        self.add_retry(packet, descriptor, resolved=False)
        self.assert_insufficient(packet, descriptor, "unresolved_retry")

    def test_successful_retry_makes_old_result_ineligible(self):
        packet, descriptor = fixture()
        self.add_retry(packet, descriptor)
        self.assert_insufficient(packet, descriptor, "not_unique_latest_observation")
        descriptor["result"] = copy.deepcopy(descriptor["history"]["observations"][-1])
        descriptor["history"]["latest_observation_record_id"] = "result-2"
        self.assertTrue(gate_pair(packet, descriptor).eligible)

    def test_retry_cannot_be_resolved_by_earlier_result(self):
        packet, descriptor = fixture()
        self.add_retry(packet, descriptor, resolved=False)
        descriptor["history"]["retries"][0]["resolved_by_record_id"] = "result-1"
        self.assert_insufficient(packet, descriptor, "not_strictly_after_action")

    def test_tied_latest_observation_abstains(self):
        packet, descriptor = fixture()
        second = dict(packet["records"][0], id="ambiguous-result")
        packet["records"].append(second)
        descriptor["history"]["observations"].append(dict(descriptor["result"], record_id=second["id"]))
        rehash(packet, descriptor)
        self.assert_insufficient(packet, descriptor, "not_unique_latest_observation")

    def test_duplicate_inventory_entries_and_unseen_resolution_block(self):
        packet, descriptor = fixture()
        descriptor["history"]["observations"].append(copy.deepcopy(descriptor["result"]))
        self.assert_insufficient(packet, descriptor, "duplicate_observation_record")
        packet, descriptor = fixture()
        self.add_retry(packet, descriptor, resolved=False)
        descriptor["history"]["retries"][0]["resolved_by_record_id"] = "missing-result"
        self.assert_insufficient(packet, descriptor, "not_in_observation_inventory")

    def test_verification_cannot_predate_the_claim(self):
        packet, descriptor = fixture()
        descriptor["verification"]["verified_at"] = "2026-01-01T10:04:00Z"
        self.assert_insufficient(packet, descriptor, "verification_predates_claim")

    def test_wrong_episode_and_malformed_packet_block(self):
        packet, descriptor = fixture()
        descriptor["episode_id"] = "different-episode"
        self.assert_insufficient(packet, descriptor, "episode_id_mismatch")
        packet, descriptor = fixture()
        packet["records"][0]["agent_id"] = 42
        self.assert_insufficient(packet, descriptor, "invalid_packet_schema")


if __name__ == "__main__":
    unittest.main()
