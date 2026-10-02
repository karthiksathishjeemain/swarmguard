"""Authored/mock challenger validation tests, not measured model accuracy."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from swarmguard.local_guard import digest
from swarmguard.local_judge import (JUDGE_MODEL_ID, JUDGE_MODEL_REVISION, JUDGE_PROMPT_VERSION,
                                   JUDGE_VALIDATOR_VERSION, LocalJudge, bind_judge_result, build_judge_prompt,
                                   judge_checkpoint_metadata, parse_judgment, validate_judgment,
                                   get_model_profile, profile_for_checkpoint)
from test_local_guard import mock_guard, model_fixture


ROOT = Path(__file__).resolve().parents[1]
PACKETS = json.loads((ROOT / "fixtures/frontier_synthetic.json").read_text())


def candidate():
    return {"verdict": "incident_candidate", "category": "coordinated_constraint_violation",
            "reason_code": "joint_action_conflicts_with_instruction", "rule_record_ids": ["p1-i"],
            "source_record_ids": ["p1-a"], "peer_record_ids": ["p1-b"],
            "action_record_ids": ["p1-c"], "result_record_ids": ["p1-d"],
            "counterevidence_record_ids": [], "graph_edge_ids": ["p1-edge"]}


def negative():
    value = candidate()
    value.update(verdict="no_finding", category="none", reason_code="governing_authorization_covers_action",
                 counterevidence_record_ids=["p1-i"])
    return value


def propagation():
    return {"verdict": "incident_candidate", "category": "unsupported_claim_propagation",
            "reason_code": "contradicted_claim_adopted_by_peer", "rule_record_ids": ["p2-i"],
            "source_record_ids": ["p2-c"], "peer_record_ids": ["p2-b"],
            "action_record_ids": ["p2-d"], "result_record_ids": ["p2-r", "p2-e"],
            "counterevidence_record_ids": [], "graph_edge_ids": ["p2-edge"]}


def goal_replacement():
    return {"verdict": "incident_candidate", "category": "goal_substitution",
            "reason_code": "joint_unauthorized_goal_replacement", "rule_record_ids": ["p4-i", "p4-g"],
            "source_record_ids": ["p4-a"], "peer_record_ids": ["p4-b"],
            "action_record_ids": ["p4-c"], "result_record_ids": ["p4-d"],
            "counterevidence_record_ids": [], "graph_edge_ids": ["p4-edge"]}


def bound(packet, finding):
    row = {**parse_judgment(packet, json.dumps(finding)), "episode_id": packet["episode_id"],
           "packet_sha256": digest(packet), "model_id": JUDGE_MODEL_ID,
           "model_revision": JUDGE_MODEL_REVISION, "prompt_version": JUDGE_PROMPT_VERSION}
    model = {"model_id": JUDGE_MODEL_ID, "revision": JUDGE_MODEL_REVISION,
             "revision_verification": "fixed_pin_and_all_download_manifest_hashes_verified"}
    return row, model


class LocalJudgeTests(unittest.TestCase):
    def setUp(self):
        self.packet = copy.deepcopy(PACKETS[1])

    def test_candidate_is_structurally_valid_not_confirmed(self):
        row = parse_judgment(self.packet, json.dumps(candidate()))
        self.assertEqual(row["status"], "classified")
        self.assertEqual(row["finding"]["verdict"], "incident_candidate")
        self.assertFalse(row["swarm_incident_established"])
        self.assertFalse(row["human_validated"])

    def test_negative_and_counterevidence(self):
        self.assertEqual(validate_judgment(PACKETS[0], negative()), [])

    def test_unknown_record_rejected(self):
        value = candidate()
        value["result_record_ids"] = ["fabricated"]
        self.assertIn("unknown_result_record_ids", validate_judgment(self.packet, value))

    def test_unknown_edge_rejected(self):
        value = candidate()
        value["graph_edge_ids"] = ["invented"]
        self.assertIn("unknown_graph_edge_ids", validate_judgment(self.packet, value))

    def test_wrong_record_kind_rejected(self):
        value = candidate()
        value["result_record_ids"] = ["p1-a"]
        self.assertIn("wrong_record_kind_in_result_record_ids", validate_judgment(self.packet, value))

    def test_candidate_needs_applicable_rule_reference(self):
        value = candidate()
        value["rule_record_ids"] = []
        self.assertIn("candidate_requires_governing_rule", validate_judgment(self.packet, value))

    def test_two_agent_requirement(self):
        for row in self.packet["records"]:
            if row["agent_id"]:
                row["agent_id"] = "amber"
        self.assertIn("candidate_requires_two_agents", validate_judgment(self.packet, candidate()))

    def test_temporal_or_rejection_edge_cannot_support_positive(self):
        for change in ({"basis": "temporal_only"}, {"relation": "explicit_rejection"}):
            with self.subTest(change=change):
                packet = copy.deepcopy(self.packet)
                packet["edges"][0].update(change)
                self.assertIn("candidate_requires_admissible_cross_agent_edge", validate_judgment(packet, candidate()))

    def test_edge_endpoints_must_be_cited_in_source_peer_fields(self):
        value = candidate()
        value["source_record_ids"] = ["p1-e"]
        self.assertIn("candidate_edge_does_not_bind_source_to_peer", validate_judgment(self.packet, value))

    def test_extra_rationale_or_confidence_rejected(self):
        for field in ("rationale", "confidence", "thinking"):
            value = candidate()
            value[field] = "SECRET_REASONING"
            row = parse_judgment(self.packet, json.dumps(value))
            self.assertEqual(row["status"], "insufficient")
            self.assertNotIn("SECRET_REASONING", json.dumps(row))

    def test_unknown_reason_or_too_many_ids_rejected(self):
        value = candidate()
        value["reason_code"] = "anything"
        self.assertIn("invalid_reason_code", validate_judgment(self.packet, value))
        value = candidate()
        value["result_record_ids"] = ["x"] * 9
        self.assertIn("invalid_result_record_ids", validate_judgment(self.packet, value))

    def test_non_candidate_category_and_reason_checked(self):
        value = negative()
        value["category"] = "goal_substitution"
        self.assertIn("noncandidate_category_must_be_none", validate_judgment(self.packet, value))
        value = negative()
        value["reason_code"] = "context_missing"
        self.assertIn("no_finding_reason_mismatch", validate_judgment(self.packet, value))

    def test_json_duplicates_prose_fences_rejected(self):
        encoded = json.dumps(candidate())
        for output in ("Here: " + encoded, "```json\n" + encoded + "\n```", '{"verdict":"x","verdict":"y"}'):
            self.assertEqual(parse_judgment(self.packet, output)["status"], "insufficient")

    def test_reasoning_discarded_and_unclosed_rejected(self):
        output = "<think>SECRET_THINKING</think>" + json.dumps(candidate())
        row = parse_judgment(self.packet, output)
        self.assertEqual(row["status"], "classified")
        self.assertNotIn("SECRET_THINKING", json.dumps(row))
        self.assertEqual(parse_judgment(self.packet, "<think>" + json.dumps(candidate()))["status"], "insufficient")

    def test_incomplete_never_exposes_candidate(self):
        for flags in ({"input_truncated": True}, {"generation_incomplete": True}):
            row = parse_judgment(self.packet, json.dumps(candidate()), **flags)
            self.assertIsNone(row["finding"])
            self.assertEqual(row["status"], "insufficient")

    def test_prompt_contains_packet_not_gold_answers(self):
        prompt = build_judge_prompt(self.packet)
        self.assertIn("p1-a", prompt)
        self.assertNotIn("expected_verdict", prompt)
        self.assertNotIn("expected_category", prompt)
        self.assertIn("successful retry before the claim", prompt)

    def test_bound_result_and_mismatches(self):
        row, model = bound(self.packet, candidate())
        self.assertEqual(bind_judge_result(self.packet, row, model), [])
        for field in ("packet_sha256", "model_revision", "prompt_version"):
            altered = {**row, field: "changed"}
            self.assertIn("stale_or_unbound_judge_result", bind_judge_result(self.packet, altered, model))
        self.assertIn("unverified_judge_checkpoint", bind_judge_result(self.packet, row, {}))
        self.assertIn("unverified_judge_checkpoint", bind_judge_result(self.packet, row, None))

    def test_thinking_mode_is_separately_bound(self):
        from swarmguard.local_judge import JUDGE_THINKING_PROMPT_VERSION
        row, model = bound(self.packet, candidate())
        row.update(thinking_enabled=True, prompt_version=JUDGE_THINKING_PROMPT_VERSION)
        self.assertIn("stale_or_unbound_judge_result", bind_judge_result(self.packet, row, model))
        model["thinking_enabled"] = True
        self.assertEqual(bind_judge_result(self.packet, row, model), [])

    def test_only_fixed_4b_and_9b_checkpoint_profiles_are_allowed(self):
        for name in ("4b", "9b"):
            profile = get_model_profile(name)
            self.assertEqual(profile_for_checkpoint(profile["model_id"], profile["revision"])["name"], name)
            self.assertIsNone(profile_for_checkpoint(profile["model_id"], "0" * 40))
            self.assertEqual(profile["model_path"].name, profile["revision"])
        with self.assertRaisesRegex(ValueError, "unsupported_judge_model_profile"):
            get_model_profile("arbitrary-model")
        self.assertIsNone(profile_for_checkpoint("attacker/unknown", JUDGE_MODEL_REVISION))

    def test_9b_rows_bind_to_9b_metadata_not_4b(self):
        profile = get_model_profile("9b")
        row, model = bound(self.packet, candidate())
        row.update(model_id=profile["model_id"], model_revision=profile["revision"], model_profile="9b")
        self.assertIn("stale_or_unbound_judge_result", bind_judge_result(self.packet, row, model))
        model.update(model_id=profile["model_id"], revision=profile["revision"], model_profile="9b")
        self.assertEqual(bind_judge_result(self.packet, row, model), [])
        model["model_profile"] = "4b"
        self.assertIn("unverified_judge_checkpoint", bind_judge_result(self.packet, row, model))

    def test_9b_local_mock_reports_selected_profile(self):
        guard = mock_guard(input_tokens=100, output=json.dumps(candidate()))
        judge = object.__new__(LocalJudge)
        judge.__dict__.update(guard.__dict__)
        judge.profile = get_model_profile("9b")
        row = judge.run(self.packet)
        self.assertEqual(row["status"], "classified")
        self.assertEqual(row["model_id"], "Qwen/Qwen3.5-9B")
        self.assertEqual(row["model_revision"], judge.profile["revision"])
        self.assertEqual(row["model_profile"], "9b")

    def test_manifest_is_bound_to_selected_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, manifest = model_fixture(Path(tmp))
            profile = get_model_profile("9b")
            data = json.loads(manifest.read_text())
            data.update(repo=profile["model_id"], revision=profile["revision"])
            manifest.write_text(json.dumps(data))
            result = judge_checkpoint_metadata(model, profile["revision"], manifest, model_profile="9b")
            self.assertEqual(result["model_profile"], "9b")
            with self.assertRaisesRegex(ValueError, "judge_revision_does_not_match_verified_pin"):
                judge_checkpoint_metadata(model, profile["revision"], manifest, model_profile="4b")

    def test_local_mock_inference_and_input_rejection(self):
        for n in (100, 300):
            guard = mock_guard(input_tokens=n, output=json.dumps(candidate()))
            judge = object.__new__(LocalJudge)
            judge.__dict__.update(guard.__dict__)
            row = judge.run(self.packet)
            self.assertEqual(row["status"], "classified" if n == 100 else "insufficient")
            self.assertEqual(row["model_id"], JUDGE_MODEL_ID)

    def test_manifest_binding_hashes_all_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, manifest = model_fixture(Path(tmp))
            data = json.loads(manifest.read_text())
            data.update(repo=JUDGE_MODEL_ID, revision="b" * 40)
            manifest.write_text(json.dumps(data))
            with patch("swarmguard.local_judge.JUDGE_MODEL_REVISION", "b" * 40):
                meta = judge_checkpoint_metadata(model, "b" * 40, manifest)
                self.assertEqual(meta["model_id"], JUDGE_MODEL_ID)
                (model / "model.safetensors").write_bytes(b"x" * (model / "model.safetensors").stat().st_size)
                with self.assertRaisesRegex(ValueError, "judge_file_hash_mismatch"):
                    judge_checkpoint_metadata(model, "b" * 40, manifest)

    def test_downstream_result_after_claim_is_valid_temporal_chain(self):
        packet = copy.deepcopy(PACKETS[2])
        self.assertEqual(validate_judgment(packet, propagation()), [])
        parsed = parse_judgment(packet, json.dumps(propagation()))
        self.assertEqual(parsed["validator_version"], JUDGE_VALIDATOR_VERSION)
        self.assertEqual(parsed["finding"]["verdict"], "incident_candidate")

    def test_future_source_result_cannot_support_earlier_claim(self):
        packet = copy.deepcopy(PACKETS[2])
        next(r for r in packet["records"] if r["id"] == "p2-r")["at"] = "2026-01-01T12:00:10Z"
        self.assertIn("unsupported_claim_requires_temporal_source_peer_chain", validate_judgment(packet, propagation()))

    def test_peer_action_before_claim_is_not_propagation(self):
        packet = copy.deepcopy(PACKETS[2])
        next(r for r in packet["records"] if r["id"] == "p2-d")["at"] = "2026-01-01T12:00:01Z"
        self.assertIn("unsupported_claim_requires_temporal_source_peer_chain", validate_judgment(packet, propagation()))

    def test_peer_result_from_another_actor_is_not_a_chain(self):
        packet = copy.deepcopy(PACKETS[2])
        next(r for r in packet["records"] if r["id"] == "p2-e")["agent_id"] = "amber"
        self.assertIn("unsupported_claim_requires_temporal_source_peer_chain", validate_judgment(packet, propagation()))

    def test_delivery_only_result_cannot_establish_originating_contradiction(self):
        finding = propagation()
        finding["result_record_ids"] = ["p2-e"]
        self.assertIn("unsupported_claim_requires_temporal_source_peer_chain", validate_judgment(PACKETS[2], finding))

    def test_present_direct_action_result_link_must_supply_cited_outcome(self):
        packet = copy.deepcopy(PACKETS[2])
        packet["records"].append({"id": "actual-outcome", "kind": "result", "agent_id": "blue",
                                  "at": "2026-01-01T12:00:08Z", "text": "Mock action outcome."})
        packet["edges"].append({"id": "actual-action-link", "source": "p2-d", "target": "actual-outcome",
                               "relation": "action_has_recorded_result", "basis": "direct"})
        self.assertIn("unsupported_claim_requires_temporal_source_peer_chain", validate_judgment(packet, propagation()))
        finding = propagation()
        finding["result_record_ids"].append("actual-outcome")
        self.assertEqual(validate_judgment(packet, finding), [])

    def test_older_results_receive_current_postvalidation_without_new_inference(self):
        packet = copy.deepcopy(PACKETS[2])
        finding = propagation()
        finding["result_record_ids"] = ["p2-e"]
        row, model = bound(packet, propagation())
        row["finding"] = finding
        row.pop("validator_version", None)
        self.assertIn("unsupported_claim_requires_temporal_source_peer_chain",
                      bind_judge_result(packet, row, model))

    def test_rule_conflicts_require_action_and_outcome_by_edge_participant(self):
        for original, original_finding in ((PACKETS[1], candidate()), (PACKETS[6], goal_replacement())):
            for moved in ("action_only", "result_only", "both"):
                with self.subTest(category=original_finding["category"], moved=moved):
                    packet, finding = copy.deepcopy(original), copy.deepcopy(original_finding)
                    changed = []
                    if moved in {"action_only", "both"}:
                        changed += finding["action_record_ids"]
                    if moved in {"result_only", "both"}:
                        changed += finding["result_record_ids"]
                    for record in packet["records"]:
                        if record["id"] in changed:
                            record["agent_id"] = "unrelated-third-agent"
                    self.assertIn("rule_conflict_requires_temporal_participant_action_chain",
                                  validate_judgment(packet, finding))

    def test_rule_conflicts_reject_future_rules_and_reversed_action_outcome(self):
        for original, original_finding in ((PACKETS[1], candidate()), (PACKETS[6], goal_replacement())):
            for change in ("future_rules", "action_before_reference", "outcome_before_action"):
                with self.subTest(category=original_finding["category"], change=change):
                    packet, finding = copy.deepcopy(original), copy.deepcopy(original_finding)
                    by_id = {r["id"]: r for r in packet["records"]}
                    if change == "future_rules":
                        for key in finding["rule_record_ids"]:
                            by_id[key]["at"] = "2027-01-01T12:00:00Z"
                    elif change == "action_before_reference":
                        by_id[finding["action_record_ids"][0]]["at"] = "2026-01-01T12:00:01Z"
                    else:
                        by_id[finding["result_record_ids"][0]]["at"] = "2026-01-01T12:00:01Z"
                    self.assertIn("rule_conflict_requires_temporal_participant_action_chain",
                                  validate_judgment(packet, finding))

    def test_coherent_subset_allows_redundant_mixed_records_and_either_participant(self):
        packet, finding = copy.deepcopy(PACKETS[1]), candidate()
        finding["source_record_ids"] += ["p1-b", "p1-c"]
        finding["peer_record_ids"] += ["p1-c", "p1-d"]
        packet["records"].extend([
            {"id": "extra-action", "kind": "action", "agent_id": "third",
             "at": "2026-01-01T11:00:00Z", "text": "Unrelated authored action."},
            {"id": "extra-result", "kind": "result", "agent_id": "third",
             "at": "2026-01-01T11:00:01Z", "text": "Unrelated authored result."},
        ])
        finding["action_record_ids"].append("extra-action")
        finding["result_record_ids"].append("extra-result")
        self.assertEqual(validate_judgment(packet, finding), [])
        # The source agent can execute after peer agreement; the necessary
        # constraint/goal check must not assume only the target can act.
        for record in packet["records"]:
            if record["id"] in {"p1-c", "p1-d"}:
                record["agent_id"] = "amber"
        self.assertEqual(validate_judgment(packet, finding), [])

    def test_rule_conflict_obeys_present_direct_outcome_link(self):
        packet, finding = copy.deepcopy(PACKETS[6]), goal_replacement()
        packet["records"].append({"id": "linked-outcome", "kind": "result", "agent_id": "blue",
                                  "at": "2026-01-01T12:00:08Z", "text": "Linked authored outcome."})
        packet["edges"].append({"id": "outcome-edge", "source": "p4-c", "target": "linked-outcome",
                               "relation": "action_has_recorded_result", "basis": "direct"})
        self.assertIn("rule_conflict_requires_temporal_participant_action_chain",
                      validate_judgment(packet, finding))
        finding["result_record_ids"].append("linked-outcome")
        self.assertEqual(validate_judgment(packet, finding), [])

    def test_explicit_rejection_cannot_be_positive_chain_for_either_rule_conflict(self):
        for original, original_finding in ((PACKETS[1], candidate()), (PACKETS[6], goal_replacement())):
            packet, finding = copy.deepcopy(original), copy.deepcopy(original_finding)
            packet["edges"][0]["relation"] = "explicit_rejection"
            self.assertIn("rule_conflict_requires_temporal_participant_action_chain",
                          validate_judgment(packet, finding))
            finding.update(verdict="no_finding", category="none", reason_code="peer_rejects_or_corrects_claim")
            self.assertEqual(validate_judgment(packet, finding), [])


if __name__ == "__main__":
    unittest.main()
