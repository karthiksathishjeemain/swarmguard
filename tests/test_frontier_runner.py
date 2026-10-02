"""Deterministic transport tests using only author-written synthetic trace data."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_evidence_validation import fixture
from swarmguard import runner
from swarmguard.prompts import build_investigator_messages


def config():
    return {
        "model": {"name": "gpt-6-astra", "reasoning_effort": "high",
                  "max_output_tokens": 6000, "timeout_seconds": 180},
        "limits": {"max_episodes": 20, "max_request_bytes": 128000, "budget_usd": 20.0},
        "pricing": {"input_ceiling_per_million": 12.5, "output_per_million": 50.0},
    }


def provider_response(finding):
    return {
        "status": "completed", "model": "gpt-6-astra",
        "output": [{"type": "message", "content": [
            {"type": "output_text", "text": json.dumps(finding)},
        ]}],
        "usage": {"input_tokens": 100, "output_tokens": 80},
    }


def benign(finding, verdict="no_incident"):
    result = copy.deepcopy(finding)
    result.update(verdict=verdict, category="none", implicated_agents=[],
                  summary="The supplied record does not establish a collective violation.",
                  first_violation_record_id=None, graph_edge_ids=[])
    return result


class MockTransport:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, body, key, timeout):
        self.calls.append(copy.deepcopy(body))
        if not self.responses:
            raise AssertionError("Unexpected provider call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return copy.deepcopy(response)


class FrontierRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.packet, self.finding = fixture()
        self.config = config()
        self.ledger = {"requests": {}}

    def run_mock(self, transport):
        return runner.run_packet(self.packet, self.config, self.directory,
                                 self.ledger, "synthetic-unused-key", transport=transport)

    def test_two_pass_agreement_is_candidate_not_human_adjudication(self):
        transport = MockTransport(provider_response(self.finding), provider_response(self.finding))
        result = self.run_mock(transport)
        self.assertEqual(result["review_status"], "candidate_incident")
        self.assertTrue(result["pass_agreement"])
        self.assertFalse(result["human_adjudicated"])
        self.assertEqual([p["phase"] for p in result["passes"]], ["investigator", "skeptic"])
        self.assertEqual(len(transport.calls), 2)
        self.assertTrue(all(r["state"] == "completed" for r in self.ledger["requests"].values()))

    def test_skeptical_disagreement_abstains(self):
        transport = MockTransport(provider_response(self.finding), provider_response(benign(self.finding)))
        result = self.run_mock(transport)
        self.assertFalse(result["pass_agreement"])
        self.assertEqual(result["review_status"], "insufficient")

    def test_matching_benign_or_insufficient_passes_preserve_verdict(self):
        for verdict in ("no_incident", "insufficient"):
            with self.subTest(verdict=verdict), tempfile.TemporaryDirectory() as directory:
                finding = benign(self.finding, verdict)
                result = runner.run_packet(
                    self.packet, self.config, directory, {"requests": {}}, "synthetic-unused-key",
                    transport=MockTransport(provider_response(finding), provider_response(finding)),
                )
                self.assertEqual(result["review_status"], verdict)

    def test_fabricated_citation_stops_before_skeptic(self):
        self.finding["supporting"][0]["quote"] = "fabricated text absent from the record"
        transport = MockTransport(provider_response(self.finding))
        with self.assertRaisesRegex(RuntimeError, "finding_failed_evidence_validation"):
            self.run_mock(transport)
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(next(iter(self.ledger["requests"].values()))["state"], "invalid")

    def test_budget_reservation_precedes_transport(self):
        self.config["limits"]["budget_usd"] = 0.000001
        transport = MockTransport()
        with self.assertRaisesRegex(RuntimeError, "budget_cap_reached"):
            self.run_mock(transport)
        self.assertEqual(transport.calls, [])
        self.assertEqual(self.ledger["requests"], {})

    def test_existing_reservations_count_towards_budget(self):
        self.ledger["requests"]["previous-unrelated-call"] = {
            "state": "started", "reservation_usd": self.config["limits"]["budget_usd"],
        }
        transport = MockTransport()
        with self.assertRaisesRegex(RuntimeError, "budget_cap_reached"):
            self.run_mock(transport)
        self.assertEqual(transport.calls, [])

    def test_complete_cache_resume_makes_no_duplicate_calls(self):
        result = self.run_mock(MockTransport(provider_response(self.finding), provider_response(self.finding)))
        self.ledger = json.loads((self.directory / "ledger.json").read_text())
        transport = MockTransport()
        resumed = self.run_mock(transport)
        self.assertEqual(transport.calls, [])
        self.assertEqual(resumed, result)
        self.assertEqual(len(self.ledger["requests"]), 2)

    def test_transport_failure_reservation_survives_and_refuses_retry(self):
        failed = MockTransport(RuntimeError("provider_transport_error_no_retry"))
        with self.assertRaisesRegex(RuntimeError, "provider_transport_error_no_retry"):
            self.run_mock(failed)
        self.ledger = json.loads((self.directory / "ledger.json").read_text())
        self.assertEqual(next(iter(self.ledger["requests"].values()))["state"], "started")
        retry = MockTransport()
        with self.assertRaisesRegex(RuntimeError, "previous_attempt_unresolved_no_retry"):
            self.run_mock(retry)
        self.assertEqual(retry.calls, [])

    def test_unresolved_skeptic_reuses_investigator_then_refuses_retry(self):
        failed = MockTransport(provider_response(self.finding), RuntimeError("provider_transport_error_no_retry"))
        with self.assertRaisesRegex(RuntimeError, "provider_transport_error_no_retry"):
            self.run_mock(failed)
        self.ledger = json.loads((self.directory / "ledger.json").read_text())
        retry = MockTransport()
        with self.assertRaisesRegex(RuntimeError, "previous_attempt_unresolved_no_retry"):
            self.run_mock(retry)
        self.assertEqual(retry.calls, [])
        self.assertEqual(len(self.ledger["requests"]), 2)

    def test_missing_cache_refuses_completed_request_retry(self):
        self.run_mock(MockTransport(provider_response(self.finding), provider_response(self.finding)))
        for path in (self.directory / "cache").glob("*.json"):
            path.unlink()
        transport = MockTransport()
        with self.assertRaisesRegex(RuntimeError, "previous_attempt_unresolved_no_retry"):
            self.run_mock(transport)
        self.assertEqual(transport.calls, [])

    def test_tampered_cached_citation_is_revalidated(self):
        self.run_mock(MockTransport(provider_response(self.finding), provider_response(self.finding)))
        for path in (self.directory / "cache").glob("*.json"):
            cached = json.loads(path.read_text())
            cached["finding"]["supporting"][0]["quote"] = "invented cached quote"
            path.write_text(json.dumps(cached))
        transport = MockTransport()
        with self.assertRaisesRegex(RuntimeError, "cached_finding_failed_current_validation"):
            self.run_mock(transport)
        self.assertEqual(transport.calls, [])

    def test_incomplete_and_refused_outputs_are_not_findings(self):
        responses = [
            ({"status": "incomplete", "output": []}, "provider_response_not_completed"),
            ({"status": "completed", "output": [{"type": "message", "content": [
                {"type": "refusal", "refusal": "Synthetic refusal"},
            ]}]}, "provider_refusal"),
            ({"status": "completed", "output": []}, "provider_expected_one_finding"),
        ]
        for response, error in responses:
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                ledger = {"requests": {}}
                transport = MockTransport(response)
                with self.assertRaisesRegex(ValueError, error):
                    runner.run_packet(self.packet, self.config, directory, ledger,
                                      "synthetic-unused-key", transport=transport)
                self.assertEqual(len(transport.calls), 1)
                self.assertEqual(next(iter(ledger["requests"].values()))["state"], "started")

    def test_requests_disable_storage_and_expose_no_execution_tools(self):
        transport = MockTransport(provider_response(self.finding), provider_response(self.finding))
        self.run_mock(transport)
        for body in transport.calls:
            self.assertIs(body["store"], False)
            self.assertNotIn("tools", body)
            self.assertNotIn("previous_response_id", body)
            self.assertEqual(body["truncation"], "disabled")
            self.assertEqual(body["max_output_tokens"], self.config["model"]["max_output_tokens"])
            self.assertTrue(body["text"]["format"]["strict"])
            self.assertNotIn("synthetic-unused-key", json.dumps(body))

    def test_changing_model_changes_cache_key_but_needs_price_review(self):
        messages = build_investigator_messages(self.packet)
        original_digest = hashlib.sha256(runner.canonical(runner.request_body(messages, self.config))).hexdigest()
        self.config["model"]["name"] = "synthetic-other-model"
        changed_digest = hashlib.sha256(runner.canonical(runner.request_body(messages, self.config))).hexdigest()
        self.assertNotEqual(original_digest, changed_digest)
        transport = MockTransport()
        with self.assertRaisesRegex(ValueError, "unsupported_model_configuration"):
            self.run_mock(transport)
        self.assertEqual(transport.calls, [])

    def test_changing_reasoning_configuration_invalidates_request_cache(self):
        self.run_mock(MockTransport(provider_response(self.finding), provider_response(self.finding)))
        self.config["model"]["reasoning_effort"] = "medium"
        transport = MockTransport(provider_response(self.finding), provider_response(self.finding))
        self.run_mock(transport)
        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(len(self.ledger["requests"]), 4)

    def test_budget_stop_between_passes_preserves_first_pass_for_resume(self):
        investigator_body = runner.request_body(build_investigator_messages(self.packet), self.config)
        self.config["limits"]["budget_usd"] = runner.reservation(investigator_body, self.config) + 0.000001
        transport = MockTransport(provider_response(self.finding))
        with self.assertRaisesRegex(RuntimeError, "budget_cap_reached"):
            self.run_mock(transport)
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(next(iter(self.ledger["requests"].values()))["state"], "completed")
        self.ledger = json.loads((self.directory / "ledger.json").read_text())
        self.config["limits"]["budget_usd"] = 20.0
        resumed = MockTransport(provider_response(self.finding))
        result = self.run_mock(resumed)
        self.assertEqual(len(resumed.calls), 1)
        self.assertEqual(result["review_status"], "candidate_incident")

    def test_changing_investigator_prompt_invalidates_its_cache(self):
        self.run_mock(MockTransport(provider_response(self.finding), provider_response(self.finding)))

        def revised_messages(packet):
            messages = build_investigator_messages(packet)
            messages[0]["content"] += "\nSynthetic prompt revision."
            return messages

        transport = MockTransport(provider_response(self.finding))
        with patch("swarmguard.runner.build_investigator_messages", side_effect=revised_messages):
            self.run_mock(transport)
        # An identical investigator finding permits reuse of the identical skeptic request.
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(len(self.ledger["requests"]), 3)

    def test_plan_never_calls_network_or_reads_credentials(self):
        with patch("swarmguard.runner.call_provider", side_effect=AssertionError("network forbidden")), \
             patch("swarmguard.runner.read_key", side_effect=AssertionError("credential access forbidden")):
            result = runner.plan([self.packet], self.config)
        self.assertEqual(result["maximum_calls"], 2)
        self.assertEqual(result["api_calls_made"], 0)

    def test_request_size_limit_prevents_transport(self):
        self.config["limits"]["max_request_bytes"] = 1024
        transport = MockTransport()
        with self.assertRaisesRegex(ValueError, "request_size_limit"):
            self.run_mock(transport)
        self.assertEqual(transport.calls, [])

    def test_invalid_packet_never_reaches_transport(self):
        self.packet["edges"][0]["source"] = "nonexistent-record"
        transport = MockTransport()
        with self.assertRaisesRegex(ValueError, "invalid_evidence_packet"):
            self.run_mock(transport)
        self.assertEqual(transport.calls, [])

    def test_invalid_prices_never_reach_transport(self):
        for price in (-1, float("nan"), float("inf")):
            with self.subTest(price=price), tempfile.TemporaryDirectory() as directory:
                candidate_config = copy.deepcopy(self.config)
                candidate_config["pricing"]["input_ceiling_per_million"] = price
                transport = MockTransport()
                with self.assertRaises((ValueError, RuntimeError)):
                    runner.run_packet(self.packet, candidate_config, directory, {"requests": {}},
                                      "synthetic-unused-key", transport=transport)
                self.assertEqual(transport.calls, [])


if __name__ == "__main__":
    unittest.main()
