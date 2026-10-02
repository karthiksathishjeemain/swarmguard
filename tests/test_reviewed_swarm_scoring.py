"""Known authored confusion counts; no inference or private corpus reads."""
import copy
import importlib.util
from pathlib import Path
import unittest

from swarmguard.local_guard import digest
from test_local_judge import PACKETS, bound, candidate, negative

SPEC = importlib.util.spec_from_file_location("score_reviewed_swarm_cases", Path(__file__).resolve().parents[1] / "scripts/score_reviewed_swarm_cases.py")
scoring = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scoring)


def label(packet, verdict="incident"):
    by_id = {record["id"]: record for record in packet["records"]}
    cite = lambda record_id: {"record_id": record_id, "quote": by_id[record_id]["text"]}
    finding = {"verdict": verdict, "category": "coordinated_constraint_violation" if verdict == "incident" else "none",
               "implicated_agents": ["amber", "blue"] if verdict == "incident" else [],
               "summary": "Author-written reference label for checking aggregate arithmetic.",
               "governing_rule": [cite("p1-i")], "supporting": [cite(i) for i in ("p1-a", "p1-b", "p1-c", "p1-d")],
               "counterevidence": [], "benign_alternative": "Authorization could alter the interpretation.",
               "missing_context": [], "first_violation_record_id": "p1-c" if verdict == "incident" else None,
               "graph_edge_ids": ["p1-edge"] if verdict == "incident" else []}
    return {"episode_id": packet["episode_id"], "packet_sha256": digest(packet), "packet_snapshot": copy.deepcopy(packet),
            "review_source": "interactive_reviewer_attestation", "independent_gold": False,
            "reviewer_declared": "synthetic-test-author", "annotated_at_utc": "2026-10-02T00:00:00Z", "finding": finding}


def packet_named(name):
    return {**copy.deepcopy(PACKETS[1]), "episode_id": name}


class ReviewedSwarmScoringTests(unittest.TestCase):
    def test_zero_references_produces_pending_and_undefined_metrics(self):
        result = scoring.evaluate([packet_named("authored-zero")], [], None)
        self.assertEqual(result["status"], "pending_reference_review")
        for field in ("coverage", "precision", "selective_recall"):
            self.assertIsNone(result[field]["value"])
            self.assertEqual(result[field]["denominator"], 0)
        self.assertEqual(result["independent_gold_cases"], 0)

    def test_stale_hash_or_fabricated_citation_rejected(self):
        packet = packet_named("authored-stale")
        for field in ("hash", "snapshot", "citation"):
            row = label(packet)
            if field == "hash":
                row["packet_sha256"] = "0" * 64
            elif field == "snapshot":
                row["packet_snapshot"]["records"][0]["text"] += " changed"
            else:
                row["finding"]["supporting"][0]["quote"] = "invented evidence"
            with self.subTest(field=field), self.assertRaises(ValueError):
                scoring.evaluate([packet], [row], None)

    def test_model_abstention_is_never_a_negative(self):
        packet = packet_named("authored-abstention")
        for failure in ("incomplete", "stale_model_binding"):
            row, model = bound(packet, candidate())
            if failure == "incomplete":
                row.update(status="insufficient", finding=None, errors=["generation_incomplete"], generation_incomplete=True)
            else:
                row["packet_sha256"] = "0" * 64
            result = scoring.evaluate([packet], [label(packet)], {"results": [row], "model": model})
            with self.subTest(failure=failure):
                self.assertEqual(result["confusion_vs_reviewer_attested"], {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
                self.assertEqual(result["model_abstained_or_unavailable"], 1)
                self.assertEqual(result["reference_positive_abstained"], 1)
                self.assertEqual(result["coverage"], {"numerator": 0, "denominator": 1, "value": 0.0})
                self.assertIsNone(result["selective_recall"]["value"])

    def test_duplicate_reference_annotation_rejected(self):
        packet = packet_named("authored-duplicate")
        row = label(packet)
        with self.assertRaisesRegex(ValueError, "duplicate_reference_annotation"):
            scoring.evaluate([packet], [row, copy.deepcopy(row)], None)

    def test_known_authored_confusion_and_insufficient_reference_exclusion(self):
        packets = [packet_named("authored-" + str(i)) for i in range(6)]
        references = [label(packet, verdict) for packet, verdict in zip(packets,
                      ["incident", "no_incident", "incident", "no_incident", "incident", "insufficient"])]
        rows = []
        for packet, finding in zip(packets[:4], [candidate(), candidate(), negative(), negative()]):
            row, model = bound(packet, finding)
            rows.append(row)
        result = scoring.evaluate(packets, references, {"results": rows, "model": model})
        self.assertEqual(result["confusion_vs_reviewer_attested"], {"tp": 1, "fp": 1, "tn": 1, "fn": 1})
        self.assertEqual(result["excluded_insufficient_references"], 1)
        self.assertEqual(result["coverage"], {"numerator": 4, "denominator": 5, "value": .8})
        self.assertEqual(result["precision"], {"numerator": 1, "denominator": 2, "value": .5})
        self.assertEqual(result["selective_recall"], {"numerator": 1, "denominator": 2, "value": .5})
        self.assertEqual(result["reference_positive_abstained"], 1)
        self.assertEqual(result["reference_positive_cases"], 3)
        self.assertNotIn("authored-0", str(result))


if __name__ == "__main__":
    unittest.main()
