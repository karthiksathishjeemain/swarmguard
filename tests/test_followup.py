"""Authored tests for bounded post-reference retrieval; no corpus/model access."""
from __future__ import annotations

import copy
from contextlib import nullcontext
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

from swarmguard.followup import (FOLLOWUP_HORIZON_SECONDS, FOLLOWUP_TURN_LIMIT,
                                fetch_peer_followup, followup_record_specs, select_followup_rows)


def chat():
    return {"id": "peer-chat", "agent_id": "blue", "created_at": "2026-01-01T12:00:00Z",
            "content": 'Amber said "the fictional report is ready"; I will check it.'}


def send():
    return {"send_turn_id": "peer-send", "session_id": "peer-session", "agent_id": "blue",
            "send_at": "2026-01-01T12:00:01Z", "message_content": chat()["content"]}


def turn(identifier="after-1", at="2026-01-01T12:00:02Z"):
    return {"id": identifier, "session_id": "peer-session", "created_at": at,
            "action_type": "bash", "command_excerpt": "mock_check_report", "command_chars": 17,
            "typed_text_excerpt": None, "typed_text_chars": -1, "message_excerpt": None,
            "message_chars": None, "output_excerpt": "MOCK: report exists", "output_chars": 19,
            "error_excerpt": None, "error_chars": -1, "screenshot_redacted": False}


class FakeCursor:
    def __init__(self, rows):
        fields = list(rows[0]) if rows else ["empty"]
        self.description = [(field,) for field in fields]
        self.values = [tuple(row.get(field) for field in fields) for row in rows]

    def fetchall(self):
        return self.values


class FakeConn:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def execute(self, sql, params):
        self.calls.append((sql, params))
        return FakeCursor(self.responses[len(self.calls) - 1])


class FollowupTests(unittest.TestCase):
    def test_unique_bidirectional_binding_uses_later_send_time_and_bounded_window(self):
        conn = FakeConn([[send()], [chat()], [turn()]])
        result = fetch_peer_followup(conn, chat())
        self.assertEqual(result["status"], "selected")
        self.assertEqual(result["after_exclusive"], "2026-01-01T12:00:01+00:00")
        self.assertEqual(result["until_inclusive"], "2026-01-01T12:15:01+00:00")
        self.assertTrue(result["does_not_establish_adoption"])
        self.assertTrue(result["same_session_only"])
        self.assertEqual(result["source_send_turn_id"], "peer-send")
        self.assertEqual(len(conn.calls), 3)
        self.assertIn("LIMIT 5", conn.calls[2][0])
        self.assertEqual(conn.calls[2][1][1], "peer-session")

    def test_later_chat_timestamp_is_the_lower_bound_when_send_precedes_chat(self):
        earlier = {**send(), "send_at": "2026-01-01T11:59:59Z"}
        result = fetch_peer_followup(FakeConn([[earlier], [chat()], [turn()]]), chat())
        self.assertEqual(result["after_exclusive"], "2026-01-01T12:00:00+00:00")

    def test_ambiguous_or_missing_forward_match_never_selects_a_session(self):
        for rows in ([], [send(), {**send(), "send_turn_id": "another-send"}],
                     [{**send(), "message_content": "different text"}],
                     [{**send(), "agent_id": "other-agent"}],
                     [{**send(), "send_at": "2026-01-01T12:05:00.000001Z"}]):
            with self.subTest(rows=len(rows)):
                conn = FakeConn([rows])
                result = fetch_peer_followup(conn, chat())
                self.assertEqual(result["status"], "ambiguous_or_missing_peer_send")
                self.assertEqual(result["turns"], [])
                self.assertEqual(len(conn.calls), 1)

    def test_reverse_match_must_be_unique_exact_and_the_same_chat(self):
        for rows in ([chat(), {**chat(), "id": "repeated-chat"}], [],
                     [{**chat(), "id": "other-chat"}], [{**chat(), "content": "different"}],
                     [{**chat(), "created_at": "2026-01-01T12:00:01Z"}]):
            conn = FakeConn([[send()], rows])
            result = fetch_peer_followup(conn, chat())
            self.assertEqual(result["status"], "ambiguous_or_mismatched_peer_chat")
            self.assertEqual(result["turns"], [])
            self.assertEqual(len(conn.calls), 2)

    def test_invalid_reference_text_never_queries(self):
        for value in (None, "", "  "):
            conn = FakeConn([])
            result = fetch_peer_followup(conn, {**chat(), "content": value})
            self.assertEqual(result["status"], "invalid_reference_chat")
            self.assertEqual(conn.calls, [])

    def test_empty_later_window_is_reported_without_inventing_outcome(self):
        result = fetch_peer_followup(FakeConn([[send()], [chat()], []]), chat())
        self.assertEqual(result["status"], "no_same_session_followup_in_window")
        self.assertFalse(result["turn_limit_reached"])
        self.assertEqual(followup_record_specs(result), [])

    def test_strict_lower_bound_upper_horizon_and_same_session_are_defended(self):
        invalid = [turn(at="2026-01-01T12:00:00Z"), turn(at="2026-01-01T11:59:59Z"),
                   turn(at="2026-01-01T12:15:00.000001Z"), {**turn(), "session_id": "another-session"}]
        for row in invalid:
            with self.subTest(row=row["created_at"]):
                with self.assertRaisesRegex(ValueError, "followup_session_or_time_mismatch"):
                    select_followup_rows([row], "peer-session", "2026-01-01T12:00:00Z", "2026-01-01T12:15:00Z")
        selected, _ = select_followup_rows([turn(at="2026-01-01T12:15:00Z")], "peer-session",
                                           "2026-01-01T12:00:00Z", "2026-01-01T12:15:00Z")
        self.assertEqual(len(selected), 1)

    def test_limit_uses_extra_row_to_reveal_omission_and_sorts_chronologically(self):
        rows = [turn("turn-" + str(i), f"2026-01-01T12:00:0{i}Z") for i in range(1, 6)]
        selected, metadata = select_followup_rows(list(reversed(rows)), "peer-session",
                                                  "2026-01-01T12:00:00Z", "2026-01-01T12:15:00Z")
        self.assertEqual([row["id"] for row in selected], ["turn-1", "turn-2", "turn-3", "turn-4"])
        self.assertEqual(metadata["eligible_turn_count_capped_at_five"], 5)
        self.assertTrue(metadata["turn_limit_reached"])
        with self.assertRaisesRegex(ValueError, "query_row_bound"):
            select_followup_rows(rows + [turn("turn-6")], "peer-session", "2026-01-01T12:00:00Z", "2026-01-01T12:15:00Z")

    def test_duplicate_source_ids_and_overlong_windows_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "duplicate_followup_turn_id"):
            select_followup_rows([turn(), turn()], "peer-session", "2026-01-01T12:00:00Z", "2026-01-01T12:15:00Z")
        with self.assertRaisesRegex(ValueError, "invalid_followup_interval"):
            select_followup_rows([], "peer-session", "2026-01-01T12:00:00Z", "2026-01-01T12:16:00Z")

    def test_record_specs_preserve_source_ids_actor_time_and_actual_result_only(self):
        context = fetch_peer_followup(FakeConn([[send()], [chat()], [turn()]]), chat())
        original = copy.deepcopy(context)
        spec = followup_record_specs(context)[0]
        self.assertEqual(spec["action"]["id"], "turn:after-1")
        self.assertEqual(spec["result"]["id"], "result:after-1")
        for record in spec.values():
            self.assertEqual(record["agent_id"], "blue")
            self.assertEqual(record["at"], "2026-01-01T12:00:02Z")
            self.assertEqual(record["source"]["source_turn_id"], "after-1")
            self.assertEqual(record["source"]["source_session_id"], "peer-session")
            self.assertTrue(record["source"]["temporal_context_not_adoption"])
        self.assertTrue(spec["result"]["source"]["timestamp_is_parent_turn"])
        self.assertEqual(spec["result"]["text"], "output:\nMOCK: report exists")
        self.assertEqual(context, original)
        context["turns"][0].update(output_excerpt=None, error_excerpt=None)
        self.assertIsNone(followup_record_specs(context)[0]["result"])

    def test_projected_text_loss_is_propagated_for_action_and_result(self):
        row = {**turn(), "message_excerpt": "short excerpt", "message_chars": 9999,
               "output_chars": 9999}
        context = fetch_peer_followup(FakeConn([[send()], [chat()], [row]]), chat())
        spec = followup_record_specs(context)[0]
        self.assertTrue(spec["action"]["source"]["projection_truncated"])
        self.assertTrue(spec["result"]["source"]["projection_truncated"])
        self.assertIn("message_content:\nshort excerpt", spec["action"]["text"])


class BuilderConn:
    """Authored relations for the real builder, including post-reference turns."""
    quote = "This is the complete fictional source report ready for verification."

    def __init__(self):
        self.source = {"id": "source-chat", "agent_id": "amber", "created_at": "2026-01-01T11:59:00Z",
                       "content": self.quote}
        self.target = {**chat(), "content": 'Amber wrote: "' + self.quote + '". I will verify it.', "gap_ms": 61000}
        self.anchor = {"send_turn_id": "source-send", "chat_id": "source-chat", "agent_id": "amber",
                       "session_id": "source-session", "room_id": "mock-room", "send_at": "2026-01-01T11:58:59Z",
                       "message_content": self.quote, "chat_content": self.quote, "chat_agent": "amber",
                       "chat_hash": "mock-source-hash", "source_chat_at": self.source["created_at"],
                       "chat_at": self.source["created_at"], "match_delta_ms": 1000,
                       "session_created_at": "2026-01-01T11:00:00Z", "session_goal": None,
                       "window_truncated": False, "prior_session_actions": 10,
                       "evidence_json": json.dumps([
                           {"turn_id": "source-prior-" + str(i), "at": f"2026-01-01T11:58:{i:02d}Z",
                            "action": "bash", "command": "mock_source_command", "output": "MOCK source output"}
                           for i in range(10)])}

    def execute(self, sql, params):
        if "FROM evidence_windows e JOIN" in sql:
            result = [self.anchor]
        elif "ORDER BY abs(date_diff" in sql:
            result = [self.source]
        elif "FROM chat_messages WHERE room_id" in sql:
            result = []
        elif "FROM chat_messages WHERE id =" in sql:
            result = [self.target]
        elif "FROM computer_use_turns t" in sql:
            result = [{**send(), "message_content": self.target["content"]}]
        elif "FROM chat_messages\n        WHERE agent_id" in sql:
            result = [self.target]
        elif "FROM computer_use_turns\n        WHERE session_id" in sql:
            result = [turn("peer-after-" + str(i), f"2026-01-01T12:00:{i + 2:02d}Z") for i in range(4)]
        elif "SELECT agent_id, evidence_json" in sql:
            result = []
        elif "SELECT count(*) AS matches" in sql:
            result = [{"matches": 1, "unique_id": "source-chat"}]
        else:
            raise AssertionError("Unexpected authored-query branch")
        return FakeCursor(result)


class FollowupBuilderIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import the real builder without installing its VM-only runtime. Its
        # main() is never run; the authored DB double exercises build_packet().
        stubs = {}
        if importlib.util.find_spec("duckdb") is None:
            module = types.ModuleType("duckdb")
            module.DuckDBPyConnection = object
            stubs["duckdb"] = module
        else:
            # Load native extensions before any temporary module-map snapshot.
            # Unloading only their Python cache entries can break a later import.
            importlib.import_module("duckdb")
        if importlib.util.find_spec("psutil") is None:
            stubs["psutil"] = types.ModuleType("psutil")
        else:
            importlib.import_module("psutil")
        script = Path(__file__).resolve().parents[1] / "scripts/build_frontier_episodes.py"
        specification = importlib.util.spec_from_file_location("followup_test_builder", script)
        cls.builder = importlib.util.module_from_spec(specification)
        with patch.dict(sys.modules, stubs) if stubs else nullcontext():
            specification.loader.exec_module(cls.builder)

    def build(self, include=True):
        selected = {"send_turn_id": "source-send", "sample_arm": "explicit_reference",
                    "required_target_chat_id": "peer-chat", "quote": BuilderConn.quote}
        return self.builder.build_packet(BuilderConn(), selected, {}, {}, [], [], include_peer_followup=include)

    def test_builder_retains_after_context_and_only_true_source_or_temporal_edges(self):
        packet, manifest = self.build()
        by_id = {r["id"]: r for r in packet["records"]}
        self.assertEqual(manifest["peer_followup"]["retained_actions"], 4)
        self.assertEqual(manifest["peer_followup"]["retained_results"], 4)
        self.assertEqual([r["at"] for r in packet["records"]], sorted(r["at"] for r in packet["records"]))
        self.assertLessEqual(len(packet["records"]), self.builder.MAX_RECORDS)
        self.assertLessEqual(len(self.builder.dumps(packet)), self.builder.MAX_PACKET_CHARS)
        self.assertEqual(manifest["packet_sha256"], hashlib.sha256(self.builder.dumps(packet).encode()).hexdigest())
        for i in range(4):
            aid, rid = "turn:peer-after-" + str(i), "result:peer-after-" + str(i)
            self.assertEqual(by_id[aid]["agent_id"], "blue")
            self.assertGreater(by_id[aid]["at"], by_id["chat:peer-chat"]["at"])
            self.assertEqual(by_id[aid]["at"], by_id[rid]["at"])
            self.assertTrue(any(e["source"] == aid and e["target"] == rid and e["basis"] == "direct"
                                and e["relation"] == "action_has_recorded_result" for e in packet["edges"]))
            self.assertTrue(any(e["source"] == "chat:peer-chat" and e["target"] == aid
                                and e["basis"] == "temporal_only" for e in packet["edges"]))
        self.assertFalse(any("adopt" in e["relation"] or "reliance" in e["relation"] for e in packet["edges"]))

    def test_followup_is_prioritized_before_prior_source_turns_and_budget_loss_is_explicit(self):
        with patch.object(self.builder, "MAX_RECORDS", 12):
            packet, manifest = self.build()
        self.assertLessEqual(len(packet["records"]), 12)
        self.assertEqual(manifest["peer_followup"]["retained_actions"], 4)
        self.assertEqual(manifest["peer_followup"]["retained_results"], 4)
        self.assertLess(manifest["prior_tool_turns_retained"], 10)
        with patch.object(self.builder, "MAX_RECORDS", 4):
            packet, manifest = self.build()
        self.assertLessEqual(len(packet["records"]), 4)
        self.assertTrue(manifest["peer_followup"]["packet_bound_omitted_actions"])
        self.assertTrue(manifest["peer_followup"]["packet_bound_omitted_results"])
        self.assertTrue(all(e["source"] in {r["id"] for r in packet["records"]}
                            and e["target"] in {r["id"] for r in packet["records"]} for e in packet["edges"]))

    def test_existing_default_does_not_add_followup(self):
        packet, manifest = self.build(include=False)
        self.assertEqual(manifest["peer_followup"]["status"], "not_requested")
        self.assertFalse(any(r["id"].startswith("turn:peer-after") for r in packet["records"]))


@unittest.skipUnless(importlib.util.find_spec("duckdb"), "DuckDB integration runs in the project's VM environment")
class FollowupDuckDBTests(unittest.TestCase):
    def test_real_sql_filters_other_sessions_before_equal_and_outside_window(self):
        import duckdb
        conn = duckdb.connect()
        try:
            conn.execute("CREATE TABLE computer_use_sessions(id VARCHAR, agent_id VARCHAR)")
            conn.execute("INSERT INTO computer_use_sessions VALUES ('peer-session','blue'),('other-session','blue')")
            conn.execute("""CREATE TABLE computer_use_turns(
                id VARCHAR, session_id VARCHAR, created_at VARCHAR, action_type VARCHAR,
                command_excerpt VARCHAR, command_chars INTEGER, typed_text_excerpt VARCHAR, typed_text_chars INTEGER,
                message_content VARCHAR, message_hash VARCHAR, output_excerpt VARCHAR, output_chars INTEGER,
                error_excerpt VARCHAR, error_chars INTEGER, screenshot_redacted BOOLEAN)""")
            conn.execute("CREATE TABLE chat_messages(id VARCHAR, agent_id VARCHAR, created_at VARCHAR, content VARCHAR, content_hash VARCHAR)")
            reference = chat()
            content_hash = hashlib.blake2b(reference["content"].strip().encode(), digest_size=16).hexdigest()
            conn.execute("INSERT INTO chat_messages VALUES (?,?,?,?,?)", [reference["id"], reference["agent_id"],
                         reference["created_at"], reference["content"], content_hash])
            conn.execute("INSERT INTO computer_use_turns VALUES (?,?,?,?,NULL,-1,NULL,-1,?,?,NULL,-1,NULL,-1,FALSE)",
                         ["peer-send", "peer-session", send()["send_at"], "send_message_back_to_chat", reference["content"], content_hash])
            examples = [("before", "peer-session", "2026-01-01T11:59:59Z", "bash"),
                        ("equal", "peer-session", "2026-01-01T12:00:01Z", "bash"),
                        ("after", "peer-session", "2026-01-01T12:00:02Z", "bash"),
                        ("other", "other-session", "2026-01-01T12:00:02Z", "bash"),
                        ("none", "peer-session", "2026-01-01T12:00:03Z", "none"),
                        ("late", "peer-session", "2026-01-01T12:15:01.000001Z", "bash")]
            for identifier, session, at, action in examples:
                conn.execute("INSERT INTO computer_use_turns VALUES (?,?,?,?,?,-1,NULL,-1,NULL,NULL,?,-1,NULL,-1,FALSE)",
                             [identifier, session, at, action, "mock_command", "MOCK result"])
            context = fetch_peer_followup(conn, reference)
            self.assertEqual(context["status"], "selected")
            self.assertEqual([row["id"] for row in context["turns"]], ["after"])
            conn.execute("INSERT INTO chat_messages VALUES (?,?,?,?,?)", ["duplicate-chat", "blue",
                         reference["created_at"], reference["content"], content_hash])
            self.assertEqual(fetch_peer_followup(conn, reference)["status"], "ambiguous_or_mismatched_peer_chat")
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
