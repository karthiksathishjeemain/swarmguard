"""Bounded same-session evidence after a referenced chat, without causal claims.

Queries use existing projected tables. A unique exact send/chat match scopes the
session; later actions remain temporal context, never inferred adoption.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib


FOLLOWUP_TURN_LIMIT = 4
FOLLOWUP_HORIZON_SECONDS = 900
FOLLOWUP_MESSAGE_CHARS = 1300


def utc(value: str) -> datetime:
    at = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return at.replace(tzinfo=timezone.utc) if at.tzinfo is None else at.astimezone(timezone.utc)


def _rows(conn, sql, params):
    result = conn.execute(sql, params)
    return [dict(zip([field[0] for field in result.description], row)) for row in result.fetchall()]


def select_followup_rows(candidates: list[dict], session_id: str, after: str,
                         until: str) -> tuple[list[dict], dict]:
    """Defend the row bound and interval independently of the SQL filter."""
    lower, upper = utc(after), utc(until)
    if upper <= lower or (upper - lower).total_seconds() > FOLLOWUP_HORIZON_SECONDS:
        raise ValueError("invalid_followup_interval")
    if len(candidates) > FOLLOWUP_TURN_LIMIT + 1:
        raise ValueError("followup_query_row_bound_exceeded")
    if len({row["id"] for row in candidates}) != len(candidates):
        raise ValueError("duplicate_followup_turn_id")
    if any(row["session_id"] != session_id or not lower < utc(row["created_at"]) <= upper
           for row in candidates):
        raise ValueError("followup_session_or_time_mismatch")
    ordered = sorted(candidates, key=lambda row: (utc(row["created_at"]), row["id"]))
    return ordered[:FOLLOWUP_TURN_LIMIT], {
        "eligible_turn_count_capped_at_five": len(ordered),
        "turn_limit_reached": len(ordered) > FOLLOWUP_TURN_LIMIT,
    }


def fetch_peer_followup(conn, target_chat: dict) -> dict:
    """Find an unambiguous send and retrieve at most five projected later turns.

    Both directions of the ±300-second exact-content association must be unique.
    The send/chat timestamps can differ; selection starts strictly after the
    later timestamp. Equal-time turns are deliberately excluded.
    """
    text = target_chat.get("content")
    if not isinstance(text, str) or not text.strip() or not target_chat.get("agent_id"):
        return {"status": "invalid_reference_chat", "turns": []}
    match_hash = hashlib.blake2b(text.strip().encode("utf-8"), digest_size=16).hexdigest()
    sends = _rows(conn, """
        SELECT t.id AS send_turn_id, t.session_id, s.agent_id,
               t.created_at AS send_at, t.message_content
        FROM computer_use_turns t
        JOIN computer_use_sessions s ON t.session_id = s.id
        WHERE t.action_type = 'send_message_back_to_chat' AND s.agent_id = ?
          AND t.message_hash = ?
          AND abs(epoch_us(CAST(t.created_at AS TIMESTAMP)) - epoch_us(CAST(? AS TIMESTAMP))) <= 300000000
        ORDER BY t.created_at, t.id LIMIT 3
    """, [target_chat["agent_id"], match_hash, target_chat["created_at"]])
    result = {"status": "ambiguous_or_missing_peer_send", "turns": [],
              "source_chat_id": target_chat["id"], "agent_id": target_chat["agent_id"],
              "send_candidates_capped_at_three": len(sends)}
    if (len(sends) != 1 or (sends[0].get("message_content") or "").strip() != text.strip()
            or sends[0].get("agent_id") != target_chat["agent_id"]
            or abs((utc(sends[0]["send_at"]) - utc(target_chat["created_at"])).total_seconds()) > 300):
        return result
    send = sends[0]
    chats = _rows(conn, """
        SELECT id, agent_id, created_at, content FROM chat_messages
        WHERE agent_id = ? AND content_hash = ?
          AND abs(epoch_us(CAST(created_at AS TIMESTAMP)) - epoch_us(CAST(? AS TIMESTAMP))) <= 300000000
        ORDER BY created_at, id LIMIT 3
    """, [target_chat["agent_id"], match_hash, send["send_at"]])
    result.update(status="ambiguous_or_mismatched_peer_chat", chat_candidates_capped_at_three=len(chats))
    if (len(chats) != 1 or chats[0]["id"] != target_chat["id"] or chats[0]["agent_id"] != target_chat["agent_id"] or
            utc(chats[0]["created_at"]) != utc(target_chat["created_at"]) or
            (chats[0].get("content") or "").strip() != text.strip()):
        return result
    after = max(utc(target_chat["created_at"]), utc(send["send_at"]))
    until = after + timedelta(seconds=FOLLOWUP_HORIZON_SECONDS)
    candidates = _rows(conn, """
        SELECT id, session_id, created_at, action_type,
               command_excerpt, command_chars, typed_text_excerpt, typed_text_chars,
               substring(message_content, 1, ?) AS message_excerpt, length(message_content) AS message_chars,
               output_excerpt, output_chars, error_excerpt, error_chars, screenshot_redacted
        FROM computer_use_turns
        WHERE session_id = ? AND CAST(created_at AS TIMESTAMP) > CAST(? AS TIMESTAMP)
          AND CAST(created_at AS TIMESTAMP) <= CAST(? AS TIMESTAMP)
          AND action_type IS NOT NULL AND action_type <> 'none'
        ORDER BY CAST(created_at AS TIMESTAMP), id LIMIT 5
    """, [FOLLOWUP_MESSAGE_CHARS, send["session_id"], after.isoformat(), until.isoformat()])
    turns, bounds = select_followup_rows(candidates, send["session_id"], after.isoformat(), until.isoformat())
    result.update(
        status="selected" if turns else "no_same_session_followup_in_window",
        session_id=send["session_id"], source_send_turn_id=send["send_turn_id"],
        send_at=send["send_at"], chat_at=target_chat["created_at"],
        after_exclusive=after.isoformat(), until_inclusive=until.isoformat(),
        horizon_seconds=FOLLOWUP_HORIZON_SECONDS, turn_limit=FOLLOWUP_TURN_LIMIT,
        same_session_only=True, projected_actions_only=True,
        does_not_establish_adoption=True, turns=turns, **bounds)
    return result


def followup_record_specs(context: dict) -> list[dict]:
    """Create loss-labelled source record specs; caller applies redaction/budgets."""
    if context.get("status") != "selected":
        return []
    specs = []
    for turn in context["turns"]:
        body = "Action: " + str(turn.get("action_type") or "unknown")
        for source, label in (("command_excerpt", "command"), ("typed_text_excerpt", "typed_text"),
                              ("message_excerpt", "message_content")):
            if turn.get(source):
                body += f"\n{label}:\n{turn[source]}"
        common = {"table": "computer_use_turns", "source_turn_id": turn["id"],
                  "source_session_id": context["session_id"],
                  "reference_chat_id": context["source_chat_id"],
                  "retrieval": "same_peer_session_strictly_after_reference_and_send",
                  "temporal_context_not_adoption": True}
        projected_truncation = any(
            type(turn.get(length)) is int and turn[length] > len(turn.get(field) or "")
            for field, length in (("command_excerpt", "command_chars"),
                                  ("typed_text_excerpt", "typed_text_chars"),
                                  ("message_excerpt", "message_chars")))
        action = {"id": "turn:" + turn["id"], "kind": "action", "agent_id": context["agent_id"],
                  "at": turn["created_at"], "text": body, "limit": 600,
                  "source": {**common, "fields": ["agent_action"],
                             "projection_truncated": projected_truncation}}
        observed = [f"{name}:\n{turn[field]}" for field, name in
                    (("output_excerpt", "output"), ("error_excerpt", "error")) if turn.get(field) is not None]
        outcome = None
        if observed:
            output_truncated = any(
                type(turn.get(length)) is int and turn[length] > len(turn.get(field) or "")
                for field, length in (("output_excerpt", "output_chars"), ("error_excerpt", "error_chars")))
            outcome = {"id": "result:" + turn["id"], "kind": "result", "agent_id": context["agent_id"],
                       "at": turn["created_at"], "text": "\n".join(observed), "limit": 850,
                       "source": {**common, "fields": ["output", "error"], "timestamp_is_parent_turn": True,
                                  "source_output_chars": turn.get("output_chars"),
                                  "source_error_chars": turn.get("error_chars"),
                                  "projection_truncated": output_truncated}}
        specs.append({"action": action, "result": outcome})
    return specs
