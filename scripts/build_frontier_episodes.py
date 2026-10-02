#!/usr/bin/env python3
"""Build eight bounded, redacted pilot packets on Azure; print aggregates only.

Packets and their metadata manifest remain under restricted/ with mode 0600.
The packet format is defined by swarmguard.schema; redaction is best effort and
does not itself authorize sending these research records to another service.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import resource
import sys
import time
from urllib.parse import urlsplit, urlunsplit

import duckdb
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarmguard.schema import validate_packet
from swarmguard.followup import fetch_peer_followup, followup_record_specs


ROOT = Path("/home/azureuser/ai-swarm")
SCRATCH = Path("/mnt/ai-swarm-scratch")
DERIVED = SCRATCH / "derived"
RESTRICTED = ROOT / "restricted"
SEED = "swarmguard-frontier-pilot-v1-20261002"
MAX_RECORDS = 40
MAX_PACKET_CHARS = 40000
RECORD_BUDGET = 31000
URL = re.compile(r"https?://[^\s<>\"'`]+", re.IGNORECASE)
EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])")
IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
IPV6 = re.compile(r"(?<![\w:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![\w:])")
PHONE = re.compile(r"(?<!\w)\+?\d[\d ().-]{7,}\d(?!\w)")
ENTROPY_TOKEN = re.compile(r"(?<![\w])[A-Za-z0-9_+/=-]{24,}(?![\w])")
KNOWN_TOKEN = re.compile(
    r"\b(?:sk[-_][A-Za-z0-9_-]{12,}|hf_[A-Za-z0-9]{12,}|"
    r"gh[pousr]_[A-Za-z0-9]{12,}|github_pat_[A-Za-z0-9_]{12,}|"
    r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)\b")
SECRET_ASSIGNMENT = re.compile(
    r"(?i)(\b(?:api[-_ ]?key|access[-_ ]?token|refresh[-_ ]?token|"
    r"client[-_ ]?secret|password|passwd|accountkey|connectionstring|"
    r"authorization|cookie|set-cookie|secret)\b[\"']?\s*[:=]\s*)"
    r"(?:\"[^\"\n]*\"|'[^'\n]*'|[^\s,;}\n]+)")
PRIVATE_KEY = re.compile(
    r"-----BEGIN [^-\n]*PRIVATE KEY-----.*?(?:-----END [^-\n]*PRIVATE KEY-----|\Z)",
    re.DOTALL)
ARTIFACT_PATH = re.compile(r"(?<!\w)(?:/home/|/tmp/|/mnt/|\./)[A-Za-z0-9_./-]{8,}")
QUOTATIONS = re.compile(r'"([^"\n]{32,500})"|“([^”\n]{32,500})”|(?m:^[ \t]*>[ \t]*([^\n]{32,500})$)')


def utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def timestamp(value: str) -> str:
    return utc(value).isoformat()


def key(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def redact(text: str) -> tuple[str, dict[str, int]]:
    counts: Counter = Counter()

    def replace(pattern: re.Pattern, label: str, value: str) -> str:
        def sub(match: re.Match) -> str:
            counts[label] += 1
            return f"[REDACTED_{label.upper()}]"
        return pattern.sub(sub, value)

    text = replace(PRIVATE_KEY, "private_key", text)
    text = replace(KNOWN_TOKEN, "token", text)
    text = replace(SECRET_ASSIGNMENT, "credential_assignment", text)
    text = re.sub(r"(?i)\bBearer\s+\S+", lambda _: counts.update(["bearer"]) or
                  "Bearer [REDACTED_TOKEN]", text)

    def safe_url(match: re.Match) -> str:
        original = match.group()
        try:
            parts = urlsplit(original)
            if not parts.hostname:
                raise ValueError
            hostname = parts.hostname
            try:
                ipaddress.ip_address(hostname)
                hostname = "redacted-host.invalid"
                counts["url_ip"] += 1
            except ValueError:
                pass
            if parts.username is not None or parts.password is not None:
                counts["url_credentials"] += 1
            if parts.query or parts.fragment:
                counts["url_query_fragment"] += 1
            port = f":{parts.port}" if parts.port else ""
            return urlunsplit((parts.scheme, hostname + port, parts.path, "", ""))
        except (ValueError, UnicodeError):
            counts["url"] += 1
            return "[REDACTED_URL]"

    text = URL.sub(safe_url, text)
    text = replace(EMAIL, "email", text)

    def safe_ip(match: re.Match) -> str:
        try:
            ipaddress.ip_address(match.group())
        except ValueError:
            return match.group()
        counts["ip_address"] += 1
        return "[REDACTED_IP]"

    text = IPV4.sub(safe_ip, text)
    text = IPV6.sub(safe_ip, text)

    def safe_phone(match: re.Match) -> str:
        if len(re.sub(r"\D", "", match.group())) < 9:
            return match.group()
        counts["phone_or_number"] += 1
        return "[REDACTED_PHONE_OR_NUMBER]"

    text = PHONE.sub(safe_phone, text)

    def safe_entropy(match: re.Match) -> str:
        value = match.group()
        frequencies = Counter(value)
        entropy = -sum((n / len(value)) * math.log2(n / len(value))
                       for n in frequencies.values())
        if re.fullmatch(r"[a-fA-F0-9]{24,}", value) or (
                entropy >= 3.5 and any(c.isdigit() for c in value)
                and any(c.isalpha() for c in value)):
            counts["high_entropy"] += 1
            return "[REDACTED_HIGH_ENTROPY]"
        return value

    text = ENTROPY_TOKEN.sub(safe_entropy, text)
    text = re.sub(r"/(?:home|Users)/[^/\s]+", lambda _: counts.update(["user_path"]) or
                  "/home/[REDACTED_USER]", text)
    return text, dict(counts)


def clip(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    marker = "\n[TRUNCATED_BY_PACKET_BUILDER]\n"
    room = max(0, limit - len(marker))
    head = room * 3 // 4
    return text[:head] + marker + (text[-(room-head):] if room > head else ""), True


def read_small_table(name: str, maximum_rows: int = 1000) -> list[dict]:
    rows = []
    with gzip.open(SCRATCH / "raw" / f"{name}.jsonl.gz", "rt") as stream:
        for line in stream:
            if len(rows) >= maximum_rows:
                raise RuntimeError("SmallTableRowBoundExceeded")
            rows.append(json.loads(line))
    return rows


def rows(conn: duckdb.DuckDBPyConnection, sql: str, params: list | None = None) -> list[dict]:
    result = conn.execute(sql, params or [])
    return [dict(zip([c[0] for c in result.description], row)) for row in result.fetchall()]


def active(goal: dict, at: datetime) -> bool:
    # A missing scheduled start is not evidence the assignment existed before
    # its creation in the dataset.
    start, end = goal.get("start_time") or goal.get("created_at"), goal.get("end_time")
    return bool(start) and utc(start) <= at and (not end or at < utc(end))


def artifacts(text: str) -> set[str]:
    found = set(ARTIFACT_PATH.findall(text))
    for match in URL.finditer(text):
        try:
            parts = urlsplit(match.group().rstrip(".,);]"))
            if len(parts.path.strip("/")) > 3 and "REDACTED" not in match.group():
                found.add(urlunsplit((parts.scheme, parts.netloc, parts.path, "", "")))
        except ValueError:
            pass
    return {item for item in found if "REDACTED" not in item}


def quotations(text: str) -> list[str]:
    found = []
    for match in QUOTATIONS.finditer(text):
        quote = next(group for group in match.groups() if group is not None).strip()
        if len(quote) >= 32 and quote not in found:
            found.append(quote)
        if len(found) == 3:
            break
    return found


def build_packet(conn: duckdb.DuckDBPyConnection, selected: dict,
                 agents: dict, rooms: dict, village_goals: list,
                 agent_goals: list, *, include_peer_followup: bool = False) -> tuple[dict, dict]:
    result = rows(conn, """
        SELECT e.*, l.chat_at, l.match_delta_ms,
               c.content AS chat_content, c.agent_id AS chat_agent,
               c.content_hash AS chat_hash, c.created_at AS source_chat_at,
               s.created_at AS session_created_at
        FROM evidence_windows e JOIN sent_message_links l USING (send_turn_id)
        JOIN chat_messages c ON e.chat_id = c.id
        JOIN computer_use_sessions s ON e.session_id = s.id
        WHERE e.send_turn_id = ?
    """, [selected["send_turn_id"]])
    if len(result) != 1:
        raise RuntimeError("AnchorSourceJoinNotUnique")
    anchor = result[0]
    content = anchor["message_content"]
    if not isinstance(content, str) or content.strip() != (anchor["chat_content"] or "").strip():
        raise RuntimeError("AnchorMessageMismatch")
    if anchor["agent_id"] != anchor["chat_agent"] or anchor["match_delta_ms"] > 300000:
        raise RuntimeError("AnchorAgentOrTimeMismatch")

    # A deterministic hash join is called provenance only if the candidate is
    # unambiguous. Audit all alternative source chats without printing identifiers.
    candidates = rows(conn, """
        SELECT id, content, created_at FROM chat_messages
        WHERE agent_id = ? AND content_hash = ?
          AND abs(date_diff('second', CAST(? AS TIMESTAMP), CAST(created_at AS TIMESTAMP))) <= 300
        ORDER BY abs(date_diff('millisecond', CAST(? AS TIMESTAMP), CAST(created_at AS TIMESTAMP))), id
        LIMIT 3
    """, [anchor["agent_id"], anchor["chat_hash"], anchor["send_at"], anchor["send_at"]])
    exact = [r for r in candidates if (r["content"] or "").strip() == content.strip()]
    proven_anchor = len(candidates) == 1 and len(exact) == 1 and exact[0]["id"] == anchor["chat_id"]

    episode_id = "pilot-" + key(selected["send_turn_id"])[:20]
    packet = {"episode_id": episode_id, "records": [], "edges": []}
    metadata = {
        "diagnostic_only": True,
        "sample_arm": selected["sample_arm"], "source_send_turn_id": anchor["send_turn_id"],
        "source_chat_id": anchor["chat_id"], "source_session_id": anchor["session_id"],
        "anchor_time": timestamp(anchor["send_at"]), "records": {},
        "source_window_truncated": anchor["window_truncated"],
        "source_prior_session_actions": anchor["prior_session_actions"],
        "anchor_provenance_unique": proven_anchor,
        "anchor_candidate_count_capped_at_three": len(candidates),
        "selection": ("unique prior-quotation sources with observed reference target" if selected.get("required_target_chat_id")
                      else "four lowest-rank distinct-session leads and four stable-hash controls"),
        "context_is_retrospective": True, "cross_agent_window_seconds": 3600,
        "goal_context_is_not_per_action_authorization": True,
        "limitations": [
            "Goal dates and export identity mappings establish scoped context, not historical receipt or authorization.",
            "Same-room proximity and shared artifacts do not establish influence or coordination.",
            "No screenshot, full tool output, or cross-session search is included.",
            "Redaction is best effort and may remove evidence; it does not guarantee absence of sensitive information.",
            "Tool output timestamps use their source turn timestamp; they are not separate completion timestamps."]}

    def add(record_id: str, kind: str, agent: str | None, at: str, text: str,
            source: dict, limit: int = 1300) -> str | None:
        if len(packet["records"]) >= MAX_RECORDS:
            return None
        if any(r["id"] == record_id for r in packet["records"]):
            return record_id
        sanitized, redactions = redact(text)
        centered_excerpt = False
        quote = selected.get("quote")
        if quote and kind == "chat" and quote in text:
            clean_quote = redact(quote)[0]
            position = sanitized.find(clean_quote)
            if position >= 0 and clean_quote not in clip(sanitized, limit)[0]:
                start = max(0, position - 180)
                marker = "[SOURCE_EXCERPT_AROUND_EXPLICIT_QUOTATION]\n"
                sanitized = marker + sanitized[start:start + limit - len(marker)]
                centered_excerpt = True
        sanitized, truncated = clip(sanitized, limit)
        record = {"id": record_id, "kind": kind, "agent_id": agent,
                  "at": timestamp(at), "text": sanitized}
        if len(dumps(packet)) + len(dumps(record)) > RECORD_BUDGET:
            return None
        packet["records"].append(record)
        metadata["records"][record_id] = {
            **source, "input_chars": len(text), "packet_chars": len(sanitized),
            "redactions": redactions, "builder_truncated": truncated or centered_excerpt or bool(source.get("projection_truncated")),
            "excerpt_centered_on_reference": centered_excerpt,
            "existing_truncation_marker": "[TRUNCATED]" in text}
        return record_id

    def edge(source: str | None, target: str | None, relation: str, basis: str) -> None:
        if not source or not target or source == target:
            return
        by_id = {r["id"]: r for r in packet["records"]}
        if utc(by_id[source]["at"]) > utc(by_id[target]["at"]):
            source, target = target, source
        item = {"id": "edge-" + key(f"{source}:{target}:{relation}")[:20],
                "source": source, "target": target, "relation": relation, "basis": basis}
        if item not in packet["edges"] and len(packet["edges"]) < 48:
            packet["edges"].append(item)

    send_id = add("turn:" + anchor["send_turn_id"], "action", anchor["agent_id"],
                  anchor["send_at"], "Action: send_message_back_to_chat\nContent:\n" + content,
                  {"table": "computer_use_turns", "field": "agent_action.content"}, 1200)
    chat_id = add("chat:" + anchor["chat_id"], "chat", anchor["agent_id"],
                  anchor["source_chat_at"], anchor["chat_content"],
                  {"table": "chat_messages", "field": "content"}, 1800)
    edge(send_id, chat_id, "same_message_provenance" if proven_anchor else "ambiguous_message_match",
         "direct" if proven_anchor else "temporal_only")
    metadata["provenance_edge_reversed_for_timestamp_order"] = utc(anchor["send_at"]) > utc(anchor["source_chat_at"])
    if anchor["session_goal"]:
        add("session-goal:" + anchor["session_id"], "goal", anchor["agent_id"],
            anchor["session_created_at"],
            "[Scope: this computer-use session's stated goal; external authorization is not established.]\n"
            + anchor["session_goal"], {"table": "computer_use_sessions", "field": "session_goal",
                                     "applicability": "same_session_self_stated"}, 1000)

    anchor_at = utc(anchor["send_at"])
    agent_village = agents.get(anchor["agent_id"], {}).get("village_id")
    room_village = rooms.get(anchor["room_id"], {}).get("village_id")
    village_match = agent_village is not None and agent_village == room_village
    metadata["anchor_agent_room_village_match"] = village_match
    applicable_village = [g for g in village_goals if village_match
                          and g.get("village_id") == agent_village and active(g, anchor_at)]
    for goal in sorted(applicable_village, key=lambda g: g.get("start_time") or "")[:2]:
        add("village-goal:" + goal["id"], "goal", None,
            goal.get("start_time") or goal["created_at"],
            "[Dataset village-wide goal. Export agent/room village identities match; historical membership is unverified. "
            f"Scheduled interval: {goal.get('start_time')} to {goal.get('end_time') or 'open-ended'}. "
            "A current goal is not permission for every action.]\n" + goal["goal"],
            {"table": "village_goals", "field": "goal", "applicability": "village_identity_and_interval",
             "active_at_anchor": True}, 1100)
    metadata["applicable_village_goals"] = len(applicable_village)

    # Retrieve a bounded set before prioritizing explicit references/shared objects.
    other_chats = rows(conn, """
        SELECT id, agent_id, created_at, content,
               abs(date_diff('millisecond', CAST(? AS TIMESTAMP), CAST(created_at AS TIMESTAMP))) AS gap_ms
        FROM chat_messages WHERE room_id = ? AND agent_id IS NOT NULL AND agent_id <> ?
          AND abs(epoch_us(CAST(created_at AS TIMESTAMP)) - epoch_us(CAST(? AS TIMESTAMP))) <= 3600000000
        ORDER BY gap_ms, id LIMIT 48
    """, [anchor["send_at"], anchor["room_id"], anchor["agent_id"], anchor["send_at"]])
    anchor_artifacts = artifacts(redact(content)[0])
    other_chats.sort(key=lambda r: (
        -int(anchor["chat_id"] in (r["content"] or "")),
        -int(any(quote in content for quote in quotations(r["content"] or ""))),
        -int(bool(anchor_artifacts & artifacts(redact(r["content"] or "")[0]))),
        r["gap_ms"], key(r["id"])))
    chosen_chats = other_chats[:10]
    if selected.get("required_target_chat_id"):
        required = rows(conn, """
            SELECT id, agent_id, created_at, content,
                   abs(date_diff('millisecond', CAST(? AS TIMESTAMP), CAST(created_at AS TIMESTAMP))) AS gap_ms
            FROM chat_messages WHERE id = ? AND room_id = ? AND agent_id <> ?
              AND CAST(created_at AS TIMESTAMP) > CAST(? AS TIMESTAMP)
        """, [anchor["send_at"], selected["required_target_chat_id"], anchor["room_id"],
                anchor["agent_id"], anchor["source_chat_at"]])
        if len(required) != 1 or selected.get("quote", "") not in (required[0]["content"] or ""):
            raise RuntimeError("RequiredReferenceTargetInvalid")
        chosen_chats = required + [c for c in chosen_chats if c["id"] != required[0]["id"]][:9]
        metadata["required_target_outside_one_hour"] = required[0]["gap_ms"] > 3600000
    metadata["other_agent_chat_pool_capped_at_48"] = len(other_chats)
    metadata["other_agent_chats_selected_before_record_bounds"] = len(chosen_chats)

    actor_ids = {anchor["agent_id"], *(r["agent_id"] for r in chosen_chats)}
    applicable_agent = [g for g in agent_goals if g.get("agent_id") in actor_ids and active(g, anchor_at)]
    applicable_agent.sort(key=lambda g: (g["agent_id"] != anchor["agent_id"], key(g["id"])))
    for goal in applicable_agent[:3]:
        body = "\n".join(str(goal[k]) for k in ("name", "short_name", "description") if goal.get(k))
        add("agent-goal:" + goal["id"], "goal", goal["agent_id"],
            goal.get("start_time") or goal["created_at"],
            "[Dataset per-agent goal assignment. Exact agent identity and scheduled interval match the anchor time; "
            "receipt and applicability to other times are unverified. "
            f"Effective interval: {goal.get('start_time') or goal.get('created_at')} to {goal.get('end_time') or 'open-ended'}.]\n"
            + body, {"table": "agent_goals", "fields": ["name", "short_name", "description"],
                     "applicability": "exact_agent_identity_and_interval", "active_at_anchor": True}, 850)
    metadata["applicable_agent_goals_before_bound"] = len(applicable_agent)

    for chat in chosen_chats:
        record_id = add("chat:" + chat["id"], "chat", chat["agent_id"], chat["created_at"],
                        chat["content"] or "", {"table": "chat_messages", "field": "content",
                                                  "retrieval": "same_room_plus_or_minus_one_hour_other_agent"}, 900)
        if record_id:
            by_id = {r["id"]: r for r in packet["records"]}
            # A literal source-ID reference must survive redaction and precede its
            # target. A quotation/overlap alone is not a strong influence edge.
            if anchor["chat_id"] in by_id[record_id]["text"] and utc(anchor["source_chat_at"]) <= utc(chat["created_at"]):
                edge(chat_id, record_id, "explicit_source_message_id_reference", "explicit_reference")
            elif artifacts(by_id[chat_id]["text"]) & artifacts(by_id[record_id]["text"]):
                edge(chat_id, record_id, "mentions_shared_artifact_without_proven_influence", "shared_artifact")
            else:
                edge(chat_id, record_id, "same_room_temporal_context_only", "temporal_only")

    # Retain later peer behavior before the source's prior window can exhaust
    # the packet budget. Temporal follow-up never becomes an adoption edge.
    peer_followup = {"status": "not_requested", "turns": []}
    followup_actions = followup_results = expected_followup_results = 0
    if include_peer_followup and selected.get("required_target_chat_id"):
        reference = next((chat for chat in chosen_chats if chat["id"] == selected["required_target_chat_id"]), None)
        reference_id = "chat:" + selected["required_target_chat_id"]
        if reference and any(record["id"] == reference_id for record in packet["records"]):
            peer_followup = fetch_peer_followup(conn, reference)
            for specs in followup_record_specs(peer_followup):
                action = specs["action"]
                aid = add(action["id"], action["kind"], action["agent_id"], action["at"],
                          action["text"], action["source"], action["limit"])
                followup_actions += aid is not None
                if aid:
                    edge(reference_id, aid, "same_peer_session_followup_temporal_context", "temporal_only")
                outcome = specs["result"]
                expected_followup_results += outcome is not None
                if aid and outcome:
                    rid = add(outcome["id"], outcome["kind"], outcome["agent_id"], outcome["at"],
                              outcome["text"], outcome["source"], outcome["limit"])
                    followup_results += rid is not None
                    edge(aid, rid, "action_has_recorded_result", "direct")
        else:
            peer_followup = {"status": "reference_target_not_retained", "turns": []}
    metadata["peer_followup"] = {key: value for key, value in peer_followup.items() if key != "turns"}
    metadata["peer_followup"].update(
        selected_actions=len(peer_followup["turns"]), retained_actions=followup_actions,
        selected_results=expected_followup_results, retained_results=followup_results,
        packet_bound_omitted_actions=followup_actions < len(peer_followup["turns"]),
        packet_bound_omitted_results=followup_results < expected_followup_results)
    if include_peer_followup:
        metadata["limitations"].append(
            "Peer follow-up uses at most four projected actions within fifteen minutes, in the uniquely matched sending session, "
            "strictly after both the reference chat and send turn. It excludes other sessions and does not prove adoption or causality.")

    previous = json.loads(anchor["evidence_json"])[-10:]
    retained_tool_turns = 0
    # Reserve space for cross-agent context, then prioritize the most recent
    # source actions. Final records are ordered chronologically below.
    for turn in reversed(previous):
        action_text = "Action: " + str(turn.get("action") or "unknown")
        for field in ("command", "typed_text"):
            if turn.get(field):
                action_text += f"\n{field}:\n{turn[field]}"
        action_id = add("turn:" + turn["turn_id"], "action", anchor["agent_id"], turn["at"], action_text,
                        {"table": "computer_use_turns", "fields": ["agent_action"],
                         "retrieval": "last_ten_of_existing_twenty_action_window"}, 600)
        retained_tool_turns += action_id is not None
        results = []
        for field in ("output", "error"):
            if turn.get(field) is not None:
                results.append(f"{field}:\n{turn[field]}")
        if results and action_id:
            result_id = add("result:" + turn["turn_id"], "result", anchor["agent_id"], turn["at"],
                            "\n".join(results),
                            {"table": "computer_use_turns", "fields": ["output", "error"],
                             "source_output_chars": turn.get("output_chars"),
                             "source_error_chars": turn.get("error_chars"),
                             "timestamp_is_parent_turn": True}, 850)
            edge(action_id, result_id, "action_has_recorded_result", "direct")
    metadata["prior_tool_turns_selected"] = len(previous)
    metadata["prior_tool_turns_retained"] = retained_tool_turns

    target_window = []
    metadata["target_tool_turns_retained"] = 0
    if selected.get("required_target_chat_id"):
        target_window = rows(conn, """
            SELECT agent_id, evidence_json, window_truncated, session_id
            FROM evidence_windows WHERE chat_id = ?
        """, [selected["required_target_chat_id"]])
        if len(target_window) == 1:
            metadata["target_source_window_truncated"] = target_window[0]["window_truncated"]
            for turn in reversed(json.loads(target_window[0]["evidence_json"])[-3:]):
                text = "Action: " + str(turn.get("action") or "unknown")
                for field in ("command", "typed_text"):
                    if turn.get(field):
                        text += f"\n{field}:\n{turn[field]}"
                aid = add("turn:" + turn["turn_id"], "action", target_window[0]["agent_id"],
                          turn["at"], text, {"table": "computer_use_turns",
                          "retrieval": "up_to_three_actions_before_explicit_reference_target"}, 600)
                metadata["target_tool_turns_retained"] += aid is not None
                result = "\n".join(f"{field}:\n{turn[field]}" for field in ("output", "error")
                                   if turn.get(field) is not None)
                if result and aid:
                    rid = add("result:" + turn["turn_id"], "result", target_window[0]["agent_id"],
                              turn["at"], result, {"table": "computer_use_turns",
                              "timestamp_is_parent_turn": True}, 850)
                    edge(aid, rid, "action_has_recorded_result", "direct")

    # Exact quotations are stronger than shared artifacts, but establish only
    # a textual reference. Check uniqueness over all earlier projected chats.
    chat_sources = [{"id": anchor["chat_id"], "agent_id": anchor["agent_id"],
                     "created_at": anchor["source_chat_at"], "content": anchor["chat_content"]}, *chosen_chats]
    by_id = {r["id"]: r for r in packet["records"]}
    quotation_checks = 0
    for target in chat_sources:
        target_id = "chat:" + target["id"]
        if target_id not in by_id:
            continue
        for quote in quotations(target["content"] or ""):
            possible = [source for source in chat_sources
                        if source["agent_id"] != target["agent_id"]
                        and utc(source["created_at"]) < utc(target["created_at"])
                        and quote in (source["content"] or "")
                        and "chat:" + source["id"] in by_id]
            if len(possible) != 1:
                continue
            source = possible[0]
            source_id = "chat:" + source["id"]
            clean_quote = redact(quote)[0]
            if clean_quote not in by_id[source_id]["text"] or clean_quote not in by_id[target_id]["text"]:
                continue
            uniqueness = rows(conn, """
                SELECT count(*) AS matches, min(id) AS unique_id
                FROM chat_messages WHERE id <> ?
                  AND CAST(created_at AS TIMESTAMP) < CAST(? AS TIMESTAMP)
                  AND strpos(content, ?) > 0
            """, [target["id"], target["created_at"], quote])[0]
            quotation_checks += 1
            if uniqueness["matches"] == 1 and uniqueness["unique_id"] == source["id"]:
                edge(source_id, target_id, "quoted_prior_message", "explicit_reference")
    metadata["quotation_uniqueness_checks"] = quotation_checks
    metadata["quotation_uniqueness_scope"] = "all_projected_chat_before_target"

    # Keep the on-disk packet inside the strict serialized character budget.
    while len(dumps(packet)) > MAX_PACKET_CHARS:
        removable = next((i for i, e in enumerate(packet["edges"])
                          if e["basis"] in ("temporal_only", "shared_artifact")), None)
        if removable is not None:
            packet["edges"].pop(removable)
            continue
        longest = max(packet["records"], key=lambda r: len(r["text"]))
        if len(longest["text"]) < 200:
            raise RuntimeError("PacketSizeBoundUnachievable")
        longest["text"], _ = clip(longest["text"], len(longest["text"]) - 100)
        metadata["records"][longest["id"]]["packet_bound_truncated"] = True
    packet["records"].sort(key=lambda r: (utc(r["at"]), r["id"]))
    validation = validate_packet(packet)
    if not validation.valid:
        raise RuntimeError("PacketSchemaValidationFailed")
    metadata["validation_warning_count"] = len(validation.warnings)
    metadata["serialized_chars"] = len(dumps(packet))
    metadata["packet_sha256"] = hashlib.sha256(dumps(packet).encode()).hexdigest()
    source_bundle = {"anchor": anchor, "candidate_chats": candidates,
                     "prior_tool_window": previous, "other_chats": chosen_chats,
                     "reference_target_window": target_window,
                     "peer_followup_context": peer_followup,
                     "village_goals": applicable_village[:2], "agent_goals": applicable_agent[:3]}
    metadata["source_bundle_sha256"] = hashlib.sha256(dumps(source_bundle).encode()).hexdigest()
    metadata["source_hash_scope"] = "materialized source fields used for this packet, not full dataset files"
    metadata["edge_basis_counts"] = dict(Counter(e["basis"] for e in packet["edges"]))
    metadata["cross_agent_strong_edges"] = sum(e["basis"] == "explicit_reference" for e in packet["edges"])
    return packet, metadata


def private_write(path: Path, text: str) -> None:
    staged = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(staged, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(text)
    staged.replace(path)
    path.chmod(0o600)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-candidates", type=Path)
    parser.add_argument("--include-peer-followup", action="store_true",
                        help="Add bounded same-session peer actions after explicit references; writes a new followup output stem")
    args = parser.parse_args()
    if sys.platform != "linux" or not SCRATCH.is_mount():
        parser.error("Run only on the Azure VM with scratch mounted")
    if args.include_peer_followup and not args.reference_candidates:
        parser.error("--include-peer-followup requires --reference-candidates")
    names = ("review_queue", "evidence_windows", "sent_message_links", "chat_messages", "computer_use_sessions")
    if args.include_peer_followup:
        names += ("computer_use_turns",)
    if any(not (DERIVED / f"{name}.parquet").is_file() for name in names):
        parser.error("Required projected tables and review queue are missing")
    for name in ("agents", "chat_rooms", "village_goals", "agent_goals"):
        if not (SCRATCH / "raw" / f"{name}.jsonl.gz").is_file():
            parser.error("Required small identity/goal tables are missing")
    if psutil.disk_usage(SCRATCH).free < 12 * 1024**3:
        parser.error("At least 12 GiB free scratch is required")
    started = time.monotonic()
    cpu_started = time.process_time()
    conn = duckdb.connect()
    try:
        conn.execute("SET threads = 8")
        conn.execute("SET memory_limit = '8GB'")
        temp = SCRATCH / "duckdb-frontier-packets-temp"
        temp.mkdir(exist_ok=True)
        conn.execute(f"SET temp_directory = '{temp}'")
        conn.execute("SET max_temp_directory_size = '4GB'")
        for name in names:
            conn.execute(f"CREATE TEMP VIEW {name} AS SELECT * FROM read_parquet('{DERIVED / (name + '.parquet')}')")
        queue = rows(conn, """
            SELECT q.*, e.session_id FROM review_queue q
            JOIN evidence_windows e USING (send_turn_id)
        """)
        selected = []
        used_sessions = set()
        for row in sorted((r for r in queue if r["sample_arm"] == "top_ranked"),
                          key=lambda r: (r["rank"], key(r["send_turn_id"]))):
            if row["session_id"] not in used_sessions:
                selected.append(row)
                used_sessions.add(row["session_id"])
            if len(selected) == 4:
                break
        controls = []
        for row in sorted((r for r in queue if r["sample_arm"] == "random_control"),
                          key=lambda r: key(r["send_turn_id"])):
            if row["session_id"] not in used_sessions:
                controls.append(row)
                used_sessions.add(row["session_id"])
            if len(controls) == 4:
                break
        selected.extend(controls)
        if args.reference_candidates:
            if not args.reference_candidates.resolve().is_relative_to(RESTRICTED):
                raise RuntimeError("ReferencesMustStayUnderRestricted")
            if args.reference_candidates.stat().st_size > 20_000_000:
                raise RuntimeError("ReferenceFileBoundExceeded")
            candidates = [json.loads(line) for line in args.reference_candidates.read_text().splitlines() if line.strip()]
            selected, used_sessions = [], set()
            for candidate in candidates:
                source = rows(conn, """
                    SELECT send_turn_id, session_id FROM evidence_windows WHERE chat_id = ?
                """, [candidate["source_chat_id"]])
                if len(source) != 1 or source[0]["session_id"] in used_sessions:
                    continue
                used_sessions.add(source[0]["session_id"])
                selected.append({**source[0], "sample_arm": "explicit_reference",
                                 "required_target_chat_id": candidate["target_chat_id"],
                                 "quote": candidate["quote"]})
                if len(selected) == 8:
                    break
            if not selected:
                raise RuntimeError("NoEligibleReferenceSources")
        if len(selected) > 8 or len({r["send_turn_id"] for r in selected}) != len(selected):
            raise RuntimeError("PilotSelectionInvalid")
        agents = {r["id"]: r for r in read_small_table("agents")}
        rooms = {r["id"]: r for r in read_small_table("chat_rooms")}
        village_goals = read_small_table("village_goals")
        agent_goals = read_small_table("agent_goals")
        packets, manifest = [], {}
        for row in selected:
            packet, metadata = build_packet(conn, row, agents, rooms, village_goals, agent_goals,
                                            include_peer_followup=args.include_peer_followup)
            metadata["source_queue_sha256"] = hashlib.sha256((DERIVED / "review_queue.parquet").read_bytes()).hexdigest()
            metadata["builder_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            packets.append(packet)
            manifest[packet["episode_id"]] = metadata
        RESTRICTED.mkdir(mode=0o700, parents=True, exist_ok=True)
        RESTRICTED.chmod(0o700)
        stem = "frontier-reference" if args.reference_candidates else "frontier-pilot"
        if args.include_peer_followup:
            stem += "-followup"
        private_write(RESTRICTED / f"{stem}-packets.jsonl", "".join(dumps(p) + "\n" for p in packets))
        private_write(RESTRICTED / f"{stem}-manifest.json", json.dumps(manifest, indent=2) + "\n")
        all_metadata = [m for ep in manifest.values() for m in ep["records"].values()]
        redactions = Counter()
        for item in all_metadata:
            redactions.update(item["redactions"])
        report = {
            "status": "complete", "builder_version": 2, "packet_count": len(packets),
            "peer_followup_requested": args.include_peer_followup,
            "diagnostic_only": True,
            "protocol_deviations": [
                "The pilot sample is not era- or length-matched and is not a frozen scored evaluation.",
                "Prior limited excerpt review may overlap these cases; independent holdout status is not established.",
                "Only up to ten preceding tool turns are selected from existing twenty-action windows; packet bounds may retain fewer.",
                "Retrospective same-room context includes up to one hour after the anchor, so this is not prospective detection.",
                "No accuracy, recall, prevalence, or misalignment-rate estimate is supported by this pilot."],
            "selected_by_arm": dict(Counter(row["sample_arm"] for row in selected)),
            "record_count": sum(len(p["records"]) for p in packets),
            "maximum_records_per_packet": max((len(p["records"]) for p in packets), default=0),
            "maximum_serialized_chars": max((len(dumps(p)) for p in packets), default=0),
            "unique_anchor_provenance_packets": sum(m["anchor_provenance_unique"] for m in manifest.values()),
            "cross_agent_strong_edges": sum(m["cross_agent_strong_edges"] for m in manifest.values()),
            "packets_with_target_tool_actions": sum(m["target_tool_turns_retained"] > 0 for m in manifest.values()),
            "target_tool_turns_retained": sum(m["target_tool_turns_retained"] for m in manifest.values()),
            "peer_followup_status_counts": dict(Counter(m["peer_followup"]["status"] for m in manifest.values())),
            "peer_followup_actions_retained": sum(m["peer_followup"]["retained_actions"] for m in manifest.values()),
            "peer_followup_results_retained": sum(m["peer_followup"]["retained_results"] for m in manifest.values()),
            "peer_followup_packets_with_actions": sum(m["peer_followup"]["retained_actions"] > 0 for m in manifest.values()),
            "edge_basis_counts": dict(Counter(e["basis"] for p in packets for e in p["edges"])),
            "redaction_counts": dict(redactions),
            "builder_truncated_records": sum(m["builder_truncated"] for m in all_metadata),
            "source_truncated_windows": sum(m["source_window_truncated"] for m in manifest.values()),
            "wall_seconds": round(time.monotonic() - started, 3),
            "process_cpu_seconds": round(time.process_time() - cpu_started, 3),
            "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "scratch_free_bytes": psutil.disk_usage(SCRATCH).free,
            "duckdb_threads": 8, "duckdb_memory_limit_gib": 8,
            "raw_text_in_report": False, "record_ids_in_report": False,
            "redaction_is_complete_privacy_guarantee": False,
            "strong_cross_agent_links_are_not_assumed": True,
        }
        output = ROOT / f"reports/{stem}-packets.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        staged = output.with_suffix(".tmp")
        staged.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        staged.replace(output)
        print(json.dumps(report, sort_keys=True))
        return 0
    except Exception as exc:
        # Exception text/tracebacks can contain corpus values; suppress both.
        print(json.dumps({"status": "error", "error_type": type(exc).__name__,
                          "raw_text_in_report": False, "record_ids_in_report": False}))
        return 2
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
