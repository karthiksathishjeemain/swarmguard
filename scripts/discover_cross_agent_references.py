#!/usr/bin/env python3
"""Discover exact, uniquely attributable marked quotations on Azure.

Source chat is scanned with Aho-Corasick over 32-character quote prefixes,
then full strings are verified. A quotation establishes a textual reference,
not endorsement, adoption, coordination, causality, or misalignment.
Only aggregate diagnostics are printed; quotations and source IDs stay in
0600 files under the VM's restricted directory.
"""

import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import hashlib
import heapq
import json
import os
from pathlib import Path
import re
import resource
import sys
import time

import psutil
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path("/home/azureuser/ai-swarm")
SCRATCH = Path("/mnt/ai-swarm-scratch")
DERIVED = SCRATCH / "derived"
RESTRICTED = ROOT / "restricted"
SEED = "swarmguard-exact-quote-discovery-v1"
QUOTES = re.compile(r'"([^"\n]{32,500})"|“([^”\n]{32,500})”|(?m:^[ \t]*>[ \t]*([^\n]{32,500})$)')
PREFIX_LENGTH = 32
MAX_OCCURRENCES = 20000
MAX_SOURCES_PER_QUOTE = 256
MAX_SOURCE_MATCHES = 300000
MAX_CANDIDATES = 2000
RSS_LIMIT = 8 * 1024**3


class BoundReached(RuntimeError):
    pass


def utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def stable(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def extract_quotes(text: str) -> list[str]:
    found = []
    for match in QUOTES.finditer(text):
        quote = next(g for g in match.groups() if g is not None).strip()
        if len(quote) >= PREFIX_LENGTH and quote not in found:
            found.append(quote)
        if len(found) == 3:
            break
    return found


class PrefixMatcher:
    """Fixed-length Aho-Corasick patterns require no inherited terminal lists."""

    def __init__(self, prefixes: list[str], guard):
        self.next: list[dict[str, int]] = [{}]
        self.fail = [0]
        self.terminal = [-1]
        for pattern_index, prefix in enumerate(prefixes):
            state = 0
            for char in prefix:
                child = self.next[state].get(char)
                if child is None:
                    child = len(self.next)
                    self.next[state][char] = child
                    self.next.append({})
                    self.fail.append(0)
                    self.terminal.append(-1)
                state = child
            self.terminal[state] = pattern_index
            if pattern_index % 256 == 0:
                guard()
        pending = deque(self.next[0].values())
        steps = 0
        while pending:
            parent = pending.popleft()
            for char, child in self.next[parent].items():
                pending.append(child)
                fallback = self.fail[parent]
                while fallback and char not in self.next[fallback]:
                    fallback = self.fail[fallback]
                self.fail[child] = self.next[fallback].get(char, 0)
            steps += 1
            if steps % 10000 == 0:
                guard()

    def matches(self, text: str, prefix_quotes: list[list[int]], quotes: list[str]) -> set[int]:
        found = set()
        state = 0
        transitions, failures, terminal = self.next, self.fail, self.terminal
        for position, char in enumerate(text):
            while state and char not in transitions[state]:
                state = failures[state]
            state = transitions[state].get(char, 0)
            pattern_index = terminal[state]
            if pattern_index >= 0:
                start = position - PREFIX_LENGTH + 1
                for quote_index in prefix_quotes[pattern_index]:
                    if quote_index not in found and text.startswith(quotes[quote_index], start):
                        found.add(quote_index)
        return found


def private_write(path: Path, value: str) -> None:
    staged = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(value)
    staged.replace(path)
    path.chmod(0o600)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args()
    if sys.platform != "linux" or not SCRATCH.is_mount():
        parser.error("Run only on the Azure VM with scratch mounted")
    if not 10 <= args.timeout_seconds <= 180:
        parser.error("Choose a 10-180 second timeout")
    inputs = {name: DERIVED / f"{name}.parquet" for name in
              ("chat_messages", "sent_message_links", "evidence_windows")}
    if any(not path.is_file() for path in inputs.values()):
        parser.error("Required projected chat, source links, or evidence windows missing")
    os.environ["OMP_NUM_THREADS"] = "8"
    pa.set_cpu_count(8)
    pa.set_io_thread_count(2)
    started = time.monotonic()
    cpu_started = time.process_time()
    process = psutil.Process()
    stage = "source_link_index"
    report = {
        "version": 1, "status": "running", "diagnostic_only": True,
        "raw_text_in_report": False, "record_ids_in_report": False,
        "reference_does_not_imply_adoption": True,
        "uniqueness_scope": "all earlier projected chats, including same-agent and other-room records",
        "source_and_target_require_send_links": True,
        "coverage_limits": [
            "Only ASCII/curly-double-quoted spans and Markdown blockquote lines of 32-500 characters are examined.",
            "At most three marked quotations are extracted per target message.",
            "At most 20000 target quotation occurrences are selected by deterministic hash reservoir.",
            "Targets must have saved send-to-chat links; every projected chat is scanned for duplicate sources.",
            "Quotes with more than 256 matching source messages are excluded.",
            "Unmarked quotations, paraphrases, and explicit name references are outside this diagnostic.",
            "A marked quotation can reject or criticize its source; no endorsement is inferred."],
    }
    candidates = []
    exit_code = 0

    def guard() -> None:
        if time.monotonic() - started > args.timeout_seconds:
            raise BoundReached("RuntimeBound")
        if process.memory_info().rss > RSS_LIMIT:
            raise BoundReached("MemoryBound")

    try:
        links = {}
        ambiguous_linked_chats = set()
        for batch in pq.ParquetFile(inputs["sent_message_links"]).iter_batches(
                batch_size=4096, columns=["chat_id", "send_turn_id", "session_id", "agent_id"], use_threads=False):
            for row in batch.to_pylist():
                chat_id = row["chat_id"]
                if chat_id is not None:
                    if chat_id in links:
                        ambiguous_linked_chats.add(chat_id)
                    else:
                        links[chat_id] = row
            guard()
        for chat_id in ambiguous_linked_chats:
            links.pop(chat_id, None)
        windows = {}
        for batch in pq.ParquetFile(inputs["evidence_windows"]).iter_batches(
                batch_size=4096, columns=["chat_id", "send_turn_id"], use_threads=False):
            for row in batch.to_pylist():
                if row["chat_id"]:
                    windows[row["chat_id"]] = row["send_turn_id"]
            guard()
        report["unique_linked_chats"] = len(links)
        report["ambiguous_linked_chats_excluded"] = len(ambiguous_linked_chats)
        report["chats_with_existing_evidence_windows"] = len(windows)

        stage = "target_quote_reservoir"
        reservoir = []
        target_rows, eligible_targets, occurrences = 0, 0, 0
        chat_columns = ["id", "agent_id", "room_id", "created_at", "content"]
        for batch in pq.ParquetFile(inputs["chat_messages"]).iter_batches(
                batch_size=2048, columns=chat_columns, use_threads=False):
            for row in batch.to_pylist():
                target_rows += 1
                if not row["agent_id"] or row["id"] not in links or not isinstance(row["content"], str):
                    continue
                eligible_targets += 1
                for quote in extract_quotes(row["content"]):
                    occurrences += 1
                    identity = stable(row["id"] + "\0" + quote)
                    score = int(identity, 16)
                    target = (row["id"], row["agent_id"], row["room_id"], row["created_at"], quote)
                    entry = (-score, identity, target)
                    if len(reservoir) < MAX_OCCURRENCES:
                        heapq.heappush(reservoir, entry)
                    elif score < -reservoir[0][0]:
                        heapq.heapreplace(reservoir, entry)
            guard()
        targets = [entry[2] for entry in sorted(reservoir, key=lambda item: -item[0])]
        quote_index = {}
        quotes = []
        for target in targets:
            if target[4] not in quote_index:
                quote_index[target[4]] = len(quotes)
                quotes.append(target[4])
        report.update({"target_pass_chat_rows": target_rows,
                       "eligible_linked_target_messages": eligible_targets,
                       "extracted_target_quote_occurrences": occurrences,
                       "retained_target_quote_occurrences": len(targets),
                       "distinct_retained_quotes": len(quotes),
                       "target_occurrences_discarded_by_cap": max(0, occurrences - len(targets))})
        stage = "prefix_automaton"
        prefix_index, prefixes, prefix_quotes = {}, [], []
        for index, quote in enumerate(quotes):
            prefix = quote[:PREFIX_LENGTH]
            if prefix not in prefix_index:
                prefix_index[prefix] = len(prefixes)
                prefixes.append(prefix)
                prefix_quotes.append([])
            prefix_quotes[prefix_index[prefix]].append(index)
        matcher = PrefixMatcher(prefixes, guard)
        report["prefix_patterns"] = len(prefixes)
        report["automaton_nodes"] = len(matcher.next)

        stage = "complete_source_scan"
        source_matches = [[] for _ in quotes]
        saturated = set()
        source_rows, agent_source_rows, matched_source_quote_pairs, source_chars = 0, 0, 0, 0
        for batch in pq.ParquetFile(inputs["chat_messages"]).iter_batches(
                batch_size=512, columns=chat_columns, use_threads=False):
            for row in batch.to_pylist():
                source_rows += 1
                if not isinstance(row["content"], str):
                    continue
                agent_source_rows += row["agent_id"] is not None
                source_chars += len(row["content"])
                found = matcher.matches(row["content"], prefix_quotes, quotes)
                if found:
                    source = (row["id"], row["agent_id"], row["room_id"], utc(row["created_at"]))
                    for index in found:
                        if index in saturated:
                            continue
                        if len(source_matches[index]) >= MAX_SOURCES_PER_QUOTE:
                            saturated.add(index)
                            source_matches[index].clear()
                            continue
                        source_matches[index].append(source)
                        matched_source_quote_pairs += 1
                        if matched_source_quote_pairs > MAX_SOURCE_MATCHES:
                            raise BoundReached("SourceMatchBound")
            report["source_scan_rows"] = source_rows
            report["source_scan_characters"] = source_chars
            guard()
        report.update({"complete_source_scan": True, "source_scan_agent_rows": agent_source_rows,
                       "matched_source_quote_pairs": matched_source_quote_pairs,
                       "saturated_quotes_excluded": len(saturated)})

        stage = "reference_selection"
        reasons = Counter()
        retained_relations = {}
        for target in targets:
            target_id, target_agent, target_room, target_time, quote = target
            index = quote_index[quote]
            if index in saturated:
                reasons["saturated_source_matches"] += 1
                continue
            before = utc(target_time)
            eligible = [source for source in source_matches[index] if source[3] < before]
            if not eligible:
                reasons["no_prior_source"] += 1
                continue
            if len(eligible) > 1:
                reasons["multiple_prior_sources_across_all_chats"] += 1
                continue
            source_id, source_agent, source_room, source_time = eligible[0]
            if not source_agent or source_agent == target_agent or source_room != target_room:
                reasons["unique_source_not_other_agent_in_same_room"] += 1
                continue
            if source_id not in links:
                reasons["unique_source_without_send_link"] += 1
                continue
            source_link, target_link = links[source_id], links[target_id]
            if source_link["agent_id"] != source_agent or target_link["agent_id"] != target_agent:
                reasons["source_link_agent_mismatch"] += 1
                continue
            reasons["unique_prior_linked_source"] += 1
            item = {
                "source_chat_id": source_id, "target_chat_id": target_id,
                "send_turn_id": source_link["send_turn_id"],
                "source_send_turn_id": source_link["send_turn_id"],
                "target_send_turn_id": target_link["send_turn_id"],
                "source_session_id": source_link["session_id"],
                "target_session_id": target_link["session_id"],
                "source_agent_id": source_agent, "target_agent_id": target_agent,
                "room_id": source_room, "source_at": source_time.isoformat(),
                "target_at": before.isoformat(), "quote": quote,
                "quote_chars": len(quote), "quote_sha256": hashlib.sha256(quote.encode()).hexdigest(),
                "source_has_evidence_window": source_id in windows,
                "target_has_evidence_window": target_id in windows,
                "relation": "quoted_prior_message", "basis": "explicit_reference",
                "uniqueness_scope": "all earlier projected chats, including same-agent and other-room records",
                "unique_prior_source_count": 1, "gap_seconds": (before-source_time).total_seconds(),
                "establishes_adoption": False, "diagnostic_only": True,
            }
            pair = (source_id, target_id)
            if pair not in retained_relations or len(quote) > retained_relations[pair]["quote_chars"]:
                retained_relations[pair] = item
            guard()
        ordered = sorted(retained_relations.values(), key=lambda r: (
            not r["source_has_evidence_window"], not r["target_has_evidence_window"],
            -r["quote_chars"], stable(r["source_chat_id"] + ":" + r["target_chat_id"])))
        candidates = ordered[:MAX_CANDIDATES]
        report.update({"target_occurrence_outcomes": dict(reasons),
                       "unique_source_target_relations": len(ordered),
                       "relations_with_source_evidence_window": sum(r["source_has_evidence_window"] for r in ordered),
                       "relations_with_both_evidence_windows": sum(r["source_has_evidence_window"] and r["target_has_evidence_window"] for r in ordered),
                       "retained_candidates": len(candidates),
                       "candidates_discarded_by_output_cap": max(0, len(ordered)-len(candidates)),
                       "retained_source_agents": len({r["source_agent_id"] for r in candidates}),
                       "retained_target_agents": len({r["target_agent_id"] for r in candidates}),
                       "retained_source_sessions": len({r["source_session_id"] for r in candidates}),
                       "status": "complete"})
    except BoundReached:
        candidates = []
        report.update({"status": "bounded_stop", "stage": stage,
                       "retained_candidates": 0,
                       "incomplete_scan_cannot_establish_uniqueness": True})
        exit_code = 2
    except Exception as exc:
        candidates = []
        report.update({"status": "error", "stage": stage,
                       "error_type": type(exc).__name__, "retained_candidates": 0})
        exit_code = 2

    RESTRICTED.mkdir(parents=True, exist_ok=True, mode=0o700)
    RESTRICTED.chmod(0o700)
    private_write(RESTRICTED / "cross-agent-reference-candidates.jsonl",
                  "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in candidates))
    report["resources"] = {
        "wall_seconds": round(time.monotonic() - started, 3),
        "process_cpu_seconds": round(time.process_time() - cpu_started, 3),
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "scratch_free_bytes": psutil.disk_usage(SCRATCH).free,
        "rss_limit_bytes": RSS_LIMIT, "timeout_seconds": args.timeout_seconds,
        "arrow_cpu_threads": 8, "python_scan_processes": 1,
    }
    output = ROOT / "reports/cross-agent-reference-discovery.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.with_suffix(".tmp")
    staged.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    staged.replace(output)
    print(json.dumps(report, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
