#!/usr/bin/env python3
"""Audit send-to-chat provenance on Azure without emitting text or record IDs.

Reproduces the existing hash/agent/five-minute candidate join, measures its
ambiguity, and independently checks saved links against the source projections.
Coverage and deterministic consistency do not establish ground-truth accuracy.
"""

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import threading
import time

import duckdb
import psutil


ROOT = Path("/home/azureuser/ai-swarm")
SCRATCH = Path("/mnt/ai-swarm-scratch")
DERIVED = SCRATCH / "derived"
REPORT = ROOT / "reports/provenance-link-audit.json"


def projected_hash(value: str) -> str:
    """Use the projection's exact Unicode whitespace and hashing semantics."""
    return hashlib.blake2b(value.strip().encode("utf-8"), digest_size=16).hexdigest()


def aggregate(conn: duckdb.DuckDBPyConnection, sql: str) -> dict:
    result = conn.execute(sql)
    names = [column[0] for column in result.description]
    return dict(zip(names, result.fetchone()))


def distribution(conn: duckdb.DuckDBPyConnection, table: str, column: str) -> dict:
    # Callers supply only constant internal table/column names, never corpus text.
    return aggregate(conn, f"""
        SELECT count({column}) AS count, min({column}) AS min,
               quantile_cont({column}, 0.50) AS p50,
               quantile_cont({column}, 0.90) AS p90,
               quantile_cont({column}, 0.99) AS p99,
               max({column}) AS max
        FROM {table}
    """)


def write_report(report: dict) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    staged = REPORT.with_suffix(".tmp")
    staged.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    staged.replace(REPORT)
    print(json.dumps(report, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-limit-gib", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args()
    if sys.platform != "linux" or not SCRATCH.is_mount():
        parser.error("Run only on the Azure VM with scratch mounted")
    if not 1 <= args.threads <= 32 or not 2 <= args.memory_limit_gib <= 32:
        parser.error("Use 1-32 threads and a 2-32 GiB DuckDB memory limit")
    if not 10 <= args.timeout_seconds <= 1800:
        parser.error("Timeout must be between 10 and 1800 seconds")
    names = ("chat_messages", "computer_use_sessions", "computer_use_turns",
             "sent_message_links")
    if any(not (DERIVED / f"{name}.parquet").is_file() for name in names):
        parser.error("Source projections and saved provenance links are required")
    if psutil.disk_usage(SCRATCH).free < 20 * 1024**3:
        parser.error("At least 20 GiB free scratch is required")

    started = time.monotonic()
    cpu_started = time.process_time()
    process = psutil.Process()
    samples = []
    stop = threading.Event()
    expired = threading.Event()
    conn = duckdb.connect()
    report = {"audit_version": 1, "status": "running", "raw_text_in_report": False,
              "record_ids_in_report": False,
              "candidate_rule": "same stored hash and agent; abs(date_diff(second)) <= 300",
              "selection_rule": "minimum absolute millisecond delta, then chat ID",
              "limits": [
                  "Coverage and deterministic consistency are not linkage accuracy.",
                  "A unique nearest candidate can still be the wrong real-world event.",
                  "No record contents or identifiers are emitted by this audit."]}

    def observe() -> None:
        while not stop.wait(0.5):
            samples.append((process.memory_info().rss,
                            psutil.virtual_memory().available,
                            psutil.disk_usage(SCRATCH).free))
            if time.monotonic() - started >= args.timeout_seconds:
                expired.set()
            if expired.is_set():
                # Repeated interruption also covers the boundary between queries.
                conn.interrupt()

    watcher = threading.Thread(target=observe, daemon=True)
    watcher.start()
    exit_code = 0
    try:
        conn.execute(f"SET threads = {args.threads}")
        conn.execute(f"SET memory_limit = '{args.memory_limit_gib}GB'")
        temp = SCRATCH / "duckdb-provenance-audit-temp"
        temp.mkdir(exist_ok=True)
        conn.execute(f"SET temp_directory = '{temp}'")
        conn.execute("SET max_temp_directory_size = '8GB'")
        conn.create_function("python_strip", lambda value: value.strip(),
                             ["VARCHAR"], "VARCHAR")
        conn.create_function("projected_hash", projected_hash, ["VARCHAR"], "VARCHAR")
        for name in names:
            conn.execute(f"CREATE TEMP VIEW {name} AS "
                         f"SELECT * FROM read_parquet('{DERIVED / (name + '.parquet')}')")

        integrity = {}
        for table, id_column in (("chat_messages", "id"),
                                 ("computer_use_sessions", "id"),
                                 ("computer_use_turns", "id"),
                                 ("sent_message_links", "send_turn_id")):
            integrity[table] = aggregate(conn, f"""
                SELECT count(*) AS rows,
                       count(*) - count({id_column}) AS null_ids,
                       count({id_column}) - count(DISTINCT {id_column}) AS duplicate_ids
                FROM {table}
            """)
        report["source_integrity"] = integrity
        if any(item["null_ids"] or item["duplicate_ids"] for item in integrity.values()):
            report["status"] = "blocked_by_nonunique_source_ids"
            exit_code = 2
        else:
            conn.execute("""
                CREATE TEMP TABLE sends AS
                SELECT t.id AS send_turn_id, t.session_id,
                       s.id AS resolved_session_id, s.agent_id,
                       try_cast(t.created_at AS TIMESTAMP) AS send_at,
                       t.message_hash, python_strip(t.message_content) AS text,
                       t.message_hash IS DISTINCT FROM projected_hash(t.message_content)
                           AS hash_mismatch
                FROM computer_use_turns t
                LEFT JOIN computer_use_sessions s ON t.session_id = s.id
                WHERE t.action_type = 'send_message_back_to_chat'
            """)
            conn.execute("""
                CREATE TEMP TABLE chats AS
                SELECT id AS chat_id, agent_id,
                       try_cast(created_at AS TIMESTAMP) AS chat_at,
                       content_hash, python_strip(content) AS text,
                       content_hash IS DISTINCT FROM projected_hash(content)
                           AS hash_mismatch
                FROM chat_messages
            """)
            report["sends"] = aggregate(conn, """
                SELECT count(*) AS rows,
                       count(*) FILTER (WHERE resolved_session_id IS NULL) AS missing_sessions,
                       count(*) FILTER (WHERE agent_id IS NULL) AS missing_agents,
                       count(*) FILTER (WHERE send_at IS NULL) AS invalid_timestamps,
                       count(*) FILTER (WHERE text IS NULL) AS null_text,
                       count(*) FILTER (WHERE text = '') AS empty_trimmed_text,
                       count(*) FILTER (WHERE message_hash IS NULL) AS missing_hashes,
                       count(*) FILTER (WHERE hash_mismatch) AS recomputed_hash_mismatches
                FROM sends
            """)
            report["chats"] = aggregate(conn, """
                SELECT count(*) AS rows,
                       count(*) FILTER (WHERE chat_at IS NULL) AS invalid_timestamps,
                       count(*) FILTER (WHERE text IS NULL) AS null_text,
                       count(*) FILTER (WHERE text = '') AS empty_trimmed_text,
                       count(*) FILTER (WHERE content_hash IS NULL) AS missing_hashes,
                       count(*) FILTER (WHERE hash_mismatch) AS recomputed_hash_mismatches
                FROM chats
            """)
            conn.execute("""
                CREATE TEMP TABLE candidates AS
                SELECT s.send_turn_id, c.chat_id,
                       abs(date_diff('millisecond', s.send_at, c.chat_at)) AS delta_ms,
                       date_diff('millisecond', s.send_at, c.chat_at) AS signed_delta_ms,
                       abs(epoch_us(c.chat_at) - epoch_us(s.send_at)) AS delta_us,
                       coalesce(s.text = c.text, false) AS exact_trimmed_text_equal
                FROM sends s JOIN chats c
                  ON s.message_hash = c.content_hash AND s.agent_id = c.agent_id
                 AND abs(date_diff('second', s.send_at, c.chat_at)) <= 300
            """)
            conn.execute("""
                CREATE TEMP TABLE ordered_candidates AS
                SELECT *, row_number() OVER (
                         PARTITION BY send_turn_id ORDER BY delta_ms, chat_id) AS choice,
                       min(delta_ms) OVER (PARTITION BY send_turn_id) AS nearest_delta_ms
                FROM candidates
            """)
            conn.execute("""
                CREATE TEMP TABLE per_send AS
                SELECT s.send_turn_id, count(c.chat_id) AS candidate_count,
                       count(c.chat_id) FILTER (WHERE c.exact_trimmed_text_equal)
                           AS exact_text_candidate_count,
                       count(c.chat_id) FILTER (WHERE c.delta_ms = c.nearest_delta_ms)
                           AS nearest_tie_count,
                       min(c.delta_ms) AS nearest_delta_ms,
                       max(c.delta_ms) FILTER (WHERE c.choice = 2) - min(c.delta_ms)
                           AS runner_up_gap_ms
                FROM sends s LEFT JOIN ordered_candidates c USING (send_turn_id)
                GROUP BY s.send_turn_id
            """)
            report["candidates"] = aggregate(conn, """
                SELECT count(*) AS pairs,
                       count(*) FILTER (WHERE NOT exact_trimmed_text_equal)
                           AS hash_join_pairs_with_unequal_trimmed_text,
                       count(*) FILTER (WHERE delta_us > 300000000)
                           AS pairs_outside_exact_300_second_window
                FROM candidates
            """)
            report["ambiguity"] = aggregate(conn, """
                SELECT count(*) AS sends,
                       count(*) FILTER (WHERE candidate_count = 0) AS no_candidate,
                       count(*) FILTER (WHERE candidate_count = 1) AS single_candidate,
                       count(*) FILTER (WHERE candidate_count > 1) AS multiple_candidates,
                       count(*) FILTER (WHERE nearest_tie_count = 1) AS unique_nearest,
                       count(*) FILTER (WHERE nearest_tie_count > 1) AS tied_nearest,
                       count(*) FILTER (WHERE exact_text_candidate_count = 0)
                           AS no_exact_text_candidate,
                       max(candidate_count) AS maximum_candidates_per_send
                FROM per_send
            """)
            report["candidate_count_distribution"] = distribution(conn, "per_send", "candidate_count")
            report["nearest_delta_ms_distribution"] = distribution(conn, "per_send", "nearest_delta_ms")
            report["runner_up_gap_ms_distribution"] = distribution(conn, "per_send", "runner_up_gap_ms")

            conn.execute("""
                CREATE TEMP TABLE checked_links AS
                SELECT l.send_turn_id, l.chat_id,
                       s.send_turn_id IS NOT NULL AS source_send_exists,
                       c.chat_id IS NOT NULL AS source_chat_exists,
                       l.chat_id IS NOT DISTINCT FROM n.chat_id AS matches_nearest_choice,
                       l.agent_id IS NOT DISTINCT FROM s.agent_id AS saved_agent_matches,
                       l.session_id IS NOT DISTINCT FROM s.session_id AS saved_session_matches,
                       coalesce(s.agent_id = c.agent_id, false) AS same_agent,
                       coalesce(s.text = c.text, false) AS exact_trimmed_text_equal,
                       coalesce(s.message_hash = c.content_hash, false) AS same_stored_hash,
                       abs(date_diff('millisecond', s.send_at, c.chat_at)) AS actual_delta_ms,
                       date_diff('millisecond', s.send_at, c.chat_at) AS signed_delta_ms,
                       abs(epoch_us(c.chat_at) - epoch_us(s.send_at)) AS actual_delta_us,
                       l.match_delta_ms IS NOT DISTINCT FROM
                           abs(date_diff('millisecond', s.send_at, c.chat_at))
                           AS saved_delta_matches,
                       try_cast(l.send_at AS TIMESTAMP) IS NOT DISTINCT FROM s.send_at
                           AS saved_send_time_matches,
                       try_cast(l.chat_at AS TIMESTAMP) IS NOT DISTINCT FROM c.chat_at
                           AS saved_chat_time_matches,
                       python_strip(l.message_content) IS NOT DISTINCT FROM s.text
                           AS saved_message_matches
                FROM sent_message_links l LEFT JOIN sends s USING (send_turn_id)
                LEFT JOIN chats c ON l.chat_id = c.chat_id
                LEFT JOIN ordered_candidates n ON l.send_turn_id = n.send_turn_id
                                               AND n.choice = 1
            """)
            report["saved_links"] = aggregate(conn, """
                SELECT count(*) AS rows, count(chat_id) AS linked_rows,
                       count(*) FILTER (WHERE chat_id IS NULL) AS unlinked_rows,
                       count(*) FILTER (WHERE NOT source_send_exists) AS absent_source_send,
                       count(*) FILTER (WHERE chat_id IS NOT NULL AND NOT source_chat_exists)
                           AS absent_source_chat,
                       count(*) FILTER (WHERE NOT matches_nearest_choice)
                           AS disagreement_with_recomputed_nearest,
                       count(*) FILTER (WHERE NOT saved_agent_matches) AS saved_agent_mismatch,
                       count(*) FILTER (WHERE NOT saved_session_matches) AS saved_session_mismatch,
                       count(*) FILTER (WHERE NOT saved_message_matches) AS saved_message_mismatch,
                       count(*) FILTER (WHERE NOT saved_send_time_matches) AS saved_send_time_mismatch,
                       count(*) FILTER (WHERE NOT saved_chat_time_matches) AS saved_chat_time_mismatch,
                       count(*) FILTER (WHERE NOT saved_delta_matches) AS saved_delta_mismatch,
                       count(*) FILTER (WHERE chat_id IS NOT NULL AND NOT same_agent)
                           AS linked_agent_mismatch,
                       count(*) FILTER (WHERE chat_id IS NOT NULL AND NOT exact_trimmed_text_equal)
                           AS linked_trimmed_text_mismatch,
                       count(*) FILTER (WHERE chat_id IS NOT NULL AND NOT same_stored_hash)
                           AS linked_hash_mismatch,
                       count(*) FILTER (WHERE actual_delta_us > 300000000)
                           AS links_outside_exact_300_second_window,
                       count(*) FILTER (WHERE signed_delta_ms < 0) AS chat_precedes_send,
                       count(*) FILTER (WHERE signed_delta_ms = 0) AS same_millisecond,
                       count(*) FILTER (WHERE signed_delta_ms > 0) AS chat_follows_send
                FROM checked_links
            """)
            report["missing_saved_links"] = aggregate(conn, """
                SELECT count(*) AS source_sends_without_saved_row
                FROM sends s LEFT JOIN sent_message_links l USING (send_turn_id)
                WHERE l.send_turn_id IS NULL
            """)
            report["selected_delta_ms_distribution"] = distribution(conn, "checked_links", "actual_delta_ms")
            report["selected_signed_delta_ms_distribution"] = distribution(conn, "checked_links", "signed_delta_ms")
            report["chat_reuse"] = aggregate(conn, """
                SELECT count(*) AS distinct_linked_chat_rows,
                       count(*) FILTER (WHERE uses > 1) AS chats_reused_by_multiple_sends,
                       coalesce(sum(uses) FILTER (WHERE uses > 1), 0) AS sends_using_reused_chats,
                       coalesce(sum(uses - 1) FILTER (WHERE uses > 1), 0) AS excess_uses,
                       coalesce(max(uses), 0) AS maximum_sends_per_chat
                FROM (SELECT chat_id, count(*) AS uses FROM checked_links
                      WHERE chat_id IS NOT NULL GROUP BY chat_id)
            """)
            report["status"] = "complete"
    except Exception as exc:
        # DuckDB error messages can include SQL values; emit only the exception type.
        report["status"] = "timeout" if expired.is_set() else "error"
        report["error_type"] = type(exc).__name__
        exit_code = 2
    finally:
        stop.set()
        watcher.join(timeout=2)
        conn.close()

    if expired.is_set():
        report["status"] = "timeout"
        exit_code = 2
    report["resources"] = {
        "wall_seconds": round(time.monotonic() - started, 3),
        "process_cpu_seconds": round(time.process_time() - cpu_started, 3),
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "minimum_system_available_bytes": min((s[1] for s in samples),
                                               default=psutil.virtual_memory().available),
        "minimum_scratch_free_bytes": min((s[2] for s in samples),
                                          default=psutil.disk_usage(SCRATCH).free),
        "sampled_peak_rss_bytes": max((s[0] for s in samples),
                                      default=process.memory_info().rss),
        "duckdb_threads": args.threads,
        "duckdb_memory_limit_gib": args.memory_limit_gib,
        "duckdb_temp_limit_gb": 8,
        "timeout_seconds": args.timeout_seconds,
    }
    write_report(report)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
