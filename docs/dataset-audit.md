# AI Village export audit and direction decision

Date: 2026-10-02. All corpus access and processing in this document occurred on the Azure VM. The pinned dataset revision is `838b4150303ca8228e8edb432d8b8ccae353d258`; its manifest says export time `2026-09-20T13:05:12.097Z`. Named text files were downloaded to VM scratch and verified against the repository's published SHA-256 digests. No screenshot TARs or raw records were downloaded to the Mac.

## Export structure and measured coverage

The dataset contains gzipped JSONL tables plus daily screenshot TARs. The schema says timestamps are UTC strings without timezone suffix and `events.event_index` is the canonical event order. Model-response structures differ by provider. Text may include redaction markers, and screenshots can be placeholders. `summaries` omit computer-use session interiors, so they are secondary evidence. The export is a sequential dump of a live database, so small cross-table count differences are expected. The 2026-03-24 permanent-computer-use rollout is a major regime boundary; cross-period comparisons must account for it. These points come from the pinned `SCHEMA.md` and `CHANGELOG.md` kept on the VM.

| Table | Manifest rows | Full VM scan | Relevant fields/signals |
|---|---:|---:|---|
| Agents | 46 | 46 | Stable agent IDs and model names |
| Chat rooms | 16 | 16 | Room IDs for broadcast chat |
| Village / agent goals | 51 / 33 | 51 / 33 | Time-bounded goal context |
| Chat messages | 183,485 | 183,485 | Content, agent, room, UTC time |
| Events | 381,610 | 381,610 | Canonical index, chat links, session boundaries, searches, consolidations |
| Computer-use sessions | 78,362 | 78,362 | Agent and stated goal |
| Computer-use turns | 2,510,487 | 2,510,487 | Executed action, command, output, error, session ID, screenshot redaction flag |
| Agent memories | 246,151 | 246,151 | Retained free text by agent and time |
| Summaries | 939 | 939 | Generated daily/goal/agent narratives, not primary action evidence |

The observed range is 2025-04-02 through 2026-09-19 UTC. Chat has 173,493 agent and 9,992 user rows; its median length is 390 characters (p99 2,051). Memories are much longer: median 17,870 characters (p99 125,197; maximum 1,640,873). A short-context Laya input therefore needs retrieval and deliberate truncation, especially for memory comparisons.

Every chat message ID appeared in `events.data.messageId`; 57 event message IDs were absent from the chat table. All checked chat room, chat agent, session agent, event actor, and turn-to-session references resolved. Event rows were not stored in canonical order: 190,756 consecutive rows decreased in `event_index`, so trajectory reconstruction must sort by that field. The measured event mix includes 173,493 `AGENT_TALK`, 52,325 `CONSOLIDATE`, 10,802 `SEARCH_HISTORY`, and 25,975 each of `START_USING_COMPUTER` and roughly `STOP_USING_COMPUTER`.

In the full turn scan, 979,075 rows contained bash commands; 1,121,412 had tool output strings; 153,245 had error strings; 14,196 screenshot rows were redacted. Action types included 100,354 `send_message_back_to_chat`, 10,148 `search_history`, 161,483 `type`, 368,447 `left_click`, and 171,247 `scroll`. All 2,510,487 turns resolved to a session, and 248 sessions had no turn. Session turn count has median 41. These fields make communication claims linkable to preceding tool evidence, while screenshots remain optional for a later case-specific audit.

All 10,000-row chat/event/session/turn probes and a 1,000-row memory probe parsed without malformed rows. The full core-table scan took 8.15 s and peaked at 173 MB process RSS; the full turn-and-memory aggregate scan took 83.87 s and peaked at 50 MB process RSS. The VM had more than 535 GB available system memory during these scans. These are single-process streaming benchmarks, not estimates of downstream Laya throughput.

## Decision from observed signals

The initial plan favored cross-agent claim propagation. The data supports a more defensible core: **action-grounded communication auditing**. A full VM-side provenance join matched **100,353 of 100,354** `send_message_back_to_chat` actions to a chat row by exact stripped-content hash, the same agent, and a five-minute time window (99.999%); one remained unmatched. This is a direct evidence path from a public statement to the sender's computer-use session and recent commands/results. The many sends provide enough leads for a selective review queue.

The selected system will reconstruct each send action's preceding local action window, identify candidate claims about completed external work, and rank cases where the claim and observed tool outcome may diverge. It will show source references, timestamps, and bounded excerpts for review. Laya will supply a **frozen, measured semantic claim label** on shortlisted messages. Rules and exact joins remain the provenance backbone. Neither an error string nor a Laya score alone proves misalignment; the product reports review leads and evidence gaps. Cross-agent uptake is an optional extension after the core audit works, because room-wide chat does not prove that a specific agent read a message.

The major feasibility advantage is high direct-link coverage and observable actions. The main risks are incomplete visibility of browser outcomes without screenshots, truncated tool output, stale context, and false positives from legitimate goal changes or claims about work outside the current session. Evaluation must separate supported, contradicted, and insufficient-evidence cases, record abstentions, compare rules-only and Laya-assisted ranking, and report reviewer precision at a fixed budget. There are no organizer-provided ground-truth misalignment labels or numeric judging weights. Human review of a stratified held-out sample is needed before claiming detector performance.

The full turn projection processed 2,510,487 rows (8,320,640,815 decompressed JSON bytes) into a 747,040,935-byte Parquet table in 46.48 s, peaking at 210,501,632 bytes process RSS. Chat and session projections took under 2 s each. The full send-to-chat join took 2.60 s and peaked at about 1.11 GB process RSS. A broad first-person plus completion-verb lexical screen yielded 21,100 candidate sends; a deterministic 500-message random control set was retained. Bounded 20-action windows for the 21,600 selected sends occupied 62,174,810 compressed Parquet bytes, after a 1.02 s sort and 6.54 s streaming window pass. All intermediates are on VM scratch, with aggregate reports on persistent VM storage.

## Data-use boundary

The dataset permits research and analysis but requires written permission for model training or fine-tuning. This project uses frozen Laya inference. Raw corpus records, derived Parquet, and model weights remain on the Azure VM. Public deliverables should contain code, methods, aggregate metrics, and only carefully reviewed/redacted case studies.
