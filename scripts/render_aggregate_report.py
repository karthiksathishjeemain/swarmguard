#!/usr/bin/env python3
"""Render a corpus-free aggregate report from VM-side experiment JSON files."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path("/home/azureuser/ai-swarm")
REPORTS = ROOT / "reports"


def read(name: str) -> dict:
    return json.loads((REPORTS / name).read_text())


def main() -> int:
    if sys.platform != "linux":
        print("Run on the Azure VM.", file=sys.stderr)
        return 2
    manifest = json.loads((ROOT / "source-docs/manifest.json").read_text())
    core = read("core-audit.json")
    turns = read("turns-memories-audit.json")
    projection = read("projection-computer_use_turns.json")
    links = read("sent-message-links.json")
    windows = read("evidence-windows.json")
    ranking = read("review-lead-ranking.json")
    english = read("laya-binary-pilot.json")
    typed = read("laya-binary-pilot_typed_decisions.json")
    review_queue = read("review-queue.json")
    review_laya = read("review-queue-laya.json")
    lines = [
        "# Action-grounded communication audit — aggregate findings",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Dataset revision: `{projection['dataset_revision']}`; export timestamp: `{manifest['exportedAt']}`.",
        "All corpus access, projection, and inference ran on the Azure VM. This report contains no raw records.",
        "",
        "## What the data supports",
        "",
        f"The export has **{core['counts']['agents']:,} agents**, **{core['counts']['chat_messages']:,} chat messages**, **{core['counts']['events']:,} events**, **{turns['counts']['computer_use_turns']:,} computer-use turns**, and **{turns['counts']['agent_memories']:,} memories**.",
        f"A content/agent/time join linked **{links['matched_chat_rows']:,} of {links['send_actions']:,}** sent-chat actions to their public chat rows ({links['matched_fraction']:.3%}); {links['unmatched_send_actions']} remained unmatched.",
        f"All scanned turn-to-session references resolved. {core['links']['event_message_ids_missing_from_chat']} event message IDs were absent from the chat table, while no chat IDs were absent from events.",
        f"Turn evidence includes **{turns['turns']['command_rows']:,} bash-command rows**, **{turns['turns']['output_rows']:,} tool-output rows**, and **{turns['turns']['error_rows']:,} error rows**. The screenshot-redacted flag is true on {turns['turns']['redacted_screenshot_rows']:,} turns. No screenshot TARs were fetched.",
        "",
        "## Working investigator",
        "",
        f"The pipeline screens {windows['lexical_candidates']:,} broad first-person/completion-verb leads and keeps {windows['random_controls']:,} deterministic random controls. It reconstructs at most {windows['window_actions']} preceding actions from the same session for each selected send.",
        f"A transparent review-queue score ranks {ranking['ranked_leads']:,} leads; {ranking['any_recent_error_count']:,} had some prior-window error, so error presence alone is too common to call a contradiction. This score is **not a probability**.",
        "The review policy separates completed claims from plans and separates supported, contradicted, insufficient, and outside-window evidence. Neither a tool error nor a model output alone establishes misalignment.",
        "",
        "## Laya pilot and decision",
        "",
        f"Frozen Laya 0.3.23 English (`{english['model_revision']}`) labeled {english['primary_labels_by_group']['lexical_candidate'].get('a', 0)}/200 broad lexical candidates and {english['primary_labels_by_group']['random_control'].get('a', 0)}/100 controls as completed claims; {english['option_order_disagreements']}/300 labels changed under an option-order permutation.",
        f"Frozen Laya typed-decisions (`{typed['model_revision']}`) labeled {typed['primary_labels_by_group']['lexical_candidate'].get('a', 0)}/200 candidates and {typed['primary_labels_by_group']['random_control'].get('a', 0)}/100 controls as completed; {typed['option_order_disagreements']}/300 labels changed under option reordering.",
        "These are **unlabeled real-data pilot distributions, not precision or recall**. The checkpoints warned about invalid temperature entries, so their confidence values are not treated as calibrated. A three-choice schema and focused-clause input failed exploratory stability checks and were not scaled. Laya annotations remain an abstaining secondary facet until human-reviewed quality is established.",
        f"A separate blinded queue contains {review_queue['top_ranked']} top-ranked leads, {review_queue['lexical_random']} other lexical leads, and {review_queue['random_control']} controls. On those 300 cases, Laya returned {review_laya['labels'].get('completed', 0)} stable completed labels, {review_laya['labels'].get('not_completed', 0)} stable negative labels, and {review_laya['labels'].get('abstain', 0)} option-order abstentions in {review_laya['wall_seconds']:.1f} s with {review_laya['peak_process_rss_bytes']/1024**3:.2f} GiB peak process RSS. These are annotations, not reviewed truth.",
        "",
        "## Runtime and limits",
        "",
        f"A streaming pass over all turns and memories took {turns['resources']['wall_seconds']:.1f} s and peaked at {turns['resources']['peak_process_rss_bytes']/1024**2:.0f} MiB process RSS. The turn projection reduced {projection['uncompressed_source_bytes']/1024**3:.2f} GiB decompressed JSON to {projection['parquet_bytes']/1024**2:.0f} MiB Parquet in {projection['wall_seconds']:.1f} s. The send join took {links['wall_seconds']:.1f} s and the evidence-window pass {windows['window_wall_seconds']:.1f} s, after a {windows['sort_seconds']:.1f} s sort.",
        "The 2026-03-24 permanent-computer-use rollout changes the meaning of sessions across time. Chat is room-wide, so this system does not infer that another agent saw a statement. Browser outcomes may require screenshots; memories and generated summaries may be incomplete or wrong. The sequential live export can produce small parent/child differences.",
        "No dataset training or fine-tuning was performed. No real message excerpt is included in this report. Human-adjudicated evidence precision remains unmeasured pending structured review of full VM-side cases.",
        "",
    ]
    output = REPORTS / "public-aggregate.md"
    staged = output.with_suffix(".tmp")
    staged.write_text("\n".join(lines))
    staged.replace(output)
    print(json.dumps({"output": str(output), "bytes": output.stat().st_size,
                      "contains_raw_records": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
