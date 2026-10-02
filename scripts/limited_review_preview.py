#!/usr/bin/env python3
"""Show at most 16 tightly screened agent excerpts after explicit user approval.

This script is for the one approved review task only. It does not print source
IDs, model labels, links, numbers, code, credentials, or tool evidence.
"""

import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

import pyarrow.parquet as pq


ROOT = Path("/home/azureuser/ai-swarm")
DERIVED = Path("/mnt/ai-swarm-scratch/derived")
SENSITIVE = re.compile(
    r"(?i)\b(password|token|secret|credential|private key|account|login|"
    r"authentication|api key|email|phone number|address|contact details|"
    r"payment|bank|credit card|medical|health record|identity|username|ssh key|pin)\b")
CONTACT = re.compile(r"(?i)(https?://|www\.|@|\b[a-z]:[/\\]|[/\\](home|users|etc)[/\\])")
CLAIM = re.compile(
    r"(?i)\b(completed|finished|sent|updated|created|fixed|posted|"
    r"deployed|wrote|built|saved|added|removed|published|submitted)\b")
KEEP_CAPITALIZED = {"I", "We", "The", "A", "An", "Done", "Completed",
                    "Finished", "Yes", "No", "Hello", "Hi", "Thanks", "Thank"}


def screen(value: object) -> str | None:
    if not isinstance(value, str) or not 45 <= len(value) <= 900:
        return None
    if SENSITIVE.search(value) or CONTACT.search(value):
        return None
    # Select one sentence so the preview stays short; the evidence review must
    # still use the full VM-side message, not this excerpt.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", value) if s.strip()]
    choices = [s for s in sentences if CLAIM.search(s)] or sentences
    text = choices[0]
    if not 35 <= len(text) <= 280:
        return None
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode()
    text = re.sub(r"\d+", "[NUM]", text)
    text = re.sub(r"[^A-Za-z .,!?'-]", " ", text)
    text = " ".join(text.split())
    if len(text) < 35 or len(text) > 200:
        return None
    # Mask capitalized names and organizations without printing originals.
    text = re.sub(r"\b[A-Z][a-zA-Z]+\b",
                  lambda match: match.group() if match.group() in KEEP_CAPITALIZED else "[NAME]",
                  text)
    return text


def main() -> int:
    if sys.platform != "linux" or not Path("/mnt/ai-swarm-scratch").is_mount():
        print(json.dumps({"error": "AzureScratchRequired"}))
        return 2
    predictions = {row["send_turn_id"]: row for row in
                   pq.read_table(DERIVED / "laya_binary_pilot_typed_decisions.parquet",
                                 columns=["send_turn_id", "choice_primary",
                                          "choice_permuted"]).to_pylist()}
    location = ROOT / "reports/limited-review-map.json"
    previous = json.loads(location.read_text()) if location.exists() else []
    seen = {row["send_turn_id"] for row in previous}
    if len(previous) >= 20:
        print(json.dumps({"error": "ApprovedExcerptLimitReached"}))
        return 2
    buckets = {"candidate_yes": [], "candidate_no": [], "control": []}
    for batch in pq.ParquetFile(DERIVED / "evidence_windows.parquet").iter_batches(
            batch_size=2048,
            columns=["send_turn_id", "selection_group", "message_content"]):
        for row in batch.to_pylist():
            prediction = predictions.get(row["send_turn_id"])
            if prediction is None or row["send_turn_id"] in seen:
                continue
            excerpt = screen(row["message_content"])
            if excerpt is None:
                continue
            if row["selection_group"] == "random_control":
                bucket = "control"
            elif prediction["choice_primary"] == "a":
                bucket = "candidate_yes"
            else:
                bucket = "candidate_no"
            order = hashlib.sha256(row["send_turn_id"].encode()).digest()
            buckets[bucket].append((order, row, prediction, excerpt))
    selected = []
    for bucket, limit in (("candidate_yes", 6), ("candidate_no", 6), ("control", 3)):
        selected.extend(sorted(buckets[bucket], key=lambda entry: entry[0])[:limit])
    mapping = []
    preview = []
    for number, (_, row, prediction, excerpt) in enumerate(selected, start=len(previous) + 1):
        mapping.append({"number": number, "send_turn_id": row["send_turn_id"],
                        "selection_group": row["selection_group"],
                        "typed_primary": prediction["choice_primary"],
                        "typed_permuted": prediction["choice_permuted"]})
        preview.append({"number": number, "excerpt": excerpt})
    if len(previous) + len(mapping) > 20:
        raise RuntimeError("ApprovedExcerptLimitExceeded")
    location.parent.mkdir(parents=True, exist_ok=True)
    location.write_text(json.dumps(previous + mapping, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"screened_pool": {key: len(value) for key, value in buckets.items()},
                      "shown": len(preview), "excerpts": preview}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
