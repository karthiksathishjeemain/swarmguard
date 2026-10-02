#!/usr/bin/env python3
"""Test whether a focused claim clause improves Laya on the same VM pilot rows."""

from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

from laya_binary_pilot import REVISION, question
from laya_claim_pilot import sample_rows


SCRATCH = Path("/mnt/ai-swarm-scratch")
ROOT = Path("/home/azureuser/ai-swarm")
FIRST_PERSON = re.compile(r"\b(?:i|we|i.ve|we.ve)\b", re.IGNORECASE)
COMPLETION = re.compile(
    r"\b(?:done|completed|finished|deployed|published|sent|created|fixed|"
    r"pushed|merged|verified|confirmed|submitted|updated|implemented|"
    r"launched|resolved|successfully)\b", re.IGNORECASE)


def focused_clause(message: str) -> tuple[str, str]:
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", message)
             if part.strip()]
    same = [part for part in parts if FIRST_PERSON.search(part) and COMPLETION.search(part)]
    if same:
        clause, category = same[0], "same_sentence"
    else:
        verbs = [part for part in parts if COMPLETION.search(part)]
        clause, category = (verbs[0], "verb_only") if verbs else (parts[0] if parts else "", "no_verb")
    if len(clause) > 300:
        match = COMPLETION.search(clause)
        center = match.start() if match else 0
        begin = max(0, min(center - 100, len(clause) - 300))
        clause = clause[begin:begin + 300]
    return clause, category


def main() -> int:
    previous_path = SCRATCH / "derived/laya_binary_pilot.parquet"
    if sys.platform != "linux" or not SCRATCH.is_mount() or not previous_path.is_file():
        print(json.dumps({"error": "PriorBinaryPilotRequired"}))
        return 2
    previous = {row["send_turn_id"]: row for row in
                pq.read_table(previous_path,
                              columns=["send_turn_id", "choice_primary"]).to_pylist()}
    chosen = sample_rows()
    clauses = [focused_clause(row["message_content"] or "") for row in chosen]
    os.environ["HF_HUB_CACHE"] = str(SCRATCH / "hf-hub-cache")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"
    import torch
    from laya import Router
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    router = Router(device="cpu", revisions={"english": REVISION}, max_loaded=1)
    started = time.monotonic()
    requests = [{"state": clause, "questions": question(("a", "b")),
                 "model": "english", "max_len": 512}
                for clause, _ in clauses]
    results = router.predict_batch(requests, batch_size=8, sort_by_length=True)
    labels = defaultdict(Counter)
    switches = defaultdict(Counter)
    records = []
    for row, (clause, category), result in zip(chosen, clauses, results):
        label = result["answers"]["claim"]["choice"]
        old = previous[row["send_turn_id"]]["choice_primary"]
        key = row["selection_group"] + "|" + category
        labels[key][label] += 1
        switches[key][old + ">" + label] += 1
        records.append({"send_turn_id": row["send_turn_id"], "group": key,
                        "clause_chars": len(clause), "label": label,
                        "full_message_label": old,
                        "truncated": result["usage"].get("truncated") is True})
    output = SCRATCH / "derived/laya_clause_pilot.parquet"
    pq.write_table(pa.Table.from_pylist(records), output, compression="zstd")
    report = {"sample_rows": len(chosen),
              "labels": {key: dict(counter) for key, counter in labels.items()},
              "paired_changes": {key: dict(counter) for key, counter in switches.items()},
              "median_clause_chars": sorted(len(clause) for clause, _ in clauses)[len(clauses) // 2],
              "wall_seconds": round(time.monotonic() - started, 3),
              "model_revision": REVISION}
    report_path = ROOT / "reports/laya-clause-pilot.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
