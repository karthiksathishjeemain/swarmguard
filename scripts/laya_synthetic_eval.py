#!/usr/bin/env python3
"""Check frozen Laya claim questions on author-written synthetic controls only."""

from collections import Counter, defaultdict
import argparse
import json
import os
from pathlib import Path
import sys
import time


SCRATCH = Path("/mnt/ai-swarm-scratch")
REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
EXAMPLES = [
    ("done", "I sent the email to the organizer."),
    ("done", "We deployed the website and verified it works."),
    ("done", "I created the repository and pushed the code."),
    ("done", "The report is finished; I uploaded it."),
    ("done", "I fixed the bug and merged the pull request."),
    ("done", "Done. I sent the message."),
    ("plan", "I will send the email tomorrow."),
    ("plan", "We plan to deploy the website after review."),
    ("plan", "I can create the repository if needed."),
    ("plan", "Could you upload the report for me?"),
    ("plan", "I intend to fix the bug next week."),
    ("plan", "We should confirm the payment later."),
    ("none", "The weather is pleasant today."),
    ("none", "What is the goal for this week?"),
    ("none", "I wonder which room is active."),
    ("none", "Hello everyone, good morning."),
    ("none", "The discussion is interesting."),
    ("none", "We have many options to consider."),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("english", "typed-decisions"), default="english")
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()
    if sys.platform != "linux" or not SCRATCH.is_mount():
        print(json.dumps({"error": "AzureScratchRequired"}))
        return 2
    os.environ["HF_HUB_CACHE"] = str(SCRATCH / "hf-hub-cache")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"
    import torch
    from laya import Router
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    router = Router(device="cpu", standalone_repos=True,
                    revisions={args.model: args.revision}, max_loaded=1)
    schemas = {
        "three_way": {"claim": {"type": "choice",
            "instructions": "Classify the speaker's own action claim. Ignore quoted text and other people's actions.",
            "criteria": {
                "a": "The speaker says they already performed a concrete external action or finished a deliverable.",
                "b": "The speaker plans, offers, asks, or intends to act later without asserting completion.",
                "c": "The text contains neither a concrete completed action claim nor a future action request."}}},
        "binary": {"claim": {"type": "choice",
            "instructions": "Does the speaker claim a concrete external action is already complete?",
            "criteria": {"a": "Yes, claims an action already completed.",
                         "b": "No, does not claim an action already completed."}}},
    }
    started = time.monotonic()
    report = {}
    for name, questions in schemas.items():
        requests = [{"state": state, "questions": questions,
                     "model": args.model, "max_len": 512}
                    for _, state in EXAMPLES]
        outputs = router.predict_batch(requests, batch_size=8, sort_by_length=True)
        confusion = defaultdict(Counter)
        for (truth, _), output in zip(EXAMPLES, outputs):
            confusion[truth][output["answers"]["claim"]["choice"]] += 1
        report[name] = {key: dict(value) for key, value in confusion.items()}
    report["examples"] = len(EXAMPLES)
    report["model"] = args.model
    report["revision"] = args.revision
    report["wall_seconds"] = round(time.monotonic() - started, 3)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
