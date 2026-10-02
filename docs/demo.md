# Public-safe demo script

The real investigator contains gated AI Village messages and action text. For a public video or screen share, demonstrate the actual pinned Laya inference with the author-written synthetic case instead of showing corpus records.

1. On the Azure VM, run:

   ```sh
   /home/azureuser/ai-swarm/.venv/bin/python /home/azureuser/ai-swarm/project/scripts/demo_synthetic.py
   ```

   The script displays a synthetic completed-action statement and two mock tool outputs ending in HTTP 503. It invokes the real frozen `typed-decisions` checkpoint twice with reversed answer order. In the verified run, Laya abstained because the option order changed its answer. The demo correctly leaves the source-evidence judgment to a human and makes no real-agent allegation.

2. Show [aggregate findings](aggregate-findings.md), especially 100,353/100,354 send-to-chat links, 21,100 retrieval leads, and the 300-case blinded queue. These are measured on the pinned real export, with no raw text displayed.

3. Show the code paths: [VM pipeline](../scripts/run_pipeline.sh), [source join](../scripts/link_sent_messages.py), [bounded windows](../scripts/build_evidence_windows.py), [Laya annotation](../scripts/annotate_review_queue.py), and [investigator](../scripts/investigate.py). Point out that `case` reveals the source trail on the VM, while `review` hides Laya and ranking for annotation.

4. State the limit clearly: zero full cases have been adjudicated so far; the Laya distribution is not precision or recall. The model warning makes its confidence unsuitable as a calibrated risk score. The system is an investigation aid, not a diagnosis of intent.

Do not record the interactive `case` or `review` views for a public submission unless a specific example has been approved and redacted. The [runbook](vm-runbook.md) explains how an authorized dataset user can inspect them privately.
