# Laya Trace: auditing AI Village claims against action histories

**Submission draft, 2026-10-02.** Team names, email addresses, and the GitHub URL must be supplied before the event form is submitted. The [event logistics](https://swarmchasing.com/logistics/) require a short write-up or video and a code link. This draft is designed to stand alone as the short write-up.

## What we built

AI agents often describe work they have done in chat. Those statements can be difficult to verify in a large swarm transcript. Laya Trace reconstructs the source trail behind a public statement: the exact `send_message_back_to_chat` action, its matching chat row, the same computer-use session, and up to 20 preceding actions with bounded command, output, and error excerpts. It ranks promising cases for an investigator, while keeping the actual judgment with a human reviewer.

The project runs on an Azure `Standard_E64ads_v7` Ubuntu VM (64 vCPUs, approximately 504 GiB RAM). The Mac holds code and reviewed aggregate reports only. All corpus access, joins, inference, and interactive case review stay on the VM. We pinned AI Village revision `838b4150303ca8228e8edb432d8b8ccae353d258`, verified downloaded text tables against their published hashes, and did not download screenshot archives or fine-tune any model on the data. [AI Village dataset](https://huggingface.co/datasets/aidigestorg/ai-village)

## Findings from the pinned export

The export contains 183,485 chat messages and 2,510,487 computer-use turns. A DuckDB content/agent/time join matched **100,353 of 100,354** sent-chat actions to a public chat row. That high-coverage direct link is what makes a source-grounded audit practical. A broad lexical screen yields **21,100** candidate messages; a deterministic control reservoir retains **500** other sends. Each selected send has a bounded preceding-action window. The pipeline also preserves the window's truncation flag and source counts, so missing evidence is visible.

The first ranking is transparent and deliberately modest: same-sentence first-person/completion wording, nearby tool errors, and presence of command/output context determine review order. **10,187 of 21,100** candidates have some error within the 20-action window. Error presence is common; it is not a finding that a statement was false. The score is not a probability, and this work does not infer whether another agent saw a room-wide message.

## Where Laya fits

We use the frozen, CPU-run [Laya](https://github.com/NandhaKishorM/laya) typed-decisions checkpoint (`1a793eb568e6718f15941d08f85432581df534e3`) to annotate whether a message asserts a concrete completed action. The code asks the same binary question twice with answer options in opposite orders and marks a disagreement as an abstention. Laya does not decide whether the action happened; the source trail and reviewer do. This uses Laya directly through its Python SDK, without JEV.

We tested Laya before scaling it. On 200 broad lexical candidates and 100 controls from the real export, typed-decisions labeled 35 candidates and 6 controls as completed; **17 of 300** decisions changed with option order. An English checkpoint changed **33 of 300**. These are unlabeled pilot distributions, **not accuracy measurements**. A small, user-approved review of redacted excerpts revealed clear completed-action claims that the typed model missed; that convenience sample was selected around model outputs and is unsuitable for a precision estimate. Laya's confidence is not used as a calibrated score because the checkpoint emits a temperature warning. On the separate blinded 300-case queue, the two-order rule returned **46 stable completed labels, 229 stable negative labels, and 25 abstentions** in 143.5 seconds with 8 CPU threads. The queue allows a proper adjudication of 100 top-ranked leads, 100 other lexical leads, and 100 controls.

## Reproduce and inspect

The repository contains pinned dependencies, SHA-checked text fetching, streaming PyArrow projection, DuckDB provenance joins, bounded window construction, ranking, Laya inference, a 300-case blinded review queue, a VM-only interactive investigator, and aggregate metric generation. `scripts/run_pipeline.sh` runs the pipeline on the VM; `scripts/investigate.py check` verified **300/300** queue cases have a matched chat row, evidence window, and Laya annotation. `scripts/investigate.py case` shows the source IDs and timestamps beside Laya's label; `review` hides that label and walks cases in randomized order, saving annotations on persistent VM storage. The [runbook](vm-runbook.md) records exact paths, revisions, resources, and recovery steps. Processing is bounded: the complete turn projection took 46.5 seconds and peaked at about 201 MiB process RSS; the join took 2.6 seconds and the evidence-window pass 6.5 seconds on the measured VM.

## Limits and next measurement

The evidence window can miss earlier work, browser-only outcomes, or a later successful retry. Session semantics changed when permanent computer use was introduced on 2026-03-24, so longitudinal comparisons need stratification. A contradicted statement is not automatically intentional misalignment. We report source coverage, model abstentions, and reviewed-case precision or contradiction yield only when the corresponding full VM-side cases have been labeled. No corpus-wide recall or misalignment rate is claimed without ground truth.

Dataset citation: AI Digest, “AI Village dataset,” 2026. [Dataset card and research terms](https://huggingface.co/datasets/aidigestorg/ai-village). We retain raw records and review annotations on Azure. The GitHub repository contains code, documentation, and aggregate results only.

A [public-safe synthetic demo](demo.md) runs the real Laya checkpoint on an author-written example and shows the option-order abstention rule without revealing gated corpus text.
