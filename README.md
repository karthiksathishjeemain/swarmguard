# Laya swarm investigation

Status on 2026-10-02: the Azure VM has 64 vCPUs, about 504 GiB RAM, and a verified 866 GiB scratch mount. The pinned export contains 2.51 million computer-use turns and 183,485 chat messages. The evidence pipeline links 100,353 of 100,354 sent-chat actions to public chat records, builds 21,100 candidate action windows and 500 controls, and ranks leads for review. Frozen Laya inference has been piloted on VM data; its labels are secondary annotations with an option-order abstention rule. No dataset or model weights have been downloaded to this Mac.

The selected project is a Laya-assisted investigator that links an agent's public action claim to its preceding tool actions and outputs. Exact data joins establish provenance. The ranked queue is a way to spend review time, not a misalignment score. Laya's real-data pilot showed misses and answer-order sensitivity, so its output is displayed as an abstaining facet, never as the basis for an accusation.

On the VM, run `bash /home/azureuser/ai-swarm/project/scripts/run_pipeline.sh` after mounting scratch, creating the pinned environment, and completing Hugging Face login there. This script fetches only named text tables, never the 177 GB screenshot corpus. When it finishes, run `.../.venv/bin/python .../project/scripts/investigate.py check` for aggregate integrity and `.../project/scripts/investigate.py review` in an authorized VM terminal to inspect and label the next blinded case. `case POSITION` displays raw source text on the VM; do not pipe that output back to a local machine or into Git. The [VM runbook](docs/vm-runbook.md) gives exact setup and recovery steps.

- [Assessment A–E](docs/reconnaissance.md): event, dataset metadata, prior work, candidate approaches, evaluation, and implementation gates.
- [Measured dataset audit and direction decision](docs/dataset-audit.md): actual rows, links, modalities, resource costs, and caveats.
- [Pre-registered review policy](docs/label-policy.md): completed-claim and evidence labels, sampling, and honest limits.
- [VM runbook](docs/vm-runbook.md): measured environment, scratch setup, and data-access procedure.
- [VM-only diagnostic](scripts/vm_preflight.py): credential-safe collector already run on the VM.
- [Laya CPU smoke benchmark](scripts/laya_cpu_smoke.py): bounded synthetic inference with runtime and resource measurements.
- [VM environment bootstrap](scripts/bootstrap_vm.sh): recreate the pinned Python environment on the mounted VM scratch setup.
- [Bounded JSONL probe](scripts/probe_jsonl.py): inspect key names and value types on the VM without printing record values.
- [Pinned schema fetcher](scripts/fetch_dataset_docs.py): download only three small documentation files using the VM's existing Hugging Face login.
- [Explicit text-table fetcher](scripts/fetch_text_files.py): plan and download selected pinned `.jsonl.gz` files to VM scratch with space checks and persistent hashes.
- [Aggregate join audit](scripts/audit_core_tables.py) and [turn/memory audit](scripts/audit_turns_memories.py): measured counts, links, modalities, and resource use.
- [Streaming Parquet projection](scripts/build_projected_tables.py): retain only evidence fields needed for the investigator on VM scratch.
- [Send-to-chat provenance join](scripts/link_sent_messages.py) and [bounded evidence windows](scripts/build_evidence_windows.py): link public claims to preceding tool actions.
- [Laya claim pilot](scripts/laya_claim_pilot.py): measured frozen inference and option-order check on 300 VM-side messages.
- [Laya binary and focused-clause pilots](scripts/laya_binary_pilot.py), [laya_clause_pilot.py](scripts/laya_clause_pilot.py): compare schemas and input construction before scaling inference.
- [Review lead ranking](scripts/rank_review_leads.py): transparent aggregate triage; scores are not misalignment probabilities.
- [VM pipeline runner](scripts/run_pipeline.sh), [blinded review queue](scripts/prepare_review_queue.py), [Laya annotations](scripts/annotate_review_queue.py), [interactive investigator](scripts/investigate.py), and [review metrics](scripts/summarize_reviews.py).
- [VM-local derived checkpoint](scripts/checkpoint_derived.py): save/restore hashed Parquet on the VM's persistent disk before deallocation.
- [Aggregate report renderer](scripts/render_aggregate_report.py) and [submission write-up](docs/submission-writeup.md).
- [Exact VM package versions](requirements.lock.txt): initial lockfile; install the CPU PyTorch wheel from its CPU index before applying it.
- [Frozen experiment configuration](config/experiment.toml): dataset/model revisions, retrieval bounds, review allocation, and CPU settings.
- [Research and experiment status](docs/experiment-log.md): facts, unknowns, and execution status.

The Mac is a control and documentation machine. Corpus access, extraction, indexing, model downloads, inference, and performance monitoring run on the Azure VM `hackathon`. Only code, documentation, and reviewed aggregate reports belong in this repository.

The dataset permits research and analysis, with written permission required for training or fine-tuning. The design uses frozen Laya inference. Package and model revisions are pinned in the VM runbook and lockfile. Human-adjudicated claim and evidence quality is not yet measured; the bundled review workflow is designed to report it without moving corpus records off Azure.
