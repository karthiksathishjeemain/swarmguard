# Azure-local SwarmGuard pipeline

The selected path uses frozen Qwen3.5-9B and makes no inference API calls. Raw AI Village data, model weights and natural-data dashboards stay on Azure. The Mac holds code, authored fixtures, documentation and aggregate measurements. Publication is not authorized.

## What runs

1. Stream pinned text exports to Parquet; link tool sends to exact chat records with DuckDB. Reuse ingestion from `vm-runbook.md` and `frontier-runbook.md`.
2. Retrieve bounded cross-agent reference packets containing goals, actors, times, actions/results and exact source IDs. A separate manifest records hashes, excerpts and missing context. A quotation does not establish acceptance.
3. Run frozen Qwen3.5-9B locally with thinking disabled. Request a structured candidate, no-finding or insufficient result, a category, an enumerated reason and cited record/edge IDs. The model has no tools and receives corpus text as untrusted evidence.
4. Verify model revision/file hashes, packet identity, prompt version, output completeness, schema, record kinds and cross-agent links. Claim-propagation candidates need a temporally ordered source-result → source-claim → peer-reference → peer-action/result chain. Constraint/goal candidates need a cited participant action/result and a governing rule present by action time. Invalid output abstains; no prose is repaired into a finding. The exact rendered prompt hash is recorded for audit.
5. Route valid candidates to review. A no-finding result on incomplete natural history cannot clear an episode. Retain coverage gaps, citations, latency and CPU/RAM/disk measurements in a private standalone HTML dashboard and JSON reports.
6. Collect blinded reviewer attestations in a VM terminal and compare model results with them. No labels have been collected yet. The dashboard reads evidence; it does not save annotations.

AgentDoG remains a separately runnable trajectory-safety baseline. Optional DeBERTa NLI scores short pairs only after a supplied mapping passes actor, artifact/version, exact-quote and retry-history checks. It neither gates nor overrides the selected contextual model.

## Reproduce the environment

Run these commands inside SSH on `hackathon`:

```sh
cd /home/azureuser/ai-swarm/project
PY=/home/azureuser/ai-swarm/.venv/bin/python
$PY scripts/vm_preflight.py
$PY scripts/fetch_local_guard.py --model judge9b \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a
```

The VM's existing Python3.12.3, torch2.14.1+cpu and Transformers5.18.0 passed loading/inference unchanged. For a rebuild, install the CPU torch wheel as documented in `requirements-local.txt`, then install that file. `requirements.lock.txt` records the original full environment. No SentencePiece or Accelerate is needed by this loader.

Downloads use explicit non-executable files, pinned revisions, safe tensors and checksum manifests. Every loader file is verified before inference. Caches are under `/mnt/ai-swarm-scratch`; reports/manifests are on persistent root storage. Scratch is disposable. If lost, restore the hashed derived-data checkpoint and redownload pinned models. Never read or copy the Hugging Face token or SSH key into code.

## Authored diagnostic run

```sh
$PY scripts/run_swarm_review.py --synthetic \
  --output-dir /home/azureuser/ai-swarm/restricted/swarmguard-authored-run
$PY scripts/evaluate_local_judge.py \
  --report /home/azureuser/ai-swarm/restricted/swarmguard-authored-run/judge.json \
  --output /home/azureuser/ai-swarm/reports/swarmguard-authored-evaluation.json
```

Use a fresh output directory for every inference run. Defaults in `config/local.toml` select the 9B profile, four workers × eight threads, eight episodes, 16,384 input tokens, 512 output tokens and 180 seconds per episode. Each worker is bounded to 96 GiB RSS. The wrapper refuses more than 32 total threads or insufficient available RAM. Command-line overrides are recorded in worker reports. `--judge-report PATH` validates and renders an existing run without inference; it does not resume missing work. Optional `--guard-report PATH` adds the separately measured baseline to the dashboard.

The measured selected-model development run is `/home/azureuser/ai-swarm/restricted/qwen9b-fast-eight`: eight matched verdict/category outcomes across four pairs, 146.31s wall. See `local-decision-models.md` for failures of smaller models, resource use and evaluation limits. These are authored development cases, not held-out accuracy.

## Private natural-data pilot

Build the expanded context into new files, preserving the original eight packets:

```sh
$PY scripts/build_frontier_episodes.py \
  --reference-candidates /home/azureuser/ai-swarm/restricted/cross-agent-reference-candidates.jsonl \
  --include-peer-followup
$PY scripts/run_swarm_review.py \
  --packets /home/azureuser/ai-swarm/restricted/frontier-reference-followup-packets.jsonl \
  --manifest /home/azureuser/ai-swarm/restricted/frontier-reference-followup-manifest.json \
  --max-episodes 2 --workers 2 --max-episode-seconds 240 \
  --output-dir /home/azureuser/ai-swarm/restricted/swarmguard-followup-smoke
```

The original eight packets produced four no-finding outputs and four timed-out generations in 374.51 seconds. All eight routed to insufficient because their histories are incomplete. The expanded builder adds at most four projected actions within 15 minutes, strictly after both the peer chat and its uniquely linked send, in that peer's same session. The measured build retained 32 actions and 32 outcomes across eight packets in 1.528 seconds. These are temporal context, not inferred adoption. Other sessions, screenshots and authorization updates remain outside scope.

The command above is a two-case integration smoke test with a longer per-case cap after the observed timeouts. It is not an accuracy sample. Use the exact packet version and report scope for later review/scoring; do not mix hashes from the original and expanded packets. Complete reference review before increasing the inference scan.

Outputs are `judge.json`, `decisions.json`, `dashboard.html` and unique private worker reports/logs. Reports record model/file/prompt/packet hashes, wall and CPU seconds, per-worker RSS peaks, CPU utilization samples, available RAM, disk headroom, tokens and failure counts. Worker peaks are not a measured simultaneous process-tree peak. Generated reasoning and raw final text are discarded; only allowlisted parsed fields remain. Nothing in the CLI output contains corpus records.

Input overflow is rejected, not silently clipped by inference. Retrieval may already have excerpted source records; the dashboard exposes this distinction. Generation limits are checked between decoder steps, with an outer worker timeout as backup. An incomplete generation abstains.

## Blinded reference review and scoring

In an interactive VM terminal, with no model findings displayed:

```sh
$PY scripts/review_swarm_cases.py --review --reviewer reviewer-1 \
  --packets /home/azureuser/ai-swarm/restricted/frontier-reference-followup-packets.jsonl \
  --manifest /home/azureuser/ai-swarm/restricted/frontier-reference-followup-manifest.json \
  --labels /home/azureuser/ai-swarm/restricted/swarm-followup-reference-labels.jsonl
```

Select the verdict, category, evidence and counterevidence IDs, consider a benign alternative and record missing context. These packets may warrant insufficient. The tool preserves exact source/coverage snapshots in a locked 0600 log. Its TTY requirement and reviewer pseudonym are workflow safeguards, not proof of an independent human identity. Do not automate human labels or pipe raw review output to the Mac.

Replace `--review --reviewer reviewer-1` with `--check` for aggregate review counts. Then compare saved model results:

```sh
$PY scripts/score_reviewed_swarm_cases.py \
  --packets /home/azureuser/ai-swarm/restricted/frontier-reference-followup-packets.jsonl \
  --labels /home/azureuser/ai-swarm/restricted/swarm-followup-reference-labels.jsonl \
  --judge-report /home/azureuser/ai-swarm/restricted/swarmguard-followup-smoke/judge.json \
  --output /home/azureuser/ai-swarm/restricted/swarm-followup-reviewed-metrics.json
```

Metrics compare against reviewer attestations, not independent gold. They report denominators, coverage, false positives/negatives, abstentions and positive references lost to abstention. Selective recall excludes abstentions explicitly. Undefined metrics remain null. A selected eight-case pilot cannot establish population performance.

## Safe local demonstration and software checks

This command uses only authored fictional fixtures and their saved model labels; no weights are loaded:

```sh
python3 scripts/render_case_dashboard.py --synthetic \
  --judge-results benchmarks/2026-10-02/qwen9b-synthetic.json \
  --output /tmp/swarmguard-demo.html
python3 -m unittest discover -s tests -v
```

The full corpus dashboard stays on Azure. No full-dataset inference loop is scheduled. Before increasing scale, review complete source cases and freeze a separate evaluation set. Frozen inference is the current workflow; AI Village training/fine-tuning requires the dataset owner's written permission.
