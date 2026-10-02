# Frontier detector execution on Azure

The core is implemented; frontier inference has not run. This runbook describes the bounded first pilot, not a finished submission. Corpus processing stays on `hackathon`; the Mac holds source code and documentation only. Hosted inference is an explicit exception to VM-only computation: selected evidence goes to OpenAI for inference, after the owner approves that transfer. No GPU is needed on the VM for this architecture.

## Evidence and model flow

`AI Village text → streaming Parquet → audited source links → bounded multi-agent episodes → Astra investigator → Astra skeptical pass → citation/relationship validation → candidate case for human review`

The same model in two passes is not independent ground truth. Both outputs are retained. Disagreement produces an insufficient result; agreement on an incident produces a candidate requiring review. Exact quotations, agent identities and graph links must be present. Temporal-only or shared-artifact connections cannot establish a swarm incident. The pipeline does not infer hidden intent.

## Current files

- `scripts/audit_provenance_links.py`: aggregate-only verification of send/chat joins, ambiguity and resources.
- `scripts/build_frontier_episodes.py`: bounded VM evidence assembly with redaction and retrieval manifest; no inference.
- `scripts/discover_cross_agent_references.py`: bounded streaming exact-quotation discovery; references do not imply endorsement.
- `swarmguard/schema.py` and `prompts.py`: evidence contract, validators and frozen first prompts.
- `swarmguard/runner.py`: standard-library Responses API adapter; no new ML runtime dependency.
- `scripts/run_frontier_pilot.py`: offline plan by default, controlled execution only on Azure.
- `config/frontier.toml`: Astra/high, 6,000 output tokens per call, 128 KB request cap, at most 20 episodes, proposed $20 spend cap.
- `tests/`: synthetic validation and mocked provider checks. These verify code behavior, not model detection accuracy.

`requirements-frontier.txt` lists the three dependencies needed for the new retrieval and detector path on Python 3.12. The existing VM environment already contains them. Laya and PyTorch are not required by this path. Rebuilding the original dataset projections also uses the historical ingestion dependencies recorded in the full lockfile.

## Private authentication

No model API credential was configured in the VM environment at the initial check. In the user's own SSH terminal, run:

```sh
/home/azureuser/ai-swarm/.venv/bin/python /home/azureuser/ai-swarm/project/scripts/configure_frontier_access.py
```

The hidden prompt writes a mode-0600 credential outside the Git project, under `/home/azureuser/ai-swarm/credentials/`. It never prints the key and makes no API call. An existing `OPENAI_API_KEY` environment variable takes precedence. Access in a Codex conversation does not imply API access or billing for the VM.

## Planning and execution

On the VM, run the builder and inspect its aggregate report. Use its printed output path with this command:

```sh
/home/azureuser/ai-swarm/.venv/bin/python /home/azureuser/ai-swarm/project/scripts/run_frontier_pilot.py --packets /home/azureuser/ai-swarm/restricted/frontier-pilot-packets.jsonl
```

The command makes zero API calls. The first eight real-data packets are an engineering diagnostic sample from the previous review queue, not the frozen, matched evaluation proposed in `frontier-evaluation.md`. They may include messages exposed during the earlier limited-excerpt review. Reconstruct the held-out sample and its matching before scoring it.

That initial sample had no admissible cross-agent references. Corpus-wide quotation discovery then found 1,221 unique prior-source relationships; 251 sources and 85 source/target pairs have existing action windows. The improved diagnostic sample contains eight packets, all with explicit references and bounded tool actions for both agents. Rebuild and plan it with:

```sh
/home/azureuser/ai-swarm/.venv/bin/python /home/azureuser/ai-swarm/project/scripts/discover_cross_agent_references.py
/home/azureuser/ai-swarm/.venv/bin/python /home/azureuser/ai-swarm/project/scripts/build_frontier_episodes.py --reference-candidates /home/azureuser/ai-swarm/restricted/cross-agent-reference-candidates.jsonl
/home/azureuser/ai-swarm/.venv/bin/python /home/azureuser/ai-swarm/project/scripts/run_frontier_pilot.py --packets /home/azureuser/ai-swarm/restricted/frontier-reference-packets.jsonl
```

The marked-quote search samples at most 20,000 target occurrences and checks uniqueness against all earlier projected chats. It misses paraphrases and unmarked references. Exact quotation establishes a textual relationship; the model must separately determine endorsement, rejection or reliance. Evidence packets remain truncated and retrospective. The runner refuses paid execution if no packet has an admissible cross-agent relationship.

Before paid execution, obtain explicit owner approval for the selected packets, provider and cost cap. Only then add `--execute --approved-transfer`. These flags record an existing decision; they do not create consent. The entire pilot and three planned ablations are separate scopes; the full prospective design may require 140 calls. The first implementation runs just the two-pass full system.

Eight author-written packets and a separate answer key are in `fixtures/frontier_synthetic*.json`. Four pairs change authorization, recovery, reliance or goal substitution. The planner accepts this JSON array as well as VM JSONL; the answer key is never included in model requests. Passing packet validation does not establish model performance.

Requests use the fixed TLS endpoint `https://api.openai.com/v1/responses`, `store=false`, default service tier, strict JSON output, and no tools. `store=false` disables response storage for retrieval; it is not a promise of zero provider retention. Corpus instructions remain untrusted data. Redaction reduces exposure but cannot guarantee that all sensitive content is removed.

The runner reserves a conservative cost before each call using request UTF-8 byte size plus framing allowance, maximum output tokens, and checked prices. It stops at the configured cap. Reserved amounts remain charged to the local guard even if actual usage is lower. This is conservative accounting, not an exact billing total. Cache keys include the whole request: packet, prompts, schema, model and reasoning settings. Completed requests are reused. An interrupted or failed attempt remains reserved and is not retried automatically, since a timeout can still be billable. A cap reached between passes preserves the first result for a later authorized resume.

All case findings, usage and cache live under `/home/azureuser/ai-swarm/restricted/frontier-run/`; only aggregate run status is printed. Do not copy the restricted directory to the Mac or Git. The output records usage supplied by the provider, per-call time, process CPU and peak memory. Raw provider error bodies and hidden reasoning are not logged. Authentication/HTTP errors stop the run for diagnosis without exposing submitted text.

## What remains before a submission

Run an authorized synthetic API smoke test, then bounded real-data inference; inspect evidence sufficiency and retrieval gaps; obtain human reference labels; run the frozen evaluation and ablations; build a readable incident timeline with counterevidence and retrospective replay. Do not claim accuracy, novel incidents, live prevention, or completion before those steps have evidence. The older Laya submission write-up is historical and must be replaced after the new results exist. Publishing still requires the owner's separate authorization.
