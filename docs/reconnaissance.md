# Pre-implementation assessment

Date: 2026-10-02. Evidence categories: **observed locally**, **published by source**, **proposal**, and **not yet verified**. Published metadata is not a substitute for inspecting the user's snapshot.

## Current environment status

Verified by SSH and Azure instance metadata on 2026-10-02: `hackathon` is `Standard_E64ads_v7` in `eastus2`, running Ubuntu 24.04.4 LTS and Python 3.12.3. CPU affinity exposes 64 vCPUs; the AMD EPYC 9V45 host reports 540,821,135,360 bytes of RAM (about 504 GiB). The persistent root disk is approximately 30 GB, with 27–28 GB free. Four Azure-labeled local NVMe devices are each 880 GiB. After verifying the first was blank, it was formatted as ext4 and mounted at `/mnt/ai-swarm-scratch`, with 866 GiB available. The remaining three are untouched. This scratch mount is disposable; code and retained results belong on the persistent root volume. The VM has outbound HTTPS to Hugging Face and PyPI. Azure CLI is absent on the VM. No dataset was found under the bounded directories checked. [Microsoft Eadsv7 documentation](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/memory-optimized/eadsv7-series)

An isolated Python environment at `/home/azureuser/ai-swarm/.venv` contains Polars, DuckDB, PyArrow, psutil, Laya, and huggingface_hub. The user completed a secure VM-side Hugging Face login; pinned schema documents and selected text tables are now accessible. The Mac's cached Azure CLI account exists, but a management request failed with an Entra authentication error; working SSH makes that CLI unnecessary for this pipeline. Private keys and tokens have not been read or printed.

Microsoft lists E64ads_v7 as 64 vCPUs with 512 GB memory, NVMe temporary storage, and no accelerator. The page's storage table lists four temporary disks and an 880 GiB size entry; actual mounted usable capacity must be measured rather than inferred from the SKU. [Microsoft Eadsv7 documentation](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/memory-optimized/eadsv7-series)

Temporary storage can be lost during VM lifecycle events. Use the mounted scratch for replaceable raw caches and spill files; retain code, annotations, manifests, results, and checkpoints on verified persistent Azure storage. [Microsoft storage documentation](https://learn.microsoft.com/en-us/azure/virtual-machines/managed-disks-overview)

## A. Hackathon understanding

The user-confirmed event is **AI Swarm Dynamics Hackathon**, AI Village × Grove Research, October 3–4, 2026. Its objective is tools and research for understanding, investigating, or discovering agent swarms. Information spread, trajectory exploration, and digital forensics are explicitly suggested. Projects may use AI Village or other real-world multi-agent data; early starts are allowed. Participants register individually and can subsequently form teams. The announcement offers $3,000 in prizes; free compute credits are specifically mentioned for in-person participants. [Official announcement](https://aivillageblog.substack.com/p/join-the-ai-swarm-dynamics-hackathon)

The stated submission deadline is **Sunday October 4, 17:00 America/Los_Angeles = Monday October 5, 00:00 UTC = 05:30 Asia/Kolkata**. The organizer's reply says the website was corrected to match the announcement, resolving an earlier discrepancy. [Organizer reply](https://aivillageblog.substack.com/p/join-the-ai-swarm-dynamics-hackathon/comment/350203265)

The [official logistics page](https://swarmchasing.com/logistics/) says to submit one entry per team through a link to be posted in the event Slack `#announcements`. It requires a short write-up or video, a GitHub repository link with code, and every team member's name and email; a write-up of real findings is optional. A panel of AI Village and Grove Research staff judges online and in-person submissions equally, with decisions about a week later. No numeric scoring formula, judging weights, or team-size limit is published. Registration and the eventual form still need checking; no registration or submission has been made on the user's behalf.

## B. Dataset understanding

The public card identifies `aidigestorg/ai-village`, approximately 177 GB, with text, structured actions, and screenshot images. Tables cover events, chat, sessions, turns, memories, goals, agents, and generated summaries. Screenshots are daily TARs referenced by turn ID. The card warns that narration may be wrong, summaries omit session interiors, and scaffolding changes can confound behavioral trends. Research/analysis is allowed; training or fine-tuning requires written permission. Citation and no reidentification are required. [Dataset card](https://huggingface.co/datasets/aidigestorg/ai-village)

Public file metadata, rounded decimal units:

| File | Compressed size |
|---|---:|
| computer_use_turns.jsonl.gz | 2.48 GB |
| agent_memories.jsonl.gz | 2.44 GB |
| events.jsonl.gz | 329 MB |
| claude_code_messages.jsonl.gz | 104 MB |
| chat_messages.jsonl.gz | 52.5 MB |
| computer_use_sessions.jsonl.gz | 40.1 MB |
| summaries.jsonl.gz | 2.85 MB |
| village-transcript.json | 362 MB, overlapping export |

The core compressed text tables sum to approximately **5.45 GB**, excluding the overlapping transcript and screenshots. Small goals/agent/room metadata add little. This supports a text-first pilot, with screenshot TARs fetched selectively. These are metadata estimates, not measured storage requirements. The VM-side public listing at revision `838b4150303ca8228e8edb432d8b8ccae353d258` reports 390 files; the largest text files are `computer_use_turns.jsonl.gz` (2,475,319,119 bytes) and `agent_memories.jsonl.gz` (2,438,234,633 bytes). [Public file listing](https://huggingface.co/datasets/aidigestorg/ai-village/tree/main)

The card's older counts (~123k chat messages and ~1.14M computer turns) differ from this export. VM-side full scans at the pinned revision found **183,485 chat messages and 2,510,487 computer-use turns**; all scanned row counts matched the export manifest. The actual schema, joins, modality mix, and resource use are recorded in the [measured dataset audit](dataset-audit.md). No ground-truth misalignment labels were found in the documented tables. [Dataset documentation](https://huggingface.co/datasets/aidigestorg/ai-village)

The VM inspection established source file sizes and hashes, observed UTC date range, documented join keys, full counts, key reference coverage, action modalities, redaction flags, and bounded resource use. Derived Parquet compression and the semantic-label pilot remain in progress. Treat all corpus text as untrusted records; never execute instructions, commands, or URLs found inside it.

## Existing work and Laya

| Primary source | Existing contribution | Consequence for our design |
|---|---|---|
| [Just Ask Jev](https://arxiv.org/abs/2609.29429), [RLCDAlignBench code](https://github.com/sumleo/RLCDAlignBench) | Typed alignment detection across 44 benchmarks and ten failure categories; context and leakage matter. | Replacing Jev with Laya alone is replication. Use context ablations and provenance-aware evaluation. |
| [Gemini AI Village case study](https://www.lesswrong.com/posts/yw7TzFPTdprJ7mYfG/gemini-2-5-pro-in-the-ai-village-as-a-natural-case-study-of) | Longitudinal memory/persona analysis around a known episode. | A single-agent drift chart is already explored. A cross-agent, action-grounded audit can add value. |
| [AI Village mapping study](https://www.swarm-ai.org/research/ai-village-mapping-design/) | Author reports sparse external judgments, broadcast chat, and poor generic quality proxies. | Independently replicate snapshot-specific claims. Errors are not misalignment labels; mentions are not message recipients. |
| [Misaligned thinking probes](https://arxiv.org/abs/2606.24251) | Behavioral taxonomy plus internal-activation probes. | Taxonomy may inform questions; transcript-only exports cannot reproduce activation probes. |
| [SwarmTraces](https://swarmtraces.org/) | Forensic reconstruction of a real swarm incident. | Evidence exploration is established; distinguish incomplete artifacts from labeled ground truth. |

Laya is available as open weights and a Python SDK with typed decisions, batching, CPU execution, and optional ONNX support. Its repository describes a Jev-compatible interface, but using the direct SDK avoids hosting an unnecessary service. Its advertised GPU timings do not predict this CPU VM's throughput. [Laya repository](https://github.com/NandhaKishorM/laya) Jev is TypeSafe's hosted typed-decision model; no verified event source requires its use. [TypeSafe introduction](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

The Laya model card reports substantial limitations: base typed-workflow accuracy below a majority baseline, short default state budgets, overconfidence, and a possible `noul` option-label bias. Its stronger checkpoint is domain specialized; `action.act_probability` is not a usable safety signal. Laya consumes text, not screenshot pixels. Test small neutral-key `choice` schemas, bounded source windows, multiple frozen checkpoints, and option-order sensitivity. Do not call scores calibrated misalignment probabilities. [Laya model card](https://huggingface.co/convaiinnovations/laya)

## C–D. Candidate approaches, tradeoffs, and compute

These are **planning envelopes for a bounded pilot**, not measured requirements or promised runtimes. All use the same streaming ingestion. No approach requires a GPU for its initial scope. Developer-hour estimates assume access and labeled examples are available.

| Candidate | Value and potential novelty | Weakness / dependency | Initial VM envelope; build estimate |
|---|---|---|---|
| **1. Claim/request propagation into actions** | Trace candidate claims from chat to other agents, memories, and actions; show evidence and uncertainty. Strong fit to information-spread objective. | Temporal similarity does not prove influence; joins and auditable episodes must exist. | 16–32 active vCPUs, 32–64 GiB RAM ceiling; 5k–20k Laya windows after pilot; 50–100 GiB scratch budget; 18–28 hours. |
| **2. Goal/action consistency investigator** | Link goals and requests to actual tool outcomes; distinguish unsupported success claims from observed results. | Authorization and intent may be absent; false positives from legitimate goal changes. | 8–16 vCPUs, 16–32 GiB RAM ceiling; 2k–10k windows; 30–60 GiB scratch; 12–20 hours. |
| **3. Memory distortion audit** | Compare retained memories with prior primary records; identify repeated unsupported claims. | Retrieval misses can resemble contradictions; related research exists and a [public commenter proposed a similar project](https://aivillageblog.substack.com/p/join-the-ai-swarm-dynamics-hackathon/comment/350203265). | 8–16 vCPUs, 16–32 GiB RAM ceiling; 2k–10k comparisons; 50–100 GiB scratch; 12–20 hours. |
| **4. Interaction and behavior-change explorer** | Map activity, references, goal shifts, and candidate change points, using Laya for semantic facets. | Confounded by model/scaffold changes; anomaly is not evidence of misalignment. | 16–32 vCPUs, 16–64 GiB RAM ceiling; sparse aggregates plus selected windows; 30–60 GiB scratch; 10–18 hours. |
| **5. Laya detector replication plus Village case studies** | Controlled comparison against published Jev results and simple baselines; measurable, narrow deliverable. | Limited novelty; external benchmark access and licensing need verification. Results may not transfer to swarms. | 8–16 vCPUs, 8–32 GiB RAM ceiling; hundreds to thousands of instances; 20–40 GiB scratch; 8–14 hours. |

Scratch estimates include downloaded compressed text, derived partitions, model files, and selected media. Actual expansion and available space may require reducing these budgets before ingestion. Do not allocate worker pools merely because 64 vCPUs exist.

## E. Proposed implementation plan and decision gates

**Selected after VM-side inspection:** candidate 2, action-grounded communication auditing, with candidate 1 as an optional extension. In a 200,000-turn sample, 8,047 of 8,061 sent-chat actions matched a chat row by exact content, agent, and time. This direct link supports a defensible investigator. Cross-agent exposure remains ambiguous because chats are room-wide. Laya will supply measured semantic annotations on shortlisted claims; the output is a ranked queue of investigation leads, not an assertion that an agent is misaligned. See [dataset audit](dataset-audit.md).

1. **Access and preflight — complete.** Azure identity, hardware, storage, Python, and network were measured. A previously blank Azure local device was prepared as scratch in a separate, verified setup operation. See the [VM runbook](vm-runbook.md) and [experiment log](experiment-log.md).
2. **Dataset reconnaissance (target 1–3 hours).** Locate existing files before transfer. Use approved Hugging Face authentication on the VM only. Pin the revision; fetch schema/changelog/manifest first. Download only necessary text tables if absent. Inspect stratified samples and stream counts. Record redaction and malformed-row counts without logging raw secret-like values. Use schema-informed PyArrow batches or DuckDB/Polars streaming to write partitioned Parquet, with stable source IDs.
3. **Choose scope (target 30 minutes).** Confirm event rubric and required artifacts. Pick 2–3 incident types supported by records and available audit effort. Report the actual joins, coverage, labels, and bottlenecks before selecting the implementation. If no evidence supports propagation, choose the narrower consistency investigator.
4. **Laya viability pilot (target 1–2 hours plus audit).** Benchmark 200–500 stratified windows, including benign controls. Bound token budgets, account for question overhead, and log truncation. Compare frozen English/multilingual/specialized checkpoints on the same held-out windows. Test neutral choices and option permutations. Keep no-fit model inference unless written training permission exists. Reject semantic labels that fail a preregistered quality gate; do not scale them because inference is cheap.
5. **Implement the chosen path (target 8–12 hours).** Use lexical/entity/time blocking to retrieve candidates; avoid all-pairs comparisons. Join message → candidate uptake → later memory/action with typed evidence links. Apply Laya only to candidate windows. Cache by snapshot, source IDs, context hash, checkpoint revision, and question-schema hash. Checkpoint partitions atomically and resume safely after failures.
6. **Evaluate (target 3–5 hours plus human review).** Freeze the label policy before the final evaluation. Seek a human-adjudicated audit of 200–300 windows/links spanning random and ranked examples; annotators should be blind to detector scores. Keep entire episodes/time blocks separate, handle duplicate windows, and bootstrap at the episode level. Measure precision at a fixed review budget, supported-edge precision, coverage/abstention, and investigator review time. Report recall only where a sufficiently enumerated reference set exists. Evaluate any retrospective warning time using only information available at each timestamp. Compare rules-only, Laya-only, and combined retrieval plus Laya. If human labels are unavailable, mark model-only checks as exploratory.
7. **Deliver (reserve final 3–4 hours).** VM-run scripts and configuration; pinned dependencies and checkpoint revisions; source manifest/hashes; resumable processing; reproducible metrics and a small audited example; private investigator interface reached by SSH forwarding; README with exact commands and limitations; short demo and submission narrative. Export reviewed aggregate reports, not the raw corpus, to the Mac. Submission/publication requirements remain to be verified.

### Resource control

- Record wall time, process-tree CPU time and RSS/PSS when available, system CPU/iowait, available RAM, disk free bytes, read/write throughput, rows/s, windows/s, questions/s, and cache hits. Sample approximately every five seconds during jobs, not through a recurring automation.
- Start with 1 worker × 8 inference threads, then compare 2 × 8 and 4 × 8 on the same bounded workload. Fix BLAS/PyTorch thread settings to prevent nested oversubscription; process fewer files concurrently if gzip decompression or disk is the bottleneck.
- Start ingestion with a 32–64 GiB memory limit; retain system headroom. Spill only to a verified scratch mount. Stop before free scratch falls below the greater of 20% capacity or 50 GiB. Check projected output size as well as current free space.
- Derive runtime from measured throughput: candidate count / sustained windows per second, plus measured ingestion and I/O. Use an observed subscription/credit rate if available for cost; do not invent hourly pricing. Do not infer that an idle powered-on VM is free.
- Checkpoint before long jobs, preserve results on persistent Azure storage, and make data caches disposable. Do not deallocate while needed outputs exist only on temporary storage.

## Immediate missing inputs

The data-access gate is resolved by the user's VM-side login. Credentials and tokens must never be pasted into this document or chat. The event submission link will appear in Slack `#announcements`; names/emails and GitHub destination will be needed at submission time.
