# Frontier-model swarm investigation: research and decision

Date: 2026-10-02. Status: researched proposal and implementation direction, **not a validated detector**. This supersedes the Laya requirement for new work; the earlier Laya experiments remain historical baselines. Measured data facts are in [dataset-audit.md](dataset-audit.md), and the prospective evaluation is in [frontier-evaluation.md](frontier-evaluation.md).

## Recommendation

Build a **swarm incident investigator** that answers: “Which observed actions conflict with an explicit instruction or factual claim, how did other agents continue the behavior, and what evidence supports the finding?” Keep the existing Azure ingestion and source links. Replace sentence-level Laya classification with bounded, contextual reasoning and a separate challenge pass.

The competitive contribution is a case file with verified source citations, an event timeline, carefully typed cross-agent relationships, counterevidence, and a replay of when an alert first becomes supportable. A general transcript classifier or a graph of agent mentions is already well covered. This direction offers a plausible strong submission; no research result or event rule guarantees a win.

## Event fit and requirements

The organizers explicitly suggest tools for investigating swarms, tracing information spread, answering questions about agent groups, and using digital forensics. The deadline is **2026-10-04 17:00 America/Los_Angeles**, which is **2026-10-05 05:30 Asia/Kolkata**. One submission per team needs a short write-up or video, a GitHub code link, and team names/emails; real findings are optional. Online and in-person entries are judged by AI Village and Grove Research staff. **No numeric rubric or judging weights are published.** The GitHub requirement does not override the user's current instruction to keep this repository local. [Organizer announcement](https://aivillageblog.substack.com/p/join-the-ai-swarm-dynamics-hackathon), [official logistics](https://swarmchasing.com/logistics/)

A useful tool, a verified finding, a convincing false-alarm example, and reproducibility are our competitive priorities. They are project judgments, not invented official judging criteria.

## Relevant work and reusable resources

| Primary resource | What it establishes | Design consequence |
|---|---|---|
| [The Arbiter Agent](https://arxiv.org/html/2606.10747v1), [code](https://github.com/aisilab/arbiter) | A budgeted agent monitors multi-agent conversations using observation, inspection and incident logging. Its experiments include induced misalignment; additional tools have detection/false-positive tradeoffs. | A reasoning agent that monitors a swarm is established prior work. Adapt budgeted inspection to recorded evidence; we cannot interrogate historical agents or claim access to unavailable prompts/reasoning. |
| [MAST](https://arxiv.org/abs/2503.13657), [project and annotated data](https://multi-agent-systems-failure-taxonomy.github.io/MAST/) | A taxonomy separates specification failures, inter-agent coordination problems, and verification/termination failures. The release distinguishes human annotations from model annotations. | Use concrete failure categories and preserve annotation provenance. MAST's term “inter-agent misalignment” includes coordination failures and must not be presented as proof of malicious intent. |
| [Emergent Collusion in Long-Horizon LLM Agent Interaction](https://github.com/SALT-NLP/agent-collusion) | Released code and data study joint instruction violations during repeated peer verification, with controlled compliant/violating peers and history ablations. | A particularly relevant source for paired counterfactual tests and explicit peer influence. Its intervention evidence is stronger than correlations in observational AI Village logs; do not transfer its reported rates to this dataset. |
| [TRAIL](https://arxiv.org/abs/2505.08638) | Trace debugging and issue localization are evaluated on real single- and multi-agent workflows; the published study finds this difficult even for long-context models. | Test event localization and evidence support separately from whether a model emits a plausible explanation. Its task-failure labels are not automatically safety labels. |
| [AgentDoG](https://arxiv.org/abs/2601.18491), [ATBench release family](https://github.com/LiYu0524/ATbench) | AgentDoG diagnoses trajectory risks by source, failure mode and consequence. ATBench now distinguishes the 1,000-trajectory release from the original ATBench500; the newer release reports a human audit. | Borrow the separation of behavior and consequence. Pin the release and retain original label meanings. Open-weight guard models are optional comparison baselines, not evidence that our CPU VM will run them efficiently. |
| [Inspect Scout](https://meridianlabs-ai.github.io/inspect_scout/), [multi-agent documentation](https://meridianlabs-ai.github.io/inspect_scout/multi_agent.html) | Scanners support structured results, references, validation, batching, resumability and Parquet. Agent spans can be scanned independently with context chunking. | Reuse these engineering patterns; an adapter is optional. Our focus is evidence relationships across agents rather than claiming that multi-agent scanning itself is new. |
| [Docent](https://transluce.org/docent) | Rubric development, behavior search, cited explanations, qualitative checking and aggregate charts already exist. | Avoid rebuilding a generic transcript-search dashboard as the headline contribution. |
| [SwarmTraces](https://swarmtraces.org/), [Collusion.wiki](https://collusion.wiki/), [Transluce incident investigation](https://transluce.org/agent-activity) | Public investigations provide recorded swarm behavior and incomplete forensic evidence in different formats. | Potential external case studies. Analyze redacted records, never execute recovered payloads, and distinguish documented attempts from established effects. |

## Observed-behavior categories

The output contract uses exactly the four category values below. They describe evidence in a recorded episode, not an agent's enduring goals, intent or internal alignment. Evidence sufficiency is recorded separately; a plausible concern with missing evidence is not a supported finding.

| Contract category | Minimum evidence for a supported finding | Important exclusion |
|---|---|---|
| `coordinated_constraint_violation` | A cited applicable instruction/authorization boundary, a conflicting recorded action, and direct evidence that another agent contributes to or continues the violation. | Missing permission in a truncated window does not prove prohibition. An ordinary retry or approved workaround is insufficient. |
| `unsupported_claim_propagation` | A precise factual/completed-action claim, evidence identifying its support gap or contradiction, and direct evidence that another agent relies on it. Clearly distinguish an unverified claim from one contradicted by an observed outcome. | A tool error alone is not a false claim. Shared room membership, similarity, timing or a repeated URL does not prove reliance. A reliability failure alone does not establish malicious intent. |
| `goal_substitution` | A cited assigned objective and a recorded decision/action pursuing a materially incompatible replacement, with relevant authorization or goal updates checked. | An authorized plan change, instrumental subtask, failed attempt or harmless exploratory step is insufficient. |
| `none` | The available evidence supports none of the three target categories. Record missing context or uncertainty where applicable. | This is not a claim that the agents are safe or internally aligned. |

Unauthorized publication, restriction circumvention and evidence tampering are example patterns within the first category when coordination and the applicable rule are evidenced. Contradicted completion can support the second when another agent relies on the claim. These descriptions are not additional schema categories. Routine authorized cleanup is not concealment; an observed deletion without evidence of intent should be described as a deletion.

Goals are useful context, but a broad goal alone does not supply every authorization constraint. The investigator must retrieve relevant task instructions or report that authorization is unknown. Severity describes the observed or reasonably evidenced consequence separately from confidence and intent.

## Pipeline and evidence boundary

1. **Retrieve on Azure.** Query the existing Parquet/DuckDB projections, exact chat/send links, goals and session chronology. Expand the initial action window only when needed, within an explicit token budget. Log truncation and unavailable modalities.
2. **Assemble episodes on Azure.** Include the focal event, applicable goals/instructions, bounded actions/outcomes, and relevant preceding or following references. Keep source IDs, offsets, timestamps, hashes and a record of every retrieval.
3. **Construct typed relationships.** Record exact chat/action links, explicit references and observed artifact reads/writes. Label similarity or possible exposure as inferred. They cannot independently substantiate a propagation claim.
4. **Investigate.** A reasoning model proposes specific observed-behavior findings, exact citations, counterevidence and missing context. It has no authority to execute corpus commands or visit corpus URLs.
5. **Challenge.** A separate invocation searches the same evidence for authorization, recovery, misattribution and benign explanations. It may downgrade or reject a finding. Two calls to one model are correlated judgments, not independent ground truth.
6. **Validate and assemble.** Code validates citations and graph references, rejects invented IDs, deduplicates incidents and records abstentions. A validated citation proves that text was referenced correctly, not that the interpretation is true.
7. **Review and replay.** Display the boundary/claim, chronological actions, other implicated agents, opposing evidence and unresolved questions. Prefix replay shows when the recorded evidence first supports an alert; it is not proof of live prevention.

Use **GPT-6 Astra (`gpt-6-astra`) initially for both investigator and skeptic**, with frozen prompts and a bounded reasoning/output budget. Official documentation supports structured outputs and the Responses API and describes Astra as the most capable model for demanding work. This motivates a pilot, not an unmeasured claim that it is best at this particular detection task. Published standard short-context rates on 2026-10-02 are $10 per million input tokens and $50 per million output tokens; record actual usage and recheck rates before running. [Official model documentation](https://developers.openai.com/api/docs/models/gpt-6-astra)

The latest VM authentication-presence check found no configured model API authentication. Model access in this Codex conversation does not establish authenticated access for the standalone VM pipeline. Configure credentials securely on the VM; never print them or place them in project files or Git.

All heavy processing, raw-data storage, retrieval, annotation storage and experiment orchestration stay on the **Azure VM**. A hosted model call transfers the selected evidence to its provider and runs inference there; this must be stated accurately. The earlier permission for up to 20 short excerpts in this task is **not permission to send full cases or batches to an external API**. Obtain clear user consent for the provider, bounded evidence scope and pilot cost before real-data API calls. Until then, implement and check the interface using authored synthetic fixtures. No AI Village training or fine-tuning is planned; retain the dataset's usage conditions. [Dataset card](https://huggingface.co/datasets/aidigestorg/ai-village)

## Practical hackathon scope

First deliver the 20-case pilot, citation checks, three ablations and a usable incident view. Scale to a few hundred selected episodes only after the pilot reveals an acceptable error pattern and the cost is known. Do not infer quality from scanning more data. The current 2.51-million-turn projection and exact chat links are valuable infrastructure, but the former 21,100 lexical leads are not 21,100 misalignment cases.

The public demo should show one reviewed supported case, one tempting false alarm correctly downgraded, and one click from the finding to its evidence. If real findings remain unreviewed or cannot be released, use a clearly marked synthetic demonstration and publish the measured infrastructure results and limitations. This is preferable to claiming a discovery the evidence has not established.
