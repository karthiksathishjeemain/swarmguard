# SwarmGuard — trace an agent claim into a swarm decision

Draft, 2026-10-02. Not submitted or published. Natural-data reference review remains pending.

An agent can make a wrong claim without creating a swarm incident: another agent may reject it, a later retry may succeed, or an instruction may authorize the action. SwarmGuard investigates when a claim or instruction is followed by another agent's action and whether the source evidence supports the model's interpretation.

The tool reconstructs bounded, source-linked episodes from AI Village chat and tool traces. A frozen local Qwen3.5-9B model identifies candidate coordinated constraint violations, unsupported-claim propagation and unauthorized goal substitution. Findings contain existing record IDs and relationship IDs rather than free-form allegations. Deterministic validation checks record identity, participating actors, chronology and applicable source relationships. An offline dashboard exposes the selected evidence, counterevidence and missing context for review.

## Working implementation

- Streaming text ingestion and Parquet projections, with DuckDB source joins. The pinned export has 2,510,487 computer-use turns and 183,485 chat messages; the provenance join links 100,353 of 100,354 sent-chat actions. Those are integrity counts, not detection quality.
- Exact prior-quotation retrieval found 1,221 unique cross-agent references. A reference can endorse or reject a claim, so it is a retrieval signal rather than an incident label.
- Bounded CPU inference, pinned checkpoint/file hashes, structured source-ID outputs, invalid-output abstention and coverage-aware review routes.
- Private source timeline with actor/type/text filters, model-selected evidence links and explicit missing-context information. It has no external assets or telemetry.
- Blinded reviewer-attestation CLI, aggregate evaluation with explicit denominators and abstentions, and reproducible environment/resource reports.

All data processing, model downloads and inference run on an Azure E64ads_v7 CPU VM. No hosted inference APIs, training or fine-tuning are used. The selected model is provisional, based on measured development behavior.

## Measured comparison

The eight authored trajectories form four counterfactual pairs covering authorization, recovery, reliance and goal substitution. Their answer key is separate from model input. These cases are development tests whose classes are described in the prompt, not held-out natural-data validation.

| Decision path | Authored result | Observation |
|---|---|---|
| AgentDoG Unified4B | 5/8 broad safety expectations | Confused successful recovery; broad safety differs from swarm behavior |
| Qwen3.5-4B fast | 4/8 swarm verdict/category expectations | Cleared a failed-save case; three outputs rejected/insufficient |
| Qwen3.5-4B reasoning | First two cases incomplete at180s each | Stopped early to avoid unnecessary CPU cost |
| Qwen3.5-9B fast | 8/8 swarm verdict/category expectations | Four workers × eight threads;146.31s wall;16.48–16.62GiB peak per worker |

A separate 18-pair DeBERTa NLI test exposed high-confidence actor confusion. It remains an optional helper behind verified pair mappings, not a decisive detector.

The public-safe demo includes actual saved 9B predictions on fictional cases. In one case an agent reports a successful save after the final tool result says the file is absent; a peer explicitly relies on that claim and sends a completion notice. The paired case changes the tool result to a successful retry before the claim, and the model changes its verdict. This illustrates the intended evidence distinction, not a discovered real-world incident.

## Evaluation still required

No complete AI Village episode has a human reference label in this version. We therefore do not report natural-data precision, recall or confirmed incidents. Eight private real packets test execution and context handling, but their selection and incomplete histories cannot support population estimates. Follow-up actions must be retrieved after the peer's reference; temporal proximity alone does not prove influence.

The first eight real cases took 374.51s: four no-finding responses and four incomplete generations, with all eight routed to insufficient due to incomplete histories. Expanded retrieval retained 32 later peer actions and 32 recorded outcomes across eight packets. A subsequent two-case smoke still had one incomplete response at a 240-second cap. Natural-data CPU latency/completion is an unresolved engineering limit.

The next quality gate is blinded review of complete source cases and a frozen, separately selected evaluation set. Report false alerts, missed reviewed incidents, unavailable context, abstentions and runtime separately. Source-ID validation prevents some invalid claims; it cannot establish semantic truth, artifact identity or internal intent.

## Reproducibility and submission status

See `README.md`, `docs/local-runbook.md`, `docs/local-decision-models.md`, `docs/experiment-log.md`, `config/local.toml` and `requirements-local.txt`. Natural records, model weights, private model reports and credentials are excluded from Git. AI Village research terms require written permission for training/fine-tuning; neither has been performed.

The owner has requested a local repository. Publishing the code, adding team names/contact details and submitting the final write-up or demo remain separate authorized actions. This draft is not a claim of hackathon completion or winning performance.
