# Action-grounded communication audit — aggregate findings

Generated: 2026-10-02 09:54 UTC
Dataset revision: `838b4150303ca8228e8edb432d8b8ccae353d258`; export timestamp: `2026-09-20T13:05:12.097Z`.
All corpus access, projection, and inference ran on the Azure VM. This report contains no raw records.

## What the data supports

The export has **46 agents**, **183,485 chat messages**, **381,610 events**, **2,510,487 computer-use turns**, and **246,151 memories**.
A content/agent/time join linked **100,353 of 100,354** sent-chat actions to their public chat rows (99.999%); 1 remained unmatched.
All scanned turn-to-session references resolved. 57 event message IDs were absent from the chat table, while no chat IDs were absent from events.
Turn evidence includes **979,075 bash-command rows**, **1,121,412 tool-output rows**, and **153,245 error rows**. The screenshot-redacted flag is true on 14,196 turns. No screenshot TARs were fetched.

## Working investigator

The pipeline screens 21,100 broad first-person/completion-verb leads and keeps 500 deterministic random controls. It reconstructs at most 20 preceding actions from the same session for each selected send.
A transparent review-queue score ranks 21,100 leads; 10,187 had some prior-window error, so error presence alone is too common to call a contradiction. This score is **not a probability**.
The review policy separates completed claims from plans and separates supported, contradicted, insufficient, and outside-window evidence. Neither a tool error nor a model output alone establishes misalignment.

## Laya pilot and decision

Frozen Laya 0.3.23 English (`55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`) labeled 36/200 broad lexical candidates and 11/100 controls as completed claims; 33/300 labels changed under an option-order permutation.
Frozen Laya typed-decisions (`1a793eb568e6718f15941d08f85432581df534e3`) labeled 35/200 candidates and 6/100 controls as completed; 17/300 labels changed under option reordering.
These are **unlabeled real-data pilot distributions, not precision or recall**. The checkpoints warned about invalid temperature entries, so their confidence values are not treated as calibrated. A three-choice schema and focused-clause input failed exploratory stability checks and were not scaled. Laya annotations remain an abstaining secondary facet until human-reviewed quality is established.
A separate blinded queue contains 100 top-ranked leads, 100 other lexical leads, and 100 controls. On those 300 cases, Laya returned 46 stable completed labels, 229 stable negative labels, and 25 option-order abstentions in 143.5 s with 2.59 GiB peak process RSS. These are annotations, not reviewed truth.

## Runtime and limits

A streaming pass over all turns and memories took 83.9 s and peaked at 48 MiB process RSS. The turn projection reduced 7.75 GiB decompressed JSON to 712 MiB Parquet in 46.5 s. The send join took 2.6 s and the evidence-window pass 6.5 s, after a 1.0 s sort.
The 2026-03-24 permanent-computer-use rollout changes the meaning of sessions across time. Chat is room-wide, so this system does not infer that another agent saw a statement. Browser outcomes may require screenshots; memories and generated summaries may be incomplete or wrong. The sequential live export can produce small parent/child differences.
No dataset training or fine-tuning was performed. No real message excerpt is included in this report. Human-adjudicated evidence precision remains unmeasured pending structured review of full VM-side cases.
