# P61-fr-cotrain: co-training for France — train a France-specific combiner on pairs where two independent models agree
Why: single-model pseudo-labels failed (P6, P27: each helped one LOCO direction and hurt the other) because the model
re-learns its own mistakes. Co-training (Blum & Mitchell 1998) uses two models with different errors: label target pairs
only where both agree confidently. We have two: the bge-based France chain (+ P55 combiner) and pooled Qwen r (P38; France
scores in data/work/p38/). Their errors differ (P38's audit: large disagreement on P46/P40/gib classes).
Read probes/P27-france-pseudo (what failed), P55-fr-combiner (country-invariant combiner on LOCO frames; p55.py),
P38-qwen4b-fr (Qwen France scores, diag54_pairs.parquet, LOCO Qwen scores for the holdout band), P48/P54, P50 (integrator).
Own worktree; code in probes/P61-fr-cotrain/.
Plan:
1. LOCO simulation first: on the target country, pseudo-label pairs where the source-trained bge chain and the source-trained
   Qwen both say yes (e.g. chain p ≥ a and Qwen r ≥ b) or both say no; report pseudo-label accuracy vs truth by (a, b).
2. Train P55's combiner on source labels + target co-training pseudo-labels (weighted); apply to the target; compare with
   P55 trained on source only. Both directions; gate both ≥ 0 and India→US ≥ +0.0003 France-F0.5 over P55.
3. If it passes: do it for France (bge chain = sub_clean's France decisions/p; Qwen = P38's pooled France r), build the
   France change set on output/sub_clean, validator --check-ids, and send it to P50.
SageMaker CPU allowed ($10). Memory 1.5 GB local. DISK ~2.4 GB. Never end your turn to wait. Deadline: verdict 10:00, file 11:30.
