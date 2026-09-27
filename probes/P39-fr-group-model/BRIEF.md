# P39-fr-group-model: a group-aware second stage for France (pairs are scored independently today)
Finding (master, 26 Sep 23:30, see probes/P37-fr-complete/BRIEF.md): France predictions have ~14k excess one-sided
groups (S3 found, no S2 or vice versa) and 3.10 predicted SX/S1 vs India/US 3.38 (truth 3.46); India/US match train
truth's group shape. Truth constraints: each SX in exactly one S1; ≤ 5 S2, ≤ 6 S3 per S1. P37 tests simple completion
rules; P39 learns the decision instead. Do not duplicate P37's rules — build the model.
Task: a GBDT second stage over (S1, candidate) rows with group context that transfers across domains: the candidate's
p *rank/quantile within its country* (France p is off-domain), rank within S1 per source, #accepted S2 / S3 already in
the S1's group at the base t, max competitor p for that SX, the S1's best p in the other source, sibling-offset flag,
same-address flag (P28), hn match, name/website similarity already in france.py. Train on India/US holdout labels
(probes/P23-error-analysis/work, P34/P35 frames), LOCO both directions (train US → apply India, and reverse), then
decode with the truth constraints. Gate: LOCO ≥ +0.0005 LB-eq in both directions vs the current France chain analogue.
If it passes, build output/sub39 = sub34v_rescue90 with France replaced (India/US byte-identical), validator PASS.
Own worktree, code under probes/P39-fr-group-model/. RESULT.md with tables. Memory 3 GB. DISK NEARLY FULL: small
intermediates only. Never end your turn to wait. Deadline 27 Sep 11:00 IST.
