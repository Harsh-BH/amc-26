# P56-thresh-grain: finer-grained decision thresholds (per source, per group state, per country) — untested
Why: every decision today uses one threshold per country (India/US: combiner t 0.71; France: P42 recipe t 0.99 with
stage bars rescue 0.90 / gc 0.70 / hn extras 0.80). S2 and S3 have different noise and different truth counts (≤5 S2,
≤6 S3; S2 mean 1.67, S3 mean 1.79), and the value of adding a pair depends on how full the S1's group already is (F0.5
break-even differs for an empty S1, a 1-member group, a 4-member group).
Current best: output/sub50e (India/US = P44 combiner full-holdout refit + r_all; France = sub48 France). P50 integrates
everything into output/sub_all — send P50 your change set if anything passes.
Arms (≤ 45 min each; report duds):
1. India/US: separate t for S2 vs S3 pairs, and t by group state (# already-accepted members of the S1 from the same
   source: 0 / 1 / 2+), fit on holdout half 0, scored on half 1 + swap, paired SE, using the combiner's out-of-fold p
   (probes/P44-meta-stack te44.py / solution_p44.py; P49 refit it on half 1 too). Gate: India+US ≥ +0.00010 at ≥ 3 SE,
   neither half negative.
2. France: per-source bars (S2 vs S3) for the base t and the gc / hn-extra stages in the LOCO analogue
   (probes/P42-fr-joint-tune/solution_p42.py). Gate: both directions ≥ 0 and India→US ≥ +0.0003 France-F0.5.
If something passes, build output/sub56 on sub50e (validator --check-ids, flip profile, SUBMISSIONS.md row) and send the
change set to P50. Own worktree; code in probes/P56-thresh-grain/. Memory 1.5 GB local; SageMaker CPU ($5). DISK ~2 GB.
Never end your turn to wait. Deadline: verdict 07:30, file 08:30.
