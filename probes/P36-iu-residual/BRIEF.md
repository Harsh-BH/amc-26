# P36-iu-residual: fix the India/US errors that remain after #34v_rescue90 (LB 0.98596, best)
Base India/US = #8c + P34's a10 stacker (probes/P34-a10-stack), unchanged in output/sub34v_rescue90. France is out of scope.
Read first: probes/P23-research/RESULT.md (error budget table, rows a–f), probes/P34-a10-stack/RESULT.md,
probes/P35-twin-propagate/RESULT.md (the address-less/twin idea is DEAD — do not redo it), probes/P23-error-analysis/work
(holdout dumps). Own worktree; code under probes/P36-iu-residual/.

Task:
1. Re-cut the error budget on the holdout **after** the a10 stacker: blocking miss (with / without address), pruner miss,
   low-p FN, stolen FN, FP on non-singletons, FP on singletons, wrongly-empty S1. LB-eq per bucket (India 0.468, US 0.383).
   Show 30 raw examples of the two biggest recoverable buckets (names + addresses) and name the patterns.
2. Attack the biggest recoverable bucket with a label-free, rule-legal fix. Untried levers, in priority order:
   (a) blocking-miss recovery: a second, narrow blocking pass only for S1 whose best p < t (e.g. house-number + street-token
       key, address char-3gram TF-IDF, name tokens within state), scored by the existing m2_big + CE + a10 chain;
   (b) per-S1 expected-F0.5-optimal subset decoding on the stacked p (P5 tried decision layers on the old p; retry only if
       the a10 p is better calibrated), with the hard constraints from train: each SX belongs to exactly one S1,
       ≤ 5 S2 and ≤ 6 S3 per S1;
   (c) anything the pattern read in step 1 suggests.
3. Gate: holdout half-split, fit on half 0, report half 1 and the swap, paired SE, India and US separately.
   India+US ≥ +0.00015 at ≥ 3 SE and neither country < −0.00005. If it passes, build output/sub36 = sub34v_rescue90 with
   India/US replaced (France byte-identical), run the validator, and write run.py integration notes.
RESULT.md with tables. Memory 3 GB. DISK IS NEARLY FULL (~1.5 GB free): keep intermediates small, delete scratch, never
write >300 MB locally (offload to s3 with AWS_PROFILE=harsh if needed). Never end your turn to wait. Deadline 27 Sep 11:00 IST.
