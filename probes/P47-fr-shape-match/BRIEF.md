# P47-fr-shape-match: accept France pairs until France's group shape matches India/US (label-free target)
Why: top-3 LB = 0.990816 / 0.990788 / 0.990675 (tightly clustered). Ours 0.986304. India/US ≈ 0.9903 (holdout = LB) with
a 0.0037 LB Bayes floor (P40), so ≈ 85% of our 0.0045 gap is France (implied ≈ 0.964 vs ≈ 0.99 for the top). Every France
rule today recovered pairs at p 0.3–0.9 that are ≥ 95% true on train analogues: the matcher is under-confident in France.
The generator is shared: India/US test predictions reproduce train truth's per-S1 shape almost exactly (S2 count shares
0/1/2/3/4/5 = .130/.358/.296/.151/.054/.011; S3 = .121/.325/.303/.169/.066/.016/.001; two-sided 0.805, S3-only 0.075,
S2-only 0.065, empty 0.056; mean 3.46 SX/S1 truth, 3.38 India/US predicted). France after the current best stack
(output/sub46_on43 = sub43_r90 + P46; 3.256 SX/S1, empty 5.99%) still has ≈ 36k fewer SX than India/US-predicted density.
Idea: greedy, constraint-respecting acceptance — among FREE, argmax, non-sibling (P15 sib rule) France candidate pairs not
yet accepted, add pairs in descending (calibrated) p, per S1 respecting caps 5/6, until France's per-S1 shape (S2/S3 count
distribution, two-/one-sided shares, mean SX/S1) matches the India/US predicted shape (or truth shape). Variants: stop at
50/75/100% of the gap; order by p vs by Qwen-reblended p (when P38's scores land, ~05:30; data/work/p38/…); per-bucket
targets (e.g. fill one-sided groups first). Also a remover variant if France has shapes India/US lack (e.g. 5-S2 groups).
Validate: LOCO analogue (P37/P42 machinery: probes/P42-fr-joint-tune/solution_p42.py; P37 p37.py) — apply the same
shape-matching on the target country using the SOURCE country's shape as the target (label-free), report both directions
vs the current recipe analogue, added-pair precision, and whether stopping at the shape target is near the F0.5 optimum.
Base = output/sub46_on43 (India/US byte-identical). Build output/sub47_* variants, validator PASS --check-ids, rows in
SUBMISSIONS.md with expected LB. Own worktree; code in probes/P47-fr-shape-match/. Heavy grids → SageMaker CPU
(sm_launch.py, ml.m5.4xlarge $0.97/h; budget $10). Memory 2.5 GB local. DISK ~2 GB: small files only. Never end your turn
to wait. Deadline: first staged file by 27 Sep 06:00 IST, final (incl. a Qwen-ordered variant if P38 lands) by 08:30.
