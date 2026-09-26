# P26-france-da: adapt the matcher to France (adversarial reweighting + house-number features INSIDE the main model)
**Context (26 Sep 15:05).** Best LB = #9ab 0.984072 (India/US ≈ 0.990, France ≈ 0.952 implied). LB #1 = 0.990556 →
France must be ≈ 0.985: **France is ~75% of our gap.** The matcher m2_big (LightGBM, 65 features) was trained on US+India
only; France gets it cold. Organiser rulings (docs/PROBLEM.md "Organiser rulings"): unlabelled test data may be used,
incl. pseudo-labels; no external geo tables. Credits are fine; the laptop is short on RAM (~5 GB free, shared) and disk
(~4 GB) — **all training/eval on AWS** (reuse P22/P25's SageMaker train + eval path:
/home/harsh1/github-repos/amc-26-wt/p22-ensemble/probes/P22-ensemble/ and .../p25-lgbm-bag/probes/P25-lgbm-bag/; train X
in S3 under amc26-p16/; the AWS eval job reproduces #8c's holdout 0.98990 bit-identically). AWS: `--profile harsh` only.
Read first: probes/P6-france/RESULT.md (LOCO: transfer loss is a country-level recalibration; hn features made US→India
over-confident), P14 (IDF shift; feature-shift SMD table in data/work/p14/shift.csv), P15 (sibling offsets
{3,4,5,7,9,11,13,21}; France accepts 8× more), P17 (hn/support stacker, US only), P13 (LOCO protocol + label-free France
threshold), SUBMISSIONS.md. Own worktree; code under probes/P26-france-da/.

**Phase A — analyse (≤ 1 h, cheap):** adversarial validation France-test vs US/India-train on the pair features (on the
pruned candidate lists, IDF-fixed France X from P14/P19): AUC, top shifted features, and whether the shift is in
"true-pair-like" rows or in look-alikes. Decide the reweighting feature set (drop features that are pure country id).
**Phase B — LOCO proxy (the go/no-go):** simulate: train on US only, target = India (unlabelled), and the reverse.
Arms: (0) baseline m2-style; (1) importance weights w = p_target/(1−p_target) from the adversarial classifier (clipped);
(2) + generator-aware hn features in the MAIN model (P15's hn class / signed delta / eq + P17's support counts n_same,
p_same, n_s1hn — computed label-free on the candidate list); (3) (1)+(2). Measure on the held-out country's full-corpus
fold 0 at the label-free France threshold procedure (P13/P6). Gate: ≥ +1 pt F0.5 on the unseen country in BOTH
directions, and in-country (US+India holdout) ≥ −0.0003.
**Phase C — France:** train the winning arm on US+India (folds 2–5) with weights toward France test, on AWS; score France
test (IDF-fixed candidates, #6 pruner kept), re-blend the bge CE with P13's out-of-domain France weights, pick France t at
D4-equivalent empty rate AND report the P6 rule value; build the France file + `probes/P14-france-shift/combine.py
output/sub8c <file> <t> <out>` + validator. If arm (2) also helps US/India in-country, report that separately (do not
change India/US in the submission unless ≥ +0.0003 at ≥ 3 SE).
Deliver RESULT.md (tables, $ spent, exact commands). Memory 2 GB locally. Never end your turn to wait. Deadline 26 Sep 23:00.
