# P44-meta-stack: test the "0.9924" report's idea rigorously — a 2nd diverse GBDT + calibrated meta-learner
Reference: probes/P44-meta-stack/approach_9924_reference.md (a teammate-style report built on OUR pipeline: m2_big, p10 pruner,
ce_bge_pool, SIB3 offsets, LB 0.984072 = our sub9ab). Claim: a 21-feature house-number-focused GBDT ("version_anu",
500 trees, GroupKFold by S1) + an out-of-fold logistic meta-learner z = w·[p_m2, p_anu, |diff|, min, max, has_hn,
exact_hn] + margin rule (top1−top2 < 0.10 → t ≥ 0.85) + risk buckets (no hn 0.80, non-unique name 0.78) lifts
India/US dev F0.5 0.9903 → 0.9924 on a 30k-S1 fold. Red flags to respect: 30k-S1 sample (SE ~0.0005), their metric
formula is pooled-P/R per country (the real metric is per-S1 macro, docs/PROBLEM.md), possible leakage of the dev fold
into version_anu, and their final file lists only 835,934 matched SX (ours ~5.8M). We do NOT have their model or code.
Our current India/US = #8c + P34 a10 stacker (probes/P34-a10-stack; P36 base.py reproduces it out of fold; holdout
India 0.990615 / US 0.989850). P40: 0.0037 LB of India/US loss is a Bayes floor. P22/P25 (ensembles/bagging) and P23
(EF decoding) are prior art — read them and do not repeat their arms.
Task (own worktree, code in probes/P44-meta-stack/):
1. Rebuild the idea faithfully on OUR holdout protocol: train a diverse 2nd GBDT (hn-focused feature subset, ~21 feats,
   different seed/params) on the same non-holdout folds as m2_big (never the holdout), predict holdout out of fold.
2. Meta-learners, fit on holdout half 0, report half 1 and the swap, paired SE, India and US separately, real metric:
   (a) their logistic recipe on [p_m2, p_hn, |diff|, min, max, has_hn, exact_hn] (+ CE r where it exists);
   (b) the same added as columns to P34's a10 stacker; (c) margin rule + risk buckets on top of #34.
   Report each vs the #34 decision. Also reproduce their "m2 alone → meta" jump on OUR holdout to see if it exists.
3. Gate: India+US ≥ +0.00015 at ≥ 3 SE, neither < −0.00005. France: LOCO both directions (P37/P42 machinery) if a
   France variant is sensible. If it passes: India/US test probabilities for the new GBDT (AWS CPU is fine), build
   output/sub44 = output/sub43_r90 with India/US replaced (France byte-identical), validator PASS --check-ids.
Heavy training → SageMaker CPU jobs (sm_launch.py; ml.m5.4xlarge / m6i.4xlarge $0.97/h, 4 each; ml.r6i.4xlarge ≈$1.25/h
128 GB, 4; bucket prefix amc26-p44/; smoke first; hyperparameters from a config file, never argv). Budget $30.
RESULT.md with tables. Local memory 2 GB. DISK ~3 GB: small files only. Never end your turn to wait.
Deadline: verdict by 27 Sep 07:30 IST, sub44 (if it passes) by 10:00.
