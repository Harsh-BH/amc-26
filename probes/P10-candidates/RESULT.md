# P10-candidates — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **A learned pruner between P3 blocking and m2_big cuts the test candidate set 15.87 → 4.83 per S1 (−70%). It costs
  −0.00024 ± 0.00003 F0.5 on #5's own yardstick (0.9885 → 0.9882, P5 report half).** The pruner is a 300-tree, 31-leaf
  LightGBM on 30 of the 65 pair features. It keeps a pair if q ≥ 0.03, and always keeps rule-accepted pairs.
- The curve is flat down to ~5/S1: 5.93/S1 at −0.00002, 5.20/S1 at −0.00009, 4.51/S1 at −0.00044.
  - P3's cheaper blocking config is dominated: 7.80/S1 at −0.00031.
  - A per-S1 top-k cap by block_score collapses: top-5 = 5.0/S1 at −0.021, because true matches sit deep in same-name chains.
- **Adopt τ = 0.03.** The #6 inputs are `data/work/p10/prob_test_{France,India,US}.parquet`: #5's p on the 8.37M kept
  pairs.
  - With #5's own blend weights kept, the holdout is 0.98825 (−0.00023 ± 0.00003).
  - France keeps t 0.78 (P6 rule re-run), 5.69% empty, 3.24 matches/S1. That is #5's profile.
  - 3,860 S1 (0.22%) change answer vs #5.
  - `submit_probs` → the official validator **PASS**; `candidate_pairs.tsv` 377 → 130 MB (`output/p10check/` in this
    worktree).

## Recommendation   (exactly what the master should adopt: module, function signature, config)
**#6 = #5 with the pruner. The master writes it with the existing tool:**
```
.venv/bin/python code/business_entity_resolution/src/submit_probs.py \
    --probs 'data/work/p10/prob_test_{country}.parquet' --out output/sub6 --t 0.69 --t-country France=0.78
```
`candidate_pairs.tsv` is then every row of those files: the pairs m2_big + CE actually scored.

**Pipeline stage to promote** (`probes/P10-candidates/prune.py` → `src/prune.py`):
```
m = prune.load_model()                   # data/work/p10/pruner.txt (1.1 MB, LightGBM text)
q = prune.score(m, X)                    # X: the pair-feature frame (features.featurize / the X files), FEATS columns
keep = prune.prune(X, q, tau=prune.TAU)  # q >= 0.03 | rule_acc > 0
```
- **Where it goes** in `run.py --stage model`: after `features.build` (X for every blocked pair), before
  `model.split_probs` / the CE.
- Score m2_big, and send the CE band, only on `keep`. Recompute the argmax per SX over the kept pairs; `submit_probs`
  already does this.
- **Keep the X files computed over the FULL blocking list.** The context columns (s1_n, sx_n, margins, block ranks)
  are blocking-stage statistics that m2_big was trained on. With them, m2_big's p on a kept pair is bit-identical to #5,
  and the pruner only removes pairs.
- **Refit nothing.**
  - At τ 0.03 the tune-half refit gives the same t 0.69 / t0 0.76, and the blend w moves 0.678/0.715/0.338 →
    0.674/0.718/0.343.
  - Keeping #5's w scores 0.98825 against the refit's 0.98824 (`prune-tau0.03-fixed-w`). So #5's blended p is used
    unchanged, and France stays at 0.78 by P6's rule.
- **Train:** `prune.train()`: train folds 2–3 (never 0/1), 6.3M pairs, `is_true` target, 362 s, 2.3 GB.
- **Alternative operating points** (all `prune(X, q, tau)`): τ 0.02 → 5.20/S1 at −0.00009 (conservative); τ 0.05 →
  4.51/S1 at −0.00044 (at the 0.0005 budget).

## Evidence         (table: ledger name | world | key metrics | verdict)
**Yardstick.** It is P4 `big.py`'s `_blend_eval`, the #5 number:
- P5 `bench.load` / `select` / `metrics` and P4b `fit_blend` / `blend` / `d3`, imported read-only.
- Blend w and D3 (t, t0) are refit on holdout half 0; the report is on half 1.
- The one change: each SX's argmax `am` is recomputed on m2_big p over the **kept** pairs of that SX, across every train
  S1 it competes for. The competitors are 11.5M pairs, and `prep.py hold` re-scored all of them with m2_big.
- With nothing pruned it reproduces #5 exactly: **0.9885**, t 0.69, t0 0.76, w [0.678, 0.715, 0.338], P 0.9977, R 0.9703.
- Δ = mean per-S1 paired difference on half 1 ± 1 SE.
- Holdout density is 14.19 blocked/S1 against test's 15.87, so every point is reported both ways.

| ledger name (`P10-candidates/…`) | holdout cand/S1 | **test cand/S1** (Fr / In / US) | cand recall | F0.5 (India / US) | Δ F0.5 vs #5 | verdict |
|---|---|---|---|---|---|---|
| none (= #5) | 14.19 | 15.87 (16.76 / 15.71 / 15.72) | 0.9917 | 0.9885 (0.9894 / 0.9879) | — | baseline |
| prune-tau0.002 | 6.00 | 7.27 (8.52 / 7.34 / 6.68) | 0.9916 | 0.9885 | +0.00000 ± 0.00001 | curve |
| prune-tau0.005 | 5.53 | 6.58 (7.45 / 6.62 / 6.19) | 0.9915 | 0.9885 | −0.00001 ± 0.00001 | curve |
| prune-tau0.01 | 5.10 | 5.93 (6.57 / 6.00 / 5.60) | 0.9910 | 0.9885 (0.9894 / 0.9879) | −0.00002 ± 0.00002 | curve |
| prune-tau0.015 | 4.81 | 5.51 (6.02 / 5.60 / 5.20) | 0.9902 | 0.9884 | −0.00007 ± 0.00002 | curve |
| prune-tau0.02 | 4.56 | 5.20 (5.63 / 5.30 / 4.91) | 0.9891 | 0.9884 (0.9893 / 0.9878) | −0.00009 ± 0.00003 | conservative |
| **prune-tau0.03** | **4.30** | **4.83 (5.14 / 4.90 / 4.63)** | 0.9874 | **0.9882 (0.9891 / 0.9877)** | **−0.00024 ± 0.00003** | **keep** |
| prune-tau0.05 | 4.10 | 4.51 (4.66 / 4.55 / 4.41) | 0.9854 | 0.9880 (0.9888 / 0.9875) | −0.00044 ± 0.00004 | at budget |
| prune-tau0.1 | 3.92 | 4.24 (4.27 / 4.23 / 4.23) | 0.9820 | 0.9877 (0.9885 / 0.9872) | −0.00076 ± 0.00005 | drop |
| prune-tau0.01-noguard / 0.02-noguard | 5.08 / 4.53 | 5.89 / 5.14 | 0.9910 / 0.9891 | 0.9884 / 0.9884 | −0.00004 / −0.00010 | drop (the rule_acc guard costs 0.03/S1 and saves more) |
| prune-tau{0.005,0.01,0.02}+sxtop1 | 5.75 / 5.53 / 5.20 | 6.94 / 6.63 / 6.29 | ≥ 0.9897 | 0.9885 | −0.00001 / −0.00001 / +0.00001 | drop: fine on holdout, +1.1/S1 on test (test has more distractor SX, and each keeps its top-1) |
| prune-tau{0.002,0.005}-rel{0.01,0.03,0.1} | 5.13–5.78 | 6.14–6.99 | ≥ 0.9900 | 0.9885 | −0.00004 … +0.00002 | drop: never below plain τ at equal test size |
| **p3-cheap-kr1-kf5** (P3's cheaper config, ∪ rule_acc) | 6.89 | 7.80 (8.57 / 7.65 / 7.69) | 0.9884 | 0.9882 (0.9889 / 0.9877) | −0.00031 ± 0.00004 | drop (dominated) |
| topk5-block_score / +ra | 4.99 / 5.14 | 5.00 / 5.29 | 0.881 / 0.910 | 0.9679 / 0.9734 | −0.0206 / −0.0151 | drop |
| topk6 / topk8-block_score (+ra) | 5.99 / 7.95 | 5.99 / 7.98 | 0.929 / 0.964 | 0.9764 / 0.9821 | −0.0121 / −0.0064 | drop |
| topk3 / topk4 (+ra) | 3.0–4.3 | 3.0–4.5 | 0.66–0.86 | 0.912–0.963 | −0.025 … −0.076 | drop |
| pruner-featureset-f17 / f24 / f30 (f17 folds 2–3; f24 / f30 fold 2) | fold-0 proxy (am & p ≥ 0.69), at 4.5/S1 | — | 0.9838 / 0.9862 / 0.9888 | selected true pairs dropped: 1,278 / 538 / **57** | AUC 0.9942 / 0.9969 / 0.9981 | f30 keep |
| prune-tau{0.02,0.03,0.05}-fixed-w (#5's w kept, D3 retuned) | 4.56 / 4.30 / 4.10 | 5.20 / 4.83 / 4.51 | — | 0.98837 / **0.98825** / 0.98806 (τ 0.03: 0.9891 / 0.9877) | −0.00011 / **−0.00023** / −0.00042 (± 0.00003) | keep (the deliverable's own number) |
| prune-tau*_big (600 trees × 63 leaves, same 30 feats) | τ 0.02 / 0.03 / 0.05: 4.49 / 4.23 / 4.06 | 5.03 / 4.71 / 4.45 | 0.9894 / 0.9876 / 0.9858 | 0.98831 at τ 0.03 | −0.00014 / −0.00018 / −0.00041 (± 0.00002–4) | drop (see What failed) |
| sub6-candidate-files-e2e | test | 8,370,573 pairs; 22,446 S1 with no candidate (all inside the 99,382 empty answers) | — | validator PASS; `candidate_pairs.tsv` 130 MB vs #5 377 MB | 3,860 S1 answers differ from #5 (= final.py's count) | keep |
| test-pruned-tau0.03 | test | 8,370,573 pairs, 4.83/S1 | — | France t 0.78 (ref empty 5.69%), empty 5.69%, 3.238/S1 (#5: 5.69%, 3.239); India 5.75% / 3.375 (#5 3.378); US 5.73% / 3.377 (#5 3.379) | S1 changed: Fr 0.17%, In 0.25%, US 0.21%; answers lost 420 / 2,048 / 1,442, new 16 / 9 / 1 | keep |

**Where the loss comes from** (τ 0.03, half 1): 263 S1 lose and 11 gain. Every loss is a true pair that m2_big would have
accepted but the pruner scored below 0.03. The candidate recall drops 0.9917 → 0.9874, while the answer recall only
drops 0.9703 → 0.9696, and precision stays at 0.9977. Most of the pruned true pairs are ones m2_big rejected anyway.

## What failed      (approaches tried and dropped, with the number that killed them)
- **Blocking-level signals only (17 features: block score / ranks / via / counts / margins, core_eq, core_jacc, hn / city /
  state agreement, rule_acc):** AUC 0.9942. At 4.5 holdout candidates/S1 it drops 1,278 of 735k selected true pairs; f30
  drops 57. Adding 13 name / address similarity columns that already exist in X (token-sort / token-set / cosine /
  Jaccard, core token count, web flags) fixes it.
- **A 4× bigger pruner (600 trees × 63 leaves):** it moves the curve by ~0.1/S1 at equal loss (τ 0.03: 4.71/S1 at
  −0.00018 vs 4.83 at −0.00024), about 2 SE. Not adopted: it has 4× the leaves for a marginal gain, and the pruner is
  meant to stay a light stage. `data/work/p10/pruner_big.txt` + `pareto_big.csv` are kept if the master wants it.
- **Per-S1 top-k by block_score:** −0.021 at 5/S1. True clusters are big (26% of S1 have 5+ matches) and chains of
  same-name S1 push true SX past rank k, so a fixed k per S1 cannot work.
- **P3's cheaper blocking (keep_r 1, keep_f 5):** 7.80/S1 on test at −0.00031. The learned pruner reaches the same
  loss at 4.8/S1.
- **SX-side guard rails** (keep each SX's top-1 by q; keep only pairs within ρ of the SX's best q): they match plain τ at
  equal holdout size. But test has 5.8 SX/S1 against 4.7, so the top-1 guard adds ~1.1/S1 on test for nothing.
- **Dropping the rule_acc guard:** it saves 0.02–0.06/S1 and costs 0.00001–0.00002 more F0.5. Kept, because the rules'
  pairs are what a reviewer expects to see.
- (engineering) `import rerank` after `import bench` resolves to `src/rerank.py` (no `d3`); evalp imports P4b's first
  and asserts it.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- **`prune.py`:**
  - `train(folds=(2,3), feats=FEATS, rounds=300, out=MODEL)`: 362 s, 2.32 GB.
  - `load_model(path)`; `score(m, X) -> float32 q`; `prune(X, q, tau=TAU) -> bool mask`.
  - `FEATS` (30 columns of P4's X), `F17`, `TAU = 0.03`. Self-check: `python prune.py`.
  - Scoring all 27.5M test pairs: 2–3 min including reading the X columns, ≤ 1.6 GB.
- **`prep.py`:**
  - `hold`: m2_big p + pruner columns for the 11.5M holdout rows → `data/work/p10/hold_{India,US}.parquet`. This is
    the expensive step: 52 min at load 26, 2.0 GB.
  - `holdq <model> <tag>`: q for those rows, 90 s.
  - `testq <model> [tag]`: q for every test pair in X order → `q_test_<c>.parquet`, 3 min.
- **`evalp.py`:** `Bench(cols)` (12 s, 2.4 GB). `Bench.run(keep, refit=True) -> (metrics, per-S1 F)`,
  `Bench.baseline()`, `argmax_grp`.
- **`sweep.py [model] [tag] [only]`:** every variant → `data/work/p10/pareto<tag>.csv` + one ledger row each; 20 min.
- **`fixedw.py [tau …]`:** the same yardstick with #5's blend weights held fixed (the deliverable's own p); 5 min.
- **`final.py [tau]`:** the test deliverable plus the France / profile / churn check, in 2 min at 2.7 GB. It writes
  `prob_test_{c}.parquet`: #5's rows, `am` / `sx_best_p` recomputed over the kept pairs, and p / p_gbdt unchanged.
- **Model file:** `data/work/p10/pruner.txt`. Also kept for reference: `pruner_f17.txt`, `pruner_f24.txt`,
  `pruner_f30.txt` (fold 2 only).

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Licence / rules:** LightGBM (MIT) on features the pipeline already computes. No new dependency, no external data,
  no country feature, and train folds 2–3 only.
- **"The last filtering stage" reading.** The 65 features, including the string similarities, are still computed for
  all 27.5M blocked pairs, and the pruner scores all of them.
  - m2_big + CE, the final matching model, scores only the kept 8.37M. That is exactly what `candidate_pairs.tsv` lists.
  - A reviewer could call the pruner itself a model that "runs over" the blocking output. That is the definition of
    meta-blocking, and it is the brief's framing.
  - The code should present it as its own stage (`prune.py`, 1.1 MB model), which has ~65× fewer leaves than m2_big
    (300 × 31-leaf vs 4,930 × 127-leaf trees).
- **France (no labels):**
  - At a given τ France keeps more pairs than India / US (5.14 vs 4.90 / 4.63/S1): the pruner is less sure on an unseen
    country, so it keeps more. That is the safe direction.
  - France's answer profile is unchanged: 5.69% empty, 3.238/S1, t 0.78. Only 0.17% of France S1 change, below India's
    0.25%.
- **Coupling:** the pruner is trained on P3's exact blocking list and P4's feature code. Changing either means
  retraining it (6 min). If m2_big changes, re-run `sweep.py` (needs `prep.py hold` with the new booster, ~50 min).
- **Blend weights:** the deliverable keeps #5's blended p (w 0.678/0.715/0.338). Measured with fixed w: 0.98825, the
  same as the refit's 0.98824. D3's t0 (0.76) is part of the yardstick only, as for #5; `submit_probs` ships t alone.
- **Test vs holdout size:** holdout 4.30/S1 at τ 0.03 vs test 4.83, because test has more S2/S3 per S1. The loss is
  measured at holdout density. Test answers change at the same rate as holdout answers (India 0.25% of S1 vs 0.24% of
  the holdout half), so the loss should transfer.

## Open questions   (for the master; each with your best guess)
- **τ 0.03 (4.83/S1, −0.00024) or τ 0.02 (5.20/S1, −0.00009)?** Best guess: 0.03. The organisers rank smaller candidate
  sets higher. 0.0002 is below the paired SE of any leaderboard move so far, and 4.8/S1 is 1.4 above the answer size
  (3.38/S1).
- **Should the pruner move into `run.py` before #6, or should #6 go out from these files?** Best guess: ship #6 from
  `prob_test_*.parquet` now; it is byte-for-byte #5's p on the kept rows. Then promote `prune.py` into `run.py` so the
  code the organisers review produces exactly this `candidate_pairs.tsv`. Budget one `run.py --split holdout` run to
  confirm the 0.9882.
- **Does it cut the CE's GPU work?** Only by 20%: 1.55M → 1.24M band pairs (0.02 < p < 0.98) for India + US at τ 0.03
  (India −23%, US −17%). The pruner mostly removes pairs m2_big already puts below 0.02. m2_big's own work drops 70%
  (27.5M → 8.4M pairs).
