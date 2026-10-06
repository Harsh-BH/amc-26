# P4-match — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
A two-stage LightGBM matcher scores **dense holdout F0.5 0.9822** (India 0.9812, US 0.9828; pair precision 0.9947, recall 0.9604). Stage 1 uses 65 name, address, candidate-list-context and rule features. Stage 2 adds each pair's stage-1 competition. The decision rule is fixed: each S2/S3 goes to its argmax S1, then one global t = 0.73.
The rules score **0.7708** on the same full-corpus setup (P8's p8-rules; 0.8215 was with mini-world SX), so this is **+21 pts**. A single-stage M2 alone scores 0.9814; stage 2 adds +0.078 pt (paired bootstrap 95% CI [0.058, 0.099]).
Adopt `features.py` + `model.py` + `featrun.py` + `train.py` with boosters `m2fast_fold2.txt` → `m2_stage2.txt`. Test probabilities are already written (`data/work/p4/prob_test_*.parquet`); P5 should tune on `prob_holdout.parquet` / `prob_dev.parquet`.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- **Candidates:** P3's rule over the whole S1 pool (`data/work/p3/cands_test_*`) ∪ `rules.match(f1, fx, by_state=True)` accepted pairs.
  - The accepted pairs add only +0.02 pt recall (India).
  - Candidate recall on the holdout is 0.9917.
  - Adding every rule-KEY pair would give +0.17 pt recall, but for ~76 more pairs per S1. Rejected.
- **Features:** `features.featurize(pairs, s1_rec, sx_rec, counts=None, rule_acc=None) -> DataFrame[65 float32]`, where `s1_rec = features.rec(s1_df)` and likewise for SX.
  - At full scale, use `featrun.py ctx|feat <split> <country>`.
    - It is resumable per P3 SX text file and reuses the shared rules caches `data/work/rf_{train,test}_sx_<c>.parquet`, `rf_test_s1_<c>`, `rules_s1feat_train`.
    - Output goes to `data/work/p4/X/<split>_<c>_sx_NNN.parquet`.
  - **The context features need the COMPLETE candidate list of every S1 and SX of a country.** Never featurize a sample of pairs.
- **Stage 1:** `data/work/p4/m2fast_fold2.txt` (LightGBM, lr 0.1, 63 leaves, 2,028 trees, trained on fold-2 S1 pairs). Run `train.p1_pass(ma, split=...)` to get p1 for every pair plus its competition features:
  - SX side: rank, margin over the best other S1, count with p1 ≥ 0.5, sum.
  - S1 side: the same four.
  - Output: `data/work/p4/p1/<X file name>`.
- **Stage 2:** `data/work/p4/m2_stage2.txt` (499 trees, trained on fold-3 S1 pairs with out-of-fold p1). Score with `model.predict(m, X ⊕ p1-features)`.
- **Decision (fixed rule for comparability):** `model.argmax_sx` sends each SX to its argmax S1 over **all** S1, then keeps p ≥ **0.73**.
  - The F0.5 curve is flat: 0.9818–0.9822 for t in 0.6–0.8.
  - The dev-tuned t (0.72) gives the same 0.9822 on holdout.
  - P5 owns anything smarter than this.
- **Fallback, single stage:** `m2_ladder.txt` with t = 0.71 → 0.9814. The files are `prob_*_m2.parquet`.
- **End to end, one command:** `train.py final 2 holdout,test` writes all probability files once `featrun.py feat` has run.

## Evidence         (table: ledger name | world | key metrics | verdict)
**How every dense row is measured:**
- Scored S1: holdout S1 (fold 0) — 220,358 S1, 763,735 true pairs.
- Competition: EVERY train S1 (2.2M) and EVERY train S2/S3 (10.3M) compete.
- Candidates: the same P3 rule as on test, run for all train S1.
- Metric: macro F0.5 over all holdout S1. Singletons are included, and true pairs missing from the candidates count as misses.
- `all` = argmax over all train S1, the test-faithful setting.
- `fold` = argmax among holdout S1 only (the semantics of P3's file). The two differ by ≤ 0.0003 everywhere.

| ledger name | world | key metrics | verdict |
|---|---|---|---|
| (P8) p8-rules | holdout-dense (full corpus) | F0.5 0.7708 (India 0.7893, US 0.7585), precision 0.912, recall 0.639 | the baseline |
| M0-hand-rule | holdout-dense-full | F0.5 0.8257 (India 0.8359, US 0.8189); precision 0.904, recall 0.748; AUC 0.953 | drop |
| M3-fellegi-sunter-em | holdout-dense-full | F0.5 0.2695, AUC 0.600: free EM drifts to the same-name cluster (λ 0.58) | drop |
| M3-fs-em-fixed-lambda | holdout sample (200k) | λ pinned to #SX/#pairs (one parent per SX): AUC 0.867. Free EM 0.780, M0 0.959 | drop |
| M1-logreg | holdout-dense-full | F0.5 0.9334 (India 0.9201, US 0.9423); precision 0.967, recall 0.894; AUC 0.9953 | drop |
| M2-lightgbm | holdout-dense-full | **F0.5 0.9814** (India 0.9803, US 0.9821), t 0.71; precision 0.9940, recall 0.9588; AUC 0.9993, AP 0.9981; singleton acc 0.974. Re-checked with `metric.score` on string ids. 2,276 trees, 783 s fit on 3.14M pairs | keep (fallback) |
| M2-lightgbm-fast | holdout-dense-full | F0.5 0.9811 (India 0.9801, US 0.9818), t 0.72; 2,028 trees, 448 s fit | keep (stage 1) |
| **M2-lightgbm-stage2** | holdout-dense-full | **F0.5 0.9822** (India 0.9812, US 0.9828), t 0.73; precision 0.9947, recall 0.9604; AUC 0.9994, AP 0.9984 | **keep** |
| stage2-vs-M2-paired-bootstrap | holdout-dense-full | +0.078 pt, CI [0.058, 0.099]; India +0.089, US +0.071; 4,516 S1 better vs 2,912 worse | keep |
| (dev check) prob_dev | fold 1, 221,870 S1 | stage 2: F0.5 0.9823 at t 0.72 (India 0.9819, US 0.9827) | consistent |
| train-full-cands-all-S1 | train full corpus | India 12.44M pairs (14.1/S1), US 18.93M (14.3/S1). Fold-0 pairs are identical to P3's `cands_holdout_full_*`: 1,237,432 India pairs, same scores, S1 ranks, via and labels. Recall 0.9906 / 0.9923 | keep |
| recall-p3-union-rules | holdout-dense-full | P3 0.9906 / 0.9923 → + accepted 0.9908 / 0.9923 (+0.034 / +0.001 pairs/S1) → + all keys 0.9922 / 0.9941 (+77 / +75 pairs/S1, upper bound) | keep accepted, drop keys |
| stage2-test-predictions | test | 27.49M pairs, 0 duplicates, exactly 1 argmax per SX. At t 0.73: 3.37 / 3.38 / 3.37 matches per S1, 5.6% / 5.8% / 5.8% S1 empty (France / India / US) | keep |
| M2-test-predictions | test | 27.49M pairs. At t 0.71: 3.33 / 3.37 / 3.36 matches per S1 and 5.2% / 5.7% / 5.7% S1 empty (France / India / US), vs holdout 3.35 / 5.65% | keep |
| ladder-dev-mini | dev mini-world | M0 0.970, M1 0.991, M2 0.993 — no chain collisions, cannot rank models | inconclusive |

**What the model uses (gain share):**
- M2: sx_margin 0.59, sx_gap 0.066, hn_diff 0.061, block_rank_sx 0.045, name_tsort 0.026, hn_eq 0.021, core_ntok_x 0.020, name_cos 0.014, core_tsort 0.010, addr_tset 0.010, rule_acc 0.005.
  - The dense problem is mostly "which S1 of the SX's candidates wins, and by how much". House-number agreement or conflict is the main within-chain discriminator.
- Stage 2: p1 0.57, sx_p1_margin 0.29, s1_p1_margin 0.10, s1_p1_n05 0.02.

**Where M2's remaining 1.86 pts go (t 0.71):**
- By S1 outcome:
  - S1 with recall loss only: 0.99 pts.
  - S1 with a false positive: 0.40.
  - Non-singleton S1 with no correct match: 0.33.
  - Singleton false positives: 0.14.
- Of true pairs that are in the candidates:
  - 96.7% are accepted.
  - 2.2% are the SX's argmax but below t. This is P5's per-S1 decision room.
  - 1.1% lose the argmax to another S1.
- 4,393 false positives out of 736,679 predicted pairs. Only 1.2% of those SX have a true holdout parent: FPs are SX of non-holdout S1 or pure distractors.

**Throughput** (shared 16-core machine, load 15–27):
- Features: 1.5M pairs per 500k-SX file in 45–60 s idle, up to 350 s under load. That is ~30k pairs/s end to end, including `rules.match`. `strings()` alone runs at 58–90k pairs/s.
- All 27.5M test pairs: ~50 min.
- Predict only, 12 threads: M2 15.6k pairs/s, M2-fast 21.5k, stage-2 model 141k.
- Stage 1 is the test-time bottleneck: ~21 min for 27.5M pairs.

## What failed       (approaches tried and dropped, with the number that killed them)
- **Fellegi–Sunter EM (M3), label-free:**
  - Free EM: AUC 0.600, F0.5 0.27.
  - λ pinned by the one-parent prior: AUC 0.867, still below the hand-weighted M0 (0.959).
  - Conditional independence fails in chains: name agreement is shared by the true S1 and by every same-name S1.
- **M0 hand rule (0.8257) and M1 logistic regression (0.9334):** the per-SX competition features need non-linear interactions with address evidence.
- **Unioning every rule-key pair into the candidates:** +0.17 pt recall for +76 pairs per S1.
- **Mini-world evaluation:** M2 0.993 there vs 0.981 dense, and M0 0.970 vs 0.826. It cannot rank models.
- **Engineering dead ends, all fixed:**
  - Per-pair HashingVectorizer: 9k pairs/s. Replaced by per-record sparse rows + row gathers.
  - `np.isin` and Python-string id maps: 2.67 GB in the export. Replaced by arrow `index_in` + `searchsorted`: 1.77 GB, byte-identical.
  - Holding the whole rules cache and concatenating whole files: OOM at 3 GB. Replaced by streamed pieces, per-chunk parquet writes, and the system allocator + `malloc_trim`.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- **`cands.py run|export <country>`:**
  - P3 `block.CFG` for EVERY train S1, streamed and resumable → `data/work/p4/cands_train_full_<c>.parquet` (q1, qx, block_score, block_rank_s1, block_rank_sx, via, fold, is_true).
  - q1 / qx are row numbers in P3's `txt/<c>_s1` and in the concatenated `txt/<c>_sx_*`.
  - About 60 min for both countries; peak 2.7 GB.
- **`features.py`:**
  - `rec(df)` / `assemble(rf, txt, raw_name)` → record fields.
  - `mats(rec)` → per-record sparse rows.
  - `context(pairs, q1, qx)` → block + candidate-list context.
  - `strings(...)` → 50 string / field similarities.
  - `featurize(...)` → all 65. Self-check: `python features.py`.
- **`featrun.py rf|ctx|feat <split> <country>`:** full scale, one SX text file at a time. Peak 3.1 GB (train US), 2.8 GB (test India); ~1 min per file idle.
- **`model.py`:**
  - `fit(X, y, Xv, yv, params, cols)` → Booster; `predict(model, X)` → float32.
  - `argmax_sx(s1, sx, p)` → mask.
  - `sweep(pairs, truth_n, by=)` → best global t, macro F0.5 (exact metric), per-group F.
  - `m0_score`, `fs_fit` / `fs_score`. Self-check: `python model.py`.
- **`train.py`:**
  - `ladder`: M0 / M3 / M1 / M2.
  - `stage2`.
  - `final <1|2> holdout,test`: probability files.
  - `p1_pass(ma, mb, split=)`.
  - `score_folds` / `evaluate`: dense scoring with both argmax variants.
- **`recall.py`:** the candidate-union analysis.
- **Outputs, all in `data/work/p4/`:**
  - `prob_holdout.parquet` (3.13M pairs, fold 0) and `prob_dev.parquet` (3.15M, fold 1): stage 2.
    - Columns: s1, sx, p, is_true, country, am (SX argmax over ALL train S1), sx_best_p, blocked.
    - Ties at the SX max occur only below 0.73 and never on a true pair.
  - `prob_test_{France,India,US}.parquet`: stage 2; s1, sx, p, am, sx_best_p, blocked. `*_m2.parquet` = the single-stage fallback.
  - Boosters: `m2_ladder.txt`, `m2fast_fold2.txt`, `m2fast_fold6.txt`, `m2_stage2.txt`.
  - Per-pair features: `X/`, 3.4 GB for train + test. Stage-1 pass: `p1/`. Kept on purpose: P6's leave-one-country-out can train on `X/train_US_*` and score `X/train_India_*` directly (`train.load`, `model.fit`).

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Licences / rules:**
  - lightgbm (MIT), rapidfuzz (MIT), scikit-learn / scipy / numpy / pandas (BSD), pyarrow (Apache-2.0).
  - No pretrained models, no external data, no APIs. Country is never a feature.
- **France (unseen):**
  - Every feature is country-agnostic, but France's address fields rely on train-fitted P1 tables (région, no place_freq) and on train IDF.
  - On test, France gets the same prediction profile as India/US (3.33 matches/S1, 5.2% empty). The share of argmax pairs above t is 0.60 / 0.58 / 0.58 (France / India / US) vs 0.72 on the holdout: test carries more distractors everywhere, and France is not an outlier.
  - Leave-one-country-out is P6's check; I did not run it.
- **Density shift:**
  - Test has 5.5–5.8 SX per S1 vs 4.7 in train. US test has 0.66M S1 vs 1.32M in the train pool, so fewer chain collisions: the dense holdout is, if anything, harder than test for US.
  - Matches per S1 on test equal the holdout's, so the extra test SX look like distractors the model leaves alone.
- **Early stopping** used a 25% S1 sample of fold 1, so `prob_dev` is marginally optimistic. The holdout was used only to sweep t, and the dev-tuned t gives the same number.
- **Coupling to P3:**
  - `sx_margin`, `block_rank_sx` and the p1 competition depend on P3's exact rule (keep_r 2, k_noaddr 10, keep_f 10) and vectorizers.
  - Changing the blocker means rebuilding `cands.py` + features + retraining (~2.5 h).
- **Semantics trap:** P3's `cands_holdout_full_*` rank each SX among holdout S1 only, so `block_rank_sx` differs from test semantics on 63% of pairs. Use `data/work/p4/cands_train_full_*` for anything SX-side.
- **RAM:** the feature step peaks at 3.1 GB (train US), inside a 3.5 GB cap. Run one country at a time.
- **Disk:** X/ + p1/ ≈ 4 GB. The free space hit 3.6 GB mid-run. The ctx/ and picks/ intermediates are deleted after use.

## Open questions   (for the master; each with your best guess)
- **Stage 2 (+0.08 pt, one extra scoring pass, ~25 min on test) or single M2?**
  - Best guess: stage 2 — it is built, validated and its test file exists.
  - The P5 decision layer may overlap with part of its gain; compare P5 on both `prob_holdout` and `prob_holdout_m2`.
- **Train stage 1 on more folds?** It used fold 2 only (3.1M pairs); stage 2 used fold 3. Best guess: ≤ +0.05 pt, and RAM-bound here (3.1M pairs is already 3 GB at load). Not worth it before P5.
- **Per-country thresholds?** Best guess: no. India and US peak within 0.02 of each other, and France needs the global default anyway.
- **May IDF / place_freq be fitted on test text** (PROBLEM.md open question 4)? Best guess: no, so keep the train fit. France would benefit most if allowed.

## Improvement round (MERGE-2: more training folds for the shipped single-stage M2)
**Verdict: ADOPTED by master ruling** (the gain is certain at 6.7 SE and the change adds no complexity). My own call had been
"not adopted" (below the 0.001 bar). The same M2 recipe on 4× the training data, CE-blended exactly like #4, scores **0.9885**
against #4's **0.9881** on P5's report half. That is **+0.00040 ± 0.00006** (1 paired SE; 716 S1 up, 325 down).
It is real (6.7 SE) but below the ≥ 0.001 bar. Test files: see "Adoption" below. Every shipped prob file is untouched.

| ledger name | world | key metrics | verdict |
|---|---|---|---|
| M2-big-train | train | folds 2–5 = 12,595,232 pairs; same PARAMS (lr 0.05, 127 leaves); early stop on the same fold-1 25% sample → 4,930 trees, fit 2,371 s, peak 3.70 GB (float16 `lgb.Sequence`, ~0.4 GB swap) | — |
| M2-big-vs-M2-no-ce | fold-1 val sample / full holdout | Pair level: val logloss 0.02555 → **0.02285**, AUC 0.99932 → 0.99944, AP 0.99807 → 0.99842. Fixed rule (argmax + swept t), no CE: **0.9814 → 0.9834** (t 0.71 → 0.69; recall 0.9588 → 0.9624, precision 0.9940 → 0.9949; India 0.9830, US 0.9837) | keep (as a matcher) |
| improve-control-M2-ce-D3 | holdout half 1 | #4 reproduced on the same code: **0.9881** (t 0.70, t0 0.73, w [0.636, 0.757, 0.355], 193,768 band pairs; 0.9815 without the CE) | control |
| improve-big-ce-D3 | holdout half 1 | **0.9885** (India 0.9894, US 0.9879), t 0.69, t0 0.76, w refit [0.678, 0.715, 0.338]. Precision 0.9977, recall 0.9703, singleton acc 0.9882. Without the CE: 0.9834. Gain vs control +0.00040 ± 0.00006 | drop (< 0.001) |

- **Why the gain shrinks from +0.0019 to +0.0004:** the extra data mostly fixes pairs in the uncertain 0.02–0.98 band. That band is exactly what the cross-encoder already rescores, so the two overlap.
- **Band pairs:** 193,751 for the big model vs 193,768; only 11,078 needed new CE scores (`data/work/p4/r_holdout_big.parquet`).
- **How it was run:** `probes/P4-match/big.py`, in this order: `train` → `holdout` → `ce` (under `flock /tmp/amc-gpu.lock`) → `eval`.
  - P4b's `rerank.fit_blend` / `blend` / `d3` and P5's `bench.load` / `select` / `metrics` are imported read-only from the main checkout (d8e7e11).
  - Blend weights and (t, t0) are fit on half 0 only.
- **Outputs:** `m2_big.txt`, `prob_holdout_big.parquet` (prob_holdout schema; argmax over ALL train S1), `improve_eval.json`.
  - If the master changes the bar, `big.py test` + `flock … big.py blendtest India|US` writes `prob_test_<c>_big[_blend].parquet`. Prediction is ~7.7k pairs/s, so ~60 min for the 27.5M test pairs.
- **Not tried:**
  - More than 4 folds. Folds 2–5 already peak at 3.7 GB, and going from 1 to 4 folds bought +0.0004 after the CE, so 6 folds would add less.
  - Changing hyper-parameters. The brief said keep them; no evidence argued otherwise.

### Adoption: test files for m2_big (MERGE-2 ruling)
All files are in `data/work/p4/`. The shipped `prob_test_*.parquet`, `*_m2.parquet` and P4b's files are untouched.

| file | what | rows |
|---|---|---|
| `prob_test_{France,India,US}_big.parquet` | m2_big p for every test pair. The same schema as the shipped files: s1, sx, p, am (argmax over ALL S1 of the country), sx_best_p, blocked | 4,348,797 / 12,720,874 / 10,424,924 |
| `prob_test_{India,US}_big_blend.parquet` | the #4 CE blend with w = [0.678, 0.715, 0.338] on 0.02 < p < 0.98; p = blended, p_gbdt = m2_big p | 12,720,874 / 10,424,924 |

- **CE reuse:**
  - #4 kept only the blended p, so r was recovered by inverting #4's blend on M2's band pairs.
  - Against a fresh CE score of 5,000 of those pairs per country: p99 |Δ logit r| 4e-5, max |Δr| 8.5e-4 (India) / 3e-7 (US).
  - Reused: India 718,594 of 847,265 band pairs, US 599,133 of 702,137. Only 128,671 / 103,004 new pairs went to the GPU (≈ 1.5 min each).
- **France threshold:** P6's rule (`fr_threshold.py 'data/work/p4/prob_test_{c}_big.parquet' 0.69`) gives **t_France = 0.78**.
  - The assigned-share cross-check gives 0.80.
  - At t* 0.69 the US+India reference is 5.69% empty and France 5.33%.
  - #4's M2 had 0.82.
- **Test answer profile** (P5 `select`: excl via `am`, t / t0; France t0 = t). Ledger rows `test-profile-big` / `test-profile-sub4`:

  | | India | US | France |
  |---|---|---|---|
  | big: matches/S1, S1 empty | 3.377, 5.77% (t .69, t0 .76) | 3.379, 5.75% | 3.239, 5.69% (t .78) |
  | #4 (reproduced): matches/S1, S1 empty | 3.375, 5.75% (t .70, t0 .73) | 3.378, 5.74% | 3.232, 5.68% (t .82) |

- **Answer churn vs #4** (`test-churn-big-vs-sub4`):
  - India: 1.1% of S1 change (5,176 pairs added, 4,040 dropped). US: 0.95%.
  - **France: 10.4% of S1 change.** 10.3 pts of that comes from the model alone at a fixed t 0.82; the threshold move alone accounts for 3.9 pts.
  - On the labelled holdout, the same model swap (no CE) changes 2.9% of S1, and it is strongly net-positive: pairs only big answers are 85.7% true, pairs only M2 answered 43.8% true.
- **Risk:** France churns 3.5× more than the labelled analogue and there are no France labels to say which way.
  - Best guess: net-positive like the holdout, because the same features and more data give better-calibrated pair evidence.
  - A leave-one-country-out check on m2_big (train folds 2–5 of one country, score the other) would settle it. I have not run it.
  - A risk-averse master can ship big for India/US and keep #4's France (M2, t 0.82).
- **Reproduce:**
  - `big.py test`
  - `flock /tmp/amc-gpu.lock big.py blendtest India` (then `US`)
  - `fr_threshold.py … 0.69`
  - `big.py profile 0.78 0.82`
