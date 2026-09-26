# P34-a10-stack — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **Gate passes.** P23's a10 twin_p2mask stacker, fit on holdout half 0 and reported on half 1 at t 0.71 on #8c's
  yardstick, gives **LB-eq +0.000209 ± 0.000056 (3.7 SE)**: India +0.000157 ± 0.000102, US +0.000356 ± 0.000077.
  The swap gives +0.000175 ± 0.000048 (3.6 SE). Reweighted to test's mix it is +0.000202 ± 0.000050. The as-briefed
  version is +0.000212 (3.7 SE).
- **One change from the brief: a sibling guard.** A pair whose house number is a sibling offset (stack.hn_cls == 1)
  keeps #8c's p. It costs 0 on holdout (+0.000209 vs +0.000212; the swap goes up, +0.000175 vs +0.000168). Without it,
  US test adds run at 1.64× holdout's rate, and 1,391 of them sit in a class that is 5× commoner on test than holdout.
- **Staged, validator PASS (`--check-ids`):** `output/sub34` = #31 with India/US replaced (guarded, recommended).
  `output/sub34_raw` is the unguarded as-briefed file. France and candidates are byte-identical to #31. At +0.0002 it is
  inside the LB tie band: ride it in the combined final file, not in a slot of its own.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
**Adopt `output/sub34` (guarded) as the India/US half of the final file, and port `a10.apply` into run.py for P9.**
- Files (main checkout, gitignored):
  - `output/sub34/matching_results.tsv`: md5 508b6951…; 12,848 India/US S1 changed vs #31 (India 7,574, US 5,274).
  - `output/sub34/candidate_pairs.tsv`: md5 b3ccba0d…, the same as #31's.
  - Unguarded alternative: `output/sub34_raw` (md5 4fc7dd25…, 14,396 S1 changed).
- Booster: `data/work/p34/a10_h0.txt`, fit on India+US holdout half 0 pooled.
  - Params: 200 rounds × 31 leaves, lr 0.05, min_data_in_leaf 200, λ2 1, seed 0, deterministic, init_score = logit p.
  - Columns (`a10.COLS`): lp, lm2, r, n_cand_sx, n_conf_other, s1_max_other, s1_n_am, p2m, marginm, n_twins,
    n_ctwins, n_atwins, noaddr, cat (categorical), name_sup.
  - A refit from `ho.py build` + `ho.py eval` is bit-identical and takes 2 s.
- Decision: **acc = am_8c & p' ≥ 0.71**, where am_8c is #8c's argmax computed *before* the stacker. Re-deriving the
  argmax from p' changes 585 US and 611 India test decisions, so neither submit_probs.py nor P14's combine.py
  reproduces sub34. `te.py combine` does, from the prob files' `am` column.
- Test prob files: `data/work/p34/prob_test_{India,US}_a10.parquet` (s1, sx, p = p', p8c, am). They sit on #6's
  pruned rows (#8c's, 3,969,085 India / 3,069,192 US rows), and p8c equals `p17/prob_test_{c}_bge.parquet` exactly.

**run.py port for P9** (the probe code needs no change, only a copy into src/):
1. Copy `probes/P34-a10-stack/a10.py` to `src/a10.py`, and `data/work/p34/a10_h0.txt` to where the recipe points.
   `a10.py` imports `stack.hn_cls` and `model.argmax_sx` from src.
2. In `model_parts`, right after `p = after_ce(...)`, for c in the recipe's a10 countries (India, US), when `stack` is
   true:
   ```python
   ids1, idsx = id1.to_numpy(zero_copy_only=False), idx.to_numpy(zero_copy_only=False)
   # holdout: rows of the scored (fold-0) S1; compute `ids` (defined further down in model_parts) before this step
   scored = None if split == "test" else np.isin(np.arange(len(id1)), codes(id1, ids))[q1]
   p, am, _ = a10.apply(booster, xs, c, q1, qx, p, pm, r, ids1, idsx, scored)
   ```
   Store `am` in the part dict, next to "p".
3. In `run()`: `acc = (d["am"] if "am" in d else M.argmax_sx(qx, d["p"])) & (d["p"] >= ts[c])`, and drop "am"
   together with "p".
4. Add a recipe `sub34` = #31's recipe + `a10={"India", "US"}` with booster "p34/a10_h0.txt". France is untouched.
5. Checks the port must reproduce:
   - Test: sub34's matching file (md5 above).
   - Holdout, with `CE_MISSING="fallback"` as in P23's dump, India+US **half-1 F0.5 = 0.989840** (#8c: 0.989564);
     India 0.990200, US 0.989601.
   - Half 0 is in-sample for this booster, as it is for P17's US stack, so quote half 1 only.
- Cost at full scale: 78–163 s per country on CPU. Peak RSS is 2.12 GB for India; that includes the probe's outer
  merge against the p17 file, which run.py does not need.

## Evidence         (table: ledger name | world | key metrics | verdict)
All rows are at t 0.71 against the #8c yardstick (P23's dump, s1ctx f05), with paired per-S1 SE on the report half.
LB-eq = 0.47 × dIndia + 0.38 × dUS, the countries' test S1 shares.

| ledger name | world | key metrics | verdict |
|---|---|---|---|
| a10-twin_p2mask-sibguard-fit_h0-report_h1 | holdout | **LB-eq +0.000209 ± 0.000056 (3.7 SE)**; pooled +0.000276 ± 0.000062; India +0.000157 ± 0.000102 (+289/−80 pairs); US +0.000356 ± 0.000077 (+307/−195) | **keep (gate)** |
| a10-twin_p2mask-sibguard-fit_h1-report_h0 | holdout | LB-eq +0.000175 ± 0.000048 (3.6 SE); India +0.000114; US +0.000320 (US half 0 = P17-in-sample: indicative) | confirms |
| a10-twin_p2mask-fit_h0-report_h1 (as briefed, no guard) | holdout | LB-eq +0.000212 ± 0.000058 (3.7 SE); India +0.000156 ± 0.000102 (+297/−80); US +0.000365 ± 0.000083 (+330/−222) | passes; staged as sub34_raw |
| a10-twin_p2mask-fit_h1-report_h0 (no guard) | holdout | LB-eq +0.000168 ± 0.000050 (3.3 SE); India +0.000127; US +0.000286 | confirms |
| a10-sibguard-test-reweighted-R51 | holdout→test mix | India +0.000166 ± 0.000092, US +0.000326 ± 0.000066 → **LB-eq +0.000202 ± 0.000050**. Expected precision of the test adds, from holdout (p2 × p-band) strata: India 0.861, US 0.854 (break-even is 0.67–0.75) | keep |
| a10-sibguard-test-profile | test | test/holdout-h1 ratios. US: adds/accepted 1.31, adds/unaccepted 0.80, drops/accepted 0.69, S1 changed 1.08, flips/pair 0.98. India: 1.19 / 0.63 / 0.97 / 1.14 / 1.01. All ≤ 1.5× | **PASS** |
| (same profile, no guard) | test | US adds/accepted **1.64×** (5,456 adds; 49% hn_diff, 1,391 with a sibling offset) | fails a strict reading |

Other checks (not ledgered):

| check | result |
|---|---|
| feature parity: `a10.features` from the raw sources vs P23's scratch files | 0 mismatches on n_twins, n_atwins, n_ctwins, n_cand_sx, p2, cat and the e5 mask, over 353,581 India + 524,962 US argmax pairs |
| #8c yardstick | per-S1 F from the dump reproduces s1ctx f05 exactly (asserted) |
| test base = #8c | model_parts("test", sub8c) p equals the p17 files, max \|Δp\| = 0, 0 unmatched rows; #31's India/US lines equal #8c's decision (0 diffs) |
| e5 mask on test | 0 rows masked: every test runner-up is bge-blended, as the brief expected |
| fit noise (h0 → h1, guard, seeds 1/2/3) | LB-eq +0.000216 / +0.000199 / +0.000206, all ≥ 3.4 SE |
| t sensitivity (guard, h0 → h1) | t 0.69 +0.000220; t 0.71 +0.000209; t 0.73 +0.000161. Kept 0.71 as briefed |
| absolute holdout F0.5 (guard) | h1: #8c 0.989564 → 0.989840 (India 0.990044 → 0.990200, US 0.989244 → 0.989601). h0 (swap): 0.990235 → 0.990473 |
| run.py port path, dry run: `run.model_parts("holdout", sub8c, CE_MISSING fallback)` → `a10.apply(..., scored)` → acc = am & p' ≥ .71 → metric.score on half 1 | India 0.990044 → **0.990200**, US 0.989244 → **0.989601**, the same as the probe numbers, so the step-2/3 snippet is verified |
| sub34 validator (`--check-ids`) | PASS: 1,732,544 rows, 101,908 empty. France lines and candidate_pairs.tsv byte-identical to #31. Empty rate India 5.760% → 5.776%, US 5.775% → 5.786% |
| sub34_raw validator (`--check-ids`) | PASS: 101,942 empty |

Top features by gain (h0 fit): r 0.14, lp 0.14, s1_max_other 0.12, lm2 0.11, marginm 0.09, cat 0.08, s1_n_am 0.08,
n_twins 0.07.

## What failed      (approaches tried and dropped, with the number that killed them)
- **The as-briefed stacker, on the label-free profile.** Its overall test flip rate passes the brief's 1.5× bar
  (US 1.15×, India 1.01×). But US adds per accepted pair are 1.64× holdout's, and within p bands the US add rate is
  1.3–2.3× holdout's.
  - SHAP on the 0.3 ≤ p < 0.9 pairs puts a mean +0.24-logit residual shift on test US.
  - Its drivers: lm2 is higher at the same final p (M2 liked the pair; the CE and P17 pulled it down), there are
    fewer twins, and hn_diff is 32% of that band on test against 14% on holdout.
  - Sibling-offset pairs (+1/+2 and SIB3) are 23.7% of the test US band but 4.4% of the holdout band, so there the
    stacker extrapolates from ~315 holdout pairs.
  - It is not the P17-in-sample fit: the h1-fit booster adds even more on test US (6,007).
  - The guard removes exactly that class for free, so I kept the as-briefed version only as `sub34_raw`.
- **A first test run OOM-thrashed** (2.2 GB RSS + 3 GB swap). Fixed by splitting `dump` into its own process
  (model_parts peaks at 1.5 GB) and chunking the text columns in `features()` (CHUNK 200k).

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `a10.py`: the one feature builder for holdout and test. Self-check: `py.sh a10.py`.
  - `features(q1, qx, p, p_m2, r, am, scored, s1t, sxt, t=0.71) -> DataFrame`. Input is integer codes into the
    s1t (name, addr, hn) and sxt (name, addr, is_web, hn) tables. Output is one row per argmax pair of a scored S1:
    COLS + hc + row.
  - `texts(split, c, s1_ids, sx_ids)`: reads {split}_source{1,2,3} and rf_{split}_{s1,sx}_{c}, streamed by row group.
  - `stacked(b, X, guard=True)`, `fit(X, y)`, and `apply(b, split, c, q1, qx, p, p_m2, r, id1, idx, scored=None) ->
    (p', am, a)`. `apply` is the run.py entry point.
- `ho.py build | eval`: holdout features with the parity check → `data/work/p34/ho_{c}.parquet`. It fits h0/h1 with
  and without the guard, reports and logs to the ledger, and saves `a10_h{0,1}.txt` and `ho_p[_raw]_{c}_h{fh}.npy`.
  ~1 min per country, < 1 GB.
- `te.py`:
  - `dump c`: #8c's test rows → `te_pairs_{c}.parquet`. 1.2–1.5 GB, 15 s.
  - `probs c`: → `prob_test_{c}_a10.parquet` + `te_feat_{c}.parquet`. 78–163 s, ≤ 2.1 GB.
  - `profile`: the label-free flip gate.
  - `raw`: the unguarded prob files.
  - `combine SRC OUT [a10raw]`: SRC's India/US lines replaced, after asserting that SRC's India/US lines equal #8c's
    decision and its candidates equal the prob files.
- `shift.py`: flip rates by p2 stratum, expected precision of the test adds, and R51 test-mix reweighting.
- `py.sh`: python capped at 2.2 GB (+3 GB swap).

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Licence / rules:** LightGBM (MIT) on label-free features, trained on the provided labels. No new dependency, no
  external data, no model download.
- **France / unseen countries:** untouched. The step runs only for India and US, so a new label keeps #31's path.
- **US half-0 contamination.** P17's US stack was fit on holdout half 0, so the booster learned US on P17-in-sample
  p there. The clean US report is h0 → h1 (+0.000356), which is exactly the shipped model. The swap's US figure is
  indicative only.
- **Covariate shift remains.** Even with the guard, US test adds per accepted pair are 1.31× holdout's, and the
  near-t residual shift is still positive. The stratum-weighted expected precision of those adds is 0.85–0.86. The
  estimate rests on holdout precision per (p2 × p band) stratum transferring to test.
- **Assumption:** I read "India+US ≥ +0.00015" as the LB-equivalent contribution (test S1 shares 0.47 / 0.38), which is
  stricter than the pooled holdout mean (+0.000276). Both pass at ≥ 3 SE.
- **Tie band:** the expected LB gain is +0.00015 to +0.00021, which public-vs-private noise can hide. Do not judge it
  by a dedicated LB slot.

## Open questions   (for the master; each with your best guess)
- **Guarded or as briefed?** Guarded (`sub34`). It is equal on holdout and better on the swap, and it is the only one
  within 1.5× on every profile ratio. The as-briefed `sub34_raw` differs from it on 1,563 S1 lines (14,396 vs
  12,848 changed vs #31).
- **Fit on both halves for shipping?** No. The h0 fit is the one the h1 report validates. A refit on India + US-h1 (the
  P23 Brief 7 variant) has no clean report.
- **Its own LB slot?** No. Put it inside the final combined file. If a slot is free, the delta vs #31 isolates India/US,
  but ±0.0002 is noise-level.
- **P9's port:** step 5's md5 and the h1 F0.5 0.989840 are the acceptance checks. Keep #8c's argmax (`am`) in the
  decision, or ~600 decisions per country drift.
