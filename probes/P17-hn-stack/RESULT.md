# P17-hn-stack — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **Brief A passes on the #7 yardstick.** P15's residual stacker (hn class, signed delta, eq, n_same, p_same, n_s1hn)
  sits on #7's blended p over #7's pruned list, plus lambda_l2 = 1 as a leaf guard. Report half **+0.00067 ± 0.00007
  (9.6 SE)**: India +0.00008 ± 0.00008, US +0.00107 ± 0.00011. Half-swap: +0.00082 ± 0.00008.
- The yardstick is exact: the base arm reproduces run.py's half-1 0.98869, and the #7 test rebuild matches
  output/sub7 pair for pair. #8 is staged in `data/work/p17/sub8/` and the **validator PASSES**: 19,715 S1 change
  (India 5,375, US 14,340, France 0), and the candidates are byte-identical to #7's.
- **Recommendation: upload #8** (India/US t 0.71 unchanged, France = #7's). Expected LB ≈ 0.85 × (0.55 × 0.00008 +
  0.45 × 0.00107) ≈ **+0.00045**. Brief C's branch words add nothing (drop).

## Recommendation   (exactly what the master should adopt: module, function signature, config)
**#8 (LB slot).** The already-built dry run is in `data/work/p17/sub8/` (matching_results.tsv + candidate_pairs.tsv,
official validator PASS). To rebuild it:
```
.venv/bin/python code/business_entity_resolution/src/submit_probs.py --probs 'data/work/p17/prob_test_{country}.parquet' \
    --out output/sub8 --t 0.71 --t-country France=0.95
```
- `data/work/p17/prob_test_France.parquet` is a symlink to `../p10/prob_test_France.parquet` (#7's France: 776,184
  matches at 0.95, the same as sub7).
- The India/US files are s1, sx, p = stacked p over #7's pruned rows, in the same row order.
- **code.zip** must also carry the stage: `probes/P17-hn-stack/stack.py` plus the P15 helpers it imports
  (`hn_check.py`, `hn_stack.py`), and the two boosters. Alternatively, promote it into src first (below).

**run.py integration (proposal; src/ is P9's).** Add a stage between `rerank()` and the decision in `run()` /
`model_parts`, for countries in `RR.CFG["countries"]` only:
```python
# src/hnstack.py  <- stack.frame's hn lookup + X_of + stacked (~60 lines) + P15 hn_check.hn_cls, hn_stack.support
def apply(c, q1, qx, p, h1, hx) -> np.ndarray:        # h1/hx = rf_<split>_{s1,sx}_<c>.hn at q1 / qx ("" = none)
    X = X_of({"q1": q1, "p": p, "h1": h1, "hx": hx}, "sup")          # support on THIS (pruned) list
    return stacked(lgb.Booster(model_file=WORK / f"p17/stack_sup-l21.0_{c}.txt"), X)
```
- Recipe key: `stack="p17/stack_sup-l21.0_{c}.txt"`. T stays 0.71, and France / unseen labels are untouched.
- Booster: LightGBM 200 trees × 31 leaves, lr 0.05, min_data_in_leaf 200, **lambda_l2 1**, init_score = logit p,
  categorical `cls`. Fit on the holdout S1 of `run.half == 0`, labels = truth pairs.
- Rebuild: `stack.py holdout sup t=0.71 l2=1`, 5 min CPU, deterministic (seed 0).

**Config choices, each measured (Evidence):**
- **t kept at 0.71, not retuned.** The US tune-half curve is flat from 0.61 to 0.79 (within 0.00006 of its best).
  Brief A's own half-0 retune picked US 0.77: +0.00057 ± 0.00008 on the report half, which also passes. But 0.77 is
  a noisy point on the plateau; #7's t sits inside the plateau and adds no constant.
- **lambda_l2 = 1**, one guard on top of the brief's params. Without it, residual boosting from saturated logits
  (tiny hessians at p ≈ 0/1) makes Newton-step blow-ups: max |leaf| 31 in the brief-exact US model, **48,320** in the
  India swap model. With it, max |leaf| ≤ 0.94 per tree, and the gain is equal or better (India +0.00003 → +0.00008).
- **Per-country boosters**, the brief's default. Pooled was not run (see open questions).

## Evidence         (table: ledger name | world | key metrics | verdict)
**Yardstick.** `stack.frame` rebuilds #7's p from run.py's own caches:
- M2 p on the τ 0.03 pruned list: `p9/m2_holdout_{c}_d2b6569f_6a23c2ec_0.03.parquet`;
- CE r: `p9/r_holdout_{c}_bb0c29f0.parquet`;
- sub7 w via `RR.blend`.

The decision is `M.argmax_sx` over every row (competing train S1 included), then p ≥ t. The score is per-S1 F0.5 over
**all** holdout S1 (blocking misses count), halves by `run.half`, with paired per-S1 SE vs #7 at t 0.71. **Base =
0.98869 on half 1, identical to #7's pipe row.** Support columns are computed on the pruned list the model sees.
France has no labels and is not touched.

| ledger name (`P17-hn-stack/…`) | world | key metrics | verdict |
|---|---|---|---|
| hn-stack-sup-on7 | holdout half 1, brief A exactly (t tuned on half 0) | 0.98869 → 0.98925, **+0.00057 ± 0.00008**; India +0.00003 ± 0.00008 (t .72), US +0.00092 ± 0.00012 (t .77). Control (#7 p, t retuned on half 0): −0.00001 | pass |
| hn-stack-sup-on7-swap | fit + tune on half 1, report half 0 | +0.00078 ± 0.00008; India +0.00009, US +0.00124 (t .66) | confirms |
| hn-stack-hn-on7 | half 1, hn-only (no support columns) | +0.00051 ± 0.00008; India +0.00013 ± 0.00006, US +0.00076 ± 0.00013 | fallback (no list dependence) |
| hn-stack-brc-on7 | half 1, + brief C add1 / add_in_list (P12's train-mined orphan-sibling list, count ≥ 100: US 27 tokens, India 12) | +0.00070 ± 0.00008, but the extra comes from the US t landing at 0.64. At equal t 0.70 it matches sup (US +0.00112 both; India +0.00005 vs 0.0). Split gain of the two columns 84–229, vs ~20k for p_same | drop |
| hn-stack-tcurve | both halves, t 0.55–0.90 | sup US tune half: +0.00155 at .70 vs +0.00161 best (.77). Report half: +0.00112 at .70 vs +0.00123 best (.63). India report half: 0.0 at .70, +0.00012 best (.65) | keep t 0.71 |
| hn-stack-sup-on7-t0.71 | half 1, t 0.71 | +0.00067 ± 0.00007; India +0.00003, US +0.00109 ± 0.00011 | keep |
| hn-stack-sup-on7-swap-t0.71 | half 0, t 0.71 | +0.00080 ± 0.00008; India +0.00010, US +0.00127 | confirms |
| **hn-stack-sup-on7-l21.0-t0.71** | half 1, t 0.71, lambda_l2 1 (**shipped model**) | **+0.00067 ± 0.00007**; India +0.00008 ± 0.00008; US +0.00107 ± 0.00011. S1 up / down: India 174 / 93, US 934 / 249. US d is +0.0006…+0.0012 at every t from .61 to .85 | **ship** |
| hn-stack-sup-on7-l21.0-swap-t0.71 | half 0, t 0.71, lambda_l2 1 | +0.00082 ± 0.00008; India +0.00013 ± 0.00009; US +0.00128 ± 0.00011 | confirms |
| hn-stack-l2-resid-shift | holdout-S1 rows vs test | Decision flips / #7's accepted: US 0.57% holdout vs 0.70% test; India 0.19% vs 0.20%. n_same 2.25 / 2.15, n_s1hn 2.64 / 2.69 (US). **+SIB share of candidate rows: US 9.8% → 17.4%, India 2.9% → 4.7%**. Holdout: newly accepted pairs 82–95% true, newly rejected 24–53% true | in range |
| (check7.py) | test, #7 rebuild at t 0.71 | India 2,733,366 / US 2,239,291 matches, identical to output/sub7/matching_results.tsv | yardstick valid |
| hn-stack-sub8-dryrun | test, submit_probs + official validator | India 2,733,366 → 2,734,936, US 2,239,291 → 2,245,127, France 776,184 (unchanged). S1 changed: India 5,375 (0.66%), US 14,340 (2.2%), France 0. candidate_pairs.tsv byte-identical to sub7. **Validator PASS** | stage as #8 |

## What failed      (approaches tried and dropped, with the number that killed them)
- **Brief C branch words** (add1, add_in_list on P12's train list): no gain at equal t, and the two columns carry
  about 1% of the split gain. The CE already reads the words, as P15 guessed. France's list (groupe / développement)
  was not built, because it needs test text (open question 4).
- **Brief A's half-0 t retune** (US 0.77): costs 0.00015–0.0003 vs t 0.71 on the report half. On a flat plateau the
  argmax is noise.
- **Unregularised residual boosting** (the brief's exact params): max |leaf| 31–48,320 from tiny hessians at p ≈ 0/1.
  The holdout gain is the same, but the risk sits in the test tail. Fixed by lambda_l2 = 1.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- **`stack.py holdout [sup|hn|brc] [swap] [t=0.71] [l2=1]`**
  - Fits the per-country stacker on half 0 (or half 1 with `swap`) and reports the other half vs #7 at 0.71.
  - Writes `data/work/p17/holdout_<tag>.json` (per country: t, d, se, up/down, `curve_h1`, gains), the booster
    `stack_<model tag>_<c>.txt`, and one ledger row.
  - 5–7 min under load 17–35, peak 2.0 GB (US).
- **`stack.py test sup l2=1 t=0.71`**
  - Writes `data/work/p17/prob_test_{India,US}.parquet` (s1, sx, p; #7's pruned rows and order: India 3,969,085 /
    US 3,069,192 rows).
  - 5 min per country under load, peak 1.7 GB.
- **Functions:**
  - `frame(split, c) -> dict`: #7's q1, qx, p, h1, hx, id1, idx.
  - `X_of(d, arm) -> DataFrame`.
  - `stacked(booster, X) -> p'`.
  - `labels(d, c, s1, t) -> (pos, y, ntrue, half)`.
  - `f05(am, s, t, pos, y, ntrue)`: per-S1 F0.5.
  - Reused unchanged from P15: `hn_check.hn_cls`, `hn_stack.support`, `hn_stack.f05_per_s1`, `branch_words.extra` / `cores`.
- **Helpers:**
  - `branch.py`: `cols(d)` (brief C columns), `vocab(c)`; self-check in `__main__`.
  - `tcurve.py <model tags…>`: d by t on both halves.
  - `resid.py <arm> <json tag> [holdout,test]`: residual quantiles, flip bins, feature shift.
  - `check7.py <sub dir>`: #7 rebuild vs a staged matching_results.
- **Scratch:** `data/work/p17/` is 368 MB (150 MB prob files + the 227 MB sub8 dry run). Delete `sub8/` once it is
  copied to output/.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Rules:** LightGBM (MIT) and pandas / numpy (BSD) only, with no new dependency and no external data. The dropped
  branch list came from P12's train-label statistics.
- **Holdout spend:** the boosters are fit on holdout half 0, like the CE blend w, so half 1 stays the clean report.
  Any later stage stacked on #8 has no fresh half left; the swap direction is the only second opinion.
- **Competitor rows are truncated.** Train S1 outside fold 0 keep only the SX a holdout S1 competes for, so their
  support columns are undercounted.
  - Training uses holdout-S1 rows only (complete lists), i.e. the test regime.
  - Only the argmax competition in evaluation sees the truncation.
- **Test shift:** test has about 1.8× more +SIB candidate rows (US 17.4% vs 9.8% on holdout S1).
  - This is exactly the pattern the stacker acts on, so the US LB effect may exceed the holdout's.
  - The other diagnostics are in range: flip rates within 1.2× of the holdout's, and support-column means within 5–10%.
- **France is not stacked.** P15 brief B (the France sibling rule, one LB slot) is still the way to learn France's regime.
- **Assumed:** #7 = the caches `d2b6569f` (m2_big), `6a23c2ec` (pruner) and `bb0c29f0` (ce_s2P) plus `run.half`.
  check7 and the byte-identical candidate file confirm this on test.
- **float32 ties:** stacked p is stored as float32. Ties at p → 1 break by row order in both `M.argmax_sx` and
  submit_probs, so holdout and test decide the same way.

## Open questions   (for the master; each with your best guess)
- **Per-country or pooled stacker?** Best guess: per country. India's model barely moves anything (+0.00008), and
  pooling would import the US sibling rejection into India, where P15's hard rule cost −0.0017.
- **Does it compose with the France levers (P13 CE, P14 IDF, P15 brief B)?** Yes. It touches India/US p only, and
  submit_probs takes France from whichever file the master points `prob_test_France.parquet` at.
- **Ship the hn-only arm instead?** Only if the candidate list changes before #8 (a new pruner or τ). The support
  columns must be recomputed on the shipped list, while hn-only has no list dependence (+0.00051, India +0.00013).
- **Retune the India/US t on the LB after #8?** No. The holdout plateau is flat, and an LB sweep spends slots for
  ≤ 0.0001.
