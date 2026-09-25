# P9-integrate — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **#5: `run.py --split test --stage model` reproduces submission #5 byte for byte** (both files, 1,732,544 / 1,732,544
  lines, `cmp`), validator PASS; France t 0.78 from P6's rule. `--recipe sub4` still reproduces #4 byte for byte.
- **Full-corpus holdout #5 = 0.9888** (half 1 0.9885 = P4's number; #4 0.9884 on the same pipeline). The blend refit on
  half 0 returns CFG's w. `train_matcher.py` now trains like big.py (folds 2-5, float16); only a 2% smoke retrain was run.
- **Recommendation:** merge the branch; the defaults are #5, and `run.RECIPES` / `--recipe` keep #4 reachable.

## #5 — reproducing submission #5 (m2_big)
**Change (smallest):** the defaults move to #5, and #4 stays one flag away.
- `model.BOOSTER = p4/m2_big.txt`, `model.FOLDS = (2, 3, 4, 5)`.
- `rerank.CFG["w"] = [0.678, 0.715, 0.338]`.
- `run.T = 0.69`, `run.T_M2 = 0.69`.
- `run.RECIPES = {"sub4": (m2_ladder.txt, (2,), [0.636, 0.757, 0.355], 0.70, 0.71), "sub5": (...)}`; `run.py --recipe
  sub4|sub5` (default sub5) switches all five values.
- `train_matcher.main(out, folds=(2,3,4,5), frac=1, rounds=8000)` = big.py's `train`:
  - rows as float16 in an `lgb.Sequence`;
  - the Dataset built with default params and constructed before `lgb.train(M.PARAMS, …)`;
  - early stop 100 on the same 25% fold-1 sample.
- `--frac` / `--rounds` exist for smoke tests. `model.fit` (unused) is deleted.

| check | result |
|---|---|
| `run.py --split test --stage model` vs `output/sub5` | **0 differing lines in both files; `cmp` byte-identical** (ledger P9-integrate/test-model-vs-sub5). Validator PASS. France t **0.78** (empty 0.0569 vs ref 0.0569 at T_M2 0.69); India / US 0.69. 91 min wall on a loaded machine (load 20–45): m2_big 5 / 18 / 43 min France / India / US. CE only for the 129k + 103k band pairs that #4's band lacked (81 + 127 s); the rest reused from `p9/r_test_*` |
| m2_big p recomputed from X vs P4's `prob_test_France_big` | same 4,348,797 pairs and order, max \|Δp\| 0.0 |
| band sizes | India 847,265 (128,671 new), US 702,137 (103,004 new) = big.py's reuse / new split exactly |
| `run.py --split holdout --stage model` (pipe p9-model, dcd4df4) | **F0.5 0.9888** (India 0.9899, US 0.9880), half 0 0.9890 / half 1 **0.9885** (P4 improve-big-ce-D3: 0.9885). P 0.9977, R 0.9705, singleton 0.9884. Blend refit on half 0 = [0.678, 0.715, 0.337] (CFG 0.338). 38 min wall, 2.09 GB |
| `run.py --split test --stage model --recipe sub4` vs `output/sub4` | **byte-identical, both files**; France 0.82, validator PASS; 80 s to both files from caches (no GPU) |
| `train_matcher.py --folds 2,3,4,5 --frac 0.02 --rounds 50` (smoke) | 252,154 train pairs (2% S1 sample of folds 2–5, 0.240 true), val 793,402 (= P4's), 50 trees, 54 s, 1.28 GB. **Full retrain not run**: its 3.7 GB peak (P4) was too much while other probes' jobs kept memguard pausing at < 1.5 GB free |
| `test_pipeline.py` (#5 defaults) | **pass**: rules 0.8575, model **0.9970**; unseen Xland gets P6's rule (t 0.89 = T_M2 + 0.2 cap), 95% of its S1 hit; Yland empty rows. It waited 49 min for the GPU lock (P11's CE training), about 1 min of its own work |

**Why #5 is byte-identical despite big.py's shortcut.**
- big.py recovered #4's CE scores by inverting #4's blend (max \|Δr\| 8.5e-4). My pipeline uses the exact scores
  cached by the #4 run and scores only the new band pairs fresh.
- No decision moved: no p′ within the inversion error of 0.69, and no argmax flipped.

**Risk (#5).**
- France churns 10.4% of S1 vs #4 (P4's note).
- t_France moved 0.82 → 0.78 with m2_big. That is the rule working, but the reference match is again tight
  (0.0569 vs 0.0569).

## Recommendation   (exactly what the master should adopt: module, function signature, config)
Merge branch `wave/p9-integrate` (commits c731de4 … HEAD). All files are under `code/business_entity_resolution/`:

| file | what |
|---|---|
| `src/features.py` | **new.** P4 features.py + featrun.py. `rec`, `featurize` (in memory, mirrors the files exactly), `build(split, c)` → `data/work/p4/X/` (ctx_step + feat_step) |
| `src/model.py` | **rewritten** (the stub is gone). M2 behind P8's plug point: `score(pairs, f1, fx)`, `split_probs(split, c)`, `predict`, `argmax_sx`, `BOOSTER = data/work/p4/m2_big.txt` (#5; #4 = m2_ladder.txt), `FOLDS` |
| `src/train_matcher.py` | **new.** `main(out, folds)` = big.py's M2 training (#4's ladder when `--folds 2`); `holdout([boosters])` = the M2-only dense F0.5 |
| `src/rerank.py` | **new.** P4b's rerank.py without the analysis code. `band`, `train` (+`n=` smoke), `score_file` (pipeline step [ta, tb] → [r]), `blend`, `fit_blend`, `texts`, `CFG` |
| `src/run.py` | model stage (below). Rules stage unchanged: dev 0.8526 / precision 0.9947, as P8 |
| `README.md`, `requirements.txt` | the end-to-end reproduction with measured runtimes; + lightgbm / torch / transformers pins |

Commands:
```
.venv/bin/python code/business_entity_resolution/src/run.py --split test --stage model --out output/   # the submission
.venv/bin/python code/business_entity_resolution/src/run.py --split holdout --stage model             # the decision number
```

`run.py --stage model`, per country (labels from the S1 file):
- `prepare` → P3 texts and rf caches.
- `model.split_probs` → M2 p from the X files. Missing X files are built by `features.build`. The result is cached
  at `data/work/p9/m2_<split>_<c>_<booster sha1>`.
- `run.rerank` → only for `rerank.CFG["countries"]` (India, US). The band 0.02 < p < 0.98 is scored by
  `flock /tmp/amc-gpu.lock python rerank.py score` in its own process; r is cached per CE hash; p′ = `rerank.blend`.
- `run.thresholds` → a country of the train file gets `T` (#5 0.69, #4 0.70) on p′.
  - Any other label gets P6's rule on **M2's** p: match the labelled countries' S1-weighted empty-S1 rate at
    `T_M2` (#5 0.69, #4 0.71), then clip to [T_M2, T_M2 + 0.2].
  - It is computed from the test probabilities on every run. France → **0.78** (#5), 0.82 (#4).
- argmax per S2/S3, p ≥ t → both TSVs → asserts → validator.

Changed semantics the master should know:
- **Model-stage candidates are now P3 ∪ rule-accepted** (P4's set, what M2 was trained on). P8's stub used P3 ∪
  every rule-key pair.
- **P5's `decide.py` hook is removed.** P5 was not adopted (blend-verdict: +0.0001 < 0.001).
- **`--split holdout --stage model` lets each S2/S3's argmax see every train S1**, P4's `all`. For that the CE also
  scores the competitors' band pairs: 979k pairs vs P4b's 195k-per-half files.

## Evidence         (table: ledger name | world | key metrics | verdict)
#5 rows are in the #5 section above; below is the #4 integration evidence (m2_ladder).

| ledger name | world | key metrics | verdict |
|---|---|---|---|
| P9-integrate/test-model-vs-sub4 | test | matching_results 1,732,544 / 1,732,544 identical lines, candidate_pairs 1,732,544 / 1,732,544; `cmp`: both byte-identical. Validator PASS. France t 0.82 (empty 0.0568 vs ref 0.0569), India/US 0.70; 35.5 min, 2.34 GB | **keep** |
| pipe p9-model | holdout-dense | **F0.5 0.9884** (India 0.9894, US 0.9876), half 0 0.9885 / half 1 0.9882 (P5's blend-a-t0.70 half 1: 0.9881). P 0.9975, R 0.9697, singleton 0.9883, cand recall 0.9917. Blend w refit on half 0 = [0.636, 0.757, 0.355] = CFG. Cold 22 min (M2 on 11.5M pairs + CE on 979k), cached 52 s | **keep** |
| P9-integrate/m2-retrain-from-train | holdout-dense-full | `train_matcher.py` full retrain (not a sample): 3,142,744 train / 793,402 val pairs (= P4's), 2,192 trees (P4: 2,276). Fold-argmax F0.5 **0.9813 (t 0.70) vs m2_ladder 0.9813 (t 0.71)**, India 0.9802 / 0.9802, US 0.9820 / 0.9821. The eval reproduces P4's `f05_fold` 0.9813. Fit 18 min, 2.69 GB | **keep** (within ±0.001) |
| P9-integrate/x-rebuild-equiv-test-France | test | `features.build` on France_sx_000: 1,526,605 rows × 67 cols, **0 cells differ** from P4's cached X; 49 s, 2.5 GB | keep |
| P9-integrate/x-rebuild-equiv-train-India | train | India_sx_000: 1,550,123 rows incl. is_true / fold, **0 cells differ**; 79 s, 2.7 GB | keep |
| (check, no row) | test | M2 p recomputed from X vs P4's `prob_test_France_m2`: same 4,348,797 pairs in the same order, max \|Δp\| 0.0. India / US band = 886,082 / 696,606 pairs, = P4b's sub4 counts | keep |
| P9-integrate/ce-train-entrypoint-smoke | train | `rerank.band("India")` from src: 655,128 rows (band 229,570) vs P4b 654,831 (229,262), not bit-equal (unseeded 6-thread LightGBM). `train(n=4000)` runs and writes a CE; scratch dir deleted | keep (entry points run) |
| pipe p9-model | dev | F0.5 0.9949 (India 0.9958, US 0.9943), P 0.9986, R 0.987; the whole in-memory path (blocking, featurize, M2, CE subprocess); 100 s, 0.81 GB | keep |
| (test_pipeline.py) | dev slice + Xland + Yland | rules 0.8575, model **0.9962**; Xland (unseen) gets P6's rule (t 0.91, the cap) and 94% of its S1 hit; Yland empty rows; 3,203 rows valid | pass |
| (package.py) | test output | `--selftest` ok; `--team p9check --out output/p9`: 24 files, 201 MB, structure ok (deleted after) | pass |
| (check, no row) | test | P6 rule on cached probs: ref = M2 p @ 0.71 → 0.82; M2 @ 0.70 → 0.82; blended p @ 0.70 → **0.83** | reference = M2 p |

## What failed      (approaches tried and dropped, with the number that killed them)
- **P6's reference on the CE-blended US/India p.** It gives t_France **0.83**, not sub4's 0.82: the blend shifts the
  reference empty rate to 5.735% vs 5.691%. France is scored by M2, so the reference must be M2's own behaviour at M2's
  own t* (P6's definition). That gives 0.82.
- **Bit-identical retraining.** M2 retrained has 2,192 trees vs 2,276, and the CE train band differs by 0.13%.
  LightGBM runs multi-threaded without `deterministic=true`. The holdout number is unchanged (0.9813 both), so the
  cached boosters stay the submission's; a retrain gives an equivalent model, not the same file.
- **First holdout run: 40 min stuck in `refit_w`.** `np.isin` on object (string) arrays falls back to a per-element
  loop (88k × 400k compares). Fixed with set membership; the rerun from caches took 52 s.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
In `code/business_entity_resolution/src/` (probe-dir helpers: `probes/P9-integrate/{xequiv,diff_sub4,ce_smoke}.py`).

**features.py**
- `featurize(pairs, r1, rx, i, j, rule_acc)` → 65 float32 columns.
- `build(split, c)`: test 3 / 20 / 9 min (France / India / US), 2.8 GB. The `ctx` files are deleted after use.

**model.py**
- `split_probs(split, c)` → [q1, qx, p] (holdout: the pairs of every SX that fold-0 S1 compete for, 11.5M of 31.4M).
- 31k pairs/s; test 2.3 / 7 / 6 min, 1.8 GB.

**train_matcher.py**
- `main(out, folds)`: fold 2 (#4) 18 min, 2.7 GB; folds 2–5 (#5) ~45 min, 3.7 GB (P4's big.py measurement).
- `holdout(boosters)`: 7.5 min for two boosters.
- `prepare()` builds every train input (P3 train cands, rf, X) when absent.

**rerank.py**
- `score_file(in, out)`: 1,363 pairs/s (India) and 1,700 pairs/s (US), 2.0 GB RSS, 3.4 GB VRAM.
- `band(c)`: 1 min. `train()`: 25 min (P4b).

**run.py**
- `model_parts(split, world)` → per-country dicts; `rerank(c, q1, qx, p, text, cache)` → (p′, r);
  `thresholds(parts)` → {country: t}; `refit_w(parts)`; `ensure_ce()` trains the CE when `ce.pt` is absent.

**Caches written:** `data/work/p9/` (486 MB): M2 p for test and holdout, r for test and holdout, `m2_retrain.txt`.
With them a rerun skips all scoring: measured 52 s on holdout; the test rerun was not re-timed (decision + writing were 2.5 min).

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Licences.**
  - New pins: lightgbm (MIT), torch (BSD), transformers / tokenizers / safetensors / huggingface_hub (Apache-2.0).
  - Model: e5-small, MIT, 118M.
  - `rerank.py` sets `HF_HUB_OFFLINE=1`, so no Hub request at inference. P4b's run pinged the Hub; the base weights
    must be downloaded once (README step 1).
- **France t sits on a close, flat boundary.** France's empty rate is 5.682% at 0.82 vs the 5.691% reference, and
  5.737% at 0.83. Any change to M2 (e.g. P4's stage 2) or to France's candidates can flip it to 0.83. That is by
  design (the rule reruns on the test p), and P6 found both values in the flat region.
- **Cache keys.**
  - M2 p is keyed by the booster hash, r by the CE hash.
  - X files are reused if present and are not keyed. Rebuild `data/work/p4/X` (and delete `p9/m2_*`) if
    features.py changes (`ponytail:` comment in model.py).
  - The X cache is P4's directory. P4 has exited; if a later probe rewrites it, the pipeline follows silently.
- **Train candidates.**
  - `features.cands("train")` reuses P4's `cands_train_full_*`, else `block.split_cands("train")`.
  - P4 says both are the same P3 rule over all train S1, but the recompute path was not re-run at full scale
    (35 min). Only its downstream X files were checked: 0 cells differ.
- **GPU.** Test needs ~18 min of GPU under `/tmp/amc-gpu.lock`. On a CPU-only box `rerank.py score` fails (`m.cuda()`),
  and `run.py` stops rather than silently skipping the CE.
- **Cold reproduction** from raw data: timings are summed from P3 / P8 / P4 / P4b logs plus mine (README table),
  not measured as one cold run: ~4.5 h, including the train-side blocking / features and both trainings.

## Open questions   (for the master; each with your best guess)
- **Ship `ce.pt` (43 MB) and `m2_big.txt` (69 MB) in the code zip?** Best guess: no. They are artefacts; the README
  rebuilds both. A retrain is equivalent (0.9813 = 0.9813) but not bit-identical, so keep the cached files as the
  submission's source of truth.
- **Delete `src/submit_probs.py`?** Best guess: yes after merge, since `run.py --stage model` supersedes it. It is left
  in place because staging scripts may still call it.
- **Adopt P4's stage-2 model (0.9822 vs 0.9814, on m2_ladder) next?** Best guess: only if it still gains on top of m2_big + CE. It needs `split_probs` to add
  the p1 features, and it needs the CE blend weights, T and France t refitted (`--split holdout` prints the w refit).
  Then check that the holdout gain survives the CE: both fix look-alike errors.
- **Should the unseen-country threshold be recomputed on the final (blended) p once France gets a CE?** Best guess:
  yes. The reference must be the model France is scored with; today that is M2.
