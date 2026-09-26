# P9-integrate — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **#31 = the new default recipe `sub31`:** `run.py --split test --stage model` reproduces `output/sub31`, our best LB
  **0.9856**, byte for byte (both files). `--recipe sub31_veto`, `--recipe sub31_small` and `--t-country France=0.9818`
  reproduce `sub31_veto`, `sub31_small` and `sub31_t9818`, also byte for byte (both files).
- The France chain lives in **`src/france.py`** (P20 R1 rescore, P26 c_hn + `hn_feats`, P28 bxneg, P29 `paren_hn` /
  tier / cap, P30 veto), driven by `run.france31`. c_hn's p equals P31's `chn_fix.parquet` exactly (max |Δp| 0), and
  every post-rule count matches P31's log.
- **A cold run** needs the model files (bge ckpt, m2_big, pruner, US stacker, **c_hn**) and the bge score files
  (P13 / P19 / P26 / P29 / P31, + P30 for the veto); it recomputes everything else from the raw TSVs (#31 section:
  what it needs, what it rebuilds, and what a from-raw France rebuild changes).

## #31 — the final France chain (P26 + P28 + P29 + P30 + P31)
**Changes:**
- **`src/france.py` (new):**
  - `world(base)` → `data/work/p9/fr31`: links to #9ab's `fr_idf144` world, plus `rf_test_sx_France.parquet` with
    P29's `paren_hn` re-parse (23,272 S2/S3, 23,250 hn changed; equal to `p29/w_fix`'s file).
  - `chn(model, out)`: its own process under `ER_WORK` = that world.
    - P20's rescore with R1 (`NOISE`, `strip`): blocking and context columns are kept, while strings / `rule_acc` /
      rule-only extra pairs are recomputed.
    - Then the pruner, then P26's `hn_feats` over the full list (+ `meta`, `_hn` = P26 prep), then c_hn.
  - `post(q1, qx, id1, idx, p, g, t, v, learn, veto)`: P31's order and dtypes (float64 inside, float32 out).
    - `tier_mask` / `keys` (P29), `bxneg` + `sib_words` + `street_key` + `name_rel` (P28 / P20, only the `bxneg`
      variant), the veto (P30), `cap_mask` (P29).
- **`run.py`:**
  - Recipes `sub31` (default), `sub31_veto` (`veto=(0.1, "p30/r/r_pool_France.parquet")`), `sub31_small`
    (`drop_sib_cands=True`).
  - `CHN = data/work/p31/c_hn.txt`.
  - `france31()` is the France part of `model_parts` when `FRANCE == "fr31"`. Its t = `T_COUNTRY` or
    `T_FRANCE_OVERRIDE`, because the rules decide at t.
  - `r_cache` records the seed files it merged in `<cache>.seeds`. It used to merge "seeds newer than the cache",
    which would have skipped P26's file: 16:23, older than the 16:43 France cache.
  - `seed_keys()` is shared by the r cache and the veto lookup.
  - New flags: `--no-validate`, `--ce-missing keep` (a band pair without a cached r keeps its p; no GPU needed),
    `--france fr31`.
- **`train_matcher.py --hn`**: m2_big's recipe + `france.HN`. It is c_hn's retrain path when `c_hn.txt` is absent
  (`run.france31` calls it). P26 trained in float32 on AWS; this path uses float16, like m2_big: an equivalent model,
  not the file.
- **`stack.hn_of(..., work)`**: the France part's decoy stage reads the fixed SX house numbers.

| check | result |
|---|---|
| `run.py --split test --stage model` (sub31) vs `output/sub31` | **byte-identical, both files**; validator PASS. c_hn p on 1,332,223 kept pairs = P31's `chn_fix.parquet` (same order, max \|Δp\| 0). Tier 253,136 pairs (+6,088 accepted), bxneg 523,707 (+9,647), cap 21 → 808,722 France matches. r seeded: P19 65,393, P26 36,339, P29 11,248, P31 3,974 France pairs. France chain cold: rescore + c_hn 432 s (2.60 GB, own process; memguard paused it ~3 min), post ~40 s |
| `--t-country France=0.9818` vs `output/sub31_t9818` | **byte-identical, both files** (tier +3,343, bxneg +5,678, cap 27 = P31) |
| `--recipe sub31_veto` vs `output/sub31_veto` | **byte-identical, both files** (7,753 vetoed pairs outside the rules = 10,123 − 2,370; −6,141 matches; cap 18 = P31) |
| `--recipe sub31_small` vs `output/sub31_small` | **byte-identical, both files**. Decoys dropped: France 170,057 (5.13 → 4.48), India 125,346 (4.90 → 4.75), US 416,972 (4.63 → 4.00); matching = `sub31` |
| `test_pipeline.py` (sub31, e5 CE; and `--recipe sub31_small --unseen France`) | **pass**: model 0.9968 / 0.9964, rules 0.8575. The in-memory path exercises France's blend weights, fixed t and the decoy stage; the post-rules need the full-scale France files and run only at test scale (the rows above) |
| `train_matcher.py --hn --frac 0.02 --rounds 50` (c_hn's retrain path, smoke) | runs: 70 features, **same names and order as `c_hn.txt`**; hn gain share 0.15 (P26's c_hn: 0.15); 40 s, 2.34 GB |
| `france.py` self-check | ok (R1, `paren_hn`, `street_key`, `name_rel`, `hn_feats` classes / deltas / support, `cap_mask`) |

**Cold run (task 3): what `run.py --split test --stage model` (sub31) needs, and what it recomputes.**

*Must be supplied*: these cannot be rebuilt locally, or a rebuild is not the same file, so its lines would move.

| file (`data/work/…`) | size | why | offload (`gdrive:amc26-offload/`) |
|---|---|---|---|
| `p13/ckpt/ce_bge_pool.pt` | 624 MB | the bge CE (AWS). Its sha1 also names the r caches, so it is needed even with every score cached | `models/p13/ckpt/` |
| `p13/r/r_bge_pool_test_{India,US,France}.parquet` (+ `r_bge_poolck_holdout` for the holdout) | 39 MB | P13's bge scores. Without them every band pair is scored fresh: HF weights + GPU, and bf16 noise moves lines | `models/p13/r/` |
| `p19/r/`, `p26/bge/r/`, `p29/r/`, `p31/r/` `r_bge_pool_test_France.parquet` | 2.9 MB | France band top-ups (idf144, c_hn, the rebuild) | `work/p19/r` … `work/p31/r` (uploaded 26 Sep 18:43) |
| `p30/r/r_pool_France.parquet` | 15 MB | `sub31_veto` only | `work/p30/r/` |
| `p31/c_hn.txt` | 50 MB | P26's AWS matcher; `train_matcher.py --hn` makes an equivalent, not this file | `models/p31/` (uploaded) |
| `p4/m2_big.txt` | 69 MB | m2_big; a retrain is equivalent, not bit-identical | `models/p4/` |
| `p10/pruner.txt`, `p17/stack_sup-bge-l21.0_US.txt` | 1.8 MB | both retrain bit-identically, but only after the whole train side (~1 h 45) + a holdout run | `models/p10/`, `models/p17/` |

*Recomputed from the raw TSVs* when absent: the same code built every cache, except P21's région map (France only,
below).

| step | cache | cold time (laptop) |
|---|---|---|
| TSV → parquet, folds, P2 table, place counts, vectorizers (train fit) | `*_source*.parquet`, `gt_pairs`, `s1_folds`, `p2_xlit_table`, `place_freq_train.json`, `p3/vecs_hash.pkl` | ~15 min (vectorizers need the train S2/S3 sample) |
| France région map (learned from test France) | `fr_regions.json` | ~1 min |
| test texts + P3 blocking | `p3/txt_test/`, `p3/cands_test_*` | 6.5 + 5 / 18 / 6 min |
| test rules.feats | `rf_test_*` | ~20 min |
| test X, pruner + m2_big p | `p4/X/test_*`, `p9/m2_test_*` | 3 / 20 / 9 + 3 / 18 / 5 min |
| France idf144 world (IDF refit, reblocking, X, m2) | `p9/fr_idf144/` | ~47 min |
| France fr31 world ('(n)' fix, R1 rescore, c_hn) | `p9/fr31/` | ~8 min |
| r caches (seeded from the files above; `.seeds` lists them) | `p9/r_test_*_3bd0b930` | seconds |
| decision, TSVs, validator | `output/` | ~5 min |

Total ≈ 3 h on the laptop with the model and score files present. The train side (~1 h 45 + ~45 min of m2_big
training) runs only if a model file is missing. **Fastest byte-exact path:** copy `data/work` whole. That is path A in
SETUP.md, plus `p9/fr_idf144/` (X 151 MB, cands 68 MB) and `p9/fr31/` (36 + 16 MB), which are not on the drive yet. The run
then takes ~5 min.


## #9ab — France IDF fix + bge, the cap, a smaller candidate set   (history)
**Changes:**
- **`run.py`:**
  - Recipe `sub9ab` (default): `_BGE` + `france="idf144_bge"`, t_france 0.985. France's candidates, X and M2 come from
    the `idf144` world (`france_variant`), then the bge blend with `w_france` on its band.
  - `r_seed` takes a list: P13's files, then `p19/r/`. Any seed file newer than the cache is merged into it; a pair
    already present keeps its r.
  - `--cap S2,S3` (`cap()`), `--drop-sibling-cands` (`drop_sib_cands()`), `--tau-country C=τ` (→ `model.TAU_COUNTRY`,
    passed to the France world's subprocess).
  - Each part carries an `hn` accessor for the decision-time stage.
- **`model.py`:** `tau(c)`, `kept(X, c)`; the cache key uses the country's τ.
- **`test_pipeline.py`:** `--cap`, `--drop-sibling-cands`, `--france idf144_bge`.

| check | result |
|---|---|
| `run.py --split test --stage model` (sub9ab) vs `output/sub9ab` | **byte-identical, both files**; validator PASS. France: 1,305,341 kept pairs, 783,035 matches at t 0.985; 65,393 r merged from `p19/r` (P19's AWS scores); no GPU |
| `--t-country France=0.975` / `0.99` vs `output/sub9ab_t975` / `sub9ab_t990` | **byte-identical, both files**, validator PASS (copies deleted) |
| **cap** `--cap 5,6` → `output/sub9ab_cap` | train truth max per S1 = 5 S2 (24,154 S1) / 6 S3 (2,816 S1). Test: **86 S1** lose **90** matches (France 17 S1 / 18, India 26 / 27, US 43 / 45); candidates identical; validator PASS. Holdout: 4 scored S1 change, all from F0.5 sum 3.58 to 4.00, **+0.0000019**; F0.5 0.98990 unchanged at 5 dp |
| **sibling-decoy candidates** `--drop-sibling-cands` | holdout candidates per S1: **US 4.24 → 3.92**, India 4.38 → 4.31; candidate recall 0.9874 → 0.9872; F0.5 unchanged. Test (on top of τ below): US 4.63 → 4.00, India 4.90 → 4.75, France 4.37 → 3.97. Matching identical to the run without it (`cmp`) |
| **France τ 0.1** `--tau-country France=0.1` | France 5.03 → **4.37** per S1 (1,305,341 → 1,134,863 pairs); matches 783,035 → 782,804. **225 France S1 (0.087%) change their answer** (gate ≤ 0.1%: passes); India / US untouched |
| `output/sub9ab_small` (τ France 0.1 + decoy drop) | **7,527,148 pairs, 4.34 per S1** (France 3.97 / India 4.75 / US 4.00) vs #9ab 8,343,618 = 4.82 (5.03 / 4.90 / 4.63). Matching = #9ab except those 225 France S1. Validator PASS |
| `test_pipeline.py` (sub9ab, e5 CE, `--cap 5,6 --drop-sibling-cands`; and `--unseen France --france idf144_bge`) | **pass**: 0.9968 / 0.9965 |
| P21 learned région map: a France `idf144` world rebuilt from raw under the current normalize.py (`probes/P9-integrate/p21check.py`) | **#9ab's France unchanged**. The rebuild covers texts, rules.feats, the learned département map, IDF, blocking, X, pruner and M2. Kept pairs identical (1,305,341; 0 only-old / 0 only-fresh); 22 M2 p move (max 0.053, = P21's report); the band is identical (378,445, all with cached bge r). **0 of 259,452 France matching lines and 0 candidate lines differ** from `output/sub9ab`; 11 min |

**How to read these:**
- **Decoy drop.** It is a cascade stage before the final decision: a sibling-offset pair survives only if it is its
  SX's best pair above t. That keeps `candidate_pairs.tsv` = the input of the last decision and makes matching
  unchanged by construction. The organisers' ruling allows a pruned later stage.
- **France τ 0.1.** Its 225 changes cannot be signed without France labels. On India / US holdout P10 measured
  τ 0.1 at −0.00076 (e5 CE). The worst case on France is 225 / 1.73M = 0.00013 of the test score.
- **Cap.** It only removes pairs beyond the train maximum. Its holdout effect is positive but negligible.

## #8c / #9 — bge CE, US stacker, France options   (history)
**Change:**
- **`rerank.py`:**
  - `load_bge(ckpt)` / `score_bge`: P13's `Job.load('bge')` + `restore` + `score` — bge-reranker-v2-m3 fp32,
    checkpoint loaded over it, max_len 96, token-length-sorted batches, bf16.
  - `score_file(in, out, ce, base)`: the subprocess gets the CE family.
- **`stack.py`** (new): P15's `hn_cls` / `support`, P17's `X_of` / `train` / `stacked` (200 × 31, lambda_l2 1,
  init_score = logit p), P15's `sib_mask`, `hn_of` (house numbers from the rules.feats caches).
- **`block.py`:** P14's `idf_country` (France's own S2/S3 text in the IDF at share 0.144) and `idf_domain`
  (the train-only domain-word fallback).
- **`run.py`:**
  - Recipes `sub8c`, `sub9a`, `sub9b`, `sub9s`, `sub9d` (default `sub8c`); `--france {m2,idf144,idfdom,bge,sib}` +
    `--t-country France=<t>`; `--ce-missing {score,fallback}`.
  - `r_cache()` seeds a CE's score cache from P13's `p13/r/r_bge_pool_*` files (ids → rows).
  - `rerank()`: per-country weights (France: `W_FRANCE` when France = bge), with the fallback option.
  - `after_ce()`: the US stacker and France's sibling rule; `ensure_stack()` fits the stacker when absent.
  - `france_variant()`: a France-only world (`data/work/p9/fr_<name>`: links + the variant vectorizers) where a
    subprocess reblocks France, rebuilds X, prunes and runs M2 — P14's own method, ported.
- **`model.probs_path`:** split out of `split_probs`. **`test_pipeline.py`:** `--recipe`, `--ce`, `--unseen`, `--france`.

| check | result |
|---|---|
| `--recipe sub8c` vs `output/sub8c` | **byte-identical, both files**; US 2,245,739 matches (= P17). 4 min from caches + validator PASS |
| `--recipe sub9a` (France idf144, t 0.975) vs `output/sub9a` | **byte-identical, both files**. France reblocked in its variant world: 4,041,182 P3 pairs, 1,305,341 kept; 47 min (33 of it blocking at 15 threads); validator PASS |
| `--recipe sub9b` (bge France, w [0.9245, 0.2595, 0.0515], t 0.98) vs `output/sub9b` | **byte-identical, both files**; France r seeded from `r_bge_pool_test_France` (490,033 pairs); validator PASS |
| `--recipe sub9s` (sibling rule, t 0.95) vs `output/sub9s` | **byte-identical, both files**; 170,752 France pairs zeroed; validator PASS |
| `--recipe sub9d` (France idfdom, t 0.98) vs `output/sub9d` | **byte-identical, both files**; 1,311,346 kept; 46 min; validator PASS |
| `--split holdout --recipe sub8c --ce-missing fallback` (pipe p9-model) | **F0.5 0.98990** (India 0.9905, US 0.9895); half 0 0.99024, **half 1 0.98956**: India 0.99004 / US 0.98924, both = P17's `holdout_sup-bge-l21.0-t0.71.json`. P 0.9983, R 0.9726; 5 min |
| `--recipe sub7` vs `output/sub7` (regression: the generalised `rerank()` on an e5 recipe) | **byte-identical, both files** |
| `ensure_stack('US')` refit into a scratch path | **bit-identical to `p17/stack_sup-bge-l21.0_US.txt`** (`cmp`), using `w_stack_fit` = the rounded w P17 fit on |
| `test_pipeline.py --recipe sub8c --ce p11/ce_s2P.pt --ce-base e5` | **pass**: model 0.9968, US stacked, Xland P6 rule. `--unseen France --france sib`: 0.9965 (France t 0.95, sibling rule applied). `--france bge`: 0.9968 |

**Two numerics that had to match exactly:**
- **#8c's blend w are P13's fit at full precision:** [0.5901490903244407, 0.6319357246516834, 0.29732167912424273].
  - The documented [0.5901, 0.6319, 0.2973] moves every band pair by up to 2.4e-5 and flips 4 S1 lines (1 India,
    3 US).
  - The staged `p13/prob_test_*_blend` files used the unrounded fit of `eval_bge_poolck_holdout_vs_p11s2P_plain.json`.
  - With it, my blend equals P13's p on all 7.04M India + US rows.
- **The stacker was fit on holdout p blended with the rounded w** (P17's `WB`). `ensure_stack` therefore fits with
  `w_stack_fit`, and it is bit-identical only that way. With the full-precision w the refit differs slightly.

**What was not run:**
- **Fresh bge scoring.** The laptop has the bge tokenizer (checked offline) but not the 2.3 GB base weights, and its
  GPU is still stuck at 210 MHz.
  - Every bge score in these checks is P13's cached one: test India / US / France, and the holdout S1's own band pairs.
  - `load_bge` / `score_bge` follow P13's `Job` code line for line but are untested here.
  - The holdout's 570k competitor band pairs use `--ce-missing fallback`, P17's yardstick. A true all-bge holdout
    needs `--ce-missing score` on a GPU with the weights (~1 h on an A10G).
- **Retraining the bge CE.** It is AWS-only (P13); `run.ensure_ce` stops with instructions if the checkpoint is
  missing.

**Where:** all outputs were cmp'd and then deleted. They are identical to the staged
`/home/harsh1/github-repos/amc-26/output/{sub8c,sub9a,sub9b,sub9s,sub9d}`.
- Caches: `data/work/p9/r_*_<bge hash>` (seeded), `data/work/p9/fr_idf144/`, `fr_idfdom/` (the France variant
  worlds, ~0.5 GB, reusable).

## #7 — P11's 2-epoch cross-encoder, France t 0.95 by default   (history)
**Change (smallest):**
- `rerank.CE = data/work/p11/ce_s2P.pt`, `rerank.EPOCHS = 2`, `rerank.CFG["w"] = [0.617, 0.666, 0.261]`.
  `rerank.train(epochs, out=…)` writes wherever the recipe points.
- `run.T = 0.71`, `run.T_FRANCE_OVERRIDE = 0.95` (#7's default; `None` = P6's rule).
- `run.RECIPES` gains `ce`, `ce_epochs` and `t_france`: `sub7` (default), and `sub6` / `sub5` / `sub4` with
  `p4b/ce.pt`, 1 epoch and France by P6's rule.
- `--t-country France=<t>` still overrides the recipe's France t.
- **The CE path is passed to the scoring subprocess** (`rerank.py score in out <ce>`). Without it the subprocess
  would load the module default, so `--recipe sub6` would silently score with the new CE.
- The r cache was already keyed by the checkpoint's sha1 (`p9/r_<split>_<c>_<ce hash>`), so #6's scores stay in their
  own file.
- `ensure_ce()` trains the recipe's CE (`train epochs=<n> out=<ce>`) under the GPU lock when it is absent.

| check | result |
|---|---|
| `run.py --split holdout --stage model` (pipe p9-model) | **F0.5 0.98881**: India 0.9900, US 0.9880; half 0 0.98894, **half 1 0.98869**. #6 on the same pipeline: 0.98852 / 0.98831 → **+0.00029 overall, +0.00038 on the report half** (P11: +0.00039 ± 0.00006). P 0.9980, R 0.9700, 4.30 cands/S1. Blend refit on the pruned half 0 = [0.624, 0.670, 0.267] (CFG [0.617, 0.666, 0.261] is P11's unpruned fit). CE on 725k holdout band pairs (278k India + 447k US) in 5.5 min |
| `run.py --split test --stage model` → `output/sub7` | validator PASS. France t 0.95 (P6's rule would give 0.78, still printed), India / US 0.71. 656k + 583k band pairs scored by the new CE (5 + 5 min GPU); 11 min end to end, 1.5 GB |
| vs `output/sub6c` (ledger test-model-vs-sub6c) | `matching_results`: 1,719,753 / 1,732,544 lines identical; 12,791 differ (India 7,402 = 0.91% of S1, US 5,389 = 0.81%, France 0). Pairs: India +3,528 / −4,049, US +2,637 / −2,836. `candidate_pairs`: identical (same pruner) |
| `--recipe sub6 --t-country France=0.95` vs `output/sub6c` | **byte-identical, both files** (CE scores from `r_test_*_127f69fb`, the p4b CE's cache) |
| `test_pipeline.py` (#7 defaults) | **pass**: rules 0.8575, model 0.9964; Xland (unseen) keeps P6's rule (t 0.89) since the override is France-only |
| `rerank.py train epochs=2 out=p9/ce_retrain.pt` (from src, P4b's in-loop recipe) | **partial.** (1) Run 1 (23:56): the same **778,520** train pairs as P11 and 40,000 val. It ran at 750 pairs/s (P11: 719); epoch 0 val band AUC CE **0.9286** (P11 epoch 0: 0.9275; P4b's 1-epoch ce.pt: 0.9263). It died in epoch 1 at 00:31 (battery 0%); the checkpoint is written only at the end. (2) Run 2 (01:47): the GPU was stuck at 210 MHz, P8, throttle `0x24` = SW power cap + SW thermal slowdown, with the battery charging from 0% (42→58%). No step 500 in 21 min, i.e. ≥ 15× slow; killed to free the lock. **To finish when the GPU clocks are back:** `flock /tmp/amc-gpu.lock .venv/bin/python code/business_entity_resolution/src/rerank.py train epochs=2 out=data/work/p9/ce_retrain.pt`, then `.venv/bin/python probes/P9-integrate/ce_retrain_eval.py data/work/p9/ce_retrain.pt`. That is ~35 + 6 min; it logs `ce-retrain-2ep-holdout` against 0.98869 on half 1. Not bit-identical by design: bf16 GPU training, and P11 pre-tokenised and bucketed by token length where src buckets by character length |

**Where #7 is:** `/home/harsh1/github-repos/amc-26-wt/p9-integrate/output/sub7/{matching_results,candidate_pairs}.tsv`
(this worktree; no code.zip built).

**Environment (for the master):** after the 01:41 reboot the RTX 3050 stays at **210 MHz / P8** (`nvidia-smi` throttle 0x24: SW power cap + SW thermal) even at 100% utilisation.
- Every GPU job (CE scoring in `run.py`, training) is ~15× slower until that clears.
- Likely the laptop's battery-protection power limit while it recharges from 0%. I did not touch system power settings.
- Check `nvidia-smi --query-gpu=clocks.sm,pstate --format=csv` before a run that needs the CE fresh.
- `run.py` for #7 no longer needs the GPU: every band pair's CE score is cached (`p9/r_*_bb0c29f0`).

**Docs:**
- README: the recipes table gains CE and France columns, plus #7 commands and runtimes.
- SETUP.md:
  - status → #7;
  - best LB → 0.981142 (D4);
  - `models/p11/ce_s2P.pt` added to the artefact table with its sha256 `4dc0d882…`. The Drive upload is pending: the
    master's call.

## #6 — the pruner in run.py, France's threshold as a setting   (history)
**Change:**
- **`src/prune.py`** = P10's `prune.py` with `src` paths.
  - `MODEL = data/work/p10/pruner.txt`, `FEATS` (30 columns), `PARAMS` (`deterministic=True`), `TAU = 0.03`.
  - Functions: `train(folds=(2,3))`, `score(m, X)`, `prune(X, q, tau)` = q ≥ τ | rule_acc.
  - `load_model()` trains the pruner when the file is absent (after `train_matcher.prepare()` builds the train X).
- **`model.split_probs`** prunes each X file before m2_big predicts.
  - The X files keep the FULL blocking list, so the context columns are unpruned and a kept pair's p is its #5 p.
  - The cache key adds the pruner hash + τ.
- **`model.score`** (in-memory worlds) returns (keep, p): the features come from the full list, then the pruner.
- **CE:** only kept band pairs. The argmax runs per SX over the kept pairs, and `candidate_pairs.tsv` is the kept pairs.
- **`model.TAU`:** `None` switches the pruner off.
- **`run.RECIPES`** are now dicts with `tau`: `sub6` (default, τ 0.03), `sub5` / `sub4` (τ None).
- **France's threshold:**
  - `run.T_FRANCE_OVERRIDE = None` (None = P6's rule).
  - `run.T_COUNTRY` / `--t-country C=t` set any country's t; the P6 value is still printed next to the override.
  - Any other unseen label keeps P6's rule; test_pipeline's Xland still gets it.

| check | result |
|---|---|
| `run.py --split test --stage model` vs `output/sub6` | **byte-identical, both files** (P9-integrate/test-model-vs-sub6). 8,370,573 kept pairs (France 1,332,296 / India 3,969,085 / US 3,069,192), = P10's. France t 0.78 by the rule. All CE scores came from the r cache. 28 min incl. validator, 1.8 GB |
| `--t-country France=0.88` vs `output/sub6b` | **byte-identical, both files** (test-model-vs-sub6b); France 811,100 matches; validator PASS |
| `--t-country France=0.95` vs `output/sub6c` | **byte-identical, both files** (test-model-vs-sub6c); validator PASS |
| `--recipe sub5` vs `output/sub5` | byte-identical, both files (cached, no pruner) |
| `run.py --split holdout --stage model` (pipe p9-model, ad774a7) | **F0.5 0.9885** (0.98852): India 0.9896, US 0.9878, half 0 0.98872, **half 1 0.98831** (P10 fixed-w D3: 0.98825; the plain t 0.69 vs D3's t0 gate explains the +0.00006, as for #5). P 0.9977, R 0.9698, cand recall 0.9874, 4.30 cands/S1. Blend refit [0.674, 0.718, 0.343] = P10's refit. 7 min, 2.2 GB. #5 on the same pipeline: 0.9888 → pruner cost −0.0003 overall |
| `prune.py train` (from src) | 6,296,233 pairs (0.241 true), 300 trees, 74 s; **bit-identical to `data/work/p10/pruner.txt`** (`cmp`) |
| `test_pipeline.py` (#6 defaults) | **pass**: rules 0.8575, model **0.9964**; Xland (unseen) P6 rule t 0.89, 94% hit; Yland empty |

**Not changed:** the holdout number does not depend on the France override (France is not in the holdout). The dev
world and `test_pipeline` have no France label.

## #5 — reproducing submission #5 (m2_big)   (history)
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
**#31 (26 Sep):** merge `wave/p9-integrate` (59f8918 + the docs commit). The default recipe is `sub31`; nothing else
changes for India / US. For tomorrow's clean reproduction, use the cold-run table in the #31 section: copy the model
and score files, or the whole `data/work` for a ~5-minute byte-exact run.

The rest of this section is the original #4 integration, kept for reference.

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

**prune.py** (#6)
- `train(folds=(2, 3), out=MODEL)`: 74 s, 2.3 GB, deterministic.
- `score` + `prune` on all 27.5M test pairs: inside `split_probs`, 3 / 18 / 5 min with m2_big on the kept pairs.

**rerank.py**
- `score_file(in, out)`: 1,363 pairs/s (India) and 1,700 pairs/s (US), 2.0 GB RSS, 3.4 GB VRAM.
- `band(c)`: 1 min. `train()`: 25 min (P4b).

**france.py** (#31)
- `world(base)` → `data/work/p9/fr31` (20 s the first time).
- `chn(model, out)` under `ER_WORK` = that world: 432 s, 2.60 GB. Kept rows go through a scratch parquet that is
  deleted afterwards.
- `post(...)`: ~40 s, ≤ 2.7 GB (P31's peak 2.72 GB; the source reads are country-filtered here).

**run.py**
- `france31()` → the France part dict (c_hn p as `p_m2`, post-rule p as `p`).
- `model_parts(split, world)` → per-country dicts; `rerank(c, q1, qx, p, text, cache)` → (p′, r);
  `thresholds(parts)` → {country: t}; `refit_w(parts)`; `ensure_ce()` trains the CE when `ce.pt` is absent.

**Caches written:** `data/work/p9/` (486 MB): M2 p for test and holdout, r for test and holdout, `m2_retrain.txt`.
With them a rerun skips all scoring: measured 52 s on holdout; the test rerun was not re-timed (decision + writing were 2.5 min).

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **#31 port fidelity is deliberate.**
  - bxneg reads the *unfixed* `rf_test_sx_France` (hn / core / places) and the tier reads the fixed one, as P31's code
    did (P28's `anchor.py` had no `ER_WORK`).
  - The learned sibling words come from the main world's P3 blocked list, not idf144's.
  - Changing either would move lines. Both are noted in `france.py`.
- **c_hn is a model file.** Without `p31/c_hn.txt`, `train_matcher.py --hn` retrains the recipe in float16 on the laptop
  (> 3 GB, ~1 h). The result is an equivalent booster, not the same file: France lines would move.
- **The r caches depend on their seed log.** The first run with `sub31` writes `p9/r_*.seeds`. A cache copied without
  its `.seeds` re-merges every seed once, first-wins, cache first: harmless.

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
