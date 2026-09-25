# P8-pipeline — RESULT
## TL;DR
- **Done:** `src/run.py` runs raw TSVs → both files, with pluggable stages. Its `--split test --stage rules` output is **byte-identical to submission #2** and passes the official validator. `test_pipeline.py` passes, and the zip builds and runs standalone.
- **Headline:** at the brief's definition of dense (every train S1 *and every train S2/S3* competing), the rules score **F0.5 0.7708**, not 0.8201.
  - Precision is 0.912 and singleton accuracy 0.710.
  - The 0.8201/0.8215 number used only the holdout world's 1.27M S2/S3. That world samples ~5% of the look-alike distractors; `--split holdout-world` reproduces 0.8215 exactly.
- **Recommendation:** adopt the branch. Make `--split holdout` (full corpus) the only decision number for P4/P5/P6. Expect submission #2 to score ≈0.77 on the leaderboard, not 0.82.

## Recommendation
Merge `wave/p8-pipeline` (commits 5d38943, 0e36be4, 4411c9f):

| file | what it is |
|---|---|
| `src/block.py` | P3 promoted: `block.py` + `full.py` fit/prep/split run |
| `src/run.py` | the pipeline |
| `src/model.py` | the `score(pairs, f1, fx)` interface + LR stub |
| `src/test_pipeline.py` | end-to-end guard |
| `src/package.py` | builds and checks the zip |
| `README.md`, `requirements.txt` | reproduce steps, pinned dependencies |
| `src/rules.py` (two-line fix) | `feats()` on an empty frame; dropped the dead `s0_exact` import that pulled the repo-root `ledger.py` into every import |

Commands (all from the repo root):
- **Submission:** `.venv/bin/python code/business_entity_resolution/src/run.py --split test --stage rules --out output/`. It asserts one row per S1, matches ⊆ candidates, and S2-/S3- ids of the same split and country, then runs the official validator.
- **Decision number:** `run.py --split holdout --stage rules|model` scores the fold-0 S1 while all 2.2M train S1 and all 10.3M train S2/S3 compete, and appends a `pipe` ledger row.
  - Use it, and never the holdout world, to accept P4/P5/P6 changes.
  - `--split holdout-world` is kept only as the regression link to the old 0.8215.

Plugging in P4 and P5:
- **P5:** copy `probes/P5-decide/decide.py` to `src/decide.py`. `run.py` imports it when present and calls `decide(pairs[s1, sx, p, country], CFG) -> {s1: set}` per country. Tested unchanged on dev: 0.9765 with P5's placeholder CFG.
- **P4:** replace the body of `model.score(pairs, f1, fx)`.
  - `pairs` carries `q1`/`qx` (row positions into f1/fx) plus block_score, block_rank_s1, block_rank_sx, via and acc.
  - f1/fx are `rules.feats` frames with `id`. P4's `rec()` also needs the P3 texts, which it can join by id from `data/work/p3/txt*/`.
  - Or, the simplest adapter: join on `(f1.id[q1], fx.id[qx])` against P4's `prob_{test,holdout}_*.parquet`.
- **P4's fixed decision rule** (argmax per S2/S3 + global T) is already the fallback when there is no `src/decide.py`.

What P4/P5 should know from the dense numbers:
- 92.5% of the rules' dense false positives are **parentless** S2/S3 that share the S1's exact name core.
- 87.5% of them have address score exactly 4 (for comparison, 23% of true pairs score 4).
- At score 4 with a house number on both sides, precision is **0.51**, and that bucket holds most of the false positives. House-number (street) disagreement is the negative feature the matcher needs.
- No rule threshold fixes it: need 4 → 0.7556, need 5 → 0.6603.

## Evidence
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| pipe `p8-rules` | holdout-dense (full corpus) | F0.5 **0.7708** (India 0.7893, US 0.7585). Precision 0.9121, recall 0.6392, singleton acc 0.7103. Candidate recall 0.9917 at 14.2/S1. 21 min cold (feature caches built), peak 2.14 GB | keep: the honest number |
| pipe `p8-rules` | holdout-world-dense | F0.5 **0.8215**, precision 0.9940, singleton 0.9776. This is `test_rules_cands` (by_state) exactly, and within 0.002 of 0.8201 (unpartitioned); 39 s warm | reproduces |
| P8-pipeline/rules-fullcorpus-vs-world-sx | holdout | 92.5% of FPs are parentless; 87.5% score addr = 4; 29% of singletons get ≥ 1 FP; precision 0.51 when both sides carry a house number at addr = 4 | — |
| P8-pipeline/rules-need-sweep-dense-fullcorpus | holdout-dense | need 2 / 4 / 5 / 6 → 0.7708 / 0.7556 / 0.6603 / 0.6098 (precision 0.912 / 0.913 / 0.990 / 0.992) | drop (no threshold fix) |
| P8-pipeline/test-rules-run-validator | test | 1,732,544 rows; 1,540,960 S1 matched; 27.49M candidates. **Both files byte-identical to output/sub2**, and the official validator PASSes both. 148 / 308 / 174 s (France / India / US) warm, peak 2.29 GB | keep |
| pipe `p8-model` | holdout-dense | Stub LR (fitted on plain dev): **0.6962**, precision 0.676, singleton 0.018, 99.9% of S1 matched. The interface runs at full density (2.2M pairs, 264 s, 2.22 GB) | interface only |
| P8-pipeline/p3-plus-rule-key-pairs-recall | holdout-dense | P3 alone: candidate recall 0.9917 at 14.2/S1. P3 ∪ rule-key pairs: 0.9920 at 19.9/S1 (India 24.2/S1) | drop keys from the candidate set |
| P8-pipeline/src-block-equiv-dev | dev | `src/block.py` == `probes/P3-block/block.py`: 511,521 pairs identical, pair recall 0.9990 | keep |
| pipe `p8-rules` / `p8-model` | dev | rules 0.8526; stub 0.9802 (plain world, in-sample). The unzipped package reproduces 0.8526 standalone | interface check |
| test_pipeline.py (not a ledger row) | dev slice + Xland + Yland | 3,203 S1 rows in both stages. Xland: 200 unseen-label S1 with S2/S3, 86% (rules) / 94% (model) get a correct hit. Yland: S1 only, empty rows. Matches ⊆ candidates, same-country ids only | pass |

## What failed
- **Reproducing 0.8201 on the full corpus:** 0.7708.
  - It isn't a pipeline bug: the same code on the world's S2/S3 gives 0.8215 to the fourth decimal.
  - Recall is equal (0.6392 vs 0.6392). The loss is 47k false positives from distractor records the world never sampled.
- **Raising the rule's address threshold:** it trades recall for precision one-for-one (table above). The distractors pass on city + place + state, like a quarter of the true pairs do.
- **Stub LR trained on the plain dev world:** it matches 99.9% of S1 at full density (0.6962). A matcher and its threshold must be fitted on full-density candidates (P4's `cands_train_full_*`).
- **Rule-key pairs as extra candidates:** +0.03 pt recall for +40% candidates. Keep the rule flags as features and the accepted pairs in the candidate set (needed for matches ⊆ candidates), but not every key-join pair.
- **Official validator inside a 2.6 GB cap:** it holds all 27.5M candidate ids as Python strings, 4.9 GB RAM + swap. It passed, but spent 17 min swapping. `run.py` sets its `oom_score_adj` so the OOM killer takes it, never `run.py`, and then re-validates `matching_results.tsv` alone.

## Interfaces
Everything lives in `code/business_entity_resolution/src/` (run from the repo root; `ER_WORK` / `ER_DATA` override the paths).

**`run.py`**
- `run(split, stage, out, world=None, log=True) -> metrics`; `--split test|holdout|holdout-world|dev`, `--stage rules|model`.
  - `world=(s1, sx, truth)` runs injected frames instead of a split.
  - Artifacts are rebuilt when absent:
    - `ensure()`: parquet copies, gt_pairs + folds, worlds.
    - `translit.table()`, `rules.freq()`, `block.vecs()`, `block.split_cands()`.
    - `feats()`: `data/work/rf_<split>_<side>_<country>.parquet`, streamed 500k raw rows at a time.
    - `train_stub()`.
- Helpers:
  - `inputs(split)` yields per country (f1, fx, lazy cands, scored ids, truth).
  - `assemble` builds the int64 keys q1 << 32 | qx and feeds `rules.match` 400k S2/S3 at a time.
  - `decide`, `argmax_sx`.
  - `write_lists` (P7) writes rows from int keys, so pair ids are only ever strings one 200k-S1 chunk at a time.
- Test rules run: ~10.5 min warm, peak 2.29 GB. Cold adds ~20 min of record features at `PROCS = 2` and 35 min of P3 blocking.

**`block.py`**
- P3's API unchanged: `CFG`, `fit_views`, `texts`, `index`, `search`, `country_pairs`, `finalize`, `candidates(s1, sx, cfg, vecs)`.
- Added:
  - `vecs()`: load or fit `data/work/p3/vecs_hash.pkl`.
  - `prep(split, country)`.
  - `split_cands(split, country) -> data/work/p3/cands_<split>_<country>.parquet`.
- CLI: `block.py [fit | test <country> | train <country>]`.
- Test run: 35 min, 2.7 GB on India at `shard=300k` (P3-measured). That is above a 2.6 GB cap, so lower `CFG["shard"]` there.

**`model.py`**
- `score(pairs, f1, fx) -> float32[n]`.
- `pair_feats`, `fit(X, y)`, `save(lr, t)`, `load() -> {"lr", "t"}`.
- The artifact is `data/work/p8/model_stub.pkl`.

**`test_pipeline.py`**
- `world()` builds a 3k dev slice + Xland + Yland; `check(out, s1, sx)` verifies both files. ~1 min, 0.6 GB.

**`package.py`**
- `build(team, out, dest) -> <team>_submission.zip`, with `check()` for members, headers and 1,732,544 rows per TSV.
- `--selftest`. The real build takes 38 s and gives 194 MB.
- The documentation is picked up from `Documentation_template.md` or `.pdf` at the repo root or in `docs/`, and only once it's filled.

**Probe scripts:** `probes/P8-pipeline/{equiv_block,fp_anatomy,need_sweep,sc4}.py` hold the evidence above.

## Risks
- **The leaderboard will likely come in well under 0.82 for submission #2.** Test has 5.8 S2/S3 per S1 against train's 4.7, so if anything there are more look-alike distractors per S1. P4/P5 decisions made on holdout-world numbers are suspect. The P3 `cands_holdout_full_*` and P4 `cands_train_full_*` files are full-corpus, so the work built on them is fine.
- **Model stage at test scale was not run.**
  - Rule-key pairs push India to ~20M pairs. With fx (0.6 GB) held, that would be ~3 GB at the 2.6 GB cap.
  - Upgrade path: drop key-join pairs from the model candidates (they add ~0 recall), or score per S2/S3 chunk.
- **Never submit `--stage model` with the stub** (0.6962 dense).
- **Holdout model-stage competition:** `cands_holdout_full_*` lists holdout S1 only, so an S2/S3's argmax can't see non-holdout rivals. P4's `cands_train_full_*` (all train S1) now exist and should be the model stage's holdout candidates once P4's scorer lands.
- **France:** unchanged from rules and blocking (unvalidated; P6). Countries come from the S1 file. An unseen label, or one with no S2/S3, is covered by the test.
- **IDs:** the official validator skips ID existence by default. `run.codes()` asserts that every written id is an S1/S2/S3 of the same split and country.
- **Licences:** requirements are numpy / pandas / scipy / scikit-learn (BSD), pyarrow / sparse_dot_topn (Apache-2.0), anyascii (ISC), rapidfuzz / duckdb (MIT). No models, no network. All fitting uses train data only (`vecs`, `place_freq`, translit table, stub).
- **Merge:** the two small `rules.py` edits may touch lines the master also edited on main.

## Open questions
- Which holdout definition decides? Best guess: the full corpus (`--split holdout`), since it's the one shaped like test. The first leaderboard score of submission #2 arbitrates: I predict ≈0.76–0.78.
- What is `<team_name>` in the zip name (PROBLEM.md Q7)? `package.py --team` takes it; nothing is assumed.
- Is there an upload size limit for the final zip (194 MB with both TSVs)? Best guess: none stated, and deflate is already applied.
- Should the model-stage candidate set include every rule-key pair? Best guess: no. The recall gain is +0.03 pt for +40% candidates, and the candidate file is audited for reduction ratio.
