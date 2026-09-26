# Business entity resolution — Amazon ML Challenge 2026

This code rebuilds both submission files from the raw TSVs, from the repository root. It uses no external data and
no API. The one pretrained model is `intfloat/multilingual-e5-small` (MIT, 118M parameters). Its base weights are
downloaded once from Hugging Face, then fine-tuned on the train data. Every other artefact is fitted on the train
data and rebuilt when absent.

## 1. Setup

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r code/business_entity_resolution/requirements.txt
HF_HUB_OFFLINE=0 .venv/bin/python -c "from transformers import AutoModel, AutoTokenizer as T; \
  AutoModel.from_pretrained('intfloat/multilingual-e5-small'); T.from_pretrained('intfloat/multilingual-e5-small')"
```

After that download, `rerank.py` sets `HF_HUB_OFFLINE=1`, so the pipeline never touches the network.

Unzip `student_resource.zip` so the data sits at `data/student_resource/dataset/{train,test}/*.tsv`, with the official
validator at `data/student_resource/utils/validate_submission.py`. Two environment variables override the paths:
`ER_DATA` (the `dataset/` directory) and `ER_WORK` (intermediate files, default `data/work`).

Hardware: the measurements below are from a shared 16-core laptop, 15 GB RAM, one RTX 3050 6 GB. Every step fits
a 3 GB memory cap. The cross-encoder steps need the GPU; they take the lock `/tmp/amc-gpu.lock` (`flock`).

## 2. Reproduce the submission

```bash
.venv/bin/python code/business_entity_resolution/src/run.py --split test --stage model --out output/
```

It writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`, asserts the format rules, then runs the
official validator. Nothing else has to run first: on a cold `data/work` the command rebuilds every step below in
order (training included). With the caches present it only scores, decides and writes.

**The default recipe is `sub8c`:**
- the pruner, then m2_big on the kept pairs;
- **P13's bge-reranker-v2-m3 cross-encoder** on India / US (w fit on holdout half 0), then **P17's house-number
  stacker on US only**;
- t 0.71 for India / US, and France (m2_big p) fixed at t 0.95.

With P13's cached bge scores (`data/work/p13/r/`, seeded into `p9/r_*` on first use) it needs no GPU, takes 4 min
plus the validator, and reproduces `output/sub8c` byte for byte.
- Scoring the bge band fresh needs the HF base weights (`BAAI/bge-reranker-v2-m3`, 2.3 GB, downloaded once), the
  fine-tuned checkpoint `p13/ckpt/ce_bge_pool.pt` (624 MB) and a GPU.
- ~1.24M kept band pairs take ~20 min on an A10G and ~2 h on the laptop's RTX 3050.
- The bge CE is fine-tuned on AWS (`probes/P13-aws-rerank`); `run.py` does not retrain it.

**France variants** on the same India / US base (`--recipe`, or `--france <scorer>` + `--t-country France=<t>`):

| recipe | `--france` | France scorer | France t | staged |
|---|---|---|---|---|
| `sub8c` | `m2` | m2_big p on the pruned list (#6c / #7) | 0.95 | `output/sub8c` |
| `sub9a` | `idf144` | P14: France reblocked and refeaturised with its own S2/S3 text in the IDF at its 14.4% share (`block.idf_country`) | 0.975 | `output/sub9a` |
| `sub9b` | `bge` | the bge CE on France's band, P13's label-free out-of-domain w [0.9245, 0.2595, 0.0515] | 0.98 | `output/sub9b` |
| `sub9s` | `sib` | P15's sibling rule: p := 0 when hn_x − hn_1 ∈ {3,4,5,7,9,11,13,21} (`stack.sib_mask`) | 0.95 | `output/sub9s` |
| `sub9d` | `idfdom` | P14's train-only fallback: 158 French domain words at US street-type IDF (`block.idf_domain`) | 0.98 | `output/sub9d` |

- `idf144` / `idfdom` build a France-only world, `data/work/p9/fr_<name>/`: links to this one plus the variant
  vectorizers. A subprocess reblocks France, rebuilds its X, prunes and runs M2 there (~15–25 min CPU).
- US / India never see the variant IDF.
- `idf144` uses unlabelled **test** text (the same category as chain counts and P6's rule; PROBLEM.md open question 4).
  `idfdom` uses train data and domain word lists only.

**France's threshold** is fixed per recipe (`run.T_FRANCE_OVERRIDE`). The leaderboard showed France over-matching
under P6's label-free rule (0.78): 0.88 scored +0.0010, and 0.95 (D4) scored 0.981142.
- `--t-country France=<t>` sets another value; it takes any `Country=t`.
- `T_FRANCE_OVERRIDE = None` returns France to P6's rule.
- Every other unseen label keeps P6's rule.

`--recipe` switches the booster, blend weights, thresholds, pruner, cross-encoder, stacker and France scorer back to
an earlier submission. Each recipe reproduces its staged files byte for byte, and each CE keeps its own score cache
(keyed by the checkpoint's hash). The recipes live in `run.RECIPES`:

| recipe | M2 booster (train folds) | CE | blend w (India / US) | stacker | t (India / US) | France | pruner τ → cands / S1 |
|---|---|---|---|---|---|---|---|
| `sub8c` (default), `sub9*` | `p4/m2_big.txt` (2–5) | `p13/ckpt/ce_bge_pool.pt` (bge) | [0.59015, 0.63194, 0.29732] | US: `p17/stack_sup-bge-l21.0_US.txt` | 0.71 | see above | 0.03 → 4.83 |
| `sub7` | `p4/m2_big.txt` (2–5) | `p11/ce_s2P.pt` (e5, 2 epochs) | [0.617, 0.666, 0.261] | — | 0.71 | m2, 0.95 | 0.03 → 4.83 |
| `sub6` | `p4/m2_big.txt` (2–5) | `p4b/ce.pt` (e5, 1 epoch) | [0.678, 0.715, 0.338] | — | 0.69 | m2, P6 rule: 0.78 | 0.03 → 4.83 |
| `sub5` | `p4/m2_big.txt` (2–5) | `p4b/ce.pt` (e5, 1 epoch) | [0.678, 0.715, 0.338] | — | 0.69 | m2, P6 rule: 0.78 | none → 15.87 |
| `sub4` | `p4/m2_ladder.txt` (2) | `p4b/ce.pt` (e5, 1 epoch) | [0.636, 0.757, 0.355] | — | 0.70 | m2, P6 rule: 0.82 | none → 15.87 |

- `--recipe sub6 --t-country France=0.88` reproduces `output/sub6b` byte for byte.
- `--recipe sub6 --t-country France=0.95` reproduces `output/sub6c` byte for byte.
- The #8c blend weights are P13's fit at full precision; the documented rounding [0.5901, 0.6319, 0.2973] moves 4 S1
  lines.

| step | what | module / function | cached at (`data/work/…`) |
|---|---|---|---|
| data | TSV → parquet, read with `QUOTE_NONE` and `keep_default_na=False` | `io_tsv.convert` | `{train,test}_source{1,2,3}.parquet` |
| folds | ground-truth pairs; 10 hashed folds of train S1 (0 = holdout, 1 = dev / early stopping, 2–9 = fitting) | `worlds.build_base` | `gt_pairs`, `s1_folds` |
| transliteration | Indic → Latin word table, learned from train folds 2–9 true pairs | `translit.table` | `p2_xlit_table.parquet` |
| record features | `to_latin` → `name_core`, `website_core`, `address_fields` (state, city, house number, places) | `run.feats` → `rules.feats` | `rf_<split>_{s1,sx}_<country>.parquet` |
| vectorizers | hashed TF-IDF (name char-3-grams, address words) on a seeded 1M-row sample of train S2/S3 | `block.vecs` | `p3/vecs_hash.pkl` |
| blocking | per country, within state group: each S2/S3 → its top-2 S1 (top-10 if it has no address) ∪ each S1's top-10 S2/S3 | `block.split_cands` | `p3/txt*/`, `p3/cands_<split>_<country>.parquet` |
| rule pairs | exact (country, name core) or website key + address evidence ≥ 2 → the unique best S1 | `rules.match` | — (inside the features) |
| pair features | 65 label-free features per candidate pair (P3 ∪ rule-accepted): name / address similarities, house number / city / state agreement, website, chain size, and each pair's rank and margin among its S1's and its S2/S3's candidates | `features.build` | `p4/X/<split>_<country>_sx_NNN.parquet` |
| pruner | 300-tree LightGBM on 30 of the 65 features (blocking signals + name / address similarities), train folds 2–3. A pair goes on iff q ≥ 0.03 or it is rule-accepted. The features stay computed over the full blocking list | `prune.train`, `prune.prune` | `p10/pruner.txt` |
| M2 | LightGBM on the pairs of the train folds 2–5 S1 (12.6M), early-stopped on fold 1 | `train_matcher.main` | `p4/m2_big.txt` |
| M2 scores | p for every kept candidate pair | `model.split_probs` | `p9/m2_<split>_<country>_<booster hash>_<pruner hash>_<τ>.parquet` |
| cross-encoder | #8c: bge-reranker-v2-m3 (Apache-2.0, 568M) fine-tuned 2 epochs on the same "name \| addr" pairs (M2-like uncertain bands of train folds 4–9, India, US), on AWS (P13). #7: e5-small, 2 epochs, locally | P13 `solution_p13.py` (bge); `rerank.py band`, `rerank.py train epochs=2` (e5) | `p13/ckpt/ce_bge_pool.pt`; `p11/ce_s2P.pt`, `p4b/ce.pt` |
| rerank | India / US pairs with 0.02 < p < 0.98: p′ = σ(0.5901 logit p + 0.6319 logit r + 0.2973) (#8c) | `run.rerank` → `rerank.py score … bge` | `p9/r_<split>_<country>_<ce hash>.parquet` (#8c: seeded from `p13/r/`) |
| stacker (#8c, US) | residual LightGBM (200 × 31) on logit p′ + house-number class / signed delta / equality + support among the S1's other candidates (P15 / P17); fit on the US holdout S1 of half 0 | `stack.X_of`, `stack.stacked`, `run.ensure_stack` | `p17/stack_sup-bge-l21.0_US.txt` |
| decision | each S2/S3 to its argmax-p S1 among the kept pairs, kept if p ≥ t. India / US: t = 0.71 on p′. France: its scorer (above), t = 0.95 for #8c (fixed, leaderboard-validated). Any other label without training data: M2's p, t from P6's label-free rule (below) | `run.thresholds` | — |
| output | one row per test S1 (all countries); candidates = the kept pairs (what M2 + CE scored) | `run.write_lists` | `output/*.tsv` |

**Unseen-country threshold, label-free** (France until #6, any other unseen label now). France has no labels, and a
model applied to an unseen country is over-confident
(leave-one-country-out: US→India loses 4 pts at the source threshold). So the threshold is where France's predicted
empty-S1 rate equals the labelled countries' S1-weighted rate under M2 at M2's own optimum `T_M2` = 0.69. It is
clipped to [0.69, 0.89], and computed from the test probabilities on every run. It gives 0.78, both with and
without the pruner (#4's M2: 0.82). The leaderboard later showed that this under-corrects France: #7 fixes France
at 0.95 instead, and the rule still sets t for any other unseen label.

Runtimes and memory (measured, 3 GB cap):

| step | cold | peak RAM |
|---|---|---|
| text prep + blocking, test | 6.5 min + 5 / 18 / 6 min (France / India / US) | 2.7 GB |
| record features, test (`PROCS = 2`) | ~20 min | 1.5 GB |
| pair features, test (`features.build`) | 3 / 20 / 9 min (France / India / US) | 2.8 GB |
| TRAIN side for the models: blocking for all 2.2M train S1, record + pair features | ~35 + 20 + 50 min | 2.9 GB |
| pruner training (`prune.py train`: 6.3M folds 2–3 pairs; `deterministic=True`, the retrain is bit-identical) | 74 s | 2.3 GB |
| M2 training (`train_matcher.py`: 12.6M folds 2–5 pairs as float16, 793k fold-1 validation pairs) | ~45 min (P4: 2,371 s fit, 4,930 trees) | 3.7 GB (~0.5 GB swap) |
| cross-encoder: `band` ×2 + `train epochs=2` (GPU) | 5 + ~50 min (P11: 40 min with pre-tokenised batches; 1 epoch 25 min) | 2.6 GB, 2.4 GB VRAM |
| France IDF variant world (`sub9a` / `sub9d`: France reblocked, X rebuilt, pruned, M2) | 47 / 46 min (33 / 38 min of it blocking at 15 threads) | 2.5 GB |
| pruner + M2 scores, test (the pruner reads all 27.5M pairs; m2_big scores the kept 8.37M) | 3 / 18 / 5 min (France / India / US; India under load) | 1.8 GB |
| M2 scores without the pruner (`--recipe sub5`, all 27.5M pairs) | 5 / 18 / 43 min (loaded machine) | 2.0 GB |
| CE rescoring, test (656k India + 583k US kept band pairs; 847k + 702k without the pruner; GPU) | 5 + 5 min cold (#7); 0 with the r cache | 2.1 GB (own process), 3.4 GB VRAM |
| decision + writing + validator | 2.5 min | 2.3 GB |

## 3. Validate on the training data

```bash
.venv/bin/python code/business_entity_resolution/src/run.py --split holdout --stage model   # dense holdout F0.5
.venv/bin/python code/business_entity_resolution/src/run.py --split dev --stage model       # small dev world
.venv/bin/python code/business_entity_resolution/src/train_matcher.py eval data/work/p4/m2_big.txt      # M2 alone
```

`holdout` is the test-shaped proxy. The fold-0 S1 (220,358) are scored while every train S1 and every train S2/S3
of the country compete (2.2M S1, 10.3M S2/S3). Each S2/S3's argmax sees all its train S1 (11.5M pairs). The
pruner keeps 2.78M of them (4.30/S1), and those go to m2_big and the cross-encoder. The metric is macro F0.5 per S1, singletons included
(`metric.py`). The run also refits the blend weights on holdout half 0, as a check.
`dev` runs the same stages in memory (blocking, features, M2, CE) on the 30k-S1 dev world.

For #8c, P13 scored the bge CE on the holdout S1's own band pairs only: 59k India and 95k US. The competing train S1's
band pairs have no bge score (219k India, 351k US). `--ce-missing fallback` gives them #7's e5 blend: P17's yardstick,
which the run reproduces to the 5th decimal. With the bge base weights and a GPU, `--ce-missing score` (the default)
scores them instead, ~1 h on an A10G. The US stacker is refit the same way when its file is absent:
`run.ensure_stack` on holdout half 0 gives P17's booster bit for bit.

| run | F0.5 | time (cold / cached) | peak RAM |
|---|---|---|---|
| `--split holdout --stage model --ce-missing fallback` (#8c) | **0.98990** (India 0.9905, US 0.9895; half 0 0.99024, half 1 0.98956: India 0.99004 / US 0.98924 = P17's; P 0.9983, R 0.9726) | 5 min (all cached) | 2.1 GB |
| `--split holdout --stage model --recipe sub7` (#7) | **0.9888** (0.98881; India 0.9900, US 0.9880; half 0 0.98894, half 1 0.98869; P 0.9980, R 0.9700, 4.30 cands/S1) | 6 min (CE band fresh) / 1 min | 1.4 GB |
| `--split holdout --stage model --recipe sub6` | **0.9885** (India 0.9896, US 0.9878; half 0 0.98872, half 1 0.98831 vs P10's 0.98825 with D3; P 0.9977, R 0.9698, candidate recall 0.9874, 4.30 cands/S1) | 7 min / 1 min | 2.2 GB |
| `--split holdout --stage model --recipe sub5` | 0.9888 (India 0.9899, US 0.9880; half 0 0.9890, half 1 0.9885; P 0.9977, R 0.9705) | 38 min (loaded machine) / 1 min | 2.1 GB |
| `--split holdout --stage model --recipe sub4` | 0.9884 (India 0.9894, US 0.9876; half 0 0.9885, half 1 0.9882; P 0.9975, R 0.9697) | 22 min / 1 min | 2.2 GB |
| `--split dev --stage model` | 0.9949 (plain world) | 1.7 min | 0.8 GB |
| `train_matcher.py eval` M2 alone, fold argmax | 0.9813 (`m2_ladder.txt`) = 0.9813 (retrained, #4 recipe); m2_big 0.9834 (P4, argmax over all S1, no CE) | 7.5 min for two | 2.5 GB |
| `--split holdout --stage rules` | 0.7708 | 21 min cold | 2.1 GB |

The blend refit on holdout half 0 returns [0.678, 0.715, 0.337] for m2_big (CFG [0.678, 0.715, 0.338]),
[0.674, 0.718, 0.343] for m2_big + pruner (P10: keeping #5's w costs nothing, 0.98825 vs 0.98824),
[0.624, 0.670, 0.267] for m2_big + pruner + the 2-epoch CE (CFG [0.617, 0.666, 0.261], P11's fit without the pruner), and
[0.636, 0.757, 0.355] for m2_ladder.

## 4. Tests

```bash
.venv/bin/python code/business_entity_resolution/src/test_pipeline.py      # end to end: every label incl. an unseen one
.venv/bin/python code/business_entity_resolution/src/test_pipeline.py --ce data/work/p11/ce_s2P.pt --ce-base e5 \
    [--unseen France --france sib|bge]     # #8c's stages with the local e5 CE (the bge CE needs its HF weights + a GPU)
.venv/bin/python code/business_entity_resolution/src/stack.py              # stacker / sibling-rule self-check
.venv/bin/python code/business_entity_resolution/src/features.py           # feature self-check
.venv/bin/python code/business_entity_resolution/src/prune.py              # pruner self-check
.venv/bin/python code/business_entity_resolution/src/rerank.py check       # blend-rule self-check
.venv/bin/python code/business_entity_resolution/src/block.py              # blocking self-check
.venv/bin/python code/business_entity_resolution/src/metric.py             # metric self-check (PDF example)
.venv/bin/python code/business_entity_resolution/src/package.py --selftest
```

`--stage rules` reproduces submission #2, the rules-only baseline, byte for byte.

## 5. Package

```bash
.venv/bin/python code/business_entity_resolution/src/package.py --team <team_name>   # -> <team_name>_submission.zip
```

The zip holds `output/`, `code/business_entity_resolution/` (src, this README, requirements) and the filled
`Documentation_template.md`. The script checks its structure and that both TSVs have one row per test S1.
The fine-tuned weights (`data/work/p4b/ce.pt`, 43 MB), the booster (`data/work/p4/m2_big.txt`, 69 MB) and the
pruner (`data/work/p10/pruner.txt`, 1.1 MB) are artefacts, not source. All three are rebuilt from the train data
by the commands above, and the pipeline trains any that is missing.
