# Business entity resolution — Amazon ML Challenge 2026

This code rebuilds both submission files from the raw TSVs, from the repository root. It uses no external data and
no API. Two pretrained models are fine-tuned as cross-encoders; their base weights are downloaded once from Hugging
Face:
- `BAAI/bge-reranker-v2-m3` (Apache-2.0, 568M parameters) is the default (#8c on). It was fine-tuned on AWS
  (`probes/P13-aws-rerank`), and `run.py` loads its checkpoint rather than retraining it.
- `intfloat/multilingual-e5-small` (MIT, 118M parameters) is #4–#7's cross-encoder and is fine-tuned by `rerank.py train`.

Every other artefact is fitted on the train data and rebuilt when absent. The exceptions use unlabelled test rows
only, which the organisers allow: France's département → région map (learned from confident pairs among test France),
#9ab / #31's France IDF (its own S2/S3 text at its 14.4% share) and #31's sibling-word table for bxneg (learned from
test France's blocked candidate list). There is no geo table. France is the one test label absent from train S1;
the code finds it that way (`run.unseen()`) and names no country.

## 1. Setup

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r code/business_entity_resolution/requirements.txt
HF_HUB_OFFLINE=0 .venv/bin/python -c "from transformers import AutoModel, AutoTokenizer as T; \
  AutoModel.from_pretrained('intfloat/multilingual-e5-small'); T.from_pretrained('intfloat/multilingual-e5-small')"
# only to score the bge band fresh (2.3 GB; not needed with P13's cached scores in data/work/p13/r/)
HF_HUB_OFFLINE=0 .venv/bin/python -c "from transformers import AutoModelForSequenceClassification as M, AutoTokenizer as T; \
  M.from_pretrained('BAAI/bge-reranker-v2-m3'); T.from_pretrained('BAAI/bge-reranker-v2-m3')"
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
order (training included), except the bge cross-encoder: its checkpoint `data/work/p13/ckpt/ce_bge_pool.pt` must be
downloaded (SETUP.md step 5), and `run.ensure_ce` stops with instructions when it is missing. With the caches present
it only scores, decides and writes.

**The final zip's files are `--recipe sub_clean --drop-sibling-cands`** ([FINAL: the recipe of the final upload]).
`--recipe sub_clean` is the final pipeline (27 Sep) on top of everything below:

- **India / US = the combiner (IU_PLACEHOLDER).**
- **The unseen country (France) = `france.py`'s #31 chain below, then `france.chain`** (t 0.99 on the post-rules of
  0.9868): P42's joint recipe (rescue 0.90, group completion 0.70, hn-equal extras 0.80) → P41 brand recovery (0.30,
  computed on the #37-style lists and unioned) → P48 CE veto (rescue / gc band pairs, p unmoved, bge r < 0.2) → P54
  in-band CE rescue (r ≥ 0.99) → P47 hn-change class (p ≥ 0.80) → P55 combiner (3 LightGBM boosters,
  `data/work/p55/fr_plain_s{0,1,2}.txt`; veto band recipe pairs c < 0.6, add free pairs c > 0.95).
- **Last, every country: P54's duplicate consistency** (`run.dup_add`): an unassigned content-identical copy of an
  accepted S2 / S3 goes to the same S1 and joins its candidate list.
- `--recipe sub_clean_iu34` keeps #34's India / US (the a10 stacker); `--recipe sub54` is the uploaded #54 (LB
  0.986685): the same chain with P40's address-less names (0.70) and P46's noise-word tier, in #54's order, no P55.
- Reproduction ([VERIFY]): `sub_clean`, `sub_clean_iu34` and `sub54` rebuild `output/sub_clean`, `output/sub_clean_iu34`
  and `output/sub54` as sets per line (the staged files append the duplicate copies at the line's end; run.py sorts).
- Extra inputs of the final chain: `data/work/p55/fr_plain_s{0,1,2}.txt` (P55's boosters, fit on the LOCO frames by
  `probes/P55-fr-combiner/p55.py scores`), `data/work/p48/r_lo_France.parquet` (P48's bge scores of the France pairs
  outside the CE band, AWS GPU job), `data/work/p36/morph_test.json` (the brand syllables; recomputed from the test
  names when absent).

**Before 27 Sep:** `--recipe sub37_small` holds #37_gc70's matching file (public LB 0.986304) with the small candidate
set; `--recipe sub37` the same with the pruner's full candidate list.

- **India / US = #34:** the pruner, then m2_big on the kept pairs, **P13's bge-reranker-v2-m3 cross-encoder** on the
  band (w fit on holdout half 0), then **P17's house-number stacker on US only**, then **P34's a10 residual stacker**
  (`a10.py`, India + US): a 200 × 31 LightGBM with `init_score` = logit p re-scores each S2/S3's argmax pair from its
  competition (runner-up p and margin, the S1's other argmax pairs, same-name / same-address "twin" S1 among the
  candidates, a name-only flag, an error category). A pair at a generator sibling offset keeps its p, and the argmax
  stays #8c's. Fit on the India + US holdout half 0 (`run.ensure_a10`); report half 1 **+0.00021 ± 0.00006** LB-eq
  (India +0.00016, US +0.00036; F0.5 0.98956 → 0.98984). t 0.71.
- **The unseen country = `france.py`'s chain.** `run.unseen()` = the test S1 labels absent from train S1 (France is
  the only one); every unseen-country branch keys on it, no country is named in the code (the organisers allow "a
  stricter threshold for countries unseen in training" defined that way). `--france fr31`, t 0.9868, label-free:
  1. **rebuild** (P29): #9ab's IDF-fixed world (`idf144`), with house numbers re-parsed where an address
     segment opens with `(n)` (`(520) RUE X`: 23,272 S2/S3), and **R1** (P20): the generator's French name fillers
     `fils, cie, associes, freres, frs` leave the name core. #9ab's X is rescored with them (P20's rescore: blocking
     and context columns stay; `features.strings`, `rule_acc` and the rule-only extra pairs are recomputed).
  2. **c_hn** (P26): m2_big's recipe + 5 house-number features over each S1's full candidate list (hn class / signed
     delta / other SX sharing its hn / their best name_cos / other SX carrying the S1's hn), on the pruner's kept pairs
     (1,332,223, 5.13 per S1).
  3. **bge blend** on the band with P13's out-of-domain w [0.9245, 0.2595, 0.0515] (#9ab's).
  4. **post-rules at t** (P31's order): **key tier** (P29: the SX's argmax pair with equal R1 core, hn and city, the
     core unique among the country's S1, p ≥ 0.5 → p := 1; +6,088 pairs) → **bxneg** (P28: equal hn, name equal or one
     word inserted / dropped that is not a learned sibling word, a shared place and the same street, the SX's argmax or
     matched nowhere → p := max(p, 0.99); +9,647) → **veto** (P30: a pair with c_hn p ≥ 0.98 and bge r < 0.1 gets
     p := 0 unless tier / bxneg raised it; 7,753 pairs) → **cap** (P29: an SX that would give an S1 more than 5 S2 /
     6 S3 matches loses every pair; 18 pairs).
  5. **decision-time additions** (`france.complete`, after argmax + t; siblings never join): **empty-S1 rescue**
     (#34v: an S1 left empty takes its best argmax pair if p ≥ 0.90; +1,205 S1) → **group completion** (P37: an S1
     whose group holds only S2s or only S3s takes every free argmax pair of the missing source with p ≥ 0.70, up to
     5 S2 / 6 S3; +10,084 pairs to 9,322 S1, one-sided groups 20.7% → 17.1% of the country's S1).

With the caches it needs no GPU and reproduces `output/sub37_gc70` byte for byte (both files; `sub37_small`: the same
matching file). The bge scores are seeded into `p9/r_*` on first use, first file wins: `data/work/p13/r/` (P13), then
the France top-ups of `p19/r/` (idf144's new band pairs), `p26/bge/r/`, `p29/r/` and `p31/r/` (c_hn's and the
rebuild's). `p9/r_*.seeds` lists what a cache took. a10's p′ is cached per booster and input p in `p9/a10_*` (the
features take 4.5 / 1.6 min for India / US).

| recipe / flag | what changes | staged | LB |
|---|---|---|---|
| **`sub37`** (default) | all of the above | `output/sub37_gc70` | **0.986304** |
| **`sub37_small`** (final zip) | `sub37`'s matching; the sibling-decoy candidate stage (`--drop-sibling-cands`, below) on all three countries: 8.37M → **7,658,102 candidate pairs, 4.42 per S1** (France 5.13 → 4.48, India 4.90 → 4.75, US 4.63 → 4.00) | `output/sub37_final_small` | = `sub37` |
| `sub37_hn90` (`--hn-extra 0.9`) | + two-sided extras: a free argmax non-sibling SX with p ≥ 0.90 and the S1's house number joins a group that has both sources | `output/sub37_gc70_hn90` | slot (27 Sep) |
| `sub37 --rescue 0.7` | the rescue at p ≥ 0.70 | `output/sub37_gc70_r70` | slot (27 Sep) |
| `sub41` (= `sub37 --brand 0.3`), `sub41_hn90` | + P41's synthetic-brand recovery: a free argmax non-sibling SX whose name is one bare generator brand (a single token spelled by ≥ 2 of the 44 syllables P36 learns label-free from the test names) joins a non-empty group of the only S1 with its exact hn + street + city key, p ≥ 0.30 (+5,742 pairs to 5,641 S1) | `output/sub41`, `output/sub41_hn90` | slot (27 Sep) |
| `sub40x`, `sub40x_t70` (= `sub41 --noaddr 0.9 / 0.7`) | + P40's address-less exact-name variants: a free argmax non-sibling SX with an empty address whose full normalised name (legal form kept) equals its S1's, a name no other S1 of the country carries; any S1, empty ones too | `output/sub40x`, `output/sub40x_t70` (same ID sets; P40's builder leaves new IDs unsorted) | slot (27 Sep) |
| `sub43_r90` (= `sub42_both --rescue 0.9 --brand 0.3 --noaddr 0.7`) | #43_r90's levers applied in order. The staged `output/sub43_r90` is a union of pair sets and lacks 507 pairs that #37_gc70 accepts (p ≥ 0.9868, brand / address-less guards): the union's brand and address-less pairs were computed on #37_gc70, so pairs that t 0.99 dropped were never offered to them again | `output/sub43r90_recipe` | slot (27 Sep) |
| `sub46_on43`, `sub46g_on43` (= `sub43_r90 --swap 0.05 / 0.09`) | + P46's noise-word swap tier (`france.swap_mask`): the SX's argmax pair, not accepted at the post-rules t, whose core swaps one S1 word for a word w at the same house number, place and street; P33's guards (w not an R1 filler or a learned sibling word, no other candidate S1 holds the rest of the core, the SX core is no S1's core in that city, p ≥ 0.02) and P46's lens on the argmax pairs (≥ 20 hn-present swap pairs, sibling-offset share ≤ s_max, hn-equal share ≥ 0.8) | P46's builder on `output/sub43r90_recipe` | slot (27 Sep) |
| `sub42_both` (= `sub37 --t-country France=0.99 --t-post 0.9868 --rescue 0.97 --hn-extra 0.8`) | P42's joint re-tune: the decision at t 0.99 on the post-rules of t 0.9868, rescue 0.97, gc 0.70, hn extras 0.80 | `output/sub42_both` | slot (27 Sep) |
| `sub34_veto` = `sub37 --rescue 0 --gc 0` | no decision-time additions | `output/sub34_veto` | 0.985647 |
| `sub34` | India / US + a10, the unseen country = `sub31` (no veto) | `output/sub34` | — |
| `sub31` | India / US = #8c (no a10), the chain without the veto | `output/sub31` | 0.9856 |
| `sub31 --t-country France=0.9818` | the same at t 0.9818 (the rules act at that t: 6.31% empty, 3.161 per S1) | `output/sub31_t9818` | — |
| `sub31_veto`, `sub31_small` | `sub31` + the veto; `sub31` with the decoy stage (4.42 per S1) | `output/sub31_veto`, `output/sub31_small` | — |
| `sub9ab` | #9ab: `idf144`'s m2_big p + the bge blend, t 0.985 (`--t-country France=0.975 / 0.99`: `sub9ab_t975 / t990`) | `output/sub9ab` | 0.984072 |

The unseen country's bars are flags on any recipe: `--rescue T`, `--gc T`, `--hn-extra T`, `--brand T`, `--noaddr T`,
`--swap S_MAX` (0 turns one off),
`--veto TAU` (0 = off), `--t-country <label>=T`, and `--t-post T` (the post-rules at T, the decision at the country's
t). Add `--drop-sibling-cands` for the 4.42-per-S1 candidate file of any of them.

- Cold, the chain builds `data/work/p9/fr31/` (links to `fr_idf144/` + the fixed `rf_test_sx_France.parquet`, 36 MB),
  then rescores in its own process (`france.py chn`, ~4 min, 2.7 GB) into
  `p9/fr31/p9/chn_test_France_<c_hn hash>_<pruner hash>_0.03.parquet`. The post-rules take ~1 min (2.7 GB).
- c_hn is `data/work/p31/c_hn.txt` (50 MB), trained by P26's AWS job (float32, 20 GB RAM). When it is absent,
  `train_matcher.py --hn` trains the same recipe from the train data (float16, like m2_big): an equivalent model, not
  the file, so the France lines would move.
- Scoring the bge band fresh needs the HF base weights (`BAAI/bge-reranker-v2-m3`, 2.3 GB, downloaded once), the
  fine-tuned checkpoint `p13/ckpt/ce_bge_pool.pt` (624 MB) and a GPU.
- ~1.24M kept band pairs take ~20 min on an A10G and ~2 h on the laptop's RTX 3050.
- The bge CE is fine-tuned on AWS (`probes/P13-aws-rerank`); `run.py` does not retrain it.

**Unseen-country (France) variants** on the #8c India / US base (`--recipe`, or `--france <scorer>` +
`--t-country France=<t>`):

| recipe | `--france` | France scorer | France t | staged |
|---|---|---|---|---|
| `sub8c` | `m2` | m2_big p on the pruned list (#6c / #7) | 0.95 | `output/sub8c` |
| `sub9a` | `idf144` | P14: France reblocked and refeaturised with its own S2/S3 text in the IDF at its 14.4% share (`block.idf_country`) | 0.975 | `output/sub9a` |
| `sub9b` | `bge` | the bge CE on France's band, P13's label-free out-of-domain w [0.9245, 0.2595, 0.0515] | 0.98 | `output/sub9b` |
| `sub9s` | `sib` | P15's sibling rule: p := 0 when hn_x − hn_1 ∈ {3,4,5,7,9,11,13,21} (`stack.sib_mask`) | 0.95 | `output/sub9s` |
| `sub9ab` | `idf144_bge` | `idf144`'s candidates + `bge`'s blend (P19); LB 0.984072 | 0.985 | `output/sub9ab` |
| `sub31` (the chain of `sub37`) | `fr31` | `france.py`: the rebuild + c_hn + the bge blend + tier / bxneg / cap (above); LB 0.9856 | 0.9868 | `output/sub31` |
| `sub9d` | `idfdom` | P14's train-only fallback: 158 French domain words at US street-type IDF (`block.idf_domain`) | 0.98 | `output/sub9d` |

- `idf144` / `idfdom` build a France-only world, `data/work/p9/fr_<name>/`: links to this one plus the variant
  vectorizers. A subprocess reblocks France, rebuilds its X, prunes and runs M2 there (~15–25 min CPU).
- US / India never see the variant IDF.
- `idf144` uses unlabelled **test** text (the same category as chain counts and P6's rule; PROBLEM.md open question 4).
  `idfdom` uses train data and domain word lists only.

**The unseen country's threshold** is fixed per recipe (`run.T_FRANCE_OVERRIDE`, applied to every label in
`run.unseen()`). The leaderboard showed France over-matching under P6's label-free rule (0.78): 0.88 scored +0.0010,
and 0.95 (D4) scored 0.981142.
- `--t-country France=<t>` sets another value; it takes any `Country=t`.
- `T_FRANCE_OVERRIDE = None` returns the unseen labels to P6's rule.
- A label in neither train nor test S1 (e.g. `test_pipeline.py`'s injected Xland) keeps P6's rule.

`--recipe` switches the booster, blend weights, thresholds, pruner, cross-encoder, stacker and France scorer back to
an earlier submission. Each recipe reproduces its staged files byte for byte, and each CE keeps its own score cache
(keyed by the checkpoint's hash). The recipes live in `run.RECIPES`:

| recipe | M2 booster (train folds) | CE | blend w (India / US) | stacker | t (India / US) | France | pruner τ → cands / S1 |
|---|---|---|---|---|---|---|---|
| `sub37` (default), `sub37_*`, `sub34*` | `p4/m2_big.txt` (2–5; France `fr31`: `p31/c_hn.txt`) | `p13/ckpt/ce_bge_pool.pt` (bge) | [0.59015, 0.63194, 0.29732] | US: `p17/stack_sup-bge-l21.0_US.txt`, then India + US: `p34/a10_h0.txt` | 0.71 | see above | 0.03 → 4.83 (`_small`: 4.42) |
| `sub31`, `sub31_*`, `sub9ab`, `sub8c`, `sub9*` | `p4/m2_big.txt` (2–5; France `fr31`: `p31/c_hn.txt`) | `p13/ckpt/ce_bge_pool.pt` (bge) | [0.59015, 0.63194, 0.29732] | US: `p17/stack_sup-bge-l21.0_US.txt` | 0.71 | see above | 0.03 → 4.83 |
| `sub7` | `p4/m2_big.txt` (2–5) | `p11/ce_s2P.pt` (e5, 2 epochs) | [0.617, 0.666, 0.261] | — | 0.71 | m2, 0.95 | 0.03 → 4.83 |
| `sub6` | `p4/m2_big.txt` (2–5) | `p4b/ce.pt` (e5, 1 epoch) | [0.678, 0.715, 0.338] | — | 0.69 | m2, P6 rule: 0.78 | 0.03 → 4.83 |
| `sub5` | `p4/m2_big.txt` (2–5) | `p4b/ce.pt` (e5, 1 epoch) | [0.678, 0.715, 0.338] | — | 0.69 | m2, P6 rule: 0.78 | none → 15.87 |
| `sub4` | `p4/m2_ladder.txt` (2) | `p4b/ce.pt` (e5, 1 epoch) | [0.636, 0.757, 0.355] | — | 0.70 | m2, P6 rule: 0.82 | none → 15.87 |

- `--recipe sub6 --t-country France=0.88` reproduces `output/sub6b` byte for byte.
- `--recipe sub6 --t-country France=0.95` reproduces `output/sub6c` byte for byte.
- The #8c blend weights are P13's fit at full precision; the documented rounding [0.5901, 0.6319, 0.2973] moves 4 S1
  lines.

**Options on any recipe** (built for #9ab: `output/sub9ab_cap`, `output/sub9ab_small`, both validator PASS; `fr31` has
its own cap inside the chain, and `sub31_small` / `sub37_small` = `sub31` / `sub37` + `--drop-sibling-cands`):

| option | what | #9ab test | holdout |
|---|---|---|---|
| `--cap 5,6` | at most 5 S2 and 6 S3 matches per S1, the highest p kept; the train truth never has more (24,154 S1 at 5 S2, 2,816 at 6 S3) | 86 S1 lose 90 matches (France 17 S1, India 26, US 43) | 4 S1 change, all to F0.5 1.0: +0.0000019 |
| `--drop-sibling-cands` | a stage before the final decision: a pair at a generator sibling offset (hn_x − hn_1 ∈ P15's {3,4,5,7,9,11,13,21}) survives only if it is its SX's best pair above t. Matching is unchanged by construction | candidates per S1: US 4.63 → 4.00, India 4.90 → 4.75, France 4.37 → 3.97 (after τ below); matches identical | US 4.24 → 3.92, India 4.38 → 4.31; candidate recall 0.9874 → 0.9872; F0.5 unchanged |
| `--tau-country France=0.1` | a stricter pruner for France only (India / US keep 0.03) | France 5.03 → 4.37 candidates per S1; 225 France S1 (0.087%) change their matches | — (France has no labels; P10 measured τ 0.1 at −0.00076 on India / US) |

```bash
.venv/bin/python code/business_entity_resolution/src/run.py --split test --stage model --cap 5,6 --out output/sub9ab_cap
.venv/bin/python code/business_entity_resolution/src/run.py --split test --stage model --tau-country France=0.1 \
    --drop-sibling-cands --out output/sub9ab_small       # candidates 4.82 -> 4.34 per S1 (8.34M -> 7.53M pairs)
```

| step | what | module / function | cached at (`data/work/…`) |
|---|---|---|---|
| data | TSV → parquet, read with `QUOTE_NONE` and `keep_default_na=False` | `io_tsv.convert` | `{train,test}_source{1,2,3}.parquet` |
| folds | ground-truth pairs; 10 hashed folds of train S1 (0 = holdout, 1 = dev / early stopping, 2–9 = fitting) | `worlds.build_base` | `gt_pairs`, `s1_folds` |
| transliteration | Indic → Latin word table, learned from train folds 2–9 true pairs | `translit.table` | `p2_xlit_table.parquet` |
| région map | France S1 always names its région, and the 13 région names are the only hand-written French place words. An S2/S3 segment is learned as a département of région r when it stands in the région's slot (≤ 5% of its rows also name a région), S1 never names it, and ≥ 95% of ≥ 20 confident anchors put their S1 in r. An anchor is an S2/S3 whose name core equals exactly one S1's and which shares an address segment with it. Test France learns nord and pas de calais → hauts de france, gironde → nouvelle aquitaine, loire atlantique → pays de la loire | `normalize.fr_departements` | `fr_regions.json` |
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
| a10 stacker (#34, India + US) | residual LightGBM (200 × 31, `init_score` logit p) on each SX's argmax pair: runner-up p / margin, the S1's other argmax pairs, twin S1 (same name / address / legal-free core) among the SX's candidates, name-only flag, error category; a sibling-offset pair keeps its p; the argmax stays #8c's. Fit on the India + US holdout S1 of half 0 (P23 → P34) | `a10.features`, `a10.apply`, `run.a10_stage`, `run.ensure_a10` | `p34/a10_h0.txt`; `p9/a10_<split>_<c>_<booster>_<input p>.parquet` |
| France rebuild (#31) | #9ab's idf144 France world with the `(n)` house numbers re-parsed (P29 `paren_hn`) | `france.world` | `p9/fr31/` (links + `rf_test_sx_France.parquet`) |
| France c_hn (#31) | #9ab's X rescored with R1's cores (P20), pruned, then c_hn = m2_big's recipe + `france.HN` (P26 `hn_feats`, over the full list) | `france.chn`; `train_matcher.py --hn` when `c_hn.txt` is absent | `p31/c_hn.txt`; `p9/fr31/p9/chn_test_France_<c_hn>_<pruner>_<τ>.parquet` |
| France post-rules (#31) | after the bge blend, at France's t: key tier (P29) → bxneg (P28; sibling words learned on P3's blocked list) → [veto, P30] → cap (P29) | `france.post` | — |
| France additions (#34v, #37) | after the decision: empty-S1 rescue (p ≥ 0.90), group completion of one-sided groups (p ≥ 0.70), optional hn-equal extras (hn90), brand recovery (P41), address-less exact names (P40) and noise-word swaps (P46); free argmax non-sibling pairs only | `france.complete` (`rescue`, `gc_add`, `brand_mask`, `noaddr_mask`, `swap_mask`) | `p36/morph_test.json` (the brand syllables) |
| decision | each S2/S3 to its argmax-p S1 among the kept pairs (a10's countries: #8c's argmax), kept if p ≥ t. India / US: t = 0.71 on p′. The unseen label(s) (`run.unseen()`: France): its scorer (above), t fixed per recipe (#31 / #37 0.9868, #9ab 0.985, #8c 0.95; leaderboard-validated). Any other label without training data: M2's p, t from P6's label-free rule (below) | `run.thresholds` | — |
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
| CE rescoring, test (656k India + 583k US kept band pairs; 847k + 702k without the pruner; GPU) | e5 (#7): 5 + 5 min cold. bge (#8c): ~20 min on an A10G, ~2 h on the RTX 3050. 0 with the r cache | 2.1 GB (own process), 3.4 GB VRAM (e5) |
| #31 France chain, cold (`france.py`: '(n)' fix 20 s; R1 rescore + c_hn in its own process; post-rules) | 7 min (432 s) + 40 s, on top of the `idf144` world | 2.6 GB |
| a10 stacker, test (#34: text lookups + features from the raw sources, then the booster; cached per input p) | 4.5 / 1.6 min (India / US) | 1.4 GB under a 1.5 GB cap (swapping); 2.1 GB uncapped (P34) |
| unseen-country rescue + group completion (`france.complete`) | < 10 s | — |
| whole `sub37` run, every stage cached except a10 (27 Sep, 1.5 GB cap) | 12.5 min; 6 min with the a10 cache | 1.44 GB |
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
| France (#31) | no labels: c_hn's own holdout (P26, matcher only, half 1) India +0.0015 / US +0.0032 over m2_big; LOCO +1.8–2.0 pt; the rules' LOCO / proxies in P28 / P29 / P30 | — | — |
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
    [--recipe sub37_small --unseen France]  # #37's in-memory stages (blend w, fixed t, rescue + group completion,
                                            # decoy stage) with the local e5 CE (the bge CE needs its HF weights + a GPU)
.venv/bin/python code/business_entity_resolution/src/france.py             # unseen-country chain self-check (R1, paren_hn,
                                                                           # street key, name_rel, hn_feats, cap, rescue,
                                                                           # group completion + hn extras)
.venv/bin/python code/business_entity_resolution/src/a10.py                # a10 features / guard self-check
.venv/bin/python code/business_entity_resolution/src/train_matcher.py --hn --frac 0.02 --rounds 50 --out /tmp/x.txt  # c_hn smoke
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
.venv/bin/python code/business_entity_resolution/src/package.py --team Yeti [--out output/] [--dest .]   # -> Yeti_submission.zip
```

The zip holds `output/`, `code/business_entity_resolution/` (src, this README, requirements) and the filled
`Documentation_template.md`. The script checks its structure and that both TSVs have one row per test S1.
The model files are artefacts, not source, so they are not in the zip:
- the booster `data/work/p4/m2_big.txt` (69 MB), the pruner `data/work/p10/pruner.txt` (1.1 MB), the US stacker
  `data/work/p17/stack_sup-bge-l21.0_US.txt` (0.7 MB) and the e5 CEs (`p11/ce_s2P.pt`, `p4b/ce.pt`, 43 MB each).
  They are rebuilt from the train data by the commands above; the pipeline trains any that is missing.
- the bge CE `data/work/p13/ckpt/ce_bge_pool.pt` (624 MB). It is fine-tuned on AWS by P13's SageMaker job
  (`probes/P13-aws-rerank/solution_p13.py`, 76 min on an A10G) and loaded, not retrained, by `run.py`. Its sha1 also
  names the r caches (`p9/r_*_3bd0b930`), so the file must be present even when every score is cached.
- #31's France matcher `data/work/p31/c_hn.txt` (50 MB): P26's SageMaker job (`probes/P26-france-da`, float32,
  ~20 GB RAM, 15 min). `train_matcher.py --hn` retrains the recipe locally when it is absent (equivalent, not the
  file).
- #34's a10 stacker `data/work/p34/a10_h0.txt` (0.7 MB): `run.ensure_a10` fits it on the India + US holdout half 0
  when it is absent (P34's `ho.py` refit of the same rows is bit-identical).
