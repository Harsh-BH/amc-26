# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Yeti
**Team Members:** Harsh Bhatt, Shriya Sinha, Aditya Kashyap, Anurag Pokhariyal (IIT Kharagpur)
**Submission Date:** 27 September 2026

> Every number below is traceable to a row in `ledger/*.jsonl` and to the merge memos `probes/MERGE-*.md`.

---

## 1. Executive Summary

The pipeline has four stages:
1. **Transliteration** of Indic-script records.
2. **Normalisation.**
3. **Blocking:** a state-partitioned, hashed TF-IDF search from each Source 2/3 record to Source 1. It finds 99.2% of true
   pairs at about 14 candidates per Source 1 entity, at full corpus density.
4. **Matching:** a LightGBM model on 65 label-free pair and candidate-list features, re-scored on its uncertain band by a
   fine-tuned multilingual cross-encoder.

Decisions exploit the data's one-to-many structure: each Source 2/3 record goes to at most one Source 1 entity, its argmax,
above a threshold tuned for F0.5. On a holdout scored at full corpus density (every train Source 1 and every train
Source 2/3 competing), F0.5 is **0.9885**, against 0.771 for an exact-key rule baseline.

## 2. Methodology

### 2.1 Problem Analysis (EDA)
- **Structure.** Source 1 is deduplicated. Every Source 2/3 record belongs to at most one Source 1 entity: the relation is
  strictly one-to-many. 26% of Source 2/3 records match nothing (orphans / distractors). 5.6% of Source 1 entities have no
  match. The remaining 94.4% have 1–5+ matches (mean 3.5).
- **Look-alike chains.** 50.6% of Source 1 entities share their normalised name core with another Source 1 of the same
  country (France 54.5%). Names alone cannot decide a match: name-only exact matching drops from 0.684 F0.5 on a small
  world to 0.431 at test density. Address evidence (house number, city, state) is required.
- **Noise generator.** Legal-suffix swaps (Pvt Ltd ↔ Private Limited), honorific and junk prefixes, `(ID: n)` tails,
  alias constructs (f/k/a, dba), websites and handles as names, digit-for-letter swaps, accent injection, shuffled address
  segments, `NULL` placeholders, landmark phrases, and state code ↔ name swaps.
- **Cross-script (India).** About 23% of Indian Source 2 names are in Indic scripts, and 18% of Indian true pairs cross
  scripts. The Indic names are token-for-token transliterations of the Latin Source 1 name, drawn from a closed
  1,518-word vocabulary (no out-of-vocabulary types on test). Indic text in addresses is only ever the state name.
- **Unseen country.** Test adds France (15% of test, no labels). Postcodes are nearly absent everywhere (US five-digit
  numbers are house numbers).
- **Evaluation yardstick.** A holdout (train fold 0 of 10, by Source 1) is scored while **all** train Source 1 and
  **all** train Source 2/3 records compete. Smaller "mini-world" evaluations under-sample distractors: they overstated rule
  precision by 8 points (0.994 vs 0.912).

### 2.2 Solution Strategy
**Approach type:** blocking + gradient-boosted classifier + cross-encoder reranker + structured decoding.
**Core ideas:**
- A learned transliteration table.
- Reverse (Source 2/3 → Source 1) blocking, which exploits the one-parent structure.
- Label-free candidate-context features (chain sizes, ranks, competition).
- Many-to-one exclusive decoding.

## 3. Candidate Generation (Blocking)
- **Text:** P2 transliteration → normalisation → name character 3-grams + address words, weighted 50/50. Hashed TF-IDF
  (2^18 / 2^20 buckets) fitted on train text only, so unseen French tokens get the maximum IDF instead of vanishing.
- **Search:** one sparse top-k product per country, from each Source 2/3 record to Source 1, within its state group. The
  state agrees on 99.94% of true pairs. Each record keeps its top 2 Source 1 entities (top 10 when its address is empty),
  and each Source 1 keeps its top 10 among the records that listed it.
- **Also added:** exact-key rule candidates (name core or website within state, address-confirmed).
- **Result:** pair recall **0.9916 at 14.2 candidates/Source 1** at full corpus density (India 0.9906, US 0.9923).
  27.4M test candidate pairs (15.8 per Source 1), built in about 35 minutes on a laptop.
- **Keeping true matches:** the recall of each blocker variant was measured against the whole train corpus, not a sample
  (a small world overstates recall by about 2 points). Records without an address get a wider search. The reverse direction
  recovers pairs that the forward direction misses.

## 4. Matching Model
**Features (65, all label-free, computed identically on train and test):**
- **Name:** normalised-core equality, token Jaccard, TF-IDF cosine, Jaro–Winkler / token-set ratios (rapidfuzz), acronym
  match, website core vs squashed name, transliterated variants.
- **Address:** house-number agreement, city key (frequency-picked, shuffle-proof), fuzzy place overlap, state agreement,
  `NULL` / empty flags. House-number (street) disagreement is the key negative signal: 92.5% of rule false positives are
  orphans with the same name core and matching city/state.
- **Context:** blocking score and ranks in both directions, the score gap to the best candidate, candidate-list size,
  name-chain size within the country, the rule stage's decision, and source (S2/S3).
- Country is **not** used as a raw category (France is unseen).

**Model type:** LightGBM (lr 0.05, 127 leaves, 4,930 trees). The final model is trained on train folds 2–5 (12.6M
candidate pairs) and early-stopped on fold 1. The fold-2-only model scored 0.9814; four folds give 0.9834 before reranking.
Ladder at full density:

| Model | F0.5 |
|---|---|
| Hand-weighted rule score | 0.826 |
| Fellegi–Sunter EM | 0.27 |
| Logistic regression | 0.933 |
| **LightGBM** | **0.9814** |

**Reranker:** multilingual-e5-small (MIT, 118M) fine-tuned as a cross-encoder on the LightGBM uncertain band of train
fold 3. On that band its AUC is 0.926, against 0.809 for the tree model. Blend: p′ = σ(0.678·logit p + 0.715·logit r
+ 0.338), fitted on one half of the holdout and evaluated on the other. It is applied to India and the US only:
trained on one country, it does not transfer to the other (−0.0016), so France keeps the tree model's p.

**Threshold selection:** the Source 2/3 record goes to its argmax Source 1; keep it if p ≥ t, with t swept for F0.5 on
held-out folds. The F0.5 curve is flat within 0.001 for t ∈ [0.57, 0.83]. France uses a label-free threshold: the one
whose predicted matches per Source 1 fit the expected profile. In leave-one-country-out tests it comes within 0.003 of the
oracle threshold.

## 5. Results & Error Analysis
- **F0.5 (macro, full-density holdout, report half):** **0.9885**, precision 0.9977, recall 0.9703, singleton accuracy 0.988
  (India 0.9894, US 0.9879). Two independent audits found no label leakage, and the threshold-tuning optimism is at most
  0.00003.
- **Ablation (same yardstick):**

  | System | F0.5 |
  |---|---|
  | Exact-key rules | 0.771 |
  | Logistic regression | 0.933 |
  | LightGBM, fold 2 | 0.9814 |
  | + cross-encoder | 0.9881 |
  | **LightGBM, folds 2–5 + cross-encoder** | **0.9885** |
  | P5's stacked decision layer on top | +0.0001 (not adopted) |
- **Common false positives:** 87% are orphan Source 2/3 records (no true parent among their candidates) that look like a
  real Source 1: same name core and city, a different street or house number.
- **Common false negatives:** pairs whose true Source 1 is not among the candidates (0.8%), heavy word-subset or DBA name
  changes, and records with no address.
- **Transfer to an unseen country:** a model trained on the US and applied to India scores 0.905, and India → US scores
  0.930, both with tuned thresholds. A label-free Fellegi–Sunter model scores 0.27. Tuning the threshold matters more than
  the model choice.

## 6. Conclusion
The data's one-to-many structure (one parent per Source 2/3 record) shaped every stage: reverse blocking, per-record
exclusive decoding, and competition features. Most of the gain over exact-key rules came from learning when a same-name
look-alike at a different street is not the same business, and from the cross-encoder on the uncertain band.
Measuring on the full corpus rather than a sample kept every decision honest: sampled evaluations overstated precision
by 8 points.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`:
- `src/run.py` is the entry point: `python src/run.py --split test --stage model --out output/`.
- Other modules: `io_tsv.py`, `translit.py` + `pairs.py`, `normalize.py`, `block.py`, `rules.py`, `model.py`,
  `decide.py`, `metric.py`, `worlds.py`.
- `test_pipeline.py` / `test_rules*.py` are the regression guards. `package.py` builds this zip.
- Every learned artefact (transliteration table, place frequencies, TF-IDF vectorisers, models) is rebuilt from the
  training data when absent. Nothing is downloaded except the MIT-licensed multilingual-e5-small weights.

### B. Models and licences
| Component | Model | Licence | Params |
|---|---|---|---|
| Matcher | LightGBM | MIT | n/a |
| Reranker | intfloat/multilingual-e5-small | MIT | 118M |
| Transliteration | learned token table (no model) | — | — |

No external data, APIs, geocoding or registries are used.
