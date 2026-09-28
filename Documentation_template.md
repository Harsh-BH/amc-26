# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Yeti
**Team Members:** Harsh Bhatt, Shriya Sinha, Aditya Kashyap, Anurag Pokhariyal (IIT Kharagpur)
**Submission Date:** 27 September 2026 (final leaderboard upload `sub90u`, 23:40 IST); write-up completed 29 September 2026

---

**Reading guide.**
- **`#n` / `subN`** is our n-th staged submission. The ones that were uploaded are listed with their public score in §5.2.
- **`Pnn`** is experiment nn of our research log. The experiments that produced lines of the final file ship as code in
  `code/business_entity_resolution/src/experiments/` (index: `experiments/INDEX.md`).
- **Holdout** means the full-corpus holdout (§2.1). **Report half** is hash-half 1 of it. Anything fitted on the holdout
  (blend weights, thresholds, stackers) was fitted on half 0 only, unless stated otherwise.
- **LB-eq** is a holdout delta weighted by each country's share of the leaderboard (India 0.468, US 0.383).
- **LOCO** means leave-one-country-out: train on the US, test on India, and the reverse. It stands in for the unlabelled,
  unseen country (France).
- Every Δ is a paired per-S1 difference ± 1 standard error.

---

## At a glance: evaluation criteria → evidence

| criterion | what we deliver | where to verify |
|---|---|---|
| Leaderboard score | public F0.5 **0.989382** (`sub90u`); +0.0094 over our first upload, every step measured (§5.2) | §5.1–5.2 |
| Candidate-set size and efficiency | **4.43 candidates per Source 1** (7,668,008 pairs), cut from 15.87 by a learned pruner for −0.00024 F0.5; recall 0.9917 → 0.9872; blocking in ~36 min at 2.7 GB RAM on a laptop | §3; `src/block.py`, `src/prune.py`; `src/verify_outputs.py` recomputes the count |
| Soundness of the approach | one-parent structure used end to end; every change passed a paired full-corpus holdout test (+0.0002 at ≥ 3 SE); failed ideas reported with their numbers | §2, §4, §5.4 |
| Unseen country (France) | label-free chain, chosen on LOCO proxies and France-only uploads; France found as "test label absent from train", never named in code | §4.5; `run.unseen()`, `src/france.py` |
| Model licences / size | every model MIT or Apache-2.0, ≤ 8B (largest 4.02B), run locally, fine-tuned on the provided data only | §7 table; `requirements.txt` header |
| No external data / fair play | no APIs, geocoding, registries, postal or city tables; hand-written lists only of the allowed kinds; third-party code credited | §7 |
| Output validity | one line per test S1 (all three countries); every match inside the candidate set; one owner per S2/S3; official validator `--check-ids` PASS | `src/verify_outputs.py`, `src/reproduce.sh` |
| Reproducibility | one command rebuilds both files and checks their md5 against the leaderboard upload; the recorded change sets rebuild byte for byte; their generating scripts ship in `src/experiments/` | §8; `src/reproduce.sh`; `src/experiments/INDEX.md` |
| Code quality | every module, function and class has a docstring; each module has a runnable self-check; end-to-end test with an injected unseen country | `src/test_pipeline.py`, `python src/<module>.py`, README §6 |

---

## 1. Executive Summary

- **The pipeline.** Reverse TF-IDF blocking, then a learned pruner, then a LightGBM matcher on 65 label-free features,
  then a fine-tuned bge-reranker-v2-m3 cross-encoder on the uncertain band. Then residual stackers and an exclusive
  one-parent decision.
- **Core idea: every Source 2/3 record has at most one true Source 1 parent.** Each stage works with that:
  - blocking runs from Source 2/3 to Source 1;
  - the features describe the competition among a record's candidates;
  - the decision gives each record to its single best Source 1.
- **France.** France is 15% of the test set and has no labels. It is handled label-free: its own-text IDF,
  generator-aware house-number reasoning, cross-encoders with out-of-domain blend weights, and structural rules
  validated by leave-one-country-out proxies and France-only leaderboard uploads.

| key figure | value |
|---|---|
| **Public leaderboard F0.5 (final entry `sub90u`)** | **0.989382** (first upload 0.980017; 17 uploads in all, §5.2) |
| Holdout F0.5, India + US, report half | 0.98984 at #34 (India 0.99020, US 0.98960); the final combiner adds **+0.000656 ± 0.000062** LB-eq on top (§4.3) |
| Rule baseline on the same holdout (exact keys) | 0.7708 |
| Candidate set fed to the matcher | **7,668,008 pairs = 4.43 per Source 1** (France 4.51 / India 4.75 / US 4.00), down from 15.87 after blocking. Holdout pair recall is 0.9917 after blocking and 0.9872 in the final set |
| Matches written | 5,848,223 pairs; 3.38 per Source 1 (France 3.31, India 3.38, US 3.39); 5.8% of Source 1 empty in each country |
| Models | LightGBM (MIT) throughout; cross-encoders bge-reranker-v2-m3 (Apache-2.0, 568M), Qwen3-Reranker-4B + LoRA (Apache-2.0, 4.02B), ms-marco-electra-base (Apache-2.0, 109M). All run locally, all ≤ 8B, all fine-tuned only on the provided data (§7) |
| Compute | one laptop (16 cores, 15 GB RAM, RTX 3050 6 GB, every step under a 3 GB RAM cap) plus rented AWS SageMaker instances for cross-encoder fine-tuning / scoring and two large LightGBM fits (§8) |

## 2. Methodology

### 2.1 Problem Analysis

- **Structure.**
  - Source 1 is deduplicated.
  - Every Source 2/3 record belongs to at most one Source 1 entity: the relation is strictly one-to-many.
  - 26% of Source 2/3 records match nothing (orphans / distractors).
  - 5.6% of Source 1 entities have no match. The other 94.4% have 1–5+ matches (mean 3.5). The training truth never has
    more than 5 Source 2 or 6 Source 3 matches per Source 1.
  - True pairs always share the country label.
- **Look-alike chains.**
  - 50.6% of Source 1 entities share their normalised name core with another Source 1 of the same country (France
    54.5%).
  - So names alone cannot decide a match. Name-only exact matching drops from 0.684 F0.5 on a small world to 0.431 at
    test density.
  - Address evidence (house number, street, city, state) is required.
- **Noise generator.** The data carries:
  - legal-suffix swaps (Pvt Ltd ↔ Private Limited), honorific and junk prefixes, `(ID: n)` tails;
  - alias constructs (f/k/a, dba), websites and handles used as names, digit-for-letter swaps, accent injection;
  - shuffled address segments, `NULL` placeholders, landmark phrases, and state code ↔ name swaps.
- **The generator's "siblings" (P15).**
  - A distractor sibling of a business sits at the Source 1's house number + d, with d ∈ {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}.
  - Noise on a true pair is instead a dropped/added digit or ±1/±2.
  - A matcher that only sees "house numbers equal / differ" cannot tell the two apart. A signed-delta feature can
    (§4.3).
- **Cross-script (India).**
  - About 23% of Indian Source 2 names are in Indic scripts, and 18% of Indian true pairs cross scripts.
  - The Indic names are token-for-token transliterations of the Latin Source 1 name, drawn from a closed 1,518-word
    vocabulary (no out-of-vocabulary words on test).
  - Indic text in addresses is only ever the state name.
- **Unseen country (France).**
  - France is 15% of test S1 and has no labels.
  - It is Latin with accents. Legal forms (SARL/SAS/EURL…) appear in 69% of Source 1 names, street words in 95% of
    addresses, and postcodes in almost none (0.4%).
  - It comes from the same noise generator as the US and India, but two properties shift:
    - **35% of France Source 2/3 addresses have no state** (training: 3%);
    - France's commonest address tokens ("rue", city and région names) never appear in the training text, so a
      train-fitted IDF gives them near-maximum weight (P14).
- **Evaluation yardstick.**
  - The holdout is train fold 0 of 10 (hashed by Source 1): 220,358 Source 1.
  - They are scored while **all** 2.2M train Source 1 and **all** 10.3M train Source 2/3 records compete, as on test.
  - Smaller "mini-world" evaluations under-sample distractors. They overstated rule precision by 8 points (0.994 vs
    0.912).
  - Every decision after 25 Sep 12:17 used the full-corpus yardstick with paired per-S1 standard errors.

### 2.2 Solution Strategy

**Approach Type:** Hybrid. Blocking → learned pruning (meta-blocking) → gradient-boosted classifier → cross-encoder
reranker → residual stackers over each record's competition → exclusive (many-to-one) decoding. The unseen country gets
label-free rules on top.

**Core Innovation:**
- **The one-parent structure drives every stage.** Blocking runs in reverse (Source 2/3 → Source 1). The features and
  stackers describe the competition among a record's candidates (ranks, margins, the runner-up, same-name "twin"
  Source 1). The decision is exclusive: each record goes to its best Source 1.
- **Generator-aware house-number reasoning.** A signed house-number delta, plus support counts among the Source 1's
  other records, separates the generator's sibling distractors from noisy true pairs.
- **Honest measurement, then keep only what moves the number.** Every change passed a paired test on the full-corpus
  holdout with a bar of about +0.0002 at ≥ 3 SE. Many ideas failed it (§5.4).
- **Unseen-country handling without labels.**
  - LOCO proxies (US→India, India→US) choose each France recipe.
  - Label-free profiles (predicted empty-S1 rate, matches per S1, the share of accepted pairs at a sibling offset) set
    France's thresholds.
  - A few leaderboard uploads that change only France calibrate it.
  - Test-side unsupervised statistics (IDF, word roles, synthetic pairs, pseudo-labels) are used. The organisers allow
    this.

**Pipeline at a glance** (final submission):

```
raw TSVs ──► transliteration (learned Indic→Latin word table) ──► normalisation (name core, legal form, house no., city, state)
        ──► BLOCKING: hashed TF-IDF top-k, S2/S3 → S1 within state  +  S1 → top-10 S2/S3  +  exact-key rule pairs
              27.5M pairs, 15.87 / S1, holdout recall 0.9917
        ──► PRUNER: LightGBM on 30 cheap features, keep q ≥ 0.03 or rule-accepted    8.37M pairs, 4.83 / S1
        ──► MATCHER m2_big: LightGBM, 65 features (pair + candidate-list context)
        ──► CROSS-ENCODER on the uncertain band 0.02 < p < 0.98 (bge-reranker-v2-m3, fine-tuned), logistic blend
        ──► India / US: US house-number stacker → bagged residual combiner over each record's argmax pair
              (competition, exclusivity mass, house-number corruption, CE on every argmax pair, 16 zoo models,
               generator traces + anchors) → t 0.71 → US owner re-decision with a cross-fitted stage C
        ──► France (unseen): own-text IDF re-blocking, rebuilt records, house-number-aware matcher, Qwen/bge CE blend
              with out-of-domain weights, rules at t 0.9868, decision-time completion rules, recorded change sets
        ──► one-owner assignment (caps 5 S2 / 6 S3), content-identical duplicates follow their copy
        ──► sibling-decoy candidate stage ──► matching_results.tsv + candidate_pairs.tsv (7.67M pairs, 4.43 / S1)
```

**How we worked.**
- Four people split the problem into numbered experiments (data analysis, transliteration, blocking, matching,
  decisions, reranking, France, pruning, …), each in its own git branch.
- Each experiment wrote its holdout result to a shared log, and nothing was merged without its number.
- About 90 experiments ran over the 72 hours. §5.4 lists the ones that failed.

## 3. Candidate Generation (Blocking)

- **Blocking keys used:**
  - **Hashed TF-IDF search.**
    - The text is transliteration → normalisation → name character 3-grams + address words, weighted 50/50.
    - The TF-IDF is hashed (2^18 / 2^20 buckets) and fitted on train text only. France gets its own IDF, which includes
      its unlabelled test text (§4.5).
    - One sparse top-k product (`sparse_dot_topn`) runs per country, from each Source 2/3 record to Source 1, within its
      state group. The state agrees on 99.94% of true pairs.
    - Each record keeps its top 2 Source 1, or its top 10 when its address is empty.
    - In the reverse direction, each Source 1 keeps its top 10 among the records that listed it.
  - **Exact-key rule pairs.** Name core or website key within the state, confirmed by address evidence (≥ 2 of house
    number / city / place).
  - **Acronym block (France, P62).** The Source 1 core's acronym equals the record's one-token core, with the same house
    number and city. It adds 2,638 candidates.
- **Learned pruner** (meta-blocking; added after the organisers' note that smaller candidate sets rank higher).
  - A 300-tree, 31-leaf LightGBM on 30 of the 65 pair features: blocking signals plus name/address similarities.
  - Trained on train folds 2–3 (6.3M pairs). It takes 74 s to train and 2–3 min to score all test pairs.
  - A pair reaches the matcher only if its pruner score q ≥ 0.03 or the rules accepted it.
  - The context features stay computed over the full blocking list, so a kept pair's matcher score is unchanged. The
    pruner only removes pairs.
- **Sibling-decoy stage** (the last cascade stage before the decision).
  - A pair at a generator sibling offset (hn_x − hn_1 ∈ {3, 4, 5, 7, 9, 11, 13, 21}) survives only if it is its
    record's best pair above t.
  - Matching is unchanged by construction.
- **Candidate pairs generated** (test, all three countries):

  | stage | pairs | per Source 1 (France / India / US) | holdout pair recall |
  |---|---|---|---|
  | blocking ∪ rule pairs | 27.5M | 15.87 (16.76 / 15.71 / 15.72) | 0.9917 (14.2 / S1 on the holdout) |
  | after the pruner (what the matcher and the cross-encoder score) | 8,370,573 | 4.83 (5.14 / 4.90 / 4.63) | 0.9874 (4.30 / S1) |
  | after the sibling-decoy stage | 7,658,102 | 4.42 (4.48 / 4.75 / 4.00) | 0.9872 |
  | **final `candidate_pairs.tsv`** (the above + the acronym block + every matched pair a later stage added) | **7,668,008** | **4.43 (4.51 / 4.75 / 4.00)** | — |

  - Every pair in `matching_results.tsv` is also in `candidate_pairs.tsv`.
  - The file is 121 MB; the unpruned list would be 377 MB.
- **How we ensured true matches were not lost:**
  - The recall of every blocker variant was measured against the whole train corpus, not a sample (a small world
    overstates recall by ~2 points).
  - Records without an address get a 5× wider search.
  - The reverse direction (each Source 1's top 10) recovers pairs the forward search misses.
  - Rule-accepted pairs always survive the pruner.
  - The pruner's operating point was chosen on the final F0.5, not on recall:

    | τ | cands / S1 | ΔF0.5 |
    |---|---|---|
    | 0.01 | 5.93 | −0.00002 |
    | 0.02 | 5.20 | −0.00009 |
    | **0.03** | **4.83** | **−0.00024 ± 0.00003** |
    | 0.05 | 4.51 | −0.00044 |

  - A per-Source 1 top-5 cap would have cost −0.021, because true matches sit deep inside same-name chains.
  - The leaderboard confirmed the pruner's cost: #6c vs D4 = −0.00019 (the holdout predicted −0.00024).
- **Efficiency.** On the laptop, test blocking takes 6.5 min of text prep plus 5 / 18 / 6 min (France / India / US) at
  2.7 GB peak RAM. The pruner scores all 27.5M test pairs in 3 / 18 / 5 min.

## 4. Matching Model

**Features used** (65, all label-free, computed identically on train and test; `features.py`):
- **Name features:**
  - normalised-core equality, token Jaccard, TF-IDF cosine;
  - Jaro–Winkler and token-sort / token-set ratios (rapidfuzz), acronym match;
  - website core vs squashed name, transliterated variants, an out-of-vocabulary-Indic flag.
- **Address features:**
  - house-number agreement / difference;
  - city key (frequency-picked, robust to shuffled segments), fuzzy place overlap, state agreement;
  - `NULL` / empty flags, TF-IDF address cosine.
  - House-number disagreement is the key negative signal: 92.5% of rule false positives are orphans with the same name
    core and matching city/state.
- **Other (context):**
  - the blocking score and ranks in both directions, and the score gap / margin to each side's best candidate;
  - candidate-list sizes, the name-chain size within the country, the rule stage's decision, and the source (S2/S3).
  - Country is **never** a feature, because France is unseen.

**Model type:** a cascade of learned stages. §4.1–4.3 describe India / US, §4.4 the decision and §4.5 France.

### 4.1 Matcher: LightGBM `m2_big`
- lr 0.05, 127 leaves, 4,930 trees.
- Trained on every candidate pair of train folds 2–5 (12.6M pairs, float16) and early-stopped on fold 1.
- Alone (argmax, no reranker) it scores 0.9834. The fold-2-only model scored 0.9814.
- Retraining it on folds 2–9 (25.1M pairs, P16) gave +0.00014 ± 0.00006 (2.6 SE). That is below the bar, so m2_big stays.

### 4.2 Cross-encoder: bge-reranker-v2-m3
- **Input.** Both records as "name | address" text.
- **Which pairs.** India/US pairs in the matcher's uncertain band, 0.02 < p < 0.98: 1.24M test pairs after the pruner.
- **Training.**
  - 778,520 uncertain-band pairs of train folds 4–9, plus a half-weight sample of easy pairs.
  - 2 epochs, bs 128, lr 2e-5, 500 warm-up steps, max_len 96, bf16 autocast.
  - One SageMaker ml.g5.xlarge (NVIDIA A10G 24 GB), 76 min (`src/bge_train.py`).
- **Blend.** p′ = σ(0.5901·logit p + 0.6319·logit r + 0.2973), a logistic stack fitted on holdout half 0.
- **Band AUC.** India 0.971 / US 0.956. For comparison: 0.955 / 0.934 for the earlier multilingual-e5-small CE (MIT,
  118M), and 0.809 for the tree model alone.
- **Gains.**
  - The first CE (e5-small) gained **+0.0066 ± 0.0002**. It fixed look-alike street mismatches (look-alike AUC
    0.82 → 0.98).
  - bge over e5-small: **+0.00059 ± 0.00007** (8 SE).
- **Why a bigger model helped only a little.** e5-base (2.4× the parameters) gained only +0.00008 ± 0.00006 over
  e5-small, and ensembles ≤ +0.00002. The step to bge was a better pretrained reranker, not raw size.

### 4.3 India / US: stackers, combiner, owner re-decision
1. **US house-number / support stacker (P15 → P17).**
   - A residual LightGBM (200 trees × 31 leaves, `init_score` = logit p′, so with no trees it returns p′ exactly).
   - Seven features:
     - the logit;
     - the house-number class: equal / sibling offset / digit drop-add / small / far;
     - the signed delta hx − h1 and house-number equality;
     - three support counts from the Source 1's other records.
   - US +0.00055 ± 0.00009 (India + US +0.00033 ± 0.00005).
   - It loses on India (−0.00013), where bge already resolves those pairs, so it runs on the US only.
2. **Residual stacker over each record's argmax pair (P23 → P34, "a10").**
   - It sees what pair-level models cannot: the runner-up's p and the margin to it, the Source 1's other confident pairs,
     "twin" candidate Source 1 (same exact name / same address / same legal-free core), a no-address flag and an
     error-category column.
   - Report half: **+0.00021 ± 0.00006** LB-eq (F0.5 0.98956 → 0.98984).
3. **Bagged combiner (P44 → P45 → P50, `iu_stack.py`).** 5 residual LightGBMs over the argmax pair's p. Their columns:
   - the a10 columns;
   - the exclusivity mass (the record's p summed over its candidate Source 1);
   - same-source group counts, Source 1 rank gaps and competitor counts;
   - house-number corruption features and two house-number GBDTs' logits;
   - the pruner logit;
   - **r_all**, the bge CE scored on *every* argmax pair, not only the band. It vetoes confident pairs the CE rejects and
     rescues low-p pairs it accepts;
   - **16 "zoo" models** as logits: LightGBM bags, GOSS, DART, extra-trees and feature-subset variants, CatBoost, two small
     MLPs, two ranking models.

   | step | holdout (India + US, report half) |
   |---|---|
   | combiner over #34 | LB-eq +0.000306 ± 0.000048 (6.4 SE) |
   | + r_all | **LB-eq +0.000656 ± 0.000062 (10.6 SE)** over #34; India +0.000813, US +0.000719 |
   | full-holdout refit (shipped, 5 seeds) | +0.000039 ± 0.000019 for 1.5× vs 1× the data |
   | + 16 zoo columns | +0.000025 ± 0.000029 (noise level; kept, harmless) |

4. **Generator-trace and anchor columns (P70, #86).** 12 more columns for the same combiner:
   - 7 generator traces:
     - the pair shares the exact house number + street + city key;
     - how many Source 1 sit at that key, and whether this one is alone;
     - the cores differ only in the last token;
     - another record of the same Source 1 and source has the identical address;
     - the name is a synthetic generator brand.
   - 5 "verified-merge anchor" similarities against the Source 1's other confident records.
   - Paired holdout: India +0.000046 ± 0.000033 / US +0.000056 ± 0.000030 LB-eq.
   - The idea is from the public repository credited in §7; the code is our own.
5. **US owner re-decision (P80, #90u).** A cross-fitted "stage C" re-decides which Source 1 owns each US record.
   - **Out-of-fold scores.** m2 is cross-fitted on folds {2,3} / {4,5}, and the bge CE is cross-fitted the same way on
     SageMaker, so every pair's p and r are out of sample.
   - **The stage-C LightGBM** has 34 competition columns: the pair's rank within its Source 1 and within its record,
     the gaps, the best competitor and the margin to it, the rival Source 1's anchors, and 6 CE columns.
   - **Decision.** The shipped combiner p is kept on the argmax pairs (blend weight 1.0). Stage C scores only the pairs
     the combiner does not cover (non-argmax pairs and rivals). Each record then goes to its best Source 1, t 0.72, caps
     5 / 6.
   - **Holdout (US fold 0, 132,190 S1):** **+0.000287 ± 0.000033** (half 0 +0.000301, half 1 +0.000274; tuned on one
     half and reported on the other: +0.000216 / +0.000301).
   - **Test:** 2,706 US Source 1 change (+2,326 / −424 pairs).
6. **Duplicates (P54, last).** An unassigned content-identical copy (case-folded name + address + country) of an accepted
   record goes to the same Source 1. In training truth all 60,319 such groups have a single owner.

### 4.4 Threshold selection method
- Each Source 2/3 record goes to its argmax Source 1 among its kept candidates, and the pair is kept if p ≥ t.
- India / US: t = 0.71 on the combiner's p. It was swept for F0.5 on holdout half 0 and reported on half 1. The curve is
  flat within 0.001 over t ∈ [0.57, 0.83]. The US re-decision uses t 0.72 (above).
- Caps: at most 5 Source 2 / 6 Source 3 matches per Source 1, highest p first (the training maximum).
- A richer decision layer (P5: a stacked per-S1 decision, top-k rules) gained +0.0010 on the bare matcher but only
  +0.0001 after the cross-encoder, so plain argmax + t stayed until the combiner (§4.3) replaced it.
- **France:** a fixed stricter t, set label-free and validated on the leaderboard (§4.5). The organisers allow "a
  stricter threshold for countries unseen in training". The code defines the unseen countries that way: `run.unseen()`
  returns the test Source 1 labels absent from train Source 1. No country name appears in a branch.

### 4.5 France: the unseen country

**Threshold.**
1. **The first rule (P6).** Pick the t at which France's predicted empty-S1 rate equals the labelled countries' rate. In
   LOCO it matched the oracle threshold (US→India 0.9054). On France it gave t 0.78.
2. **The leaderboard showed France over-matching.** Four uploads changed *only* France, on the same India/US files:

   | upload | France | public LB |
   |---|---|---|
   | #5 | P6's rule, t 0.78 | 0.980017 |
   | D1 | France emptied | 0.849163 |
   | D2 | t 0.88 | 0.981024 |
   | D4 | t 0.95 | **0.981142** |

   - D1 isolates France: France F0.5 ≈ 0.93, while India + US ≈ 0.989, the holdout's value. So the whole
     holdout-vs-leaderboard gap was France.
   - France accepted 22 sibling-offset pairs per 1,000 at t 0.78, against 2.6 (US) and 3.9 (India). Raising t mostly
     removes those.
3. **Rule for later variants.** Any France variant that changes France's p gets the t at which France's empty-S1 rate
   equals the leaderboard-validated profile. That gave **t 0.9868** for the final chain.

**Scorer and rules** (`france.py`, recipe `sub66_final`). Every step is label-free and touches France only.

| step | what | France test effect | evidence |
|---|---|---|---|
| own-text IDF (P14) | France is re-blocked and re-featurised with its own S2/S3 text in the IDF, at its 14.4% test share | band shrinks 11.9 → 9.5% of argmax pairs | India proxy: a train-only IDF costs −1.23 pt; own-text IDF recovers 96% of it |
| rebuild (P29, P20) | house numbers re-parsed where a segment opens with `(n)` (`(520) RUE X`); the generator's French name fillers (`fils, cie, associes, freres, frs`) leave the name core | 23,250 records gain or change a house number | fillers keep the S1's house number 81–88% of the time, and take a sibling offset only 2–5% |
| c_hn matcher (P26) | m2_big's recipe + 5 house-number features over each S1's full candidate list (class, signed delta, other records sharing this number, their best name cosine, other records carrying the S1's number); fitted on India/US | 1,332,223 kept pairs, 5.13 / S1 | LOCO +1.8–2.0 pt over m2_big |
| CE band blend (P13, P38) | Qwen3-Reranker-4B + LoRA (fine-tuned on India + US) blended with **out-of-domain** weights [0.8943, 0.3729, −0.0521]: each LOCO model's blend fitted on the *other* direction's target, because in-domain weights over-trust the CE on an unseen country (LOCO −0.0016) | 6,885 lines vs the bge blend | LOCO Qwen − bge **+0.00141 ± 0.00017 / +0.00109 ± 0.00014** |
| key tier (P29) | argmax pair with equal core, house number and city, the core unique among France S1, p ≥ 0.5 → p := 1 | +6,088 accepted | — |
| bxneg (P28) | equal house number, name equal or one non-sibling word inserted/dropped, shared place and street → p := max(p, 0.99) | +9,647 accepted | — |
| CE veto (P30) | a pair the matcher is sure of (p ≥ 0.98) that bge scores below 0.1 loses it, unless a rule above raised it | −6,141 accepted | LOCO +0.0283 / +0.0056 |
| cap (P29) | an SX that would give an S1 more than 5 S2 / 6 S3 matches loses the pair | −21 | train maximum |
| decision | argmax + t 0.9868 | 6.46% empty S1 | leaderboard (#31: 0.9856) |
| empty-S1 rescue (#34v) | an S1 left empty takes its best argmax candidate if p ≥ 0.90 | +1,205 S1 | leaderboard +0.00031 |
| group completion (P37) | an S1 whose group holds only S2 or only S3 records takes free argmax candidates of the missing source with p ≥ 0.70 (in training 81% of S1 hold both sources; France predictions were one-sided for 20.7%) | +10,084 pairs | leaderboard +0.00034; LOCO +0.00082 / +0.00204 |
| P55 band veto | a LightGBM on country-invariant inputs (p and CE densities, competitor mass and gap, group counts/ranks, house-number relation), fitted on both LOCO frames; band pairs with c < 0.6 leave | −619 | LOCO +0.0052 / +0.0046 |
| acronym rule (P62) | the acronym block above + address check (token Jaccard ≥ 0.6, every S1 street token present) | +3,538 | train precision 0.997 / 0.9997 |
| legal-form veto (P64) | both names carry a legal form and the forms are disjoint (SARL vs SAS, SCI vs SNC) → the pair leaves | −344 | LOCO +0.00300 / +0.00074 |
| address-less exact names (P66) | an argmax record with an empty address whose full name equals its S1's, a name no other France S1 carries, p ≥ 0.30 | +1,884 | train precision 0.996 / 0.987 |

That is `sub66_final` (public LB 0.987391). The remaining France gains came from these change sets, applied in order:

| change set | rule | France test | evidence |
|---|---|---|---|
| P67q synthetic-French CE | a bge CE fine-tuned on synthetic pairs generated from France's own S1 records with the generator's typo / swap templates (no labels), as an extra column of P55's combiner; adds kept where the Qwen CE gives r ≥ 0.5 | +2,793 | LOCO +0.00308 / +0.00437; an untuned-CE control gives +0.00006 / +0.00081, so the gain is the synthetic tuning |
| P70a word roles | P55's combiner + 3 label-free columns: the sibling-offset / same-house-number share of the swapped, added or dropped word over France's own argmax pairs | +424 | LOCO +0.00081 / +0.00233 |
| P69 CE-gated looser t | pairs between t 0.975 and 0.9868 whose bge r ≥ 0.99 | +4,904 | a plain t 0.975 loses −0.0012 to −0.0028; the CE gate makes it pay |
| N54 decoy words (P74) | pairs whose names differ by one organisation word that France's own argmax pairs mark as a decoy ('X Sportive' → 'X Comite') leave | −2,120 | LOCO positive in 8 of 8 readings; removed pairs 0.3–3.6% true |
| P71a noise-word swaps | same house number, the street contains the S1's, one candidate S1, and the name differs by a swap into a noise word the labelled countries show is harmless ('developpement', 'associes') | +4,523 | the US/India analogue classes are 99.8% true |
| P41 synthetic brands | the generator renames some records to a brand glued from a closed syllable set (44 syllables, learned label-free from the test names) and keeps the address; such a record joins the only S1 with its exact house number + street + city, p ≥ 0.30 | +5,824 | train analogue 99.8% (US) / 96.1% (India) right |
| → `sub77` | | | **LB 0.988292 (+0.000901)** |
| legal-form-only names (P70 xlf) | an address-less record whose core equals a unique S1 core and whose full name differs only by the legal form | +2,019 | train precision 0.95–0.99 |
| 'groupe' swaps (P71b) | identical house number, one S1 word replaced by 'groupe' | +4,062 | US analogue 0.87 |
| France-independent CE (P73 / P79) | an ms-marco-electra-base CE fine-tuned on synthetic France pairs + confident France pseudo-labels (no India/US labels); its adds gated by Qwen r ≥ 0.5, plus its vetoes | +167 / −159 | LOCO +0.00175 / +0.00201 |
| two-member frames (P76 R3) | France S1 with exactly 2 argmax records lose base-stage pairs with p in [0.9868, 0.9925), keeping a source's last member | −263 | all 8 LOCO readings positive; beats 8 permuted-label nulls |
| legal-form-kept address-less names (P80) | an address-less record that keeps its legal form and whose full name matches exactly one France S1 | +1,473 | holdout pool precision 0.92 (India) / 0.91 (US) |
| type-word veto, never-candidate keys, never-candidate swaps | −132 broad type-word swaps (compagnie / institut / pharmacie); +814 (core, house number, city)-unique pairs the blocker never proposed; +661 same-house-number noise-suffix swaps the blocker never proposed | −132 / +814 / +661 | lens + India analogue 0.85 |
| narrow CE veto (P73) | +1/+2 / sibling-step house-number pairs with a France-only electra CE logit < −4.5, never emptying a line | −65 | LOCO +0.00210 / +0.00001 |
| → `sub84` | | | **LB 0.988475 (+0.000183)** |
| same-site ADD rules (P81 → P88, passes 1–2) | post-hoc ADD rules from a public MIT repository (§7), run on our file; **our filter** kept only adds at the same street or with an address-less record, and dropped re-adds of pairs our own vetoes had removed and different/typo-street pairs (that class had measured ≈ 0.3 precise on the LB with `sub77_hnst`) | +10,689 (same street 7,895, address-less 2,537, type-only 179, street-less S1 78) | with #86: **LB 0.989023 (+0.000548)**, i.e. the kept adds are ≈ 0.95 precise |
| same filter, pass 3 | the repository's third pass, same filter | +5,429 | with #90u: LB 0.989382 |

Final France profile: 3.31 matches per Source 1, 5.82% empty. India / US: 3.38 / 3.39 and 5.8%.

## 5. Results & Error Analysis

- **F_0.5 Score (macro), full-corpus holdout, report half: 0.98984** (India 0.99020, US 0.98960) at #34.
  - The shipped combiner adds +0.000656 ± 0.000062 LB-eq on top, and the US owner re-decision +0.000287 ± 0.000033 on US
    fold 0. The final boosters were refit on the whole holdout, so the report-half figure is the last clean one.
  - Without the stackers (#8c): 0.98990 on all 220,358 Source 1; precision 0.9983, recall 0.9726.
  - Two independent audits found no label leakage. The threshold-tuning optimism is at most 0.00003.
- **Public leaderboard: 0.989382** (final entry `sub90u`).

### 5.1 Ablation: every step on the same full-corpus holdout

Δ is the paired per-S1 difference on the report half ± 1 SE, against the previous row.

| step (submission) | holdout, all S1 | report half | Δ | experiment |
|---|---|---|---|---|
| exact normalised name + first address segment (#1) | 0.1775 | — | — | #1 |
| exact-key rules: transliteration → normaliser → name / website key + address evidence (#2) | 0.7708 | — | — | P1, P2, P8 |
| hand-weighted rule score | 0.8257 | — | — | P4 |
| Fellegi–Sunter EM (label-free) | 0.2695 | — | — | P4 |
| logistic regression, 65 features | 0.9334 | — | — | P4 |
| LightGBM, fold 2 (#3) | 0.9814 | — | +21 pts vs rules | P4 |
| + e5-small cross-encoder, India/US (#4) | 0.9884 | 0.9882 | **+0.0066 ± 0.0002** | P4b |
| matcher on folds 2–5, m2_big (#5) | 0.9888 | 0.9885 | +0.00040 ± 0.00006 | P9 |
| + learned pruner, 15.87 → 4.83 cands/S1 (#6) | 0.9885 | 0.9883 | −0.00024 ± 0.00003 | P10 |
| e5-small CE trained 2 epochs (#7) | 0.98881 | 0.98869 | +0.00039 ± 0.00006 | P11 |
| bge-reranker-v2-m3 CE instead of e5 (#8a) | — | — | **+0.00059 ± 0.00007** | P13 |
| + US house-number / support stacker (#8c) | **0.98990** | 0.98956 | +0.00033 ± 0.00005 (US +0.00055) | P15, P17 |
| + a10 residual stacker, India + US (#34) | — (half 0 in-sample) | **0.98984** | +0.00021 ± 0.00006 LB-eq | P23, P34 |
| combiner + r_all instead of a10 (#44r) | — | — | **+0.000656 ± 0.000062 LB-eq (10.6 SE)** | P44, P45 |
| + full-holdout refit + 16 zoo columns (#52) | — (fit on every holdout S1) | — | +0.000039 ± 0.000019 / +0.000025 ± 0.000029 | P50 |
| + generator traces + anchors (#86) | — | — | India +0.000046 ± 0.000033 / US +0.000056 ± 0.000030 LB-eq | P70 |
| + US owner re-decision (#90u) | — | — | US fold 0 +0.000287 ± 0.000033 | P80 |

France cannot be scored on labels. Its changes were chosen on LOCO proxies and label-free profiles (§4.5), then tested
on the leaderboard.

### 5.2 Every leaderboard upload (17)

| upload (IST) | what it is | public LB F0.5 | reading |
|---|---|---|---|
| #5, 25 Sep 18:34 | m2_big + e5 CE; France t 0.78 | 0.980017 (rank 33) | holdout 0.9885, so a 0.0085 gap |
| D1, 21:43 | #5 with France emptied | 0.849163 | France ≈ 0.93, India + US ≈ 0.989: the gap is France |
| D2, 21:46 | #5 with France t 0.88 | 0.981024 | France over-matches |
| D4, 23:02 | #5 with France t 0.95 | 0.981142 | curve flattening |
| #6c, 23:36 | + pruner (4.83 cands/S1) | 0.980954 | −0.00019 = the pruner's cost (holdout predicted −0.00024) |
| #9ab, 26 Sep 14:31 | bge CE + US stacker; France own-text IDF + bge, t 0.985 | 0.984072 | +0.0029 |
| #31, 18:04 | France chain: rebuild + c_hn + CE + tier / bxneg / cap, t 0.9868 | 0.9856 | +0.0015 |
| #34_veto, 22:54 | + a10 stacker (India / US) + the France CE veto | 0.985647 | ≈ flat |
| #34v_r90, 23:00 | + France empty-S1 rescue | 0.98596 | +0.00031 |
| #37_gc70, 23:32 | + France group completion | 0.986304 | +0.00034 |
| #54, 27 Sep 05:32 | India / US combiner; France chain with two rules later dropped | 0.986685 | +0.00038 (a quarter of the expected gain; the two rules were audited and dropped) |
| `sub66_final`, 14:22 | combiner India / US + #37's France + Qwen CE, P55 veto, acronym, legal-form veto, address-less names | 0.987391 | +0.000706 |
| `sub77`, 16:56 | + six France change sets (§4.5) | 0.988292 | +0.000901 |
| `sub77_hnst`, 18:39 | + 3,756 France pairs at the same house number but a *different* street | 0.987844 | **−0.000448**: that class is ≈ 0.3 precise in France, never re-added |
| `sub84`, 20:02 | + nine evening France change sets | 0.988475 | +0.000183 |
| `sub88`, 22:32 | + trace/anchor combiner (India / US) + 10,689 filtered France same-site adds | 0.989023 | +0.000548 |
| **`sub90u`, 23:40** | + 5,429 pass-3 France adds + US owner re-decision | **0.989382** | **+0.000359, the final entry** |

Every France test sat on unchanged India/US files, so each of those leaderboard deltas measures one France change.

### 5.3 Errors
- **Common false positives (wrong merges):**
  - 87% are orphan Source 2/3 records (no true parent among their candidates) that look like a real Source 1: the same
    name core and city, but a different street or house number.
  - Many are the generator's house-number siblings (+3, +5, +7…). The US stacker, the decoy-candidate stage and
    France's strict threshold target them.
  - In France, over-weighted common address tokens made unrelated same-city records look alike (accepted-pair score
    margin 0.22 vs 0.42–0.44 in the US/India). The own-text IDF fixes this.
- **Common false negatives (missed matches):**
  - Pairs whose true Source 1 is not among the candidates: 0.8% after blocking, 1.3% after the pruner.
  - Heavy word-subset or DBA name changes, and records with no address.
  - True pairs with house-number noise that a strict threshold rejects. France's accepted house-number-mismatch share is
    1.7%, against 8.8% in the US, so recall is what France pays for its strict t.
- **Transfer to an unseen country** (LOCO, tuned thresholds):
  - US → India scores 0.905 and India → US 0.930, against ~0.98 in-country.
  - At the source country's threshold, US → India loses another 4 points. That is why France's threshold matters more
    than its model.

### 5.4 What failed (and the number that killed it)

| idea | result | why dropped |
|---|---|---|
| Fellegi–Sunter EM, label-free (P4) | 0.2695 | far below a trained model |
| neural transliterator IndicXlit / multilingual embeddings for Indic names (P2) | 93% / 69% exact vs the learned table's 99.6% | worse and slower |
| two-stage matcher, stage-1 competition as stage-2 features (P4) | +0.0008 before the CE | the CE absorbs most of it |
| stacked decision layer (P5) | +0.0010 on the bare matcher, +0.0001 after the CE | too small once the CE was in |
| cross-encoder with in-domain blend weights for France (P4b, P13) | LOCO −0.0016 | over-trusts the CE on an unseen country; out-of-domain weights fix it |
| bge zero-shot, no fine-tuning (P13) | band AUC 0.56 / 0.64 | does not know "same business" on near-duplicates |
| more CE capacity: e5-base, ensembles (P11, P13) | e5-base +0.00008 ± 0.00006; ensembles ≤ +0.00002 | capacity is not the bottleneck |
| CE trained on m2_big's own band (P11) | −0.00016 ± 0.00006 | learns what the matcher already knows |
| synthetic France-like training pairs for the matcher (P12) | US→India −0.8 pt, India→US +0.2 pt | needs the target's noise rates, which a label-free method cannot measure |
| matcher on 8 folds (P16) | +0.00014 ± 0.00006 (2.6 SE) | below the bar |
| house-number stacker on India / France (P17) | India −0.00013; France reshuffles 4.7–8.1% of S1 | bge already resolves India; France needed stricter, not looser |
| hard sibling rule on India (P15) | −0.0017 | India's true-pair house-number noise is wider |
| per-S1 top-k cap / cheaper blocking (P10) | top-5: −0.021; a cheaper config: 7.80 cands/S1 at −0.00031 | the learned pruner dominates both |
| empty-S1 rescue on India / US (P35) | −0.00001 to −0.00038 | the labelled countries' empties are mostly true singletons |
| France noise-word tier (P46) and address-less exact names at p ≥ 0.70 (P40) | a Qwen audit estimated their precision at 0.08 / 0.19; #54 scored a quarter of its expected gain | dropped from the final |
| different-street same-house-number France pairs (`sub77_hnst`) | LB −0.000448 | ≈ 0.3 precise in France |
| a rival-style stage C *replacing* our decision (P80) | −0.00585 ± 0.00012 LB-eq | without the CE it recovers only 18% of the gap; kept only as the owner re-decision around our p |
| the public repository's DROP rules (P81) | India −0.0047 / US −0.0097 on our holdout | only its ADD rules were kept, filtered |

## 6. Conclusion

- The data's one-to-many structure (one parent per Source 2/3 record) shaped every stage: reverse blocking, competition
  features, exclusive decoding, and the owner re-decision at the end.
- Most of the gain over exact-key rules (+21 points) came from learning when a same-name look-alike at a different
  street is *not* the same business. The cross-encoder on the uncertain band and the generator-aware house-number
  stages added the rest.
- A learned pruner cut the candidate set by 70% for 0.0002 F0.5. The final file holds 4.43 candidates per Source 1.
- Measuring on the full corpus rather than a sample kept decisions honest: sampled evaluations overstated precision by 8
  points.
- The unseen country was the real challenge. LOCO proxies, label-free profiles and France-only leaderboard uploads found
  the gap, and the threshold was the biggest single lever. The last gains came from the structure of the answer itself:
  - an empty Source 1 is rarely right;
  - a group holding only one source is usually missing its other half;
  - a generator-built record keeps its business's exact site.

---

## 7. Compliance and fair play

**Models and licences** (every model runs locally and was fine-tuned only on the provided training data or on
unlabelled test rows, which the organisers allow):

| component | model | licence | size |
|---|---|---|---|
| matcher m2_big, pruner, stackers, combiners, stage C, France c_hn / P55 | LightGBM 4.7 | MIT | ≤ 4,930 trees × 127 leaves |
| main cross-encoder (India / US band, r_all, France veto; P67's synthetic-French copy; P80's cross-fitted copies) | BAAI/bge-reranker-v2-m3, fine-tuned | Apache-2.0 | 568M |
| France CE band blend and gate (P38, P67q, P79q) | Qwen/Qwen3-Reranker-4B + LoRA, fine-tuned on India + US | Apache-2.0 | 4.02B (≤ 8B) |
| France-independent CE (P73 / P79, narrow veto) | cross-encoder/ms-marco-electra-base, fine-tuned on synthetic France pairs + pseudo-labels | Apache-2.0 | 109M |
| earlier cross-encoder (#4–#7, holdout fallback) | intfloat/multilingual-e5-small | MIT | 118M |
| zoo members of the combiner | LightGBM / CatBoost / small torch MLPs | MIT / Apache-2.0 / BSD-3 | small |
| transliteration | learned token table (no model) | — | 1,518-word vocabulary |

- **No model > 8B anywhere**, not even as a teacher.
- **No external data or lookup.** No APIs, geocoding, business registries, postal or city tables, libpostal, cleanco, or
  hosted LLM / embedding services. The only hand-written lists, all of the kinds the organisers allow:
  - legal suffixes and street types;
  - state names and codes (US, India, and the 13 French régions).
  - France's département → région map is not a table: `normalize.fr_departements` learns it from confident pairs among
    the unlabelled test rows (4 départements).
- **Pretrained weights** were downloaded once from Hugging Face. After that the pipeline runs with `HF_HUB_OFFLINE=1`.
- **Test-side use** is limited to the kinds the organisers allow: unsupervised statistics (IDF, word-role shares,
  generator syllables), synthetic pairs built from the provided France records, and pseudo-labels from our own models.
- **Unseen country.** France is found as "a test label absent from train" (`run.unseen()`). Every France rule keys on
  that set, never on the name.
- **Compute.** AWS SageMaker was used only as rented compute for our own code and data. The bge fine-tuning cost $4.70.
  No hosted model or API is in the resolution path.
- **Third-party code, credited.**
  - The France same-site ADD rules of §4.5 (+16,118 pairs) come from the public, MIT-licensed repository
    [PundarikakshNTripathi/present-day-ber-amlc-2026](https://github.com/PundarikakshNTripathi/present-day-ber-amlc-2026).
  - We ran its post-hoc passes on our own submission file, with the scripts unmodified and the rule subsets chosen on
    our holdout (`experiments/P81-rival-posthoc/chain.sh`). We kept only the ADD changes that passed our P88 filter
    (`experiments/P88-p81-graft/`). Its DROP rules failed our holdout and were not used.
  - Three ideas from the same repository were re-implemented in our own code and are credited where used: generator
    traces, verified-merge anchors, and a stage-C owner re-decision.

## 8. Reproducibility

- **One command:** `bash code/business_entity_resolution/src/reproduce.sh` runs the three steps below, then
  `verify_outputs.py` (the invariants and headline numbers), the md5 check against the leaderboard upload, and the
  official validator with `--check-ids`.
- **Regenerate the files** (repository root; `code/business_entity_resolution/README.md` has setup, caches and
  runtimes):
  ```bash
  python code/business_entity_resolution/src/run.py --split test --stage model --recipe sub66_final \
      --drop-sibling-cands --out output/sub66_final_small                         # the pipeline, raw data -> sub66_final
  python code/business_entity_resolution/src/final_build.py output/sub66_final_small output/sub77 --upto 41   # France change sets -> sub77
  python code/business_entity_resolution/src/final_patch.py output/sub77 output/sub90u                        # evening change sets -> sub90u
  ```
- **Byte-for-byte check.** `output/matching_results.tsv` md5 `a8ad38b59e75013bcb53ec13578b5529` and
  `output/candidate_pairs.tsv` md5 `f311dc4d44e4587ece2e74541072c2e2` are the leaderboard file of `sub90u`. The two
  post-model steps rebuild it exactly.
- **What is code and what is recorded.**
  - `run.py` regenerates `sub66_final` from the raw TSVs.
  - The later change sets ship as their recorded rows: `src/final_changes.parquet` (22,562 rows) and
    `src/final_patch.parquet` (lines that differ from `sub77`). Each came from an experiment's own GPU model or frame.
  - The scripts that produced them are in `src/experiments/`, and `experiments/INDEX.md` maps each change set to its
    script.
  - Recording them keeps the final file exactly reproducible without re-running those GPU jobs.
- **Hardware.**
  - One laptop (16 cores, 15 GB RAM, RTX 3050 6 GB); every step fits a 3 GB memory cap.
  - Rented AWS SageMaker instances: GPUs (mainly A10G) to fine-tune and score the cross-encoders, and CPUs for two
    large LightGBM fits.
- **Runtimes.**
  - A cold test run with trained models and cached CE scores takes ~2.5 h: blocking ~36 min, features ~50 min, matcher
    scoring ~30 min.
  - Scoring the bge band fresh adds ~20 min on an A10G.
  - Rebuilding every LightGBM from the training data takes ~5.5–6 h.
- **Tests.** `test_pipeline.py` runs end to end, including an injected unseen country. Each module has a self-check
  (`python src/<module>.py`). `package.py --selftest` checks the zip layout.

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`: `README.md` (exact commands, caches, runtimes), `requirements.txt` (pinned
versions; every library is MIT / Apache-2.0 / BSD / ISC), `src/`.

- **Entry points:** `src/run.py` (the pipeline; `--split test|holdout|dev`, `--recipe` selects any earlier submission),
  then `src/final_build.py` and `src/final_patch.py` (the recorded change sets). `src/reproduce.sh` chains them with
  every check; `src/verify_outputs.py` recomputes the headline numbers from the TSVs; `src/package.py` builds this zip.
- **Modules, in pipeline order:**
  - `io_tsv.py`: TSV ↔ parquet with `QUOTE_NONE`.
  - `worlds.py`: folds and evaluation worlds.
  - `pairs.py`, `translit.py`: the learned Indic → Latin table.
  - `normalize.py`: the name / address normaliser.
  - `rules.py`: the exact-key rule stage. `s0_exact.py`: submission #1.
  - `block.py`: TF-IDF blocking, including France's IDF variants.
  - `features.py`: the 65 pair features.
  - `prune.py`: the learned pruner.
  - `train_matcher.py`, `model.py`: m2_big and c_hn.
  - `rerank.py`: CE scoring and the blend. `bge_train.py`: the GPU fine-tuning job.
  - `stack.py`: the US house-number stacker and the sibling rule.
  - `a10.py`, `iu_stack.py`: the India / US stacker and combiner.
  - `france.py`: the unseen-country chain.
  - `submit_probs.py`: scored pairs → TSVs.
  - `metric.py`: macro F0.5.
  - `experiments/`: the scripts behind the recorded change sets.
- **Artefacts not in the zip** (size):
  - trained boosters and CE checkpoints;
  - the cached CE scores.
  - The LightGBM models are rebuilt from the training data when absent. The CE checkpoints come from `bge_train.py` on a
    GPU. We can provide every file on request.

### B. Additional Results

**Label-free France profile at each operating point** (test, argmax per Source 2/3):

| France scorer / t | empty S1 | matches per S1 |
|---|---|---|
| m2_big, t 0.78 (#5) | 5.69% | 3.24 |
| m2_big, t 0.95 (D4) | 6.85% | 2.99 |
| the chain, t 0.9868 (#31) | 6.46% | 3.12 |
| + veto + rescue + group completion (#37) | 6.05% | 3.14 |
| `sub77` | — | 3.22 |
| **`sub90u` (final)** | **5.82%** | **3.31** |
| US / India, final (for reference) | 5.8% | 3.39 / 3.38 |

**Label-free France checks.**
- The leaderboard tracked the India→US LOCO direction, or did better, on every measured France addition.
- France's addresses behave like US ones: 92% of its accepted pairs share the house number.
