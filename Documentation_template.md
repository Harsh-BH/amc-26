# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Yeti
**Team Members:** Harsh Bhatt, Shriya Sinha, Aditya Kashyap, Anurag Pokhariyal (IIT Kharagpur)
**Submission Date:** 27 September 2026

> Every number below comes from a row in `ledger/*.jsonl`, a probe report (`probes/*/RESULT.md`) or the merge memos
> `probes/MERGE-*.md`. "Holdout" always means the **full-corpus holdout** (defined in 2.1). "Report half" means
> hash-half 1 of the holdout; anything fitted on the holdout (blend weights, thresholds, the US stacker) was fitted on
> half 0 only.

---

## 1. Executive Summary

The pipeline has six stages. Every model in it runs locally, is MIT or Apache-2.0, and is at most 568M parameters.

1. **Transliteration and normalisation.** Indic-script names are mapped back to Latin with a word table learned from the
   training pairs. Names and addresses are then cleaned (legal forms, junk prefixes, `NULL`s, house number, city, state).
2. **Blocking.** A state-partitioned, hashed TF-IDF search runs from each Source 2/3 record to Source 1. It keeps 99.2%
   of true pairs at ~16 candidates per Source 1.
3. **Learned pruner** (meta-blocking). A small LightGBM drops the hopeless pairs, which cuts the candidate set to
   **4.83 per Source 1** (−70%) at a cost of 0.0002 F0.5.
4. **Matcher.** LightGBM (`m2_big`) on 65 label-free pair and candidate-list features.
5. **Cross-encoder** on the matcher's uncertain band. It is **bge-reranker-v2-m3** (Apache-2.0, 568M), fine-tuned on an
   AWS SageMaker A10G GPU. It replaced an earlier multilingual-e5-small cross-encoder.
6. **US house-number stacker, then the decision.** A residual LightGBM for the US knows the data generator's
   "sibling" house-number offsets. Then each Source 2/3 record goes to its single best Source 1, kept if p ≥ t.

**France** is 15% of the test set and has no labels. It uses the same matcher with a label-free threshold. The operating
point was validated with four diagnostic leaderboard uploads.

- **Holdout F0.5: 0.98990** (report half 0.98956), against 0.7708 for an exact-key rule baseline.
- **Best public leaderboard so far: 0.981142.**
- The whole gap between holdout and leaderboard is France: the leaderboard implies India + US ≈ 0.989 and France ≈ 0.93.

## 2. Methodology

### 2.1 Problem Analysis

- **Structure.**
  - Source 1 is deduplicated.
  - Every Source 2/3 record belongs to at most one Source 1 entity: the relation is strictly one-to-many.
  - 26% of Source 2/3 records match nothing (orphans / distractors).
  - 5.6% of Source 1 entities have no match. The remaining 94.4% have 1–5+ matches (mean 3.5).
- **Look-alike chains.** 50.6% of Source 1 entities share their normalised name core with another Source 1 of the same
  country (France 54.5%). Names alone cannot decide a match: name-only exact matching drops from 0.684 F0.5 on a small
  world to 0.431 at test density. Address evidence (house number, city, state) is required.
- **Noise generator.** The data carries:
  - legal-suffix swaps (Pvt Ltd ↔ Private Limited), honorific and junk prefixes, `(ID: n)` tails;
  - alias constructs (f/k/a, dba), websites and handles as names, digit-for-letter swaps, accent injection;
  - shuffled address segments, `NULL` placeholders, landmark phrases, and state code ↔ name swaps.
- **The generator's siblings** (found late, P15). An orphan "sibling" of a business sits at the Source 1's house
  number + d, with d ∈ {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}. Noise on a true pair is instead a dropped/added digit or ±1/±2.
  - The matcher only saw "house numbers equal / differ", so it could not tell the two apart.
  - A signed-delta feature can (section 4.4).
- **Cross-script (India).**
  - About 23% of Indian Source 2 names are in Indic scripts, and 18% of Indian true pairs cross scripts.
  - The Indic names are token-for-token transliterations of the Latin Source 1 name, drawn from a closed 1,518-word
    vocabulary (no out-of-vocabulary types on test).
  - Indic text in addresses is only ever the state name.
- **Unseen country (France).**
  - France is 15% of test S1 and has no labels.
  - It is Latin with accents. Legal forms (SARL/SAS/EURL…) appear in 69% of Source 1 names, street words in 95% of
    addresses, and postcodes in almost none (0.4%).
  - It is the same noise generator as US/India, but two properties shift:
    - **35% of France Source 2/3 addresses have no state** (training: 3%).
    - France's commonest address tokens ("rue", city and région names) are unseen in the training text, so a
      train-fitted IDF gives them near-maximum weight (P14).
- **Evaluation yardstick.** The holdout is train fold 0 of 10 (hashed by Source 1): 220,358 Source 1.
  - They are scored while **all** 2.2M train Source 1 and **all** 10.3M train Source 2/3 records compete, as on test.
  - Smaller "mini-world" evaluations under-sample distractors: they overstated rule precision by 8 points (0.994 vs 0.912).
  - Every decision after 25 Sep 12:17 used this full-corpus yardstick, with paired per-S1 standard errors (SE).

### 2.2 Solution Strategy

**Approach Type:** Hybrid. Blocking → learned pruning → gradient-boosted classifier → cross-encoder reranker →
house-number stacker → exclusive (many-to-one) decoding.

**Core Innovation:**
- **The one-parent structure drives every stage.**
  - Blocking runs in reverse (Source 2/3 → Source 1).
  - The features describe the competition among a record's candidates (ranks, margins, chain sizes).
  - The decision is exclusive: each record goes to its argmax Source 1.
- **Measure honestly, then keep only what moves the number.** Every change went through a paired test on the
  full-corpus holdout with a bar of about +0.0002 at ≥ 3 SE. Many ideas failed it (section 5.4).
- **Unseen-country handling without labels.**
  - Leave-one-country-out (LOCO) proxies (US→India, India→US) choose the France recipe.
  - Label-free profiles (predicted empty-S1 rate, matches per S1) set France's threshold.
  - A few diagnostic leaderboard uploads that change only France calibrate it.
- **Generator-aware house-number reasoning.** The signed house-number delta separates the generator's orphan siblings
  from noisy true pairs.

How we worked: a coordinating session split the problem into about 20 probes (data analysis, transliteration, blocking,
matching, decisions, reranking, France, pruning, …). Each probe had its own git worktree and wrote its result to the
shared ledger; nothing was merged without its number.

## 3. Candidate Generation (Blocking)

- **Blocking keys used:**
  - **TF-IDF search.** The text is P2 transliteration → normalisation → name character 3-grams + address words,
    weighted 50/50.
    - The TF-IDF is hashed (2^18 / 2^20 buckets) and fitted on train text only.
    - One sparse top-k product per country, from each Source 2/3 record to Source 1, within its state group. The state
      agrees on 99.94% of true pairs.
    - Each record keeps its top 2 Source 1 (top 10 when its address is empty). Each Source 1 keeps its top 10 among the
      records that listed it.
  - **Exact-key rule pairs.** Name core or website within state, confirmed by address evidence.
- **Learned pruner** (meta-blocking; added after the organisers' 25 Sep note that smaller candidate sets rank higher).
  - A 300-tree, 31-leaf LightGBM on 30 of the 65 pair features: blocking signals plus name/address similarities.
  - Trained on train folds 2–3 (6.3M pairs); 74 s to train, 2–3 min to score all test pairs.
  - A pair reaches the matcher only if its pruner score q ≥ 0.03 or the rules accepted it.
  - The context features stay computed over the full blocking list, so a kept pair's matcher score is unchanged: the
    pruner only removes pairs.
- **Candidate pairs generated** (test, all three countries):

  | stage | pairs | per Source 1 (France / India / US) | holdout pair recall |
  |---|---|---|---|
  | blocking ∪ rule pairs | 27.5M | 15.87 (16.76 / 15.71 / 15.72) | 0.9917 (14.2 / S1 on holdout) |
  | **after the pruner = `candidate_pairs.tsv`** | **8,370,573** | **4.83 (5.14 / 4.90 / 4.63)** | **0.9874** (4.30 / S1) |

  `candidate_pairs.tsv` is exactly the set the matcher and the cross-encoder score (130 MB, down from 377 MB).
  22,446 test Source 1 have no candidate, and all of them are among the empty answers.
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
  - The leaderboard confirmed the cost: #6c vs D4 = −0.00019.

## 4. Matching Model

**Features used** (65, all label-free, computed identically on train and test):
- **Name features:**
  - normalised-core equality, token Jaccard, TF-IDF cosine;
  - Jaro–Winkler and token-sort / token-set ratios (rapidfuzz), acronym match;
  - website core vs squashed name, transliterated variants, an out-of-vocabulary-Indic flag.
- **Address features:**
  - house-number agreement / difference;
  - city key (frequency-picked, shuffle-proof), fuzzy place overlap, state agreement;
  - `NULL` / empty flags, TF-IDF address cosine.
  - House-number disagreement is the key negative signal: 92.5% of rule false positives are orphans with the same name
    core and matching city/state.
- **Other (context):**
  - blocking score and ranks in both directions, the score gap / margin to each side's best candidate;
  - candidate-list sizes, name-chain size within the country, the rule stage's decision, and source (S2/S3).
  - Country is **never** a feature, because France is unseen.

**Model type:** four learned stages. 4.1–4.4 describe each; 4.5 is the decision; 4.6 is France.

### 4.1 Matcher: LightGBM `m2_big`
- lr 0.05, 127 leaves, 4,930 trees.
- Trained on every candidate pair of train folds 2–5 (12.6M pairs, float16) and early-stopped on fold 1.
- Alone (argmax, no reranker) it scores 0.9834. The fold-2-only model scored 0.9814.
- P16 retrained it on folds 2–9 (25.1M pairs) on a SageMaker CPU instance: +0.00014 ± 0.00006, 2.6 SE. That is below the
  bar, so m2_big stays.

### 4.2 Cross-encoder: bge-reranker-v2-m3 (final) and multilingual-e5-small (history)
- **Input.** Both records as "name | address" text.
- **Which pairs.** Only India/US pairs in the matcher's uncertain band, 0.02 < p < 0.98: 1.24M test pairs after the
  pruner.
- **Blend.** p′ = σ(a·logit p + b·logit r + c). The weights are a logistic stack fitted on holdout half 0.
- **History (#4–#7): intfloat/multilingual-e5-small** (MIT, 118M).
  - Training set: 778,520 uncertain-band pairs of train folds 4–9 (band of a weak fold-2 LightGBM, 0.1 < p < 0.9, plus a
    half-weight sample of easy pairs).
  - Frozen word embeddings, bs 128, lr 6e-5; 25 min per epoch on the laptop's RTX 3050.
  - On the band its AUC is 0.926, against 0.809 for the tree model. It fixed look-alike street mismatches: look-alike
    AUC 0.82 → 0.98.
  - Gain: +0.0066 ± 0.0002. A 2-epoch run added +0.00039 ± 0.00006 (#7).
- **Final (#8c): BAAI/bge-reranker-v2-m3** (Apache-2.0, 568M).
  - Same 778,520 pairs, 2 epochs, bs 128, lr 2e-5, 500 warm-up steps, max_len 96, bf16 autocast.
  - Fine-tuned in a SageMaker training job on one ml.g5.xlarge (NVIDIA A10G 24 GB): 76 min. The whole probe cost $4.70.
  - Band AUC India 0.971 / US 0.956, against 0.955 / 0.934 for e5-small.
  - Blend w = [0.5901, 0.6319, 0.2973].
  - Gain over #7's e5 CE: **+0.00059 ± 0.00007** (8 SE).
  - Test inference: ~20 min on an A10G, ~2 h on the laptop GPU.
  - SageMaker is used only as rented compute: our code, our training data, open weights. No hosted model or API is in
    the resolution path.
- **Why a bigger model helped only a little.** e5-base (2.4× the parameters) gained only +0.00008 ± 0.00006 over
  e5-small, and ensembles ≤ +0.00002. The step to bge was a better pretrained reranker, not raw size.

### 4.3 France's scorer
- France keeps the matcher's p by default.
- A cross-encoder with in-domain blend weights hurts an unseen country: e5 LOCO −0.0016, bge India→US −0.0016.
- Out-of-domain weights (each LOCO model blended with the weights fitted on the *other* direction's target) do
  transfer: +0.0028 / +0.0015 on the two LOCO proxies. That is France option `bge` below.

### 4.4 US house-number / support stacker (P15 → P17)
- A residual LightGBM (200 trees × 31 leaves, lr 0.05, min_data_in_leaf 200, λ_l2 1). It starts from `init_score` =
  logit p′, so with no trees it returns p′ exactly.
- Seven features:
  - the logit;
  - the house-number class: equal / sibling offset / digit drop-add / small / far;
  - the signed delta hx − h1 (clipped ±100) and house-number equality;
  - three support counts: how many other records of this Source 1 carry the same house number, their best p, and how
    many carry the Source 1's own house number.
- Fitted on US holdout half 0; t stays 0.71.
- Gain on the report half: **US +0.00055 ± 0.00009, India + US +0.00033 ± 0.00005**.
- On India it loses −0.00013 on top of bge (bge already resolves those pairs), so it runs on the **US only**.

### 4.5 Threshold selection method
- Each Source 2/3 record goes to its argmax Source 1 among its kept candidates, and the pair is kept if p ≥ t.
- For India and US, t = 0.71 on the final p. It was swept for F0.5 on holdout half 0 and reported on half 1; the curve
  is flat within 0.001 over t ∈ [0.57, 0.83].
- P5 tested richer decision layers (a stacked per-S1 decision, top-k rules). The best gained +0.0010 on the bare matcher
  but only +0.0001 after the cross-encoder, so plain argmax + t stayed.

### 4.6 France: label-free threshold, validated on the leaderboard
1. **P6's rule** (#3–#6). Pick the t at which France's predicted empty-S1 rate equals the labelled countries' rate.
   In LOCO it matched the oracle threshold (US→India 0.9054 = oracle). On France it gave t 0.78.
2. **The leaderboard showed France over-matching.** Four uploads changed *only* France, on the same India/US files:

   | upload | France | public LB |
   |---|---|---|
   | #5 | P6's rule, t 0.78 | 0.980017 |
   | D1 | France emptied | 0.849163 |
   | D2 | t 0.88 | 0.981024 |
   | D4 | t 0.95 | **0.981142** |

   - D1 isolates France: France F0.5 ≈ (0.980017 − 0.849163) / 0.1497 + 0.055 ≈ **0.93**, while India + US ≈ 0.989 — the
     holdout's value. So the whole holdout-vs-leaderboard gap is France.
   - Stricter was better up to 0.95, where the curve flattens.
   - At 0.95 France keeps 6.85% of its Source 1 empty and gives 2.99 matches per Source 1 (US/India: 5.7% / 3.38).
3. **Why stricter helps.** France accepts 22 sibling-offset pairs per 1,000 at t 0.78, against 2.6 (US) and 3.9 (India).
   Raising t mostly removes those. France is still an unseen country (LOCO costs 5–7 points at a source-country
   threshold), and its IDF shift widens the uncertain band.
4. **Rule for any France variant that changes France's p.** Its t is set label-free, so that France's empty-S1 rate
   equals D4's 6.85%, the leaderboard-validated profile. Any *other* unseen country label would get P6's rule.

**France options on the #8c base.** Each is one `run.py --recipe`; India/US are identical in all five.

| recipe | France change | France t | what it rests on | public LB |
|---|---|---|---|---|
| `sub8c` | none (m2_big p, as #6c / #7) | 0.95 | D4 | TBD (26 Sep) |
| `sub9a` | **IDF fix.** France is reblocked and refeaturised with its own Source 2/3 text in the IDF at its 14.4% test share (P14). This uses unlabelled test text, see Risks | 0.975 | India proxy: a train-only IDF costs −1.23 pt; own-text IDF recovers 96% of it (0.9716 → 0.9834). France's band shrinks 11.9 → 9.5% of argmax pairs | TBD (26 Sep) |
| `sub9b` | **bge cross-encoder for France** with out-of-domain weights [0.9245, 0.2595, 0.0515] | 0.98 | LOCO +0.0028 / +0.0015 | TBD (26 Sep) |
| `sub9s` | **sibling rule.** p := 0 when hx − h1 ∈ {3, 4, 5, 7, 9, 11, 13, 21} (170,752 pairs, 4,367 accepted matches) | 0.95 | US holdout +0.00097 on m2 p; India −0.0017. France's regime is decided by the LB | TBD (26 Sep) |
| `sub9d` | train-only IDF fallback: 158 French domain words at US street-type IDF | 0.98 | ~40% of the label-free profile shift of `sub9a` | spare (only if test statistics are ruled out) |

**Final France option: TBD by 26 Sep LB.**

## 5. Results & Error Analysis

- **F_0.5 Score (macro): 0.98990** on the full-corpus holdout (220,358 Source 1).
  - Report half 0.98956; India 0.9905, US 0.9895; precision 0.9983, recall 0.9726; 4.30 candidates per Source 1.
  - The report half is the honest figure, because the US stacker was fitted on half 0.
  - The holdout's competing (non-holdout) band pairs keep the e5 blend, since bge scores were computed only for the
    holdout's own band.
  - Two independent audits (at #3) found no label leakage. The threshold-tuning optimism is at most 0.00003.
  - **Public leaderboard: best 0.981142 (D4). The final submission's score is TBD (26–27 Sep uploads).**

### 5.1 Ablation — every step on the same full-corpus holdout

Δ is the paired per-S1 difference on the report half ± 1 SE, against the previous row unless noted.

| step (submission) | holdout, all S1 | report half | Δ | ledger / source |
|---|---|---|---|---|
| exact normalised name + first address segment (#1) | 0.1775 | — | — | pipe `s0-exact` |
| exact-key rules: P2 transliteration → P1 normaliser → name / website key + address evidence (#2) | 0.7708 | — | — | pipe `p8-rules` |
| hand-weighted rule score | 0.8257 | — | — | `P4-match/M0-hand-rule` |
| Fellegi–Sunter EM (label-free) | 0.2695 | — | — | `P4-match/M3-fellegi-sunter-em` |
| logistic regression, 65 features | 0.9334 | — | — | `P4-match/M1-logreg` |
| LightGBM, fold 2 (#3) | 0.9814 | — | +21 pts vs rules | `P4-match/M2-lightgbm` |
| + e5-small cross-encoder, India/US (#4) | 0.9884 | 0.9882 | **+0.0066 ± 0.0002** | `P4b-rerank/p4-D3-blend-0.02-0.98`; pipe `p9-model` |
| matcher on folds 2–5, m2_big (#5) | 0.9888 | 0.9885 | +0.00040 ± 0.00006 | pipe `p9-model` (20:17) |
| + learned pruner, 15.87 → 4.83 cands/S1 (#6) | 0.9885 | 0.9883 | −0.00024 ± 0.00003 | `P10-candidates/prune-tau0.03-fixed-w` |
| e5-small CE trained 2 epochs (#7) | 0.98881 | 0.98869 | +0.00039 ± 0.00006 | `P11-ce-big/ce-s2P-plain-rule`; pipe `p9-model` (23:46) |
| bge-reranker-v2-m3 CE instead of e5 (#8a) | — | — | **+0.00059 ± 0.00007** (P13 yardstick); +0.00055 ± 0.00007 on the pruned list | `P13-aws-rerank/pool-bge_poolck_holdout_vs_p11s2P_plain`; P17 |
| + US house-number / support stacker (#8c, **final India/US**) | **0.98990** | **0.98956** | +0.00033 ± 0.00005 (US +0.00055) | `P17-hn-stack/hn-stack-sup-on8a-bge-USonly`; pipe `p9-model` (26 Sep 07:45) |

France cannot be scored on labels. Its changes were chosen on LOCO proxies and label-free profiles (section 4.6), then
tested on the leaderboard.

### 5.2 Every leaderboard upload

The public leaderboard is a subset of test (all three countries).

| upload (IST) | what it is | public LB F0.5 | reading |
|---|---|---|---|
| #5 — 25 Sep 18:34 | m2_big + e5 CE; France t 0.78 (P6's rule) | 0.980017 (rank 33) | holdout 0.9885 → a 0.0085 gap |
| D1 — 21:43 | #5 with France emptied | 0.849163 | France ≈ 0.93, India + US ≈ 0.989: the gap is France |
| D2 — 21:46 | #5 with France t 0.88 | 0.981024 | +0.0010: France over-matches |
| D4 — 23:02 | #5 with France t 0.95 | **0.981142** | +0.00012 more; curve flattening |
| #6c — 23:36 | #6 (pruner, 4.83 cands/S1) + France t 0.95 | 0.980954 | D4 − 0.00019 = the pruner's cost (holdout predicted −0.00024) |
| #8c — 26 Sep | bge CE + US stacker; France t 0.95 | TBD | expected ≈ 0.9819 |
| #9a — 26 Sep | #8c + France IDF fix, t 0.975 | TBD | Δ vs #8c = the IDF fix alone |
| #9b — 26 Sep | #8c + bge France, t 0.98 | TBD | Δ vs #8c = bge on France alone |
| #9s — 26 Sep | #8c + France sibling rule, t 0.95 | TBD | Δ vs #8c = the rule alone |
| final — 26/27 Sep | #8c + the winning France option(s) | TBD | — |

Every France test sits on the same India/US files, so each leaderboard delta measures one France change.

### 5.3 Errors

- **Common false positives (wrong merges):**
  - 87% are orphan Source 2/3 records (no true parent among their candidates) that look like a real Source 1: the same
    name core and city, but a different street or house number.
  - Many are the generator's house-number siblings (+3, +5, +7…), which the US stacker and the France sibling rule
    target.
  - In France, the over-weighted common address tokens make unrelated same-city records look alike (accepted-pair score
    margin 0.22 vs 0.42–0.44 in US/India); the IDF fix targets this.
- **Common false negatives (missed matches):**
  - Pairs whose true Source 1 is not among the candidates: 0.8% after blocking, 1.3% after the pruner.
  - Heavy word-subset or DBA name changes, and records with no address.
  - True pairs with house-number noise that a strict threshold rejects. France's accepted house-number-mismatch share is
    1.7%, against the US's 8.8%, so recall is what France pays for its strict t.
- **Transfer to an unseen country** (LOCO, tuned thresholds):
  - US → India scores 0.905 and India → US 0.930, against ~0.98 in-country.
  - At the source country's threshold US → India loses another 4 points, which is why France's threshold matters more
    than its model.

### 5.4 What failed (and the number that killed it)

| idea | result | why dropped |
|---|---|---|
| Fellegi–Sunter EM, label-free (P4) | 0.2695 | far below a trained model |
| neural transliterator IndicXlit / multilingual embeddings for Indic names (P2) | 93% / 69% exact vs the learned table's 99.6% | worse and slower |
| two-stage matcher, stage-1 competition as stage-2 features (P4) | +0.0008 before the CE | complexity for a gain the CE mostly absorbs |
| P5 stacked decision layer (D7 stack + exclusivity + top-k gate) | +0.0010 on the bare matcher, +0.0001 after the CE | too small once the CE was in |
| cross-encoder with in-domain blend weights for France (P4b, P13) | LOCO −0.0016 (e5 up to −0.0031) | over-trusts the CE on an unseen country; out-of-domain weights fix it (`sub9b`) |
| bge zero-shot, no fine-tuning (P13) | band AUC 0.56 / 0.64 | does not know "same business" on near-duplicates |
| more CE capacity: e5-base, ensembles, bge + e5 (P11, P13) | e5-base +0.00008 ± 0.00006 (n.s.); ensembles ≤ +0.00002; bge + e5 +0.00041 < bge alone | capacity is not the bottleneck; the band's remaining headroom is +0.0048 |
| CE trained on m2_big's own band (P11) | −0.00016 ± 0.00006 | learns what the matcher already knows |
| synthetic France-like training pairs (P12) | US→India −0.8 pt, India→US +0.2 pt at the label-free t | needs the *target's* noise rates (+2.2 pt with India's labels), which a label-free method cannot measure (−1.2 pt) |
| matcher on 8 folds, 2–9, on SageMaker (P16) | +0.00014 ± 0.00006 (2.6 SE) | below the bar; the CE already rescores the pairs it fixes |
| fill France's missing state from the city (P14) | missing state costs only −0.00023 on a US proxy; fill recovers +0.00008 | noise-level lever |
| P6's French normalisation rules at the model stage (P14) | 0.9% of France S1 change, profile identical | neutral |
| house-number stacker on India (on bge) (P17) | −0.00013 ± 0.00006 | bge already resolves it |
| house-number stacker on France (P17) | loosens France, reshuffles 4.7–8.1% of France S1 | the LB says France must get stricter; the sibling rule is the clean test |
| hard sibling rule on India (P15) | −0.0017 | India's true-pair house-number noise is wider |
| per-S1 top-k candidate cap / cheaper blocking (P10) | top-5: −0.021; P3's cheaper config: 7.80 cands/S1 at −0.00031 | the learned pruner dominates both |
| a 4× larger pruner (P10) | 4.71 cands/S1 at −0.00018 vs 4.83 at −0.00024 | marginal for 4× the model |
| Qwen3-Reranker-0.6B (P13) | never ran (the job crashed on a config bug) | dropped; bge had already gained |

## 6. Conclusion

- The data's one-to-many structure (one parent per Source 2/3 record) shaped every stage: reverse blocking, competition
  features, and exclusive argmax decoding.
- Most of the gain over exact-key rules (+21 points) came from learning when a same-name look-alike at a different
  street is *not* the same business. The cross-encoder on the uncertain band and the generator-aware house-number stage
  added the rest.
- A learned pruner cut the candidate set by 70% for 0.0002 F0.5.
- Measuring on the full corpus rather than a sample kept every decision honest; sampled evaluations overstated
  precision by 8 points.
- The unseen country was the real challenge. LOCO proxies, label-free profiles and a handful of France-only leaderboard
  uploads located the gap, and the threshold was the biggest single lever for France.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`. `README.md` has the exact commands, caches and measured runtimes;
`requirements.txt` has pinned versions.

- **Entry point:** `python src/run.py --split test --stage model --out output/` writes both TSVs, asserts the format
  rules and runs the official validator.
  - The default recipe is `sub8c`. `--recipe sub9a|sub9b|sub9s|sub9d` gives the France options, and
    `--recipe sub7|sub6|sub5|sub4` gives earlier submissions.
  - Each recipe reproduces its staged files byte for byte.
  - `--split holdout` prints the full-corpus holdout score.
- **Modules** (in pipeline order):
  - `io_tsv.py`: TSV ↔ parquet with `QUOTE_NONE`.
  - `worlds.py`: folds and evaluation worlds.
  - `pairs.py` + `translit.py`: the learned Indic → Latin table.
  - `normalize.py`: the name / address normaliser.
  - `rules.py`: the exact-key rule stage; `s0_exact.py`: submission #1.
  - `block.py`: TF-IDF blocking, including France's IDF variants.
  - `features.py`: the 65 pair features.
  - `prune.py`: the learned pruner.
  - `train_matcher.py` + `model.py`: m2_big.
  - `rerank.py`: e5 / bge cross-encoder scoring and the blend.
  - `stack.py`: the US house-number stacker and France's sibling rule.
  - `submit_probs.py`: scored pairs → TSVs.
  - `metric.py`: macro F0.5.
- **Guards:** `test_pipeline.py` (end to end, including an injected unseen country), `test_rules*.py`, and each module's
  self-check. `package.py` builds this zip.
- **Artefacts.**
  - Rebuilt from the training data when absent: the transliteration table, place frequencies, vectorisers, pruner,
    m2_big, the e5 CE and the US stacker.
  - The bge CE checkpoint (`ce_bge_pool.pt`, 624 MB) was fine-tuned on SageMaker and is loaded, not retrained, by
    `run.py`. With cached cross-encoder scores the default run takes ~4 min plus the validator.
  - Downloaded once: the MIT / Apache-2.0 base weights of e5-small and bge-reranker-v2-m3.

### B. Additional Results

**Models and licences.** Every model is ≤ 568M parameters and runs locally.

| component | model | licence | size |
|---|---|---|---|
| matcher (m2_big) | LightGBM 4.7 | MIT | 4,930 trees × 127 leaves |
| pruner | LightGBM | MIT | 300 trees × 31 leaves |
| US stacker | LightGBM | MIT | 200 trees × 31 leaves |
| cross-encoder (final, #8a+) | BAAI/bge-reranker-v2-m3, fine-tuned | Apache-2.0 | 568M params |
| cross-encoder (history, #4–#7) | intfloat/multilingual-e5-small, fine-tuned | MIT | 118M params |
| transliteration | learned token table (no model) | — | 1,518-word vocabulary |

- Libraries: numpy, pandas, scipy, scikit-learn, torch (BSD-3); pyarrow, sparse_dot_topn, transformers, tokenizers,
  safetensors, huggingface_hub (Apache-2.0); rapidfuzz, duckdb, lightgbm (MIT); anyascii (ISC).
- No external data, APIs, geocoding, registries or hosted models are used. The AWS SageMaker jobs were rented compute
  for our own training code: bge fine-tuning $4.70, the 8-fold matcher test $0.63.

**Label-free France profile at each operating point** (test, argmax per Source 2/3):

| France scorer / t | empty S1 | matches per S1 |
|---|---|---|
| m2_big, t 0.78 (P6's rule, #5) | 5.69% | 3.24 |
| m2_big, t 0.88 (D2) | 6.20% | 3.13 |
| m2_big, t 0.95 (D4, #6c, #7, #8c) | 6.85% | 2.99 |
| IDF fix, t 0.975 (#9a) | 6.79% | 3.01 |
| bge France, t 0.98 (#9b) | 6.8% | 2.99 |
| US / India at t 0.71 (for reference) | 5.7% | 3.38 |

**Hardware and runtimes.**
- Everything except the bge fine-tuning ran on one laptop: 16 cores, 15 GB RAM, RTX 3050 6 GB. Every step fits a 3 GB
  memory cap.
- Cold test run with trained models and the cached bge scores: ~2.5 h (blocking ~36 min, features ~50 min, matcher
  scoring ~30 min). Scoring the bge band fresh adds ~20 min on an A10G or ~2 h on the laptop GPU.
  Rebuilding every model from scratch: ~5.5–6 h, excluding the bge fine-tuning (76 min on an A10G).
