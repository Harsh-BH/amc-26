# P12-synth — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
Synthetic pairs made from target S1 text help an unseen country only when the noise rates match that country.
With the source country's rates (the brief's recipe) the gate fails: US→India **−0.8 pt** (0.9054 → 0.8978), India→US **+0.2 pt** (0.9293 → 0.9312), both at the label-free threshold.
With India's own rates, US→India gains **+2.2 pt** (0.9273), but that uses India labels; the label-free variants lose more: rates measured on the US model's confident India pairs −1.2 pt, 0.3× dose −0.4 pt. Do not build synthetic France for the submission; keep P6's pooled model + label-free threshold.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- **Adopt nothing for the submission now.** The brief's recipe fails the gate: source-country rates, target S1 text,
  real source + synthetic target. It needs ≥ +1 pt at the label-free t in both directions; it got −0.8 pt (US→India) and
  +0.2 pt (India→US). P6's pooled model + `fr_threshold.py` stays the France path.
- **Why:** the generator is sound, but the rates must be the target's (upper bound +2.2 pt on India).
  - Source rates put source-country noise onto target records. Real India has 2× the US house-number change rate,
    2.5× the segment drops, 8× the legal-form swaps and 8× the honorific prefixes.
  - The model then learns the wrong "how much disagreement is still a match" and loses recall on real India
    (pr 0.863 vs 0.872 at the same precision).
- **The label-free way to get target rates does not recover the upper bound.**
  - Method: the source model's confident target pairs (argmax & p ≥ 0.98, 94% precise) → rates. Result: −1.2 pt.
  - Why: they match India on the easy ops but under-count exactly the ops that make pairs hard (house-number change 0.13 vs 0.20,
    empty address 0.012 vs 0.037). Those ops set the decision boundary the synthetic data teaches.
  - Even with the target's true pair rates, the source country's orphan structure caps the gain at +0.7 pt
    (decomposition row). The other +1.5 pt of the upper bound comes from the target's own sibling / orphan shape.
  - Adding the synthetic target to in-country training also costs 0.8 pt, even with the target's own rates.
  - So there is no label-free configuration that passes the gate.
- **What France would need**, if the master wants to pursue it (NOT built):
  1. The organisers' OK to use test France S1 text as generator input. PROBLEM.md open question 4; the brief also forbids
     test text for training. For the pseudo-rate variant, the OK would also have to cover test France S2/S3 text.
  2. Rates: the pooled US+India true-pair rates (the brief's recipe, which LOCO says hurts), or rates measured on the
     pooled model's confident France pairs.
  3. `synth.generate(france_test_s1, rates)`, then the P3 blocking + P4 features over the France S1 pool
     (`world.py gen|rf|block|feat`, pointed at test France), then synthetic France added to the pooled training
     (fold 2 ×0.5 per country here; P4's shipped model uses folds 2–5).
     - Cost: ~25 min CPU + a refit of the matcher.
     - Validation: none on France; only the two LOCO directions.

## Evidence         (table: ledger name | world | key metrics | verdict)
Every number is on REAL train fold 0 (full corpus: all 2.2M train S1 and 10.3M S2/S3 compete, argmax per SX over all S1).
The setup is P6's `loco_dense` protocol: fold 2 ×0.5 of the real source, early stop on the source's fold 1 ×0.2, lr 0.1.
- "empty t": the label-free threshold (the unseen country's empty-S1 rate matched to the source's at t*), P6's France rule.
- Synthetic training rows are ~1× the real target's fold-2 ×0.5 pair count (~620k India / ~950k US), pairs of source S1 only.

| ledger name | world | key metrics | verdict |
|---|---|---|---|
| loco-base-rescored | holdout-dense-full | P6's models re-scored in my code path (= P6). US→India: oracle 0.9054 (t .98), **0.9054 at empty t**, 0.8636 at source t. India→US: 0.9296 (t .87), **0.9293 at empty t**, 0.9260 at source t. In-country US 0.9808, India 0.9792 | baseline |
| loco-aug-US+synIndia_r1 | holdout-dense-full | real US + 620k synthetic India pairs (**US rates**) → India oracle 0.8983 (t .59), **0.8978 at empty t (−0.76 pt)**, 0.8976 at source t (+3.4 pt: calibration fixed); pp/pr at empty t 0.942/0.863 (base 0.944/0.872). US in-country 0.9796 (−0.12) | drop |
| loco-aug-India+synUS_r1 | holdout-dense-full | real India + 951k synthetic US pairs (**India rates**) → US oracle 0.9317 (t .86), **0.9312 at empty t (+0.19 pt)**, 0.9154 at source t (−1.1). India in-country 0.9760 (−0.32) | drop (< +1 pt) |
| loco-synIndia_from_US_only | holdout-dense-full | synthetic India (US rates) ONLY, no real data → real India 0.8920 (t .51), real US 0.9329 (t .50). A synthetic world is about as informative as a real foreign country (US→India 0.9054, India→US 0.9296). It transfers best to the country whose RATES it copied | diagnostic |
| loco-aug-US+synIndia_r1_Indiarates | holdout-dense-full | **upper bound, not label-free** (India rates from India labels): real US + synthetic India → India oracle 0.9317 (t .61), **0.9273 at empty t (+2.19 pt)**, 0.9295 at source t (+6.6); US in-country 0.9797 | the generator works when the rates are the target's |
| loco-aug-India+synIndia_r1 | holdout-dense-full | **in-country**: real India + 618k synthetic India pairs (India's own rates) → India holdout 0.9709 (t .74); **P5 report half 0.9710 vs 0.9789 (−0.8 pt)**. The same model → US 0.9246 at empty t (India-only 0.9293) | drop: synthetic pairs dilute real in-country labels |
| loco-aug-US+synIndia_r1_psIndiarates | holdout-dense-full | **label-free target rates**: the US model's confident India pairs (argmax & p ≥ 0.98, fold 2; 140k pairs, 94.3% precise) → rates → synthetic India. Real US + it → India oracle 0.8948 (t .67), **0.8930 at empty t (−1.24 pt)**; pp/pr 0.941/0.852. Pseudo rates match India on most ops (seg_drop 0.61 vs 0.62, legal_swap 0.36 vs 0.36) but under-count the hard ones (hn_change 0.13 vs 0.20, empty address 0.012 vs 0.037) | drop |
| loco-aug-US+synIndia_r0.3 | holdout-dense-full | dose check, US rates, 0.3× synthetic (186k pairs) → India oracle 0.9025, **0.9016 at empty t (−0.38 pt)**. The loss shrinks with dose but never beats base | drop |
| loco-aug-US+synIndia_r1_mixIndiarates | holdout-dense-full | **decomposition** (not label-free): India's pair-noise rates + the US's generator structure (cluster sizes, orphan / sibling shape) → India oracle 0.9148 (t .56), **0.9122 at empty t (+0.68 pt)**. Upper bound 0.9273 = +1.4 pt from India's pair-noise rates (vs US rates 0.8978) + 1.5 pt from India's orphan / sibling structure | both halves of the gain need target labels |

Synthetic world vs real (feature means over true pairs, folds ≥ 2; `diag.py`):
- India built with US rates:
  - true pairs with a house number on both sides 0.621 vs real India 0.618 (US 0.872): the target's S1 structure carries over;
  - sx_margin 0.405 vs 0.406; candidate recall of synthetic truth 0.989 (real India 0.991);
  - but core_eq 0.658 vs 0.782 (US 0.636), hn_diff 0.075 vs 0.126, city_eq 0.837 vs 0.693: the noise is the US's.
- India built with India rates: hn_diff 0.121 vs 0.126, hn_eq 0.479 vs 0.492, core_eq 0.726 vs 0.782.
  city_eq stays too high (0.808 vs 0.693).

Generator facts measured on the data (aggregates in `rates_{US,India}.json`):
- Orphan S2/S3 are generated **siblings** of an S1: 93–95% look alike their top-1 S1.
  - House number + one of {1, 2, 3, 4, 5, 7, 9, 11, 13, 21} (uniform, the same set in US and India).
  - A branch word is added to the name 66% (US) / 82% (India) of the time. US: partners/holdings/group/east/midtown…;
    India: ventures/group/public/holdings/industries/overseas…
  - Orphan share 26.6% (S2) / 25.4% (S3) in both countries; 4.67 / 4.68 S2+S3 per S1.
- Address noise is decided per (entity, source) and shared by all copies of that source.
  - Share of the op decided per entity: NULL 0.96, CITY suffix 0.92, segment drop 0.67, house-number change 0.52 (US S2).
- State form is relative to S1: S2 keeps the S1's form (US codes / India names, 99.9%); S3 swaps it (93% US, 64% India).

## What failed      (approaches tried and dropped, with the number that killed them)
- **Source-country rates** (the brief's recipe): US→India −0.76 pt at the label-free t, India→US +0.19 pt. The gate needs ≥ +1 pt both ways.
  - It fixes calibration: at the source t, US→India goes 0.8636 → 0.8976.
  - But P6's label-free threshold rule already recovers that (0.9054 = oracle), so calibration was never the loss.
  - Synthetic data only helps if it improves the RANKING on the target, and source-rate noise makes it worse.
- **Mixing synthetic other-country pairs into in-country training** costs 0.1–0.3 pt in-country: US 0.9808 → 0.9796, India 0.9792 → 0.9760.
- First-pass estimator bugs, fixed before any LOCO run:
  - name-form rates were averaged only over rows carrying the key, so they read 1.0;
  - digit-for-letter read 0 because the tokenizer folds digits back to letters.
  Both were re-measured (digit-for-letter 3.0–3.3%, matching P1's 3.0%).

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `synth.py`:
  - `estimate(country)` → rates dict: ops on TRUE pairs of folds ≥ 2 (never folds 0/1), cluster sizes, orphan share, sibling shape.
  - `pair_rows(country, s1_raw_by_id, pairs)` → per-pair detected ops.
  - `generate(s1_raw, rates, seed)` → (synthetic S2/S3 raw rows [entity_id, business_name, business_address, country],
    truth [s1, sx]). It takes ~65 µs per synthetic record.
  - Vocabulary (legal forms, words, addresses, state table) comes from the target S1 only; rates and filler words from the rates dict.
  - `python synth.py est <c>` → `rates_<c>.json`, ~4 min, 2 GB. Self-check: `python synth.py`.
- `pseudo.py <tgt> <src>`: the src model's confident argmax pairs (p ≥ 0.98) on the target's fold 2 → the same
  detectors → `rates_ps<tgt>.json`. Structural constants stay the source's. ~3 min, 1.5 GB.
- `world.py gen|rf|block|feat|all <tgt> <src>`: synthetic world for a target (pool = target S1 of folds ≥ 2; sources = a
  seeded state-group subset, P12_FRAC 0.35 India / 0.25 US).
  - Runs P3 `country_pairs` (same vectorizers and CFG), `rules.match(by_state)`, and P4's `features.strings` / `_group_ctx` / `block_cols`,
    with the same rounding.
  - Output: `data/work/p12/X/<tgt>_from_<src>_NNN.parquet`, P4 X format (q1, qx, is_true, fold + the 65 features).
  - India (1.23M synthetic SX, 4.5M pairs): gen 1.5 min, rf 1.6 min, block 6 min, rules 6 min, feat 2 min; peak 2.35 GB.
- `synloco.py base | aug <tgt> <src> [ratio] [world_src] | synthonly <tgt> <src> | incountry`: P6's protocol with
  synthetic training rows. Ledger rows, curves in `data/work/p12/curve_*.csv`, models in `data/work/p12/models/`.
  ~8–15 min per run under load, peak 2.5 GB.
- `diag.py <tgt> <src>`: synthetic vs real feature means.
- Disk: `data/work/p12/` ≈ 1.0 GB: 6 worlds × ~165 MB of X, plus models and curves. The per-world intermediates were deleted after `feat`.
  The whole directory is deletable once this result is read.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Licences / rules:** no new dependency (anyascii ISC, rapidfuzz MIT, LightGBM MIT, pandas/numpy BSD).
  - No external data, no LLM, no wordlists.
  - Every synthetic record is code applied to a train S1 record plus words mined from train. No test text was read.
- **Assumptions:**
  - The pool is the target's folds ≥ 2 only (80% of the real S1 density; chain counts 16 vs 21 per true-pair S1).
  - Sources are a state-group subset: training keeps only pairs of source S1, whose S1-side context is complete.
  - One seed. Ratio 1× everywhere, plus 0.3× for US→India (source rates).
  - The upper-bound row uses India LABELS for the rates. It is a ceiling, not a deployable result.
- **France-specific:** P6 says France is less unseen than either LOCO direction (pooled model, smaller threshold gap).
  So a France gain, if any, would be smaller than the LOCO upper bound. Pooled rates are an average of two countries
  whose rates differ 2–8× on the ops that matter; France's rates are unknown without France pairs.
- **Label-free rate estimation needs target S2/S3 text:** test S2/S3 for France. That is open question 4 again, and stricter than S1-only.

## Open questions   (for the master; each with your best guess)
- **Ask the organisers whether test S1 (and S2/S3) text may be the INPUT of a synthetic-pair generator?** Best guess: only if
  the master wants a France-specific experiment later; otherwise do not spend a question on it.
  Every label-free LOCO variant lost 0.4–1.2 pt on the unseen country. The only win (+2.2 pt) needs the target's labelled rates, which France cannot have.
- **Where does the +2.2 pt upper bound come from?** Answered by the decomposition row, about half and half:
  - India's pair-noise rates: 0.8978 → 0.9122.
  - India's orphan / sibling structure: 0.9122 → 0.9273. That structure is the same-house-number mode, the country's own
    branch words (ventures/industries/overseas…) and the 82% added-word rate.
  - With US structure, the pseudo rates score 0.8930 against the true India rates' 0.9122. That gap is the hard ops the pseudo-pairs
    under-count.
  - France would need its own versions of both, and neither is measurable without France labels.
  - Best guess: not worth more time in this challenge.
- **Is P6's label-free threshold still right if synthetic data is ever added?** Best guess: yes. With synthetic target data
  the empty-match t moved to 0.66–0.84 and stayed within 0.5 pt of each run's oracle.
