# Round 2 approach details — Team Yeti (paste-ready)

Public LB 0.989382 (final entry `sub90u`). Full write-up: `Documentation_template.md` in `Yeti_submission.zip`.

## 1. Methodology used

A hybrid cascade built around one property of the data: every Source 2/3 record has at most one true Source 1
parent. Stages: learned Indic→Latin transliteration and name/address normalisation → reverse TF-IDF blocking → a learned
pruner (meta-blocking) → a LightGBM matcher on 65 label-free features → a fine-tuned bge-reranker-v2-m3 cross-encoder on
the uncertain band → residual stackers that re-score each record's best pair from its competition → an exclusive
one-owner decision (each record to its best Source 1, kept if p ≥ t; at most 5 S2 / 6 S3 per S1). All decisions were
made on a full-corpus holdout (train fold 0: 220k Source 1 scored against all 2.2M train Source 1 and 10.3M S2/S3), with
paired per-S1 standard errors and a +0.0002 / 3 SE bar per change. France (15% of test, no labels, unseen in training)
is detected as "a test country absent from train" and handled label-free: its own-text IDF, a house-number-aware
matcher, cross-encoders with out-of-domain blend weights, a stricter threshold, and structural rules chosen on
leave-one-country-out proxies (US→India, India→US) and France-only leaderboard uploads.

## 2. Candidate generation / blocking strategy

- Hashed TF-IDF (name character 3-grams + address words, 50/50), one sparse top-k product per country from each S2/S3
  record to Source 1 within its state (state agrees on 99.94% of true pairs); each record keeps its top 2 (top 10 without
  an address), and each Source 1 keeps its top 10 in the reverse direction. Exact-key rule pairs (name core / website +
  address evidence) and a French acronym block are added.
- 27.5M pairs (15.87 per S1), holdout pair recall 0.9917.
- A learned pruner (300-tree LightGBM on 30 cheap features, q ≥ 0.03 or rule-accepted) keeps 8.37M pairs (4.83 per S1)
  for −0.00024 F0.5; the operating point was chosen on final F0.5, not recall.
- A sibling-decoy cascade stage drops pairs at the data generator's house-number offsets unless they are the record's
  best pair above t (matching unchanged).
- Final `candidate_pairs.tsv`: **7,668,008 pairs = 4.43 per Source 1** (France 4.51 / India 4.75 / US 4.00); every
  matched pair is inside it. Blocking runs on a laptop in ~36 min at 2.7 GB RAM.

## 3. Model architecture and feature engineering

- **Features (65, label-free, country never a feature):** name — normalised-core equality, token Jaccard, TF-IDF cosine,
  Jaro–Winkler, token-sort/set ratios, acronym, website-vs-name, transliterated variants; address — house-number
  agreement/difference, shuffle-proof city key, place overlap, state, NULL/empty flags, address cosine; context —
  blocking scores and ranks in both directions, margins to each side's best candidate, candidate-list sizes, name-chain
  size, rule decision, source.
- **Matcher:** LightGBM `m2_big` (127 leaves, 4,930 trees, 12.6M pairs of folds 2–5, early-stopped on fold 1).
- **Cross-encoder:** BAAI/bge-reranker-v2-m3 (Apache-2.0, 568M) fine-tuned 2 epochs on 778k uncertain-band pairs,
  applied to 0.02 < p < 0.98 and blended by a logistic stack (+0.0066 for the first CE, +0.00059 more for bge).
- **Stackers:** a US house-number stacker using the generator's signed sibling offsets and support counts (+0.00055 US);
  a bagged residual combiner over each record's argmax pair with competition, exclusivity-mass, house-number-corruption,
  CE-on-every-argmax-pair, 16 zoo-model, generator-trace and anchor columns (+0.000656 LB-eq); a cross-fitted stage-C
  owner re-decision for the US (+0.000287 on US fold 0).
- **France:** own-text IDF re-blocking, rebuilt records (`(n)` house numbers, generator name fillers), a house-number-aware
  LightGBM (`c_hn`), Qwen3-Reranker-4B + LoRA (Apache-2.0, 4.02B) band blend with out-of-domain weights, rules at a
  label-free t 0.9868 (key tier, same-address rule, CE veto, cap), empty-S1 rescue and one-sided-group completion,
  plus CE- and combiner-driven change sets validated by LOCO.

## 4. Any other relevant information

- **Compliance:** every model is MIT / Apache-2.0, ≤ 8B, runs locally, and was fine-tuned only on the provided data
  (or unlabelled test rows, as allowed). No external data, APIs, geocoding or registries; the only hand-written lists are
  legal suffixes, street types and state / région names. AWS SageMaker was rented compute for our own code only.
- **Credit:** France same-site ADD rules (+16,118 pairs) come from the public MIT repository
  PundarikakshNTripathi/present-day-ber-amlc-2026, run on our file and filtered by our own holdout-checked rule; its
  DROP rules failed our holdout and were not used. Three of its ideas (generator traces, anchors, stage-C re-decision)
  were re-implemented in our own code.
- **Reproducibility:** `run.py` regenerates the base submission from the raw TSVs; `final_build.py` and `final_patch.py`
  apply the recorded change sets and rebuild the uploaded files byte for byte (md5 a8ad38b5… / f311dc4d…); the scripts
  behind every recorded change set ship in `src/experiments/`.
- **Hardware:** one laptop (16 cores, 15 GB RAM, RTX 3050), every step under 3 GB RAM; cross-encoder fine-tuning on one
  A10G (76 min, $4.70).
