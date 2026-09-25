# P4-match: pair features and matcher
Inputs: adopted normalize/translit/block modules (see probes/MERGE-1.md for the exact versions).
Training pairs: candidates from the adopted blocker on the train world, labelled from truth. Validate on dev,
report on holdout.
Features: name (rapidfuzz ratio, partial_ratio, token_set, token_sort, Jaro-Winkler, token Jaccard, TF-IDF
cosine, name_core equality, acronym match, website_core vs name, transliterated variants), address (token
Jaccard, city_key, state, postcode, house number, street similarity, NULL/empty flags), context (block ranks
both directions, score gap to this S1's best candidate, #candidates, the S1's rank among this SX's candidate
S1s, name frequency within country = chain-ness), source S2/S3. Do NOT feed country as a raw category
(France is unseen) — P6 decides country handling.
Model ladder (log each): M0 hand-weighted rule score · M1 logistic regression · M2 LightGBM (expected
workhorse; research: Foursquare 1st place) · M3 Splink Fellegi-Sunter EM (label-free baseline).
Fixed decision rule for comparability: each SX goes only to its best S1, then a global threshold swept for F0.5.
Report pair AUC/AP, dev F0.5, feature importance, predict throughput (pairs/s).
Deliverable: probes/P4-match/{features.py, model.py}: featurize(pairs, s1_df, sx_df) -> DataFrame;
fit(X, y) -> model; predict(model, X) -> np.ndarray. Save out-of-fold probabilities for the dev and holdout
worlds to data/work/P4/ for P5.

## UPDATE from MERGE-1 (read probes/MERGE-1.md first) — these override anything above
- Scoring is **dense / full-density only**. The dev/holdout mini-worlds overstate recall (~2 pts) and hide chain collisions.
- **Eval set:** `data/work/p3/cands_holdout_full_{India,US}.parquet`. Columns: s1, sx, block_score, block_rank_s1,
  block_rank_sx, via, is_true. It holds holdout (fold 0) S1 against the FULL train corpus and S1 pool, 3.1M pairs.
  Truth for the F0.5 comes from `worlds.load_world("holdout")` truth, restricted to the holdout S1.
  An S1's true S2/S3 missing from the candidates is a recall loss you must count.
- **Training set (build it first, ~35 min):** the same P3 rule for train folds 2–9 (a sample of S1 is fine, e.g. 300k S1).
  `probes/P3-block/full.py run <country>` performs the full-train reverse search. Adapt the export to write folds ≥ 2 to
  `data/work/p4/cands_train_full_{country}.parquet` with is_true. Never train on folds 0/1.
- **Stage 1 already exists:** `code/business_entity_resolution/src/rules.py`.
  - `rules.feats(df)` gives core/web/state/city/hn/places.
  - `rules.match(f1, fx)` gives accepted pairs and key candidates. Dense holdout: F0.5 0.8201, pair precision 0.9937, pair recall 0.637.
  - Add its outputs as features: key match, website match, address score, rule-accepted flag.
  - Measure the recall of candidates = P3 ∪ rule-key pairs.
- **Target:** beat **0.8201 dense holdout F0.5** by a clear margin. Use the fixed decision rule: each SX goes to its argmax S1,
  then a global threshold swept for F0.5. Report per country.
- **Scale guidance (P7):**
  - Vectorise features with rapidfuzz `process.cpdist(workers=…)` and numpy.
  - Never loop per pair in Python. Stream 1M-pair chunks and keep only float32 probs.
  - Test candidates already exist (`data/work/p3/cands_test_*.parquet`, 27.4M pairs). Predict them at the end and save
    `data/work/p4/prob_test_{country}.parquet` (s1, sx, p) so the master can build the submission.
- Save holdout predictions to `data/work/p4/prob_holdout.parquet` (s1, sx, p, is_true, country) for P5.
