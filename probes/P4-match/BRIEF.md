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
