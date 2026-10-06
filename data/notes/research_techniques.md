# Technique Scout: Business Entity Resolution at 10M Scale — Ranked Recommendations

## 1. Blocking / candidate generation

1. **TF-IDF/BM25 char n-gram sparse top-k (Sparkly-style)** — beat 8 SOTA blockers in VLDB benchmarks, pure sparse-matrix CPU work (`sparse_dot_topn` or scipy chunked), fits 16-core/15GB laptop with no GPU needed. k=20–50 per record with one well-chosen tokenizer hits >90% recall on Magellan-style benchmarks. — [Sparkly, VLDB 2023](https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf)
2. **Splink blocking rules (union of multiple keys)** — cheap complementary layer for reordered/partial tokens that top-k TF-IDF misses; recommended practice is start strict (<10M comparisons), loosen, run on DuckDB on a laptop before Spark. — [Splink blocking rules](https://moj-analytical-services.github.io/splink3_legacy_docs/topic_guides/blocking/blocking_rules.html)
3. **MinHash-LSH** — fast set-similarity net for dirty/typo'd names, use as a second independent blocker (union with #1), not primary — lower precision ranking than TF-IDF top-k. — [Comparison of Blocking Methods](https://arxiv.org/pdf/1407.3191)
4. **Embedding ANN (FAISS/HNSW, e.g. LaBSE)** — restrict to the cross-script residual (India Devanagari/Telugu↔Latin, France) where token-overlap blocking structurally fails; HNSW/ANNOY hit memory/latency walls past ~1M records uncompressed, so use IVFPQ/HNSWPQ quantization or scope to the residual only. — [BlockingPy ANN blocking](https://arxiv.org/pdf/2504.04266)

## 2. Matchers

1. **Feature-engineered LightGBM (GBDT) on similarity features** (Levenshtein/Jaro-Winkler, token-set overlap, TF-IDF cosine, geo/address overlap) — this is literally the 1st-place recipe for Foursquare Location Matching (Kaggle, POI name+address, multi-country, millions of rows): 2-stage LightGBM cascade, candidates → engineered features → final score. Trivially satisfies the ≤8B/no-API rule since it isn't an LLM at all. — [1st place solution](https://www.kaggle.com/competitions/foursquare-location-matching/writeups/re-waiwai-1st-place-solution)
2. **Fellegi-Sunter EM (Splink, label-free)** — use for (a) the zero-label France slice, (b) generating weak/pseudo labels to bootstrap the GBDT. A cited case study shows supervised (MLPC, F1 98.94) beats unsupervised FS-EM, but FS-EM needs no labels at all. — [EM intuition, Splink](https://www.robinlinacre.com/em_intuition/)
3. **Ditto-style fine-tuned small multilingual transformer as re-ranker only** (not primary matcher) on GBDT's top-k hard/cross-script candidates — Ditto beats DeepMatcher+/Magellan by ~9.9 F1 on ER-Magellan benchmarks; too costly to run on all pairs, so scope to the ambiguous tail on the RTX 3050. — [Ditto, arXiv 2004.00584](https://arxiv.org/pdf/2004.00584)
4. **LLM-as-judge (small ≤8B open-weight, MIT/Apache)** only on the residual low-confidence pairs after GBDT+re-ranker — recent multi-step/compound prompting (ComEM) improves LLM matching accuracy but cost/latency make it unsuitable as a primary matcher at 10M scale. — [ComEM benchmark](https://aclanthology.org/2025.coling-main.8.pdf)

General tabular-data note backing #1: tuned GBDTs win or tie within noise on ~70% of tabular benchmarks vs transformers; transformers only pull ahead on large/embedding-heavy data. — [Tabular Data 2026](https://www.mlaia.com/blog-tabular-data-ml-vs-dl.html)

## 3. Cross-script name matching (Devanagari/Telugu ↔ Latin)

1. **Rule transliteration to a common romanization (Hunterian, India's official system) + phonetic key** as a blocking/matching key — deterministic, cheap, no training data needed; validated stepwise-matching approach for Hindi-transliterated names. — [hindi-fuzzy-merge](https://github.com/IDinsight/hindi-fuzzy-merge)
2. **Character n-gram cosine similarity on transliterated forms** (bigram/trigram, plus a vowel-stripped variant) fed as a GBDT feature — standard technique for cross-script fuzzy name matching. — [Fuzzy Name Matching, Babel Street](https://www.babelstreet.com/blog/fuzzy-name-matching-techniques)
3. **Multilingual embeddings (LaBSE/bge-m3) as a secondary/fallback feature only** — sources explicitly warn dense retrievers degrade severely on short 2-word names with no context and when query is romanized vs native-script; don't make this the primary cross-script signal. — [Cross-Script Name Retrieval](https://towardsdatascience.com/bytes-speak-all-languages-cross-script-name-retrieval-via-contrastive-learning/)

## 4. Decision layer for set-valued F-beta + exclusivity

1. **Decision-theoretic F-beta maximizer** (Ye et al., ICML 2012) — given calibrated P(match) from the GBDT, pick the expected-F0.5-optimal subset per S1 query rather than a fixed global threshold; matches the macro-F0.5-per-S1 metric exactly. — [Ye et al. 2012](https://icml.cc/2012/papers/175.pdf)
2. **Exact F-measure maximizer DP** (Dembczynski/Waegeman) — O(n²) dynamic program over sorted probabilities per query group; cheap here since per-S1 candidate counts are bounded by blocking top-k. — [Bayes-Optimality of F-Measure Maximizers, JMLR](https://jmlr.org/papers/volume15/waegeman14a/waegeman14a.pdf)
3. **Global one-to-many exclusivity as bipartite max-weight matching** (`scipy.optimize.linear_sum_assignment` / Hungarian) applied after the F-beta subset step, to enforce "each S2/S3 → at most one S1" — standard bipartite record-linkage framing. — [(Almost) All of Entity Resolution](https://arxiv.org/pdf/2008.04443)
4. **Skip full transitive/correlation clustering** — the problem is inherently bipartite/many-to-one (S2,S3→S1), not multi-partite clustering; correlation clustering is NP-hard and unneeded here. Only fall back to connected-components if S2 and S3 need reconciling against each other. — [Entity Resolution in Practice](https://arxiv.org/pdf/2607.26298)

## 5. Zero-label domain adaptation for France

1. **Leave-one-country-out (LOCO) as the dev proxy** — hold out India or US during development to simulate the unseen-France test; general ML precedent shows LOCO reliably captures degradation under distributional shift (no record-linkage-specific LOCO paper found — treat as adapted general-ML practice, flag the gap explicitly). — [Cross-country generalization, LOCO](https://arxiv.org/pdf/2605.26589)
2. **Fellegi-Sunter EM (Splink) run directly on France pairs** — label-free by construction, only needs blocking rules retuned for French tokens. — [Splink](https://github.com/moj-analytical-services/splink)
3. **Self-training / pseudo-labeling** from the GBDT's highest-confidence France predictions, retrain/calibrate — standard UDA practice, but noisy pseudo-labels are the known failure mode, so trust only extreme-confidence deciles and sanity-check against the LOCO proxy before shipping. — [Guiding Pseudo-labels with Uncertainty](https://arxiv.org/pdf/2303.03770)
4. **Don't assume France's no-match rate ≈ 5.6%** (the S1 baseline) — estimate it from the LOCO proxy instead; unstated because no source addresses this directly, flagged as a modeling risk.

## 6. French business name/address normalization

1. **libpostal (MIT license)** as the base normalizer — pretrained on OSM data across 100+ countries/60 languages, handles French street abbreviations and CEDEX out of the box; safe as a preprocessing dependency, doesn't touch the MIT/Apache "final model" constraint since it's not the matching model itself. — [libpostal](https://github.com/openvenues/libpostal)
2. **Manual regex table for legal-form suffixes** (SARL, SAS, SASU, SA, EURL, SCI, SNC, EI, EIRL, auto-entrepreneur, "Ets"/"Établissements") — small deterministic dict, same pattern as normalizing Ltd/LLC/Pvt Ltd for US/India, no licensing risk. — [French legal forms overview](https://stripe.com/resources/more/business-legal-types-france)
3. **Street-type + article normalization dict**: av/bd/r/pl/imp/all/chem/rte/fg/qu, St/Ste→Saint/Sainte, fold l'/d'/de/la/le/du/des into stopwords — same normalization layer already needed for other countries' pipelines, French-specific dict only. — [Address Format in France 2026](https://batchdata.io/uncategorized/address-format-in-france)