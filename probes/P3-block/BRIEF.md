# P3-block: candidate generation ("search the relevant rows")
Goal: the recall ceiling. Target >= 0.97 pair recall at <= 40 candidates per S1, measured on holdout S1 against
the FULL train S2/S3 corpus (10.3M), not only the mini-world (the mini-world has fewer confusers).
Always block within country. Blockers to compare (research_techniques.md §1):
 B1 exact keys (name_core, city_key+first name token)       B2 IDF-weighted token inverted index, top-k
 B3 TF-IDF char 3/4-gram cosine top-k with sparse_dot_topn (Sparkly-style; name, and name+city variants)
 B4 MinHash-LSH on name tokens (optional)                    B5 embedding ANN (faiss) for the cross-script residual
 B6 unions / rank fusion; both directions: S1→SX top-k and SX→S1 top-k (each SX has at most one parent, so
    the reverse direction is natural)
Use probes/P1-taxonomy/normalize.py and probes/P2-script/translit.py when they appear (poll); until then,
lowercase + strip punctuation. Fit vectorizers on train only.
Report per country and per P1 difficulty class: pair recall, S1 full-recall, cands/S1, recall@k curves
(k = 5, 10, 20, 50, 100), build/query time, peak RAM, and extrapolated full-test runtime.
Deliverable: probes/P3-block/block.py: candidates(s1_df, sx_df, cfg) -> DataFrame[s1, sx, block_score,
block_rank_s1, block_rank_sx, via] and the recommended cfg.

Note: if the full-corpus evaluation does not fit your memory cap, run it per country and in chunks of S1
queries (the corpus index is the big part — build it once per country, query in chunks), or report the
largest corpus size that fits and extrapolate. Say which you did.
