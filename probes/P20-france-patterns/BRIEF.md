# P20-france-patterns: read French records, find the noise patterns our model mishandles, turn them into code rules
You (a strong LLM) are the pattern finder. **Hard rule:** you may READ France data and describe patterns, but you must
NOT label pairs as match/non-match for training, and no LLM may take part in any match decision. Everything you
deliver must be general code (normalisation rules / features / filters) that works on any record, like
probes/P6-france/france.py or probes/P15-research/sib_rule.py. No external data, no per-record lists.
Read first: probes/P14-france-shift/RESULT.md (IDF is France's main shift), probes/P15-research/RESULT.md (sibling
house-number fingerprint, branch words), probes/P6-france/RESULT.md + france.py, normalize.py's French word lists,
SUBMISSIONS.md (#8c / #9a / #9b / #9s). Own worktree; code under probes/P20-france-patterns/.

Material (all France, test split; data/work):
- the France probability files of the variants: p10/prob_test_France.parquet (#8c's France, t 0.95),
  p14/prob_test_France_idf144.parquet (#9a, t 0.975), p13/prob_test_France_blend.parquet (#9b, t 0.98),
  p15/prob_test_France_sib.parquet (#9s); raw records in test_source{1,2,3}.parquet; record features rf_test_*_France.
- Build three samples (~150–300 S1 groups each, seeded): (a) S1 whose answers DISAGREE between the variants;
  (b) pairs in the uncertain band (0.3 ≤ p < 0.95); (c) for contrast, confident matches (p ≥ 0.99) and confident
  rejections that share the name core. For each S1 group print the S1 row and every candidate S2/S3 row with p per variant.
Task:
1. Read the samples and write a taxonomy of the differences between records that look like the same business vs
   look-alikes: abbreviations, word order, accents, legal forms, branch/sibling words, house-number formats and
   offsets, street-type variants, city/département forms, missing segments, digit swaps, etc. Count each type on the
   samples (and, where a pattern is codable, on the full France test by a regex/feature count).
2. For each pattern say whether our pipeline already handles it (normalize.py / france.py / features.py) and, if not,
   what it costs (how many uncertain/disagreeing pairs it touches).
3. Implement the top 2–4 unhandled patterns as code (a normalisation rule or a new feature/filter). Validate label-free:
   US/India holdout must not move (or must improve) if the rule is country-agnostic; France answer profile stays in
   P6's bands; report how many France S1 change. If a rule needs the matcher retrained (new feature), train on
   US/India only and check the holdout on our standard yardstick (run.py --split holdout).
4. Deliver the best variant as a France probability file on the #8c base + the exact combine.py command
   (probes/P14-france-shift/combine.py output/sub8c <file> <t> <out>, t = France's empty rate matching D4's 6.85%).
RESULT.md: taxonomy table with counts and example rows (few, short), what is already handled, what you implemented,
evidence, and the recommendation. Memory 3 GB. Never end your turn to wait. Finish by 26 Sep 17:00 IST.
