# P8-pipeline: the end-to-end pipeline (src/run.py) and the final package
You run in your own git worktree on branch wave/p8-pipeline, and you MAY edit code/business_entity_resolution/ there
(an exception to the _COMMON rule). The master merges your branch. Read probes/MERGE-1.md first.
Goal: a runnable, reproducible pipeline from the raw TSVs to both output files. Every stage is pluggable, so P4's
matcher and P5's decision layer drop in as soon as they land (P4 and P5 are running in parallel right now).

Build, in this order:
1. `src/block.py` — promote probes/P3-block/block.py and the relevant parts of full.py, keeping CFG. Re-point its
   imports to src/translit.py and src/normalize.py.
2. `src/run.py` — `python src/run.py --split test|holdout --stage rules|model --out output/`.
   a. Texts/features via `rules.feats` (P2 transliteration + P1 normalisation).
   b. Candidates = P3 blocking ∪ rule-key pairs.
   c. Scoring:
      - `rules` = the `rules.match` accept set. This is v0 and must reproduce dense 0.8201.
      - `model` = P4's matcher through ONE interface, `score(pairs, f1, fx) -> np.ndarray` of probabilities.
        Until P4 lands, use a stub (logistic regression on block_score + rule flags) so the interface is exercised.
   d. Decide: P5's `decide(pairs[s1,sx,p], cfg) -> dict[s1,set]` if present, else each SX to its argmax S1 + a global threshold.
   e. Write matching_results.tsv + candidate_pairs.tsv, per country (P7 run shape; `probes/P7-scale/chunks.py`
      write_lists). Assert every rule: one row per test S1 incl. France, matches ⊆ candidates, S2-/S3- ids only, no
      duplicates. Then run the official validator.
   - Holdout mode = dense: holdout S1 against the full train corpus and all train S1. Reuse
     data/work/p3/cands_holdout_full_*.parquet. Score it and append one `pipe` ledger row per run.
3. `src/test_pipeline.py` — the plan's end-to-end guard. Every S1 of any country label, including an injected
   unseen "Xland", gets exactly one row; matches ⊆ candidates; ids valid.
4. Packaging:
   - code/business_entity_resolution/README.md: exact reproduce steps, data → blocking → matching → output.
   - requirements.txt: pinned, only what src/ imports.
   - `src/package.py`: builds `<team>_submission.zip` (output/, code/business_entity_resolution/, and the filled
     Documentation_template.md when it exists) and checks its structure.
5. Reproducibility: every artifact the pipeline needs must be rebuilt from the training data by run.py if absent —
   the P2 transliteration table, place_freq, TF-IDF vectorisers and the model. Nothing is downloaded or hand-copied.

Acceptance:
- `run.py --split holdout --stage rules` reproduces dense F0.5 0.8201 ± 0.002.
- test_pipeline passes.
- `run.py --split test --stage rules` output passes the official validator.
- Commit your work on your branch as you go (git add + git commit inside your worktree — never push, never touch main).
Memory: your cap is 3 GB INCLUDING the CLI. Run test scale one country at a time. P3's test candidates already exist in
data/work/p3/cands_test_*.parquet — reuse them, but the code path to recompute them must exist and be tested on dev.
