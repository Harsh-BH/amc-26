# Submission queue

Upload on the play page: **Matching Results File** = `matching_results.tsv`, **Code File** = the zip.
Max 5 per day; the day-1 slots expire at 23:59 IST on 25 Sep. Each staged folder under `output/` already
passed the official validator. After an upload, tell the master the leaderboard score so it goes in the
pipeline ledger (`ledger.py show pipe`).

"Full-corpus F0.5" = holdout S1 scored while ALL train S1 and ALL train S2/S3 compete: the true train-side mirror of
test (found by P8 at 12:17). The earlier "dense world" number used only a sample of distractor records and
overstated precision (0.994 vs 0.912 for the rules).

| # | Status | Files | Holdout F0.5 | What it is |
|---|---|---|---|---|
| 1 | staged | `output/matching_results.tsv` + `output/code_s0.zip` | 0.1775 (plain) | exact name + first address segment. Sanity only — **skip it**, #2 tests the same upload path |
| 2 | **staged, validator PASS** — upload this | `output/sub2/matching_results.tsv` + `output/sub2/code.zip` | **0.7708 full-corpus** (precision 0.912; dense-world 0.8215 overstated) | P2 transliteration → P1 normaliser → exact name/website key within state + address confirmation, unique best S1. Test: 1,540,960 of 1,732,544 S1 matched (France 86%, India 87%, US 92%) |

**Upload first:** the highest-numbered staged row.

To regenerate #2: `.venv/bin/python code/business_entity_resolution/src/rules.py test` (13 min, per country), then the validator.
