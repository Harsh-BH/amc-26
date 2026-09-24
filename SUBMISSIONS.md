# Submission queue

Upload on the play page: **Matching Results File** = `matching_results.tsv`, **Code File** = the zip.
Max 5 per day; the day-1 slots expire at 23:59 IST on 25 Sep. Each staged folder under `output/` already
passed the official validator. After an upload, tell the master the leaderboard score so it goes in the
pipeline ledger (`ledger.py show pipe`).

"Dense F0.5" = holdout S1 scored while every train S1 competes, the closest proxy for the 1.73M-S1 test set.

| # | Status | Files | Holdout F0.5 | What it is |
|---|---|---|---|---|
| 1 | staged | `output/matching_results.tsv` + `output/code_s0.zip` | 0.1775 (plain) | exact name + first address segment. Sanity only — **skip it**, #2 tests the same upload path |
| 2 | **not generated** (the test-set run was stopped by Claude Code's memory safeguard at ~05:20; rerun takes ~10 min) | `output/sub2/matching_results.tsv` + `output/sub2/code.zip` | **0.8201 dense** (precision 0.994) | P2 transliteration → P1 normalizer → exact name/website key + address confirmation, unique best S1 |

**Upload first:** the highest-numbered staged row.

To (re)generate #2: `.venv/bin/python code/business_entity_resolution/src/rules.py test`, then validate from
`data/student_resource/` with `python3 utils/validate_submission.py --matching ../../output/sub2/matching_results.tsv
--candidate ../../output/sub2/candidate_pairs.tsv --test-dir dataset/test`. It forks 5 workers; with probes running,
lower `procs` in `rules.feats` to 2-3 (the 05:20 memory spike came from 5 workers alongside P3 and P7).
