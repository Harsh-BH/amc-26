# P21-fr-regions: replace the hand-written FR_REGIONS table with a mapping learned from the data (compliance)
Organiser ruling (docs/PROBLEM.md "Organiser rulings"): "large static country/state/city tables" and city/postal
tables are banned; unlabelled test data may be used (incl. pseudo-labels). `normalize.FR_REGIONS` (normalize.py:187,
13 régions → ~96 départements, used in normalize.py:220 and block.py:445) is a gray area. Own worktree; you may edit
src/normalize.py and src/block.py (nobody else edits src/ now).
Task:
1. Learn the département → région map from the data, label-free: e.g. for France S2/S3 records whose address names a
   département-like segment, take the région of the S1 they are matched to by the confident rule stage / m2_big
   (p ≥ 0.99 argmax), keep a token → région mapping when it is pure (≥ 95%) with enough support (≥ 20). Also cover
   segments that already equal a région name. Write it as a function computed at run time (cached under data/work).
2. Delete the FR_REGIONS table (keep only the 13 région NAMES if you need them — hand-written state names are
   explicitly allowed) and route normalize/block through the learned map.
3. Check: the learned map restricted to tokens that occur in test France must equal the table's mapping for those tokens
   (report any difference). Then `run.py --split test --stage model` (#8c recipe and the France variants) must stay
   byte-identical to output/sub8c / sub9a / sub9b / sub9s, or report the exact diff and its size. US/India holdout unchanged.
4. Update README/doc wording ("learned from confident pairs", no geo table). Commit on your branch.
Memory 3 GB. Never end your turn to wait. Deadline 26 Sep 18:00.
