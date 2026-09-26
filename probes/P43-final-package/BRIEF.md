# P43-final-package: make the final zip reproducible and compliant NOW, so the morning only swaps in winners
Current best = output/sub37_gc70 (LB 0.986304) = #34_veto (India/US #8c + P34 a10; France #31 chain + P30 veto)
+ P34v rescue p ≥ 0.90 (France empty S1 → best non-sibling argmax candidate) + P37 group completion gc p ≥ 0.70.
Staged candidates for later: sub37_gc70_hn90 (P37), sub39_ind80 (P39), P38/P41/P42 outputs as they land.
Read: docs/PROBLEM.md (incl. "Organiser rulings" sections: zip layout, candidate-set size ranks higher, unseen-country
threshold allowed but no hard-coded country list, no >8B model), SUBMISSIONS.md, src/run.py + src/france.py,
probes/P37-fr-complete (p37.py gc_add + run.py notes), probes/P9-integrate/RESULT.md, package.py, Documentation_template.md,
APPROACH.md, SETUP.md. Work in your own worktree/branch; merge to main only when every check passes.
Tasks:
1. run.py recipe `sub37` (default) that reproduces output/sub37_gc70/matching_results.tsv BYTE FOR BYTE from cached
   stage outputs (rescue + gc implemented in src/france.py with a self-check assert; flags for bars so hn90 / later
   winners are one flag away: e.g. --fr-extra hn90).
2. Replace every `c == "France"` branch in src/ with `c in UNSEEN` (UNSEEN = test S1 country labels absent from train
   S1). Behaviour must stay byte-identical (re-run the recipe and diff).
3. Small-candidate version: rebuild candidate_pairs.tsv the way sub31_small did (4.42 cands/S1) for the sub37 recipe;
   matching_results identical; validator PASS with --check-ids. Put it at output/sub37_final_small/.
4. Docs: fill Documentation_template.md, APPROACH.md, SETUP.md, SUBMISSIONS.md with the current recipe and the LB
   history (up to 0.986304), leaving clearly marked slots for later winners. Build Yeti_submission.zip with package.py
   (layout per the rulings), unzip into a temp dir and run the validator on it.
5. Back up the zip + output/sub37_final_small to s3://sagemaker-114872709824-ap-south-1/amc26-offload/final/ (AWS_PROFILE=harsh,
   rclone check before anything is deleted; delete nothing).
Heavy rebuilds may run as SageMaker CPU jobs (sm_launch.py, ml.m5.4xlarge $0.97/h, budget $10) if a stage would take
> 15 min locally. Memory 1.5 GB locally. DISK ~3 GB free: stage outputs via S3. Never end your turn to wait.
Deadline: steps 1–3 by 27 Sep 07:00 IST, 4–5 by 09:00.
