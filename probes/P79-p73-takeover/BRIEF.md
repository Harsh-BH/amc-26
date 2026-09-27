# P79-p73-takeover: integrate P73's running AWS training arms (P73's session died: usage limit) — user: "try every training experiment and model"
P73 (probes/P73-fr-independent: BRIEF.md, RESULT.md draft, code, stream.jsonl for its last state) trains France-INDEPENDENT
cross-encoders (no labelled India/US pair in training: P67 v2 synthetic target pairs + P61 agreement pseudo-labels on the target
band; replay arm "use our training data also" added by the user at 17:15) and gates each as a new column in P55's France combiner
at the shipped cell (India→US ≥ +0.0003 AND US→India ≥ 0 France-F0.5, both halves).
Running SageMaker jobs (AWS_PROFILE=harsh ONLY): p73r-us / p73r-in / p73r-fr (replay arms), p73c-bgl / -mxl / -ele6 / -mxb / -bgb /
-mml (backbone / epoch variants: bge large/base, mxbai, electra, mMiniLM…), p73b-fr. Its staged deliverable: output/sub73e
(--check-ids PASS; on top of sub77: +792 adds / −154 removes new).
Current best: output/sub76 LB 0.988292 (final zip). output/sub77 (= sub76 + P41 brands) scoring now. output/sub78g = sub77 + groupe
swaps (staged). Final build: code/business_entity_resolution/src/final_build.py (change sets data/work/p78/final_changes.parquet).
Tasks:
1. Read P73's code/state; poll the jobs (no sleep loops > needed; use background waits). As each finishes, fetch its LOCO/France
   scores and run P73's gate. Report a table: arm, backbone, data, India→US, US→India, France adds/removes, Qwen audit.
2. BUDGET GUARD (important): the account is a Free plan with finite credits (≈$194 at the start of 27 Sep; ~$85 billed so far in
   Sep per Cost Explorer, which lags a day). The account CLOSES at $0. Hard cap for P73+P79 from now: $45. Stop any job running
   > 60 min, any arm whose first LOCO direction already fails, and anything still Pending at 18:30 (capacity). Log $ per job.
3. If one or more arms pass: build output/sub79 = the best current base (sub77 if it beats 0.988292, else sub76) + that arm's
   France change set (combiner re-run with the new column; adds + vetoes as P73 staged sub73e), Qwen/P64/N54 consistency checks,
   validator --check-ids, SUBMISSIONS row with expected LB, and add the step to final_build.py so it rebuilds byte-for-byte.
4. Message lead amc-26-a7 with the gate table as arms finish; final verdict by 19:30.
RAM ≤ 2 GB local, disk ~2.8 GB. Never end your turn to wait.
