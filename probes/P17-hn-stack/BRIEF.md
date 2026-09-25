# P17-hn-stack: generator-aware house-number + SX↔SX support stage (India/US)
Implement **Brief A** of probes/P15-research/RESULT.md exactly (read the whole RESULT first, incl. Evidence and
Risks; reuse probes/P15-research/hn_stack.py / hn_check.py — do not rewrite them). Own worktree; code under
probes/P17-hn-stack/ (src/ is P9's; propose the run.py integration in RESULT.md).

Base = **#7** (current default recipe in run.py: pruner τ 0.03, P11's 2-epoch CE `data/work/p11/ce_s2P.pt`,
w [0.617, 0.666, 0.261], India/US t 0.71, France t 0.95). Holdout #7 = 0.98881 (half 1 0.98869). Stack on #7's
blended p over the **pruned** candidate list (support features must be computed on the list the model actually sees).
Train on holdout half 0 only; report on half 1 with paired SE on the #7 yardstick (P9's run.py holdout / P10 evalp).
Acceptance: India+US report half ≥ +0.0003 at ≥ 3 SE and India ≥ −0.0001. Add brief C's branch-word columns
(add1, add_in_list with a TRAIN-mined list) as a second arm if time allows.
If accepted: produce data/work/p17/prob_test_{India,US}.parquet (s1, sx, p = stacked p) on #7's pruned rows + the new
t, so the master can write #8 with src/submit_probs.py (France = #7's France unchanged). CPU only (no GPU needed).
Memory cap 3 GB incl. CLI, swap allowed; disk ~5 GB free — keep scratch small. Never end your turn to wait; ledger
every run. Deadline: results by 26 Sep 12:00 IST (freeze is 27 Sep 14:00).
