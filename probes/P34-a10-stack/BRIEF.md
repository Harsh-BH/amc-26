# P34-a10-stack: ship P23's a10 residual stacker for India/US in its p2-masked form (P23-research "Brief 7")
Read probes/P23-research/RESULT.md (Brief 7: a10 residual stacker, p2-masked; measured +0.000166 / +0.000209 LB after
reweighting the holdout to test's mix), probes/P17-hn-stack (the current US stacker in #8c) and probes/P23-error-analysis
(the #8c holdout dump). Base India/US = #8c (unchanged in #31, LB 0.9856). Own worktree; code under probes/P34-a10-stack/.
Task: build the a10 stacker exactly as briefed, fit on holdout half 0 only, report half 1 with paired SE on the #8c
yardstick for India and US separately (and the half-swap). Gate: India+US ≥ +0.00015 at ≥ 3 SE and neither country
< −0.00005. If it passes, produce India/US test probabilities on #6's pruned rows and build output/sub34 = #31 with
India/US replaced (France unchanged) via src/submit_probs.py or an equivalent combine (+ validator), plus the run.py
integration notes for P9. RESULT.md with tables. Memory 3 GB. Never end your turn to wait. Deadline 26 Sep 22:30.
