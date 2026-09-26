# P49-twin-prior: attack the "same-name, address-less" floor with the only levers the literature leaves
Context: P40 says 0.0037 LB of India/US loss is a Bayes floor — address-less SX whose full normalised name is shared by
≥ 2 S1 ("twins"). A literature review (Bayesian ER: Zanella/Betancourt microclustering priors; Sadinle 2017 bipartite
matching; Ye 2012 expected-F decoding; Steorts blink distortion) agrees identical observations are a posterior floor;
only (a) the grouping prior and (b) generator artifacts outside the normalised name can move it. P35 (twin propagation)
and P36 (EF decoding on #8c p) are prior art — read them; do not repeat.
Current India/US best = output/sub44r (P44 combiner + P45 r_all on #34; holdout decision reproducible from P44/P45
files: probes/P44-meta-stack, data/work/p45/oof/holdout_*.parquet). Own worktree; code in probes/P49-twin-prior/.
Run these five cheap label-based tests on the #8c holdout's ambiguous twin rows FIRST (each ≤ 30 min), report the true
rate vs 50% (or vs what the current decision already achieves) with SE:
1. Group-size likelihood ratio: f = per-source group-size distribution from train truth; for an ambiguous SX between twins
   A and B with confident accepted counts k_A, k_B (excluding ambiguous SX), ratio = f(k_A+1)/f(k_A) · f(k_B)/f(k_B+1).
   Does truth follow it (bucket by (k_A, k_B))? Is f non-geometric (esp. low P(size=0), caps 5/6)?
2. Joint exact decode per name-component (2–4 twin S1 × 1–5 ambiguous SX): enumerate assignments, score Π p × Π f(size),
   decode for expected F0.5; compare with per-SX argmax.
3. Raw-string noise channel: twins identical only after normalisation — among ambiguous rows where the twins' RAW names
   differ (legal form, punctuation, case), how often is the raw-closer twin the truth?
4. Script → region (India): does an Indic-script SX's script predict the true twin's state? (learn state→script on train
   only; check probes/P23-error-analysis/work/india-script/).
5. Within-group distinctness: do true groups ever contain two identical raw strings? If never, an SX identical to an
   accepted member of A belongs to B.
Any test with a clear signal → add it as feature(s)/decode step on top of sub44r's decision, holdout half-split with
paired SE, gate India+US ≥ +0.00015 at ≥ 3 SE and neither < −0.00005. If it passes, build output/sub49 = sub44r with
India/US replaced (France byte-identical), validator --check-ids, SUBMISSIONS.md row. If all five sit at ~50%, say so
plainly and stop — that closes the floor question.
Memory 2 GB local; SageMaker CPU allowed (sm_launch.py, ml.m5.4xlarge $0.97/h, budget $10). DISK ~1.9 GB. Never end
your turn to wait. Deadline: test table by 06:00, sub49 (if any) by 08:00.
