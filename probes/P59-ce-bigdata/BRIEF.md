# P59-ce-bigdata: retrain the bge cross-encoder on ~2.5× more labelled pairs (folds 1–5), re-score, refit
Why: P45's leakage check showed ce_bge_pool.pt was trained only on folds ≥ 4 (fold 3 = validation; P4b rerank.band() reads
fold ≥ 3; P13 build() uses fold ≥ 4 for train). The #8c holdout is fold 0. So folds 1–3 labelled pairs were never used to
train the CE. The CE is our most valuable model (France's biggest gains; r_all gave India/US +0.0005 LB-eq on holdout).
Training on folds 1–5 (keeping fold 0 out) gives ~2.5× more pairs while the holdout stays clean for the combiner.
Read probes/P13-aws-rerank (training job, pair channels, hyperparameters, the dict-comment crash), src/bge_train.py,
probes/P45-ens-stages (r_all scoring jobs for holdout + test, formats in data/work/p45/oof/), probes/P44-meta-stack /
P50-idea-batch (combiner refit: te44.py, asm50.py), probes/P51-ce-serialize/sub51.sh and P38 build38.py (France rebuild from
any band r + OOD weights), P48/P54 rules that use r. Own worktree; code in probes/P59-ce-bigdata/.
Plan (AWS, --profile harsh; quota 1 per instance type):
1. Build the fold 1–5 band-pair channel (same band / sampling recipe as P13; validation = a slice of fold 1 or 2, never fold 0).
   Smoke job (5 min), then the full fine-tune on the fastest free single GPU (try ml.g6e.2xlarge / g6e.4xlarge / g5.4xlarge /
   g6.4xlarge in parallel; keep the first to start). Same epochs/lr as P13 unless the step count forces a cap; checkpoint to S3.
2. Score in parallel shards across instance types: all holdout pairs (for r_all OOF), all test India/US argmax pairs, and the
   France pair file's band + out-of-band pairs.
3. India/US: refit the combiner (P50's full-holdout recipe) with the new r_all; holdout half-split vs the current (old-CE)
   combiner, paired SE. Gate ≥ +0.00010 at ≥ 3 SE, neither half negative.
   France: LOCO needs a CE that hasn't seen the target country — also train two LOCO variants (US folds 1–5 only → score India
   holdout band; India folds 1–5 → US) IF time allows, and compare with P13's LOCO bge in P38's gate. If there's no time for
   LOCO, only India/US ships.
4. If it passes: build output/sub59 on the current best staged file (sub_clean if present, else sub54), validator --check-ids,
   send the change set to P50 (integrator).
Budget $40 AWS. Memory 1.5 GB local; keep data on S3. DISK ~3 GB. Never end your turn to wait.
Deadlines: training started by 07:00; India/US verdict by 11:30; sub59 by 12:30 (hard stop 13:30 for the final package).
