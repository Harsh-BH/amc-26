# P48-fr-ce-all: France's CE sees only the band (0.02 < p < 0.98) — score EVERY France pair and let it veto / rescue
Why: P45 found that scoring our shipped bge CE (ce_bge_pool.pt) on EVERY India/US pair (r_all) instead of only the band
adds +0.00033 LB-eq on top of P44's combiner (10 SE; clean of leakage): it vetoes #8c p ≥ 0.98 pairs that are 74–90%
false and rescues p < 0.02 pairs that are 80–100% true. France's chain blends the CE only on the band too (P31: out-of-
domain weights [0.9245, 0.2595, 0.0515]; P30's veto extends past t at τ 0.1). Out-of-band France pairs never meet the CE.
Current best France = output/sub46_on43's France (P42 joint recipe t 0.99 / rescue 0.90 / gc 0.70 / hn extras 0.80 +
P41 gib + P40 xf_noaddr + P46 noise-word tier). The India/US best will be output/sub44r (P44; France byte-identical).
Read: probes/P45-ens-stages (r_all scoring code + data/work/p45/oof/holdout_*.parquet with p_r_all on EVERY holdout pair,
and the GPU scoring job for test), probes/P42-fr-joint-tune/solution_p42.py + P37 p37.py (LOCO France-chain analogue),
probes/P30-bge-veto, probes/P38-qwen4b-fr/build38.py (reproduces the France chain byte-exact). Own worktree; code in
probes/P48-fr-ce-all/.
Task:
1. LOCO first (no GPU needed): in the France-chain analogue, use p_r_all for out-of-band pairs. Arms: (a) veto: drop
   accepted pairs whose r_all logit < v; (b) rescue: accept free argmax non-sibling pairs with p < 0.02 but r_all > u;
   (c) full-range blend with the out-of-domain weights; grid v/u. Report both directions vs the current recipe analogue,
   with precision of the flipped pairs. Gate: both directions ≥ 0 and India→US ≥ +0.0005 France-F0.5, or both ≥ +0.0003.
2. If it passes: score the out-of-band France pairs with ce_bge_pool on ONE AWS GPU (reuse P45's job; ml.g5.2xlarge or
   ml.g6.2xlarge — P38 uses g6e; quota 1 per type), ≈ 1M pairs, ~15–20 min. Build output/sub48 = output/sub44r with
   France changed by the passing arm (India/US byte-identical to sub44r; if sub44r isn't staged yet, build on
   sub46_on43 and say so). Validator --check-ids, SUBMISSIONS.md row with expected LB (France dF × 0.1498).
Budget $10. Memory 2 GB local. DISK ~1.9 GB: small files only. Never end your turn to wait.
Deadline: LOCO verdict by 05:30, sub48 by 07:00.
