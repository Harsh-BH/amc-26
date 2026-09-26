# P50-idea-batch: six quick, untested ideas — test each cheaply, ship only what passes
Current best stack = output/sub44r (France = sub46_on43's France: P42 joint recipe + P41 gib + P40 xf_noaddr + P46 noise
tier; India/US = P44 combiner + P45 r_all). Machinery: France LOCO analogue (probes/P42-fr-joint-tune/solution_p42.py,
P37 p37.py, P48 if it has r_all in the analogue), India/US holdout half-split (probes/P44-meta-stack te44.py / solution_p44.py,
data/work/p45/oof/holdout_*.parquet), src/normalize.py (LEGAL, STREET, hn parsing), src/france.py. Own worktree; code in
probes/P50-idea-batch/. Each idea: ≤ 45 min, report even if it's a dud.
A. French house-number suffixes/ranges: "12 bis", "12bis", "12 ter", "12-14", "12 à 14", "12/14". How does our hn parser
   treat them (hn equality, sibling flags, c_hn features)? Count France test pairs affected (accepted vs rejected, p bands);
   if a fix is warranted, normalise (bis/ter → base number, ranges → match either end) and measure via the France chain
   (label-free counts + LOCO where the pattern exists in US/India, e.g. "12A", "12-14").
B. French legal forms: list France S1/SX name tokens that look like legal forms (SAS, SASU, SARL, EURL, SA, SCI, SNC, SCOP,
   SELARL, SELAS, GIE, "S.A.R.L.", "Sàrl", "S.A.S.U." …) and check coverage in normalize.LEGAL (+ dotted/accented forms).
   If gaps exist, measure how many France pairs' name similarity changes and the France-chain effect.
C. Exclusivity veto for France: P44's sx_psum (p mass over the SX's candidate S1s) — at p .5–.95 with other-S1 mass ≥ 0.3,
   India/US truth rate 0.45. Apply as a France remover/stricter bar in the LOCO analogue, both directions.
D. CE text-order TTA: does scoring (SX, S1) as well as (S1, SX) and averaging change bge r? Test on a 20k holdout band
   sample on one AWS GPU (ml.g5.2xlarge or g6.2xlarge; P38 uses g6e; quota 1 per type) — AUC / stacked gain. Skip if
   the order-swap correlation is > 0.995.
E. Fit the India/US combiner on the FULL holdout for shipping (both halves), cross-fit validated: measure the expected
   gain from 2× training data (learning curve: fit on 25/50/75% of h0, score h1) and, if the curve is still rising, build
   the full-holdout refit.
F. Seed bagging 5 → 20 for the combiner: half-split paired SE vs 5 seeds.
Ship rules: France ideas need LOCO both directions ≥ 0 and India→US ≥ +0.0003 France-F0.5 (or clear label-free
evidence for pure parsing fixes like A/B, with counts); India/US ideas need ≥ +0.00010 at ≥ 3 SE and neither half
negative. Stack all passing ideas into ONE file: output/sub50 = sub44r + passing changes (validator --check-ids,
SUBMISSIONS.md row with expected LB). Budget $10 AWS. Memory 2 GB local. DISK ~1.9 GB. Never end your turn to wait.
Deadline: per-idea table by 07:00, sub50 by 08:30.
