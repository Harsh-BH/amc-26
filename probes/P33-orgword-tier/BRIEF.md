# P33-orgword-tier: France org-word tier (P23-research "Brief 4", R05, gated s(w) ≤ 0.4) on top of #31
Read probes/P23-research/RESULT.md (Brief 4 / R05 and its evidence: org-word swaps where the swapped word w is rarely a
sibling marker; s(w) = label-free sibling-marker score), probes/P20-france-patterns (the 75k "one core word swapped,
hn equal" cell; US 89% true), probes/P28-addr-anchor (the pure swap rule FAILED on labels: in-domain 0.2–0.7% true — so
the gate on w is essential), probes/P31-final-france (the #31 France file + how rules are applied: tier/bxneg/veto/cap,
combine.py). Base: **#31 (LB 0.9856)**, France file in P31's RESULT. Own worktree; code under probes/P33-orgword-tier/.
Task: implement R05 exactly as Brief 4 specifies (same address — hn+street+city equal — and the name cores differ by one
org word w with s(w) ≤ 0.4, learned label-free from the data; sibling-word guard as P28's bxneg). Validate on labels
FIRST: in-domain US/India holdout (#8c yardstick, paired SE) and the LOCO proxy (both directions) — report rule-pair
precision and ΔF0.5; gate: LOCO ≥ +0.0005 in both directions and no in-domain loss beyond −0.0001. If it passes, apply
on top of #31's France (before the cap) and build output/sub33 via combine.py (+ validator). RESULT.md with tables.
Memory 3 GB. Never end your turn to wait. Deadline 26 Sep 22:30.
