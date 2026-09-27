# P62-public-ideas: three ideas from public repos (ideas only — read their README/docs if useful, NEVER copy code)
Source: public repos found by a search agent (not top-30 teams): aditya-bheke/business-entity-resolution (team NFG, LB 0.9810,
docs/EXPERIMENTS.md), rishabhiitj25/AmazonML26 (clone09 validation), Sid-techweb/AmazonML-New. Re-implement ideas our own way.
Current best staged: output/sub_clean (France chain without P46/P40 + P55; India/US = P44 combiner + r_all + full refit + zoo).
LB: gc70 0.986304, sub54 0.986685. P50 is the integrator (send passing change sets there). Own worktree; code in
probes/P62-public-ideas/. ≤ 45 min per idea; report duds.
1. Acronym rule: an SX whose name is the acronym of the S1 name core (e.g. "Dietetique Amis SAS" → "DA"), at an identical
   address (hn + street + city), is a match (NFG: 434/435 held-out pairs; 10× more common in France). Measure on the India/US
   holdout (precision; how many our current decision misses) and count on France test (how many sub_clean misses; are they
   free argmax non-sibling?). If holdout precision ≥ 0.95 and misses exist, add them (caps 5/6) as a change set for P50.
2. Dense-retrieval rescue for blocking/pruner misses: embed S1 and SX (name + address) with intfloat/multilingual-e5-small (MIT;
   P45 already computed e5 cosines — reuse its code), retrieve the top-k S1 per SX within country for SX that have NO accepted
   owner, and verify with our CE (bge r) + m2-style features. On the holdout: top-1 hit rate for true owners missed by blocking,
   precision after verification, LB-eq via the half-split (India/US gate +0.00010 at ≥ 3 SE, both halves positive). France:
   label-free counts plus LOCO if feasible. SageMaker CPU/GPU allowed for embedding (≤ $10; quota 1 per GPU type, P59/P60 hold
   g6e/g6 types).
3. clone09 validation (LB-tracking check): give every unmatched holdout SX ~0.9 exact clones, recompute the context features,
   and re-score #34 (gc70's India/US) vs sub54's India/US vs sub_clean's. Does the ordering change? This is a DENSITY-ROBUSTNESS
   check for the India/US choice; send the result to me and to P57 (P57-iu-transfer) as soon as you have it.
Memory 2 GB local. DISK ~2.5 GB. Never end your turn to wait. Deadlines: idea 3 by 08:30, ideas 1–2 by 10:30.
