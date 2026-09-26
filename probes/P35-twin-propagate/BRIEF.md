# P35-twin-propagate: cross-source twin reasoning for ADDRESS-LESS records (the biggest remaining India/US loss bucket)
Evidence (probes/P23-research/RESULT.md error budget, exact on the #8c holdout): address-less S2/S3 records carry **59%
(India) / 71% (US)** of the F0.5 loss ≈ 0.0054 LB. With no address the matcher only has the name; among same-name S1
look-alikes it misses (low-p FN, "stolen" FN) or picks the wrong S1. Idea (Foursquare/Shopee winners' neighbour/group
step): an address-less SX often has a TWIN in the other source (S2↔S3, near-identical name core) that HAS an address and
is confidently matched to S1 X ⇒ the address-less SX likely belongs to X too. Also the reverse check: an address-less SX
whose best S1 differs from its confident twin's S1 is suspicious (FP guard). Read P23-research (R33/R38, a10 "twin
columns" in Brief 7 — do not duplicate P34, which ships a10; go further: explicit propagation / group decision), P23's
holdout dump (probes/P23-error-analysis), P15/P17 (support features), P31 (#31 = current best LB 0.9856; France file and
rule mechanics), P14 combine.py. Own worktree; code under probes/P35-twin-propagate/. CPU; memory 3 GB (shared, tight).
Task:
1. Build twins label-free: for each address-less SX, candidate twins = SX of the OTHER source sharing its S1 candidate
   list / name core (rapidfuzz ≥ τ on cores), with an address; twin confidence = its #8c p to its argmax S1.
2. Rules/features: (a) propagate: if the twin is matched to X with p ≥ τ1 and X is among the address-less SX's
   candidates, raise p(SX, X) (e.g. max with a calibrated value) and/or make X its argmax; (b) conflict guard: if the
   address-less SX's argmax ≠ twin's X, lower it. Also try as a small residual GBDT on top of #8c's p with twin columns.
3. Validate on labels: #8c holdout yardstick (report half, paired SE) per country, and the LOCO proxy both directions
   (France-like). Gate: India+US ≥ +0.0003 at ≥ 3 SE, no country < −0.0001; LOCO ≥ +0.0005 both directions for France use.
4. If it passes: apply to India/US (on #8c/#31's India/US p) and, if LOCO passes, to France (on #31's France file before the
   cap); build output/sub35 = #31 + twins via submit_probs/combine (+ validator). RESULT.md with tables + integration notes
   for P9. Never end your turn to wait. Deadline 26 Sep 23:30.
