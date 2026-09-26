# P53-fr-drop1-sameaddr — France pairs: SX name = S1 name minus exactly one word, IDENTICAL full address
Read probes/NIGHT/AGY_COMMON.md (REVISED VALIDATION BAR, MEMORY GOTCHAS) and probes/P28-addr-anchor/RESULT.md
(same-address key), probes/P48-fr-filler-tier/RESULT.md (its "pure drop1" finding used the OLD bar — redo it).

Master's quick count (P31 proxy base, data/work/p31/pairs_France.parquet): argmax pairs with name_rel=='drop1' and
hn_cls==0 and S1 hn non-empty: 24,895; accepted 20,108; rejected with SX free 4,787, of which 1,273 have p >= 0.8 in a
non-empty group (P42 hn-extra takes those) ⇒ ≈ 3,500 candidates. P51's holdout cell drop1 × hn-equal: US precision
0.999 over ALL pairs, negatives 0.000324 per S1; India 0.964, 0.00377 per S1 ⇒ bounds ≈ 0.976 (US) / 0.72 (India).
Note France has FEWER drop1 hn-equal pairs per S1 (≈ 0.096) than the US (≥ 0.31), so no sign of excess decoys.

Task:
1. Restrict to IDENTICAL address: hn equal AND street key equal AND city equal (use P28 anchor.street_key / its same-
   address definition; for France use the same key P28 used for France). Recompute the France candidates (free argmax,
   not accepted, not sibling), minus hn-extra (p >= 0.8 & S1 non-empty), minus P37 gc (one-sided group, missing source,
   p >= 0.70), minus P46/P48 (a gained word). Pure drop: nothing gained.
2. Recompute neg_per_s1 on US / India holdout (p23 dumps, holdout==True, join rf features + raw) for EXACTLY this cell,
   and the bounds. Also check: is the dropped word a legal form (SARL/SAS/EURL/LLC/Pvt Ltd) vs a descriptive word —
   report both sub-cells separately (bounds + France counts).
3. If a sub-cell passes the bar: p53.py build <base> <out> + selftest per AGY_COMMON; report added pairs and how many
   fill empty S1. Otherwise a clean negative.
Deadline: 08:30 IST.
