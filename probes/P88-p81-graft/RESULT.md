# P88: graft of sh-feat P81 (rival post-hoc ADD rules) onto sub86

- Input: sh-feat `probes/P81-rival-posthoc/cs81A12.parquet` (passes 1+2 adds, the rival's LB-backed part), France only.
- Filter (`profile.py` profiles every add against sub84/sub77/sub66_final, P31 France p and P70 street classes):
  - drop 203 veto re-adds (P31 p >= .9868, or owned by the same S1 in sub77 / sub66_final);
  - drop 405 different-street / typo-street pairs (the sub77_hnst class, LB ~.3; the typo class fails the India pool .55).
  - kept 10,689: same street 7,895, SX address-less 2,537, type-only 179, S1 street-less 78. P31 p quartiles .05 / .43 / .88.
- `cs88.parquet` applied with P81's `apply_cs.py apply output/sub86 cs88.parquet output/sub88`: +10,689, no cap
  overflow added (35 over-cap S1 = sub86's), France links/S1 3.2513 -> 3.2925, empty .0597 -> .0587.
- `cs89_p3.parquet`: the same filter on pass 3 (5,429 adds), staged only as a follow-up if sub88 gains.
- Price (P81's per-add values): France precision .75 -> ~0, .85 -> +0.0002, .90 -> +0.0003, on top of sub86's +0.00005.
