# N57-loco-cellscan — systematic LOCO scan for France removal / addition cells (the method that found N54)
Read probes/NIGHT/AGY_COMMON.md (hard rules: you write NO labels; truth is used only to SCORE rules on the LOCO frame),
then probes/N54-fr-decoyword-removal/RESULT.md (addenda 3–5) and probes/NIGHT/{loco_chain.py,loco_gc70.py,loco_n54_n55.py}.
Local LOCO frames exist: data/work/p6/S_holdout_{US,India}.parquet (fold-0 train pairs scored by the OTHER country's
booster = the unseen-country situation). loco_chain.chain(src, tgt) builds a France-like decision (t_profile base + hn-extra
0.8 + gc 0.7 + rescue 0.9, N54 applied) and returns per-pair columns rel (name_rel), band (gained-word lens band),
hrel (hn_eq / hn_diff / hn_missing), p, y, plus per(), hv (halves).
Task:
1. Use the gc70-style chain (loco_gc70.py: no hn-extra; that is the current final France) AND the chain with hn-extra.
2. Enumerate cells = rel × band × hrel × p-band (p < 0.3, 0.3–0.7, 0.7–0.9, 0.9–t, ≥ t). For every cell with ≥ 50
   accepted pairs in BOTH directions: ΔF0.5 of REMOVING the cell's accepted pairs, per half, both directions (US→India,
   India→US). For every cell with ≥ 50 unaccepted free argmax pairs: ΔF0.5 of ADDING them (free SX, caps 5/6).
3. Keep only cells positive in BOTH directions AND BOTH halves with |Δ| > 2 SE each. For each survivor, count the France
   analogue on data/work/p31/pairs_France.parquet (same rel/band/hrel/p definitions; lens learned on France argmax pairs
   with p46.lens_table; band cut-offs as in loco_chain.py) — accepted pairs (for removals) / free argmax unaccepted (adds),
   excluding N54 cells (data/work/p54/cells.parquet).
4. Report a table of survivors with LOCO numbers + France counts + expected LB (France-F0.5 Δ × 0.15 scaled by France
   count / LOCO count is fine as a rough guide). If a survivor is big (≥ 1k France pairs), write probes/N57-loco-cellscan/
   n57.py build <base> <out> like p54.py. A clean "no survivors" is a valid result.
Memory: one python at a time, ≤ 2.8 GB (watchdog kills above 3 GB). Runtime of one chain ≈ 30 s; cache the chain frames.
Deadline: 11:00 IST.
