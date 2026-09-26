# N56-review — adversarial code + evidence review of N54 and N55 before they cost upload slots
Read probes/NIGHT/AGY_COMMON.md (hard rules; you write NO labels), then:
probes/N54-fr-decoyword-removal/{p54.py,RESULT.md}, probes/N55-fr-noiseword-fill/{n55.py,RESULT.md},
probes/NIGHT/{loco_n54_n55.py,loco_chain.py,loco_score.py,FOR_HARSH.md}, probes/P46-noise-word-tier/p46.py (lens),
probes/P28-addr-anchor/{loco.py,ho.py} (LOCO helpers).
Your job is to find BUGS or flawed reasoning, not to re-derive the results. Check specifically:
1. p54.build / n55.build: can they drop or add a pair for the wrong S1, touch India/US rows, break the header, duplicate an
   SX, exceed caps 5 S2 / 6 S3, add a pair not in candidate_pairs.tsv, or leave a trailing comma? Run both selftests and
   write extra adversarial selftests (in probes/N56-review/test_builds.py) — e.g. S1 rows with no trailing newline,
   empty rows, an SX already owned by another S1, candidates missing for an S1.
2. p54.cells / n55.analyse: are s1 / sx columns aligned after the filters and merges (index resets, reindex by ids)?
   Is the lens learned only on ARGMAX pairs (label-free)? Is p used anywhere it should not be?
3. loco_n54_n55.py / loco_chain.py: the id mapping from the S_holdout frames (country offsets OFF1/OFF2, CI order) —
   compare with P46's loco() and P28's loco.py. Is the per-S1 F0.5 (fbeta over fold-0 S1 incl. those with no pairs)
   computed like P28/P46? Is anything leaking truth into the rule (y used only for scoring)?
4. Any reason N54 'all' could remove pairs the base accepts at p ≥ 0.99 that are TRUE in France specifically (think
   about French org-word names; the lens words list is data/work/p54/decoy_words.csv)?
Deliverable: probes/N56-review/RESULT.md — a list of concrete issues (file:line, severity, fix) or "no bugs found" per
item, plus the adversarial selftests. Keep python ≤ 2 GB. Deadline: 09:30 IST.
