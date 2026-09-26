# P29-fr-tiers: France rebuild — house-number parser fix + R1 fillers + bge, then safe "key tier" acceptance
Base: our best LB = #9ab 0.984072 (#8c India/US + France = P14 IDF (idf144) candidates, m2_big, P13 bge blend with
out-of-domain weights, t 0.985). France is ~75% of our gap to #1 (0.990556). Read: probes/P19-bge-idf (how #9ab's France
file was built: data/work/p19/prob_test_France_idf144_bge.parquet), P14 (combine.py, compare.py), P20 (R1 fillers
fils/cie/associes/freres/frs, fr20.py), P6 france.py (parenthesised house number "(520) RUE X" → 520: 1.6% of France
S2/S3 get no hn today), P15/P28 (sibling offsets), P13 (bge scoring recipe; AWS scoring-only job). Own worktree; code under
probes/P29-fr-tiers/. AWS `--profile harsh` for bge top-up scoring (credits fine). Memory 3 GB locally.
Task (in order):
1. **France rebuild:** IDF world of #9ab + hn parser fix (parenthesised numbers) + R1 filler stripping in the name core
   → rules / features / pruner / m2_big → bge top-up on new band pairs (AWS, P19's recipe) → P13 France blend. Pick
   France t at #9ab's profile (empty 6.71%, 3.02 matches/S1) AND report the t for empty = 6.43% (#9ab_t975-like).
2. **Key tiers (label-free rule, general code):** accept a France pair (if it is the SX's argmax) when cleaned name core
   == S1 core AND house number equal AND city equal AND that core is unique among France S1 (chain size 1).
   First measure on the US/India full-corpus holdout (labels): tier precision (expect 97–99.8%), count, ΔF0.5 on the
   #8c yardstick (report half, paired SE) if applied there too. Then apply to France: count pairs changed (~7.8k expected)
   and empty S1 fixed (~267).
3. Also enforce the per-S1 cap seen in train (≤ 5 S2 and ≤ 6 S3 matches per S1; keep the highest p) — report how many S1
   it touches.
4. Build France files on the #8c base with `probes/P14-france-shift/combine.py output/sub8c <file> <t> output/sub29<x>`:
   sub29a = rebuild only; sub29b = rebuild + tiers (+cap). Validator on each. RESULT.md with tables, $, commands.
Never end your turn to wait. Deadline 26 Sep 21:00.
