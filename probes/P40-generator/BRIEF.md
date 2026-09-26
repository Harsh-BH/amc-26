# P40-generator: reverse-engineer the data generator, find the operations we still get wrong (India/US AND France)
Why: we are 0.0043 behind #1 (0.990556 vs 0.986304). Implied F0.5: India/US ≈ 0.9903 (0.0083 LB lost), France ≈ 0.964
(0.0054 LB lost). Even a perfect France (0.99) gives 0.9902 < #1, so #1 is ahead on both. The data is synthetic: P36
found invented business names are built from exactly 44 syllables (same set learned on train and on test incl. France);
P37 found same-address SX with invented brand names are true variants (7,763 true vs 159 false at p > 0.6); sibling
decoys use house-number offsets {3,4,5,7,9,11,13,21}; P12 estimated some op rates (seg_drop, legal_swap, hn_change,
empty address). Top teams likely reverse-engineered the generator. We have never catalogued it systematically.
Read: probes/P12-synth/RESULT.md, probes/P15-research (sib rule), probes/P36-iu-residual (syllables gib.py, budget),
probes/P37-fr-complete/RESULT.md, probes/P23-research/RESULT.md (error budget), docs/PROBLEM.md. Own worktree; code in
probes/P40-generator/.
Task:
1. From train truth, catalogue every generator operation that turns an S1 into a true S2/S3 variant (name ops: typo kinds,
   legal-form swap/drop, reorder, abbreviation, invented name, "formerly:", website/handle forms, case/diacritics;
   address ops: hn format, hn change, street abbreviation, segment drop/reorder, region/state forms, empty address), and
   every way a DISTRACTOR SX (unowned, or owned by another S1) is made (sibling offsets, same-name other city, orphan
   entities with no S1 …). Rates per country; which ops co-occur; how many variants per S1 and per source.
2. Join to the #34 India/US holdout errors (P36 data/work/p36/err_*.parquet, out of fold): error rate and LB-eq lost per
   operation. The table "operation → share of the remaining 0.0083 LB" is the main deliverable.
3. For the top 3 ops by recoverable loss, design an exact or near-exact rule/feature (op detector) and measure it on the
   holdout (half-split, paired SE; gate India+US ≥ +0.00015 at ≥ 3 SE, neither < −0.00005).
4. Check the same ops on France test (label-free: op frequencies in France accepted pairs vs India/US; where France's
   op mix differs, say what that implies for France errors) and LOCO-check any op rule that applies to France (gate:
   ≥ +0.0005 both directions, or report both directions; the LB has tracked India→US so far).
5. If a rule passes, build output/sub40 on the current best (#37_gc70 or its successor), validator PASS, run.py notes.
RESULT.md with tables. Memory 3 GB. DISK NEARLY FULL (~1.2 GB): small intermediates only. Never end your turn to wait.
Deadline: op-loss table by 27 Sep 04:00 IST, sub40 by 11:00.
