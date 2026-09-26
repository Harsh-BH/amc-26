# From Aditya's laptop (night run, 27 Sep 03:00–05:30 IST) — for Harsh / the master

## 1. Upload candidate: N54 — remove France same-address org-word-swap decoys (France-only REMOVAL)
probes/N54-fr-decoyword-removal/RESULT.md. Decoys that swap one org word (comite / ecole / amicale / club / parents …)
and KEEP the house number. In-domain the cell (sub1 swap, swapped-in word lens n_hn ≥ 20 & s > 0.2, hn equal, argmax)
is 0/65 true (US holdout) and 1/759 (India). France's lens for these 83 words mirrors India's decoy words. France's base
accepts ≈ 5.4k of them, ≈ 4.7k via post-threshold rules (mostly P42 hn-extra p ≥ 0.8, which has no word check).
  .venv/bin/python probes/N54-fr-decoyword-removal/p54.py cells
  .venv/bin/python probes/N54-fr-decoyword-removal/p54.py build output/<best> output/sub54r ruled   # then validator
Expected +0.0004–0.0006 LB (break-even at 29% false; a label-free admixture test puts the removed pairs at ≈ 90–100% decoy — see the N54 addendum). One upload measures it cleanly (France-only diff).
Also worth adding the same word guard inside hn-extra / gc in run.py.
**P47 (shape matching) MUST apply this guard**: the same-hn decoy-word cell holds ≈ 7.9k free argmax France pairs at
p 0.3–0.8 whose profile is pure decoy (N54 addendum); a descending-p fill would take them. Exclude
data/work/p54/cells.parquet pairs (after `p54.py cells`) from any France add rule.

## 1b. Upload candidate: N55 — ADD ≈ 12.7k France same-address noise-word variants (fils / cie / associés / groupe)
probes/N55-fr-noiseword-fill/RESULT.md. Same template as P46's développement swaps, but P46's R05 guards exclude R1
fillers, P28's guard blocks them, hn-extra needs p ≥ 0.8. France lens: fils s 0.020 e 0.96, associés 0.020 / 0.96,
cie 0.044 / 0.92 (decoy words s ≈ 0.4). In-domain analogue cell 0.998 (US) / 0.997 (India) true.
  .venv/bin/python probes/N55-fr-noiseword-fill/n55.py analyse
  .venv/bin/python probes/N55-fr-noiseword-fill/n55.py build output/<base> output/<base>_n55 strict   # then validator
Expected +0.0004–0.0006 (strict), +0.0005–0.0007 (all). Disjoint from N54 (apply N54 first, then N55).
Suggested upload order: best + N54 ruled → + N55 strict → + N55 all / N54 all.

## 2. Already-run answers for P49-twin-prior (save its time)
- Test 3 (raw-string channel): train US, twins by normalised core, address-less true SX: SX raw name == true twin's raw
  name 4.3% vs == other twin's 5.3% (twins' raw names differ in 26% of cases). No signal. probes/NIGHT/twin_raw.py
- Test 5 (within-group distinctness): TRUE twin's cluster holds an addressed SX with the same raw name 8.1% vs 6.1% for
  false twins — reverse and weak; split-case true-in-no-dup 43% (k=2, chance 50%). probes/NIGHT/twin_dupname.py

## 3. France negatives from tonight (don't repeat)
- N49 fuzzy street (same core + hn + city, street typo): only 2,443 pairs new after P28/P42/P37, India bound 0.72. Drop.
- N48 filler phrases: SUPERSEDED by N55 (my earlier 'drop' was wrong — the cores keep the filler tokens).
- N53 pure drop1 at identical address: 1,026 new pairs, ≈ +0.00004. Drop.
- N50 exact name at a different address (S3, unique name): valid but 150 pairs. Drop.
- N51 France FP audit: accepted sibling-offset pairs are clean (holdout accepted hc==1 0.98 / 0.99 true). No FP cell except N54.
- France blocking: only ≈ 2.3k never-candidate SX share core+city+hn with an S1, and the 516 unique-key ones are
  different streets (coincident numbers). Drop.
- Validation note: for France cells, "precision among pred==False on the India/US holdout" is the wrong bar (the
  in-domain model rejects only a hard 0.1–1% remainder). Use negatives-per-S1 × France S1 vs the pairs added.
