# N55-fr-noiseword-fill — RESULT (master, laptop, 27 Sep ~05:10 IST)
## TL;DR
- **≈ 12.7k France true-variant pairs are still rejected**: same address (hn + street key + city identical), SX = S1
  name with one word swapped for / plus one France NOISE word — fils 5,857, cie 3,035, groupe 2,517, associés 1,315
  (proxy count after removing what hn-extra p ≥ 0.8 and P46 développement already add); 721 fill an empty S1.
  Why missed: P46's R05 guards exclude R1 fillers (fils / cie / associés), P28's same-address rule has a sibling-word
  guard on the dropped org word, hn-extra needs p ≥ 0.8; the matcher scores them 0.02–0.8.
- **Label-free France lens (argmax sub1 pairs, P46 definition):** fils s 0.020 e 0.964; associés s 0.020 e 0.959;
  cie s 0.044 e 0.924; groupe s 0.089 e 0.879 — vs France decoy words s 0.29–0.42 e 0.47–0.62 (N54). Mixture read:
  ≈ 1% decoys for fils / associés, ≈ 4–7% cie, ≈ 10% groupe (same-hn share of decoys ≈ 0.5).
- **In-domain analogue** (identical address + one gained word from the country's own lens noise set s ≤ 0.05, e ≥ 0.8,
  n ≥ 20; argmax holdout pairs): **US 2,289 pairs 99.8% true, India 1,253 pairs 99.7% true** (negatives 0.00003 /
  0.00005 per S1 ⇒ bound 0.999). Caveat: France's cell is ≈ 8× larger per S1 (0.127 vs 0.018 / 0.014) — a France-
  specific template (like P46's développement), so the lens, not the bound, is the main evidence.
- Expected LB: strict (fils / cie / associés, ≈ 10.2k) ≈ +0.0004–0.0006; all (+ groupe) ≈ +0.0005–0.0007.

## Recommendation
On the current best base (after N54 if adopted — the word sets are disjoint):
  .venv/bin/python probes/N55-fr-noiseword-fill/n55.py analyse      # ≈ 5 min, writes data/work/n55/cells.parquet
  .venv/bin/python probes/N55-fr-noiseword-fill/n55.py build output/<base> output/<base>_n55 strict   # then validator
Adds only free SX that are candidates of that S1, argmax, non-sibling, within caps 5 S2 / 6 S3, best p first.
`all` also adds groupe (P46's 'g' already adds part of it; owned SX are skipped). For run.py: accept France pairs in this
cell after the other add rules (freeze the word list {fils, cie, associes} (+ groupe)).
Smoke test (local #31-era base): N54 ruled −1,075 then N55 strict +16,008 (this base predates hn-extra), validator PASS.

## Evidence
| check | world | numbers |
|---|---|---|
| cell precision, identical address + 1 gained lens-noise word | US holdout (#8c, argmax) | 2,289 pairs, 0.998 true; rejected 3 |
| same | India holdout | 1,253 pairs, 0.997 true; rejected 4 |
| France cell (argmax, non-sibling) | test, P31 pairs | 32,907; accepted in P31 base 8,584; new after hn-extra & P46: 12,724 |
| France new, p bands (all words, before identical-address filter) | test | ≤0.02 1.2k, 0.02–0.1 2.4k, 0.1–0.3 3.9k, 0.3–0.5 3.6k, 0.5–0.8 6.4k |
## Risks
- The France template is not in train; precision rests on the label-free lens (as P46's did — P46's développement
  swaps are the same template). Very low-p pairs (≤ 0.02, ≈ 1k) could hold other issues; P46 used p ≥ 0.02 — add that
  guard if cautious (`cells.parquet` has p).
