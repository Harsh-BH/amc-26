# P62-public-ideas — RESULT (DRAFT 07:00, ideas 1–2 in progress)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **Idea 3 (clone09): the India/US ordering holds under test-like distractor density.** sub54's / sub_clean's combiner
  beats #34 by +0.000730 ± 0.000067 / +0.000604 ± 0.000062 LB-eq (h0→h1 / h1→h0), against +0.000681 / +0.000528 on the
  plain frame. #34's own gain over #8c shrinks ~35–60% with density. **Keep the combiner; do not revert India/US to #34.**
- **Idea 1 (acronym rule): a France-only change set of 3,590 pairs, expected +0.00014–0.00017 LB (unvalidated on France).**
  With an address check the rule is 0.997 / 0.9997 precise on India / US. The India/US decision already takes 96–97% of
  it, so India/US is worth only +0.00001. sub_clean takes just 67% of France's rule pairs, and 2,643 of the misses are not
  even candidates. Change set sent to P50: `data/work/p62/acro_cs_france.tsv`.
- Idea 2 (e5 dense-retrieval rescue): in progress. Ceiling = +0.0033 LB-eq for perfect rescue of pruned + blocked misses.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- Idea 3: no change. sub_clean's India/US (sub54 recipe = P44 combiner + r_all + zoo, full refit) stays. It is the
  most density-robust arm tested; sub44r ties it, rel54 comes next, and #34 comes last.

- Idea 1: P50 applies `data/work/p62/acro_cs_france.tsv` (s1, sx, tier, in_cand, p8c, comp_p) on top of sub_clean.
  It adds every row to the S1's matching line, and adds the SX to the S1's candidate line where `in_cand` is False
  (2,643 rows). Recheck caps (≤ 5 S2 / ≤ 6 S3) if another change set touches the same S1. Tiers, if the master wants
  a smaller first step: T2 = 236 rows where #8c's p agrees (median 0.998); T1 = 2,643 (never candidates) and T3 = 711
  (in sub_clean's candidate file, not in #8c's scored frame) are unscored. Rule =
  `normalize.acronym(name_core(S1)) == squashed name_core(SX)`, same rf hn and city, P3 address token Jaccard ≥ 0.6,
  every S1 street token in the SX address, one rule S1 per SX, SX free in sub_clean, no other S1 on the SX at
  #8c p ≥ 0.5.

## Evidence         (table: ledger name | world | key metrics | verdict)
clone09 world = P57's full holdout frame, where every free SX (in no train truth list) gets exact clones: the same text
and the same per-pair p / r / hn / zoo p on every S1 row it has. Argmax, a10 columns, #34's p', CTX2 / CTX3 and
n_comp_rows are then re-derived with `p57.world()`. Shipped boosters (P57's cache, fit on half h of the full world) score
half 1-h. Poisson(0.9) adds +112k India rows (+9.9%) and +141k US rows (+8.6%); #8c accepts 291 / 367 clones.

LB-eq vs #34 (paired per-S1, h0→h1 / h1→h0):

| arm | plain frame | clone09 Poisson(0.9) s0 | clone09 Bernoulli(0.9) s1 |
|---|---|---|---|
| **sub54 (= sub_clean India/US)** | +0.000681 / +0.000528 | **+0.000730 ± 67 / +0.000604 ± 62** | +0.000752 ± 68 / +0.000629 ± 62 |
| sub44r | +0.000656 / +0.000505 | +0.000732 / +0.000567 | +0.000756 / +0.000570 |
| rel54 (density arm) | +0.000638 / +0.000532 | +0.000666 / +0.000614 | +0.000711 / +0.000640 |
| rel44r | +0.000625 / +0.000485 | +0.000655 / +0.000538 | +0.000707 / +0.000574 |
| sub44r_psumoff | +0.000526 / +0.000447 | +0.000602 / +0.000494 | +0.000607 / +0.000506 |
| rall_only | +0.000331 / +0.000224 | +0.000410 / +0.000321 | +0.000438 / +0.000320 |
| sub44 | +0.000306 / +0.000210 | +0.000245 / +0.000195 | +0.000246 / +0.000205 |
| sub44_psumoff | +0.000178 / +0.000147 | +0.000161 / +0.000126 | +0.000156 / +0.000138 |
| yardstick #34 vs #8c (rep h0 / h1) | +0.000176 / +0.000210 | +0.000113 / +0.000129 | +0.000065 / +0.000085 |

Ledger: `P62-public-ideas/clone09-pois0.9-s0` (keep). Every r_all arm gains under density; only sub44 (no r_all) loses.

**Idea 1 (acronym rule)**, key = acronym(S1 core) == squashed SX core, same hn + city:

| variant | India train prec (unamb) | US train prec | France test pairs (unamb) |
|---|---|---|---|
| hn + city only | 0.25 (0.78) | 0.996 | 72,245 (10,851) |
| + address Jaccard ≥ 0.6 | 0.9935 (0.9976) | 0.9997 | 17,338 (10,760) |
| **+ every S1 street token in the SX address (shipped)** | **0.9973 (0.9991)** | **0.9997** | **11,495 (11,371)** |

| ledger name | world | key metrics | verdict |
|---|---|---|---|
| acro-holdout-jac.6-st1 | holdout fold 0 | India 587 rule pairs, 566 accepted by sub54, 19 free misses at precision 0.89; US 319 / 311 / 8 at 1.00; LB-eq +0.000007 / +0.000012 | drop (India/US) |
| acro-france-changeset-vs-sub_clean | test France | 11,371 unamb rule pairs: sub_clean matches 7,608 (67%), gc70 6,595 (58%); 3,693 free misses → 3,590 after the competitor and cap filters; expected LB +0.00014 at q=0.9, +0.00017 at q=0.95 | keep (France, to P50) |

Expected-gain model: each touched S1 is assumed complete after its adds and its current matches correct; q = precision
of the adds. Break-even q ≈ 0.73. None of the 3,590 were matched by gc70's T1; only 6 of T2 and 26 of T3 were.

## What failed      (approaches tried and dropped, with the number that killed them)
- Pending (ideas 1–2).

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `clone09.py build <c> [lam] [seed] [pois|bern]` -> `data/work/p62/A_clone<mode><lam>s<seed>_<c>.parquet`
  (India 64 s, US 92 s, peak 1.8 GB; frames deleted after eval for disk). `clone09.py eval [lam] [seed] [mode]`
  -> `data/work/p62/ev_<world>.json` (~4 min, uses P57's cached boosters `data/work/p57/b_*`).
- `acro.py pairs <split> <c>` -> `data/work/p62/acro_<split>_<c>.parquet`; `acro.py holdout`.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- clone09 recomputes only the stacker-level context. m2's own input context inside #8c's p is kept from the full world,
  so a clone gets exactly its original's p.
- Exact clones are a crude stand-in for test's extra distractors: train has no content-identical group that is entirely
  unmatched (P54). They do not trip any SX-identity feature, since a10's twins are S1-side and name_sup only looks at
  accepted SX.

## Open questions   (for the master; each with your best guess)
- Pending.
