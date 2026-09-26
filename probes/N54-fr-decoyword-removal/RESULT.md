# N54-fr-decoyword-removal — RESULT (master, laptop, 27 Sep ~05:15 IST)
## TL;DR
- **France accepts ~5k decoys that the generator builds by swapping one org word at the SAME address**
  (template: "<X> Club SAS" → "<X> Ecole SAS", "<X> Parents" → "<X> Anciens", same house number and street).
  Most enter through the post-threshold rules — mainly P42's hn-equal extras (p ≥ 0.8), which carry no word check.
- **In-domain this cell is all false:** swapped-in word with lens support n_hn ≥ 20 and sibling share s > 0.2, house
  number equal, argmax pair → **US holdout 0 true / 65, India 1 / 759** (the #8c model accepts 0% / 0.2% of them).
  France's decoy-word lens profile (s 0.29–0.42, same-hn share e 0.47–0.62, 83 words: comite, ecole, amicale, club,
  parents, sportive, societe, amis, college, anciens …) mirrors India's decoy words (s 0.29–0.44, e 0.38–0.62), whose
  same-hn swaps are 99.9% false. True France noise words sit at s ≈ 0.02 (fils, cie, associes, developpement).
- **Removal (France-only, candidates untouched):** `p54.py build <base> <out> ruled` drops cell pairs with p < 0.9868
  (≈ 4,749 on the P31 + hn-extra proxy, 45 S1 emptied); `all` also drops the 704-ish matcher-accepted ones (≈ 5,453,
  79 emptied). Expected LB: +0.00045 if 90% of removed pairs are false; break-even at ≈ 29% false.

## Recommendation
1. Build on the current best base (teammate box): `.venv/bin/python probes/N54-fr-decoyword-removal/p54.py cells`
   (≈ 1 min; writes data/work/p54/cells.parquet from data/work/p31/pairs_France.parquet) then
   `.venv/bin/python probes/N54-fr-decoyword-removal/p54.py build output/<best> output/sub54r_<best> ruled`,
   validator `--check-ids`. This is a France-only REMOVAL, so one upload measures it cleanly.
2. If it gains, try `all` (adds the matcher-accepted part of the cell).
3. run.py integration (after every France add rule, including hn-extra / gc / rescue / P46): drop accepted France pairs
   that are P20 name_rel 'sub1' with a swapped-in word w, S1 hn non-empty and equal to the SX's, where w's lens
   (P46 lens_table on the France argmax pairs, label-free) has n_hn ≥ 20 and s > 0.2, and p < 0.9868.
   Better: apply the same word guard INSIDE hn-extra / gc so these pairs are never added.

## Evidence
| check | world | numbers |
|---|---|---|
| decoy-word same-hn cell precision (n_hn ≥ 20, s > 0.2) | US holdout (#8c, P23 dump) | 0 / 65 true; accepted 0% |
| same | India holdout | 1 / 759 true (0.1%); accepted 0.2% |
| same cell, all words incl. n < 20 (typo-like words) | US holdout | s .2–.4: 621 pairs 15% true; > .4: 391, 16% — the rare words carry the trues; hence n_hn ≥ 20 |
| France cell size | test, P31 pairs | 29,673 candidate pairs (0.11 / S1 vs US 0.0005, India 0.009); 704 with p ≥ 0.9868 |
| France in base (proxy: P31 acc ∪ hn-extra p ≥ 0.8) | test | 5,453 pairs; 4,749 with p < 0.9868; top words comite 507, ecole 452, amicale 379, club 283 |
| phantom-mates signal (other SX with the same core+city) | holdout / France | does not separate (cell is ~all mates = 0 everywhere) |
Scripts: probes/NIGHT/decoyword_fp.py, decoyword_ho.py, decoy_mates.py; probes/N54-fr-decoyword-removal/p54.py.

## Risks
- The France cell is 13× India's per S1: France's generator makes many more same-address org-swap decoys — or, the
  risk, uses org swaps as a TRUE noise op too. A two-population mixing read of the lens (pure decoys s ≈ 0.42) would put
  ~10–20% true in some words (club s 0.33, sportive 0.33, amicale 0.36). Break-even is 29% false, so even that case is
  ≈ neutral-to-positive for `ruled`; `all` is riskier.
- US→India LOCO of P37's hn90 rule was −0.0014 ("India's hn misleads"): consistent with this cell being the culprit.
- p comes from data/work/p31/pairs_France.parquet (P31 chain); the base's post-veto p may differ slightly near 0.9868.

## Addendum (05:45) — label-free admixture test: the pairs N54 removes look ≈ 90–100% decoy
Decoys REPLACE the S1's org word (dropped word ∈ decoy-word set D); true France noise variants drop an S1 word and add
a noise word (développement / groupe / fils / cie / associés). Share of pairs whose DROPPED word ∈ D (France sub1 argmax):
pure decoys (swap to a D word at a SIBLING offset) 0.964 (n 24,343); true-noise cell (noise word, hn equal) 0.877
(n 28,979). Mixture estimate of the true share in the same-hn decoy-word cell, by matcher p:
| p band | n | dropped∈D | est. true share |
|---|---|---|---|
| 0–0.3 | 14,945 | 0.948 | ≈ 0.18 |
| 0.3–0.6 | 4,645 | 0.967 | ≈ 0 |
| 0.6–0.8 | 3,271 | 0.967 | ≈ 0 |
| 0.8–0.9 | 2,057 | 0.956 | ≈ 0.09 |
| 0.9–0.9868 | 2,945 | 0.965 | ≈ 0 |
| ≥ 0.9868 | 724 | 0.961 | ≈ 0.03 |
The matcher does NOT concentrate the few trues at high p (they sit at p < 0.3, never accepted). In the band N54 removes
(p ≥ 0.8 via hn-extra / gc / rescue, and ≥ 0.9868 for `all`) the profile equals pure decoys (SE ≈ ±0.02 on the share).
⇒ expected LB for `ruled` ≈ +0.0004–0.0005, for `all` ≈ +0.0005–0.0006. Script: in this file's commit message folder,
probes/NIGHT (inline; reproduce with the dropped/gained word split of P20 name_rel 'sub1').

## Addendum 2 (04:55) — N54 is the only decoy family inside the France add rules
Simulated on the P31 proxy: **hn-extra** (free argmax, hn equal, p ≥ 0.8, non-empty group) adds 35,928 pairs: eq 14,683;
sub1 12,149 (3,880 with a decoy word — N54's target — the other 8,269 swap in true noise words: fils 2,102, groupe
1,543, cie 1,212, développement 1,110, associés 1,070); disjoint 4,247 (web / brand names); add1 3,282 (221 decoy-word).
**gc** (one-sided completion, p ≥ 0.7) adds 10,120: its sub1/add1 gained words are the same noise words, with a
decoy-word tail (ecole 59, comite 57, amicale 48, …) that N54 also removes. Nothing else looks decoy-like.
