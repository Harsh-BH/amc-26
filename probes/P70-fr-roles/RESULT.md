# P70-fr-roles — RESULT (final, 27 Sep 15:45 IST)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **Label-free word-role features make P55's France combiner transfer in both directions.** The features are the pair's name relation, plus the word the pair swaps, adds or drops, described by the country's own sibling-offset share and same-house-number share for that word. **P55's own add arm fails US→India at −0.009; with these three columns (`words_nn`) it is positive both ways.**
- **On top of what sub66_final ships** (reference minus P55's no-rescue band veto), h1 ± SE:
  - arm A (veto c < .4, add c > .8): India→US **+0.00496 ± 0.00021**, US→India **+0.00517 ± 0.00039** France-F0.5.
  - The precise cut (veto + adds at c > .98, add precision .989 / .965) keeps about +0.0017 / +0.0029.
- **France file `output/sub70a`: +1,991 adds at c > .98, no vetoes, validator PASS.** The Qwen audit puts the adds level with the accepted base. Expected LB is about +0.0001; 1,695 of the adds are also #67's.

## France file
- **`output/sub70a`** = #66_final + **1,991 France adds on 1,919 S1, nothing removed**. Validator `--check-ids` PASS; md5 f43e96dc…; India/US lines byte-identical; additions only.
  - Rule: a free own argmax pair (non-sibling, not capped out, p not moved by a tier / bxneg rule) with c > .98. Adds are taken best-first, up to caps 5/6 counted on the base's France line, skipping any SX already listed for a France S1. Then P64's legal-form veto runs last (−14).
- **Qwen audit** (P38 pooled r, 1,802 of the adds scored): 87.8% have r ≥ .5 and 79.7% r ≥ .9, median .997.
  - That equals the accepted base (87% / 78%).
  - LB-proven add classes are lower: rescue90 68% / 49%, gc70 62% / 38%. P55's adds had 82% / 69%.
- **The veto arm was dropped** (`output/sub70`, c < .4, band, non-rescue). Its 4,127 bge-frame vetoes land on only 295 pairs in the Qwen France, and those pairs are Qwen-supported (78.6% r ≥ .5, median .96).
- **Overlap:**
  - 1,695 of the adds are also #67's adds (0 same-SX conflicts), so on top of #67 only about 296 are new.
  - 289 are also t975ce's.
  - P72's number-mark add tier (sub72, 1,060 adds at c > .99) is almost a subset: 932 of its adds are P70's and 1,044 are P67's, and only 3 fall outside the union. Three different extra columns on P55's combiner converge on the same high-c pairs, so the lever is the precise add tier itself.
- **Change set:** `data/work/p70/changes.parquet` (s1, sx, action, p, c_new, qwen_r). Filtering on c_new ≥ .99 reproduces a u .99 build, because adds are taken best-first.
- **Rebuild** (≈ 3 min, ≤ 1.7 GB):
  ```
  p70.py scores_fr words_nn
  p70.py build sub66_final sub70a mode=p70words_nn,scope=band_norescue,v=0,u=0.98
  p70.py changes sub66_final sub70a words_nn
  ```

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- **Adopt the adds of `output/sub70a`** (or merge `changes.parquet` into the combined final, P64 last). Expected LB is about +0.0001 alone and about +0.00002 on top of #67's adds.
- **Better merge:** fit one P55 combiner with P67's synthetic column, P73's electra column and P70's `words_nn` columns, instead of taking the union of three add lists. The LOCO role columns are cached in `data/work/p70/roles_<dir>.parquet`, row-aligned with `p55.loco_frame`.

## Addendum (27 Sep ~17:15): Fable's capture plan, gated
The prompt is `data/notes/fable_capture_prompt.md`. It gave Fable the cluster-count gap: sub66_final predicts 3.155 links per France S1, against 3.46 in train truth for both US and India. Size 6+ holds 7.6% of France S1 against 11.4% in truth.

Fable's ideas, gated with `i2_loco.py` → `i2_loco.json`. Each cell is an argmax pair on P55's LOCO frames where S1 and SX share one hn + street key. Precision is measured on the pairs the reference leaves free, which is what a rule would add; "add dF" is the result of adding them (caps, one owner).

| cell (S1 and SX share one hn + street key) | India target: free n at precision | US target: free n at precision | add dF (India / US target) | verdict |
|---|---|---|---|---|
| related names, key unique | 1,031 at .61 | 5,207 at .63 | −0.0004 / −0.0014 | drop: these are just the rejected extras again |
| disjoint names, not a brand, key unique | 42 at .38 | 481 at .14 | −0.00004 / −0.00075 | drop. France holds 103k such free SX, almost all same-address decoys |
| **disjoint, synthetic brand, key unique** | 57 at 1.00 | 321 at .885 | **+0.00008 / +0.00022** | keep. This is P41's rule |
| synthetic brand, key shared | 44 at .59 | 424 at .41 | −0.00002 / −0.0003 | drop |
| raw address byte-equal (Fable's "decoys copy verbatim") | all: 98.9% true | all: 99.8% true | — | the premise is false |

- **Adopted:**
  - P41's brand rule (bare generator brand, unique hn + street + city key, key S1 = argmax with members, p ≥ .30) was LB-untested on its own. sub54 carried it, and it was dropped when France reverted to gc70.
  - It is re-applied on sub66_final by `p41_on.py`, which merges the adds onto the base's own lines; `p41.build` would drop the 2,638 acronym pairs.
  - The result is **`output/sub41_on66`: +5,834 France pairs on 5,733 S1**, p median .92. 0 SX already listed, 0 P64 conflicts, 0 matches outside the candidates.
  - It is orthogonal to everything staged: 10 of its pairs are in sub75a, 0 in sub71a, 0 SX conflicts, 0 N54 removals.
  - Evidence: train truth for a bare brand at a unique key is US 34,777 at .998 and India 15,083 at .961. P41's LOCO was +0.00008–0.00014 (precision .98–1.00), and +0.00035–0.00079 France-matched.
  - Expected LB +0.00016 to +0.00024.
- **Already done or out of date in the plan:**
  - Slot A is sub76.
  - I1 is P71a. Fable's kill test (in-domain *unassigned* precision) measures the wrong population: 0.5–0.7% true by construction. The class analogue is .947 / .969.
  - I6a is P54's dup step.
  - The independent CE is P73e / b.

## Addendum 2 (27 Sep ~18:50): the rest of the France gap, gated
This round gates each idea on P55's LOCO frames. Precision is measured on the pairs the reference leaves free, with Wilson 95% bounds and add dF. The master's bar is Wilson LB ≥ .75 with n ≥ 200 in both countries. Scripts: `gap_loco.py`, `hn_loco.py`, `exact_loco.py` / `exact_fr.py`, `street_fr.py`, `shift.py`, `clean_loco.py`, `clean_street_loco.py`; JSONs alongside.

| cell | India target (free n at precision) | US target | France (sub77) | verdict |
|---|---|---|---|---|
| fill a missing source, related name, S1's unique key | 191 at .70, −0.00004 | 841 at .81, +0.00038 | — | fails: India n < 200, Wilson ≈ .63 |
| copy of an accepted SX (same core + key) | 17,349 at .55, −0.0089 | 4,956 at .75, +0.00035 | — | drop |
| empty-S1 fill / related extras at a unique key | .42 / .60 | .40 / .61 | — | drop |
| identical name + city, \|Δhn\| = 1 / 2 | .46 / .41 | .06 / .05 | 37.9k "other" free | drop: ±1/±2 is a decoy offset in the US |
| exact duplicate (core + hn\|street key + city) | 558 at .943 | 1,310 at .996 | 41 free (29 P64-vetoed) | France already accepts them |
| **hidden sibling shift** (parsed hn equal; one other address number shifted by SIB3 or 1–2) | all 2% true; veto of accepted +0.0045 | 5% true; veto +0.00013 | 23 accepted | a decoy marker. France rejects it already; India/US in-domain accept it only when true (237 at .992, 55 at 1.00), so there is no veto anywhere |
| same core + city + hn, no shift, lf ok: **same street** | 531 at .985 | 314 at 1.00 | 28 free | passes, but there is nothing to add |
| same, **different street** | 21 at .24 (all pairs 183 at .85) | 297 at .983 (all pairs 11,195 at .999) | 6,219 free vs 10,210 accepted (38% free; LOCO 2.7% / 11.5%) | fails India n. P64 vetoes 39% of France's pool, and the adds have p median .31 → `output/sub77_hnst` is staged as a measurement only, **not recommended** |

- Parser notes:
  - `house_number` keeps the leading digit run, so "5 bis" = "5", and "Appt N" is never read as a house number.
  - The real gaps are number ranges, second numbers, and a leading postcode read as the house number. They hide sibling shifts, and `shift.flag` catches them.
- Caps don't truncate France's big clusters: truth never exceeds 5 S2 / 6 S3, and France sits at the cap less often than truth (0.77% vs 1.09%).

## Addendum 3 (27 Sep 18:30): the legal-form-only address-less class (lead's P77 q4)
The class: address-less SX whose name core equals a unique S1 core, with a full name that differs only by the legal form. `xlf_loco.py` runs it through P66's LOCO harness; `xlf_build.py` builds France.

| split | in-domain train precision (India / US) | LOCO unaccepted pool (India target / US target) | France accepted / free |
|---|---|---|---|
| (a) form dropped / added | .961 n 853 / .951 n 2,575 | 329 at .91 (+0.00016) / 95 at .46 (−0.00002) | 1,610 / 1,643 |
| (b) same form, spelled differently | .986 n 1,303 / .978 n 920 | 336 at .96 (+0.00027) / 16 at .63 | 1,149 / 377 |
| (c) conflicting | .982 n 822 / .575 n 73 | — | 44 / 96 (excluded) |

- Free-share diagnostic:
  - When a model leaves a large share of a precise class free, as the India target does (39% / 26%), the leftovers stay true (.91 / .96).
  - When it accepts nearly all of it, as the US target does (4% / 2%), the leftovers are coin flips (.46 / .63).
  - France leaves 50% / 25% free, like the India target.
- **`output/sub77_xlf`**: +2,019 adds, validator PASS. Expected +0.0002 to +0.0004 LB if France is India-like, ≈ −0.0001 if US-like.

## Evidence         (France-F0.5 dF on the LOCO target; A→B = fit on A's out-of-domain frame, bars picked on A, applied to B; h1 = report half ± SE)
| ledger name | world | key metrics (India→US / US→India) | verdict |
|---|---|---|---|
| loco-p55-vs-roles | P55 LOCO frames vs recipe + P48 veto | all 9 role columns vs P55 alone. **A:** p55 +0.00786 / −0.00754, roles +0.00560 / +0.00650. **B:** p55 +0.00341 / +0.00107, roles +0.00228 / +0.00303. AUC 0.965 → 0.970 / 0.938 → 0.964 | mixed (costs the France-like India→US) |
| loco-ablate | same | **words** (kind, w_n, w_s, w_e): A +0.00695 / +0.00624, B +0.00295 / +0.00218, add +0.00399 / +0.00408. words + lf_k / gib_x / noaddr_x: A +0.00475 / +0.00408. all but key_n: A +0.00469 / +0.00447 | words is the keeper |
| loco-inc-words_r+words_nn | same base | words_r (w_n as a rate per 100k pairs) = words. **words_nn (kind, w_s, w_e): A +0.00496 ± 0.00021 / +0.00517 ± 0.00039; B +0.00085 / +0.00057; add .8 +0.00411 / +0.00461** | words_nn ships (scale-free) |
| loco-inc-words_nn (add grid) | same base, add-only arm on the target | u .9: 6,681 at .936 +0.00335 / 9,254 at .889 +0.00564. **u .95**: 3,722 at .968 +0.00215 / 6,194 at .935 +0.00473. **u .97**: 2,075 at .985 +0.00132 / 4,150 at .954 +0.00341. **u .98**: 1,257 at .989 +0.00081 / 2,759 at .965 +0.00233. **u .99**: 533 at .996 +0.00035 / 1,178 at .984 +0.00110 | u .98 (LB lesson: only ≥ .98-precision France adds transferred) |
| loco-inc-words | base = ref minus P55's no-rescue band veto at c55 < .6 (= sub66_final's France) | ship vs ref +0.00305 / +0.00185. On top of it: **words A +0.00493 ± 0.00021 / +0.00457 ± 0.00039** (adds 11,373 at 0.868 / 13,403 at 0.812; vetoes 869 at 68% false / 916 at 55% false). words B +0.00089 / +0.00053. p55 A +0.00546 / **−0.00894** (US→India adds 18,104 at 0.602) | pass |

**Why it works:** the generator draws its noise and decoy words from per-country lists (US "partners" is noise; "group" and "holdings" are decoys; France has "développement" as noise and "participations" as a decoy). A model fit on India or US learns those lists by heart, so it cannot know France's. The role statistics re-derive each list from the target country's own argmax pairs, with no labels:
- A decoy word sits at the generator's sibling house-number offsets.
- A noise word keeps the house number.

## What failed      (approaches tried and dropped, with the number that killed them)
- The full 9-column role set. lf_k (P64 legal-form class), gib_x, noaddr_x, twins1 and key_n (S1 rows at the SX's exact hn + street) each lower India→US A by about 0.002. They let the model memorise the source country's address and legal-form mix.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `p70.py loco` / `ablate <sets>` / `inc <sets>`: LOCO comparisons.
  - Role frames are cached in `data/work/p70/roles_<dir>.parquet`. The first computation takes ~4 min and peaks at 2.61 GB; later runs take ~5 min at ≤ 2.5 GB.
- `scores_fr <set>`: bag of 3 seeds fit on both LOCO frames pooled; scores France's 1,182,279 argmax pairs with France's own role statistics. Writes `data/work/p55/cf_p70<set>.npy`.
- `build BASE OUT mode=p70<set>,scope=band_norescue,v=..,u=..`: runs `p55.build`, then P64's legal-form veto last (`frtest.build`). Candidates and code.zip are hard links.
- `changes BASE OUT`: writes `data/work/p70/changes.parquet` (s1, sx, action, p, c_new, qwen_r), the Qwen audit, and the overlap with sub67 / sub66_t975ce.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- w_n is a raw count, and France's frame is 1.9–2.9× the LOCO frames. `w_nr` (rate per 100k pairs) and `words_nn` (no count) are the scale-free variants.
- LOCO's veto can hit any region or stage. France's is limited to band and non-rescue, as shipped, so it is a subset.
- The dup step is not re-run after the adds.
