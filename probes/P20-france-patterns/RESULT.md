# P20-france-patterns — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
Two record relations explain France's uncertain band: how the S2/S3 name core differs from the S1 core, and how the house
number (hn) differs. France's generated siblings keep the hn only 2–6% of the time, like the US (2.7%) and unlike India (26%)
(the proxy is checked on labels). So France hn-equal pairs should be mostly true, yet #8c accepts only 35% of France's 75k
"one core word swapped, hn equal" pairs, and US labels put that cell at 89% true.
**Rule R1** (new, general code): strip the generator's French suffix fillers (& Fils / & Cie / Associés / Frères / Frs) from the
name core, as `GENERIC` already strips 'services'/'center'. On #9as's France it adds 17.5k pairs (92.7% hn-equal, 0% sibling offset)
and drops 2.0k. **Upload `data/work/p20/sub20a` (#8c base + IDF + R1 + sibling rule, t 0.975; validator PASS) against #9as.**

## Recommendation   (exactly what the master should adopt: module, function signature, config)
1. **LB slot (France-only test, one lever vs #9as):** the staged folder `data/work/p20/sub20a/` (validator PASS).
   - India/US rows are byte-identical to #8c. 18,599 France S1 differ from `output/sub9as`, 56,508 from `output/sub8c`.
   - Rebuild it with:
     `.venv/bin/python probes/P14-france-shift/combine.py output/sub8c data/work/p20/prob_test_France_idf_noisesib.parquet 0.975 output/sub20a`
   - The file is P14's IDF world (#9a) + R1 + R2 (P15's sibling rule), so the LB delta vs #9as measures R1 alone.
   - t 0.975 is #9as's threshold. The brief's rule (France empty rate = D4's 6.85%) gives **t 0.98** instead: 6.78% empty,
     3.03 per S1, and it drops 10.3k extra pairs that are 66% hn-equal. Use 0.98 only if #9as at 0.975 already over-matches.
     At 0.975 the France profile is 6.64% empty and 3.06 per S1, inside P6's bands.
   - **If the IDF fix (#9a) loses on the LB,** use the #8c-world file instead (t_eq_d4 0.955, or 0.95 to compare with #9s at 0.95):
     `.venv/bin/python probes/P14-france-shift/combine.py output/sub8c data/work/p20/prob_test_France_noisesib.parquet 0.955 output/sub20b`
     vs #9s: 6.85% empty, 3.02 per S1. At 0.95: +15,926 / −1,374 pairs, 16,639 S1 changed, 6.78% empty, 3.03 per S1.
2. **src change if it wins (P9):** add the words to `normalize.GENERIC`: `{"fils", "cie", "associes", "freres", "frs"}`.
   - Leave 'compagnie', 'groupe' and 'developpement' out: they are S1 org words or sibling branch words (see Evidence).
   - US/India footprint: 21 S1 and 189 S2/S3 of 12.5M train records, 10 and 132 of 11M test records. The holdout cannot move measurably.
   - Code: `probes/P20-france-patterns/fr20.py` `noise_core(rf)` applies the same change to a rules.feats frame after the fact.
   - Pipeline effect: rules, features and the pruner all read the core. Blocking does not (P3 texts), so no re-blocking is needed.
3. **Keep P15's sibling rule for France** (R2, `fr20.sib_zero`). France's siblings behave like US ones, so the US-holdout sign of the rule
   (+0.00097) is the one that applies here, not India's.
4. **Biggest remaining France gap (not fixed by a rule):** the 75k one-core-word-swap pairs with an equal hn. The org word is swapped
   ("X Ecole" → "X Comite") or a suffix is swapped in ("X & Fils SA" → "X & SA Cie"); R1 covers only the suffix part.
   - Mean p is 0.61 and 35% are accepted, against 89% true in the US cell.
   - Their logit is 4.8 below the US cell. sx_margin explains −2.1 of it (France 0.12 vs US 0.36): a competitor that the pruner later removes,
     93% of the time, so an IDF/blocking artefact. Shorter names explain about −2.
   - A blanket fix (R3, quantile-mapping sx_margin) failed. It needs a model-level change: context features over the pruned list,
     or a France-invariant retrain. See Open questions.

## Evidence         (table: ledger name | world | key metrics | verdict)
### Taxonomy: name relation × hn relation (label-free; France test, #8c's France p, each SX's argmax pair; 1,173,107 pairs)
The name relation is between normalize.name_core cores. The hn relation is one of: eq; sib3 = hx − h1 ∈ {3,4,5,7,9,11,13,21}
(the generator's sibling offsets, P12); d12 = +1/+2; other; miss. "US true" is the same cell's truth rate on the US holdout
(fold-0 argmax pairs, m2_big p). Band = 0.3 ≤ p < 0.95 (173,189 pairs). Disagree = pairs accepted by some but not all of
#8c / #9a / #9b / #9s (64,565).

| pattern (schematic example) | France pairs | band share | disagree share | #8c mean p | acc. t .95 | US true (p) | India true | handled? |
|---|---|---|---|---|---|---|---|---|
| core equal, hn equal | 591,552 | 9.7% | 26.0% | 0.986 | 96.5% | 0.998 (0.998) | 0.982 | yes |
| core equal, hn + SIB3 (generated sibling, "X SARL" at 152 → "X SASU" at 155) | 89,285 | **29.0%** | 6.3% | 0.446 | 4.5% | 0.010 (0.082) | 0.236 | P15 sib rule (#9s/#9as), not #8c |
| core equal, hn +1/+2 | 22,580 | 7.4% | 1.4% | 0.448 | 4.5% | 0.160 | 0.649 | no (sign differs by country) |
| core equal, hn other (mostly another street: "City + OrgWord" chains) | 37,801 | 4.8% | 3.6% | 0.336 | 17.6% | (0.826, not comparable: no street check) | 0.834 | model |
| core equal, hn missing | 90,672 | 9.6% | 13.7% | 0.835 | 70.2% | 0.883 | 0.900 | model; '(12) RUE' hn unparsed in #8c |
| **one core word swapped, hn equal** ("X Ecole" → "X Comite"; "X & Fils SA" → "X & SA Cie") | 74,887 | **15.2%** | 12.3% | 0.612 | **34.7%** | **0.890 (0.892)** | 0.625 | **no; R1 covers the suffix part** |
| one word swapped, hn + SIB3 | 23,317 | 1.1% | 0.2% | 0.083 | 0.4% | 0.002 | 0.008 | yes (model) |
| one word added, hn equal (noise & Fils / & Cie / Associés; some Groupe) | 17,123 | 5.1% | 3.0% | 0.625 | 25.3% | 0.852 | 0.246 | R1 (new) |
| one word added, hn + SIB3 (branch word: Distribution, Holding …) | 37,054 | ~0 | ~0 | 0.024 | 0% | 0.000 | 0.002 | yes (model) |
| one word dropped, hn equal | 21,356 | 1.7% | 3.9% | 0.942 | 84.3% | 0.999 | 0.964 | yes |
| no shared core token, hn equal (invented word as the name; initials of the core) | 19,361 | 6.0% | **20.7%** | 0.662 | 30.6% | 0.775 | 0.780 | partly: acronyms are blocked only in the IDF world (#9a) |
| typo only, hn equal | 15,626 | 0.7% | 1.0% | 0.918 | 85.7% | 0.958 | 0.900 | yes |
| website / handle name, hn equal | 52,318 | 1.4% | 3.2% | 0.984 | 95.0% | 0.997 | 0.991 | yes |

Per word: share of the word's pairs that keep the hn / that take a SIB3 offset (all France candidate pairs):
- **Noise fillers** (the generator's true-pair suffix pool; the France counterpart of US 'services'/'center'/'partners'):
  - swapped in: fils .88 / .02, cie .85 / .04, associes .86 / .02, developpement .91 / .01, groupe .81 / .08;
  - added: fils .82 / .05, associes .84 / .04, cie .81 / .05.
- **Sibling branch words** (added): distribution .01 / .74, participations .02 / .73, holding .02 / .71, international .02 / .71,
  developpement .15 / .64, groupe .25 / .51.
  - The model already rejects these (mean p 0.02–0.17; 437 of 50k accepted).
  - So 'developpement' and 'groupe' mean "noise" when swapped in and "sibling" when added. They are not in R1.
- **Org-word swaps** (club, comite, ecole, amicale, amis, parents, sportive, union, college, societe, fetes, primaire, sante …): about .50 / .36.
  - A pure sibling mix would show SIB3 .73 and d12 .18. Observed d12/SIB3 is 0.26, so these are about half siblings with an offset,
    and the other half keep the hn.
  - France siblings keep the hn only 2–6%, so that hn-equal half is not siblings.

Surface patterns, full France test (share of S1 / of S2+S3; `taxonomy_counts.py`):

| pattern | S1 | S2/S3 | handled by |
|---|---|---|---|
| legal form SARL/SAS/SASU/SA/EURL/SCI/SNC/EI | 68.9% | 60.1% | normalize.LEGAL |
| legal form in brackets / dotted (S.A.R.L.) | 0 / 0 | 1.7% / 5.4% | _SEP / _merge_letters |
| accented letter in name (incl. accents injected into plain letters) | 15.7% | 24.2% | fold (anyascii) |
| '(France)' / 'France' in name | 8.0% | 9.4% | country word dropped from the core |
| suffix filler & Fils / & Cie / Associés / Frères / Frs | 6.1% | 7.3% | **R1 (new)** |
| branch word Développement / International / Distribution / Holding / Participations | 0.4% | 12.8% | model (+ sib rule) |
| Groupe / Groupement; Compagnie | 1.9%; 1.4% | 4.5%; 1.1% | model |
| joiner & / + / et; doubled space | 6.0%; 0 | 7.4%; 8.5% | STOP 'and'/'et', _SEP |
| website / handle; alias (dba / aka / formerly); 2–4-letter acronym | 0 | 4.2%; 0.8%; 1.7% | website_core; _ALIAS; acro feature |
| empty address | 0 | 3.0% | addr_empty features |
| hn label N° / Nº / No / # | 0 | 14.7% | fold + _HN_LABEL |
| hn in parentheses '(12) RUE' | 0 | 1.6% | P6 france.py only; not in #8c/#9a (P14: neutral at model stage) |
| zero-padded hn / '17 - BD' / bis, ter, letter | 0 / 0 / 6.3% | 3.3% / 1.6% / 5.7% | lstrip('0') / digit run |
| street 'R.' / 'Rue' | 0 / 65.8% | 22.5% / 38.9% | STREET rue→r |
| Av / Ave; Avenue; Bd / Blvd; All / Imp / Rte / Crs / Pl / Sq / Fg | 0.9%; 11.9%; 0.8%; 5.1% | 5.4%; 6.7%; 2.1%; 6.3% | STREET |
| unmapped abbreviations Ch. (chemin), Q. (quai), Psg, Bvld | ~0 | ≤ 0.8% each | **no**: minor, and fixing it means re-blocking (P3 texts) |
| city St- / St. / Ste- (saint) | 0.01% | 1.7% | STREET saint→st |
| région / département name | 100% / 0 | 33.4% / 31.8% (≈35% neither) | FR_REGIONS; missing state = P14 (0.02 pt) |
| all-caps address | 0 | 14.2% | fold |

### Experiments
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| pattern-cells-France-test | test-France | cells above | keep |
| pattern-cells-labelled-analog | holdout fold-0 argmax, m2_big | sub1/eq: US 0.890 true (p 0.892), India 0.625. disjoint/eq: 0.775 / 0.780. eq/sib3: 0.010 / 0.236. drop1/eq: 0.999 / 0.964 | keep |
| sibling-same-hn-proxy | holdout + test | share of branch-word-added pairs that keep the hn: US 0.027 (all US false pairs 0.044), India 0.258 (0.246), France 0.065 (0.02 for the 4 pure branch words) | keep: France is in the US regime |
| (shap_cells sub1 eq, 8k pairs each) | test-France vs holdout | mean logit France 1.29, US 6.10, India 2.03. Largest gaps: sx_margin −2.08 (0.12 vs 0.36), name_cos −0.66, name_partial −0.41, name_tsort −0.38, core_jacc −0.28 | the gap is context (IDF) + short names |
| (runner-up check) | test-France | 93% of sub1/eq SX have no other kept candidate. The competitor behind the low sx_margin is pruned | blocking/IDF artefact |
| rescore-base-sanity | test-France | re-deriving #8c's France from its X: 1,332,296 pairs, max abs dp 0.0 | keep |
| R1-noise-core | test-France (#8c world) | t .95: +16,115 / −1,440 pairs vs #8c (added: hn eq 0.90, SIB3 0.012), 548 S1 empty → matched, 64 matched → empty, empty 6.66%, 3.05 per S1. Drops: p just under .95 (median 0.922), mostly core-equal pairs whose S1 core now collides with a sibling S1 ('X & Fils' vs 'X') | keep |
| R1R2-noise-core+sib | test-France (#8c world) | t .95 vs #9s: +15,926 / −1,374, 16,639 S1 changed, 6.78% empty, 3.03 per S1; added hn eq 0.911, SIB3 0 | keep (fallback file) |
| idf-world-rebuild-repro | test-France | P14's IDF world rebuilt with X kept: 1,305,341 pairs, max abs dp vs idf144 0.0 | keep |
| R1-noise-core-on-idf | test-France (IDF world) | t .975 vs idf144: +17,569 / −1,984 (added hn eq 0.925, SIB3 0.002), 6.61% empty, 3.07 per S1 | keep |
| R1R2-on-idf-vs-9as | test-France (IDF world) | t .975 vs #9as's France: +17,529 / −1,954, 18,599 S1 changed (added hn eq 0.927, SIB3 0), 6.64% empty, 3.06 per S1. Staged `data/work/p20/sub20a`: validator PASS, non-France rows identical | **keep → LB** |
| R3-sx_margin-quantile-map | test-France (#8c world) | at t_eq_d4 0.995: +26,435 / −34,692 (added hn eq 0.605 vs dropped 0.635). At t .95: +76k, of which only 44.8% keep the hn and 21.8% have another hn | drop |
| R1-global-footprint-US-India | rules.feats | train: US 7 S1 + 108 SX, India 14 + 81; test: 10 + 132 | keep (fold into GENERIC) |

hn lens across the finished files (pairs one file accepts and the other doesn't, each at its t):
- #9as's France vs #9s's France: the pairs only #9as accepts keep the hn 90%; the pairs only #9s accepts keep it 62%.
- #9as's France vs R1 + R2 on #8c (t 0.955): 89% vs 75%.
- IDF + R1 + R2 vs #9as's France: 93% vs 73%.
- Label-free ranking: **IDF+R1+R2 > IDF+R2 (#9as) > R1+R2 > R2 (#9s) > #8c.**

## What failed      (approaches tried and dropped, with the number that killed them)
- **R3, quantile-mapping France's sx_margin onto US+India's.**
  - The diagnosis was right: sx_margin is France's largest logit deficit on sub1/eq.
  - But the fix raises every France pair, chain look-alikes on other streets included.
  - At D4's empty rate it swaps in 26k pairs and drops 35k, and the swapped-in pairs keep the hn less often (0.605 vs 0.635).
  - At t .95 it adds 76k pairs, of which 22% have a different hn. A marginal de-shift is not selective; the fix has to be in the model.
- **Also stripping 'compagnie', 'groupe' and 'developpement' (P6's FR_FILLER list): not run, rejected by the lens.** As an added
  word, 'groupe' / 'developpement' carry a sibling offset 51% / 64% of the time. 'Compagnie' is an S1 org word, with a SIB3 share
  of 17% when swapped in. Stripping them would turn branch-word siblings into core-equal pairs, where France's model is
  over-confident (eq/sib3 mean p 0.45 vs US 0.08).
- **Run with memory limits (not a result):**
  - The first IDF rebuild and the first R1+R3 run were paused by memguard (the box was at < 1 GB free) and were killed.
  - The rebuild was redone in a quiet window and reproduces P14 exactly.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `fr20.py`: the rules.
  - `noise_core(rf) -> rf`: R1 on a rules.feats frame; NOISE = {fils, cie, associes, freres, frs}.
  - `sib_zero(d, hn) -> d`: R2 on (s1, sx, p).
  - `qm_tables(path)`, `qmap(x, qsrc, qdst)`: R3, dropped.
  - `python fr20.py sib IN OUT` writes R2 onto a probability file. Self-checks included.
- `rescore.py NAME` (rules joined by `_`, e.g. `noise`, `noise_qm`, `qm`, `base`):
  - It takes #8c's France X (`data/work/p4/X`), re-derives features.strings + rule_acc + rule-only pairs from the changed rules.feats
    records in 300k-row slices, then runs the pruner + m2_big.
  - Output: `data/work/p20/prob_test_France_[<world>_]NAME.parquet`.
  - Runtime 3–6 min, 2.5–2.6 GB. `qm` alone is 0.9 GB.
  - Another world: `ER_WORK=data/work/p14/w_idfx` (after `variant.py idfx --idf 0.144 --keep-x`, 17 min, 2.7 GB).
  - `base` reproduces #8c's France exactly.
- `compare20.py BASE[:t] VAR[:t] …`: profile (empty, matches per S1), S1 churn, and the hn lens on the pairs added / dropped.
  With no t it uses t_eq_d4. 1 min, 1 GB.
- `patterns.py [c]`:
  - `name_rel(c1, cx, is_web) -> (kind, word)`, `hn_rel(h1, hx)`, `table(...)`;
  - writes `data/work/p20/pairs_France.parquet`. 10 s, 1.2 GB.
- `analog.py US|India`: the same cells on the labelled holdout (`data/work/p20/hold_{c}.parquet`). 1 min, 1.5 GB.
- `sample.py [n]`:
  - merges the four France variants (`merged.parquet`);
  - writes seeded samples `sample_{a,b,c}.txt` (raw rows, gitignored data/ only).
- `shap_cells.py kind hrel [n]`: mean m2_big contributions, France vs holdout. 20 min (US scan), 1 GB.
- `taxonomy_counts.py`: surface-pattern counts → `data/work/p20/taxonomy_counts.csv`.
- Files in `data/work/p20/`:
  - `prob_test_France_{noise,noisesib,qm,qmsib,w_idfx_noise,idf_noisesib,idf_sib}.parquet` (s1, sx, p);
  - `sub20a/` (staged, validator PASS; 217 MB, deletable once uploaded).
- `data/work/p14/w_idfx/` holds the IDF world's X (229 MB): deletable, or keep it to rescore more rules there.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Rules:**
  - No LLM decision and no pair labels. I read samples only to name patterns.
  - Every rule is general code over record fields, and every number is an aggregate.
  - R1's word list is domain knowledge (French trade-name suffixes), on the same footing as P6's lists. Test rows were only counted to check it.
  - R2's offsets are train-measured (P12).
  - The IDF base reads unlabelled test text (P14; open question 4). The #8c-world fallback file does not.
- **The hn lens is a proxy, not a label.** It assumes non-sibling distractors are rare among hn-equal pairs.
  - US labels bound that: sub1/eq is 11% false and disjoint/eq 22% false.
  - Read R1's added pairs as roughly 85–93% true-like. They are not proven true.
  - The LB is the test: expect France +0.3–0.6 pt → LB +0.0005–0.001 if the lens is right.
- **R1 merges cores:** 'X & Fils SARL' and 'X SARL' S1 now share a core. That costs 1.4–2.0k pairs just under t (median p 0.92).
  The rule stage suffers most (P6 saw ties there), but the model stage absorbs it: net +15.5k pairs.
- **Threshold:** I kept #9as's t (0.975) so the LB isolates R1. Matching D4's empty rate would push t to 0.98 and give back
  ~10k mostly hn-equal pairs. If R1 wins at 0.975, a follow-up slot at 0.97 is the natural direction; 0.98 is not.

## Open questions   (for the master; each with your best guess)
- **How to recover the 75k hn-equal one-word-swap pairs** (35% accepted; ~89% true if US-like)? This is France's largest gap: ~41k true-like pairs missed, ≤ ~1 France pt if all were recovered with no false positives.
  - Best guess: recompute the context features (s1_*/sx_* margin, gap and n) over the pruned list.
  - Or retrain M2 once with them dropped / relativised, and check on the P14 India-under-US-IDF proxy.
  - A rule cannot do it (R3 shows why).
- **Should 'core equal, hn +1/+2' (22.6k pairs, mean p 0.45) be zeroed like SIB3?** Best guess: no. The US truth rate is 0.16 and India's 0.65.
  France is US-like, so the rule would be net positive but small (~1k accepted pairs). The LB slot is better spent on R1.
- **Parse '(12) RUE' house numbers in src (P6's `_paren_hn`)?** Best guess: yes, with the next src change. It is 1.6% of France S2/S3,
  and pairs whose hn is missing on one side are 9.6% of France's band. P14 measured it neutral alone, and it is harmless.
