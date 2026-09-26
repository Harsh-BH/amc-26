# P44-noise-word-tier — RESULT (27 Sep ~01:45 IST)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **A France generator template the India/US model never saw.** A true France variant drops one S1 word and gains "Développement" at the identical address, usually after the legal form ("Rapiere Sportive SAS" → "Rapiere SAS Développement"). The matcher scores these at median p 0.31, so they are all rejected.
- **A label-free lens picks the noise words.** It looks at argmax one-word-swap pairs with a house number on both sides: the sibling-offset share must be ≤ 0.05 and the same-house-number share ≥ 0.8. On the #8c holdout that cell is **94.7% true (US, 3,603 pairs) / 96.9% (India, 1,971)**; decoy words (s > 0.14) are ≤ 0.4% true. In France it selects **développement only** (s 0.018, e 0.974 over 4,735 pairs).
- **Staged, validator PASS, France-only additions:**
  - **`sub44_on43`** = #43_r90 (the master's recommended combined upload) + 3,185 pairs (3,147 S1, 75 empties filled);
  - `sub44_on43safe` = #43_safe + 3,186;
  - `sub44g_on43` adds groupe (+5,252 in total).
  - Expected ≈ **+0.00018 LB** (g: ≈ +0.00026). LOCO confirms precision (100%) but, being conservative, measures no gain.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
Upload the P44 file that sits on the current best public base. Every file = base + France pairs only: India/US and `candidate_pairs.tsv` are byte-identical to the base (hard links).

| folder | base | France change vs base | per S1 / empty | expected LB vs base |
|---|---|---|---|---|
| **`output/sub44_on43`** | #43_r90 | +3,185 pairs (développement), 3,147 S1, 75 empties filled | 3.244 → 3.256 / 6.018% → 5.989% | ≈ +0.00018 |
| `output/sub44g_on43` | #43_r90 | +5,252 (+ groupe 2,065), 5,165 S1, 147 empties | 3.244 → 3.264 / 5.961% | ≈ +0.00026 (riskier) |
| `output/sub44_on43safe` | #43_safe | +3,186, 3,148 S1, 81 empties | 3.241 → 3.253 / 6.258% → 6.227% | ≈ +0.00018 |

- Order: upload the plain file right after its base, i.e. after #43_r90 (or #43_safe) is uploaded. Upload `g` only after the plain one gains, and only if a slot is spare.
- Files on #41, #42_both_on41 and #41_hn90 were built at 01:35–01:37 (same numbers ±20 pairs), then deleted for disk once #43 superseded those bases. Rebuild in 10 s with `build`.
- The tier is disjoint from the other rules. It needs hn + street + city equal and a shared name word. #41 takes bare brand names, P40's xf_noaddr takes address-less SX, and #42's extras need p ≥ 0.80 while the P44 adds have median p 0.31. Pairs already present are skipped: 70–71 plain, 167 g.
- Any other base: `P=probes/P44-noise-word-tier/p44.py; .venv/bin/python $P build output/<base> output/<out> lens` (10 s, 1.0 GB). Then run the validator.

**run.py integration** (France only, after the rescue / gc / gib / extras rules, on the post-rules p):
```python
m = P44.lens_masks(d_fr[["s1", "sx", "p"]], "test", "France", t, am, learn=cands_test_France, s_max=0.05)   # 0.09 = 'g'
add = m["lens_rule"] & ~acc          # guards + lens; learned on this run's France argmax pairs (label-free)
pred, _ = P44.merge(pred, d_fr[add])  # SX free, caps 5 S2 / 6 S3, best p first
```
- It depends on P33 `r05.r05_masks` / `s_table`, P28 `anchor.rule_masks` (sibling-word guard learned from the blocked list) and P20 `patterns.name_rel`.
- For a frozen recipe, either import them or port them. Porting means freezing the learned word list: `{"developpement"}`, plus `"groupe"` for g.

## Evidence         (table: ledger name | world | key metrics | verdict)
Ledger rows from 01:30 on use the lens as defined here. Rows 01:19–01:28 used P33's all-pairs share and are superseded (see What failed); so is build-sub44_hn90 (01:35), whose folder was deleted once #42 dominated hn90.

| ledger name | world | key metrics | verdict |
|---|---|---|---|
| lens-holdout-US (01:30) | #8c holdout (P23 dump), sub1 argmax pairs with n_hn(w) ≥ 20 | by s: ≤ 0.05 3,656 pairs **0.946 true**; 0.05–0.09 234 0.868; 0.09–0.14 130 0.723; 0.14–0.2 48 0.208; > 0.2 2,403 **0.002**. Lens cell (s ≤ 0.05, e ≥ 0.8) 3,603 @ **0.947** (partners 3,081 + typo tokens); g-only cell (0.05 < s ≤ 0.09, e ≥ 0.8) 205 @ 0.873 | keep |
| lens-holdout-India (01:30) | same | ≤ 0.05 3,072 **0.959**; 0.05–0.09 21 0.000; 0.09–0.14 172 0.000; 0.14–0.2 2,448 0.004; > 0.2 3,161 0.001. Lens cell 1,971 @ **0.969** (partners, lakshmi); g-only cell 0 pairs | keep |
| lens-loco-India>US (01:31) | P6 LOCO fold 0, t_profile 0.81 | lens: 5 flips @ 1.00, dF ≈ 0; lens_g: 10 @ 1.00, +0.00001 | precision only |
| lens-loco-US>India (01:31) | same, t 0.97 | lens: 0 flips; lens_g: 24 @ 1.00 (lakshmi), +0.00006 ± 0.00003 | precision only |
| lens-France-flips (01:34) | test France, `p31/fr31v0.1_t9868` (#31_veto post-rules p, t 0.9868) | lens: 3,256 flips on 3,217 S1 (développement 3,254), p quartiles 0.13 / 0.32 / 0.54; lens_g: 5,419 on 5,329 S1 (+ groupe 2,154) | — |
| build-sub44_on43 / -sub44g_on43 / -sub44_on43safe (~01:50); earlier bases 01:35–01:37 | test | table above; validator PASS for all; independent check: only France rows change, only additions, 0 SX assigned twice, every added pair is in `candidate_pairs.tsv` | staged |

**The France lens table** (argmax sub1 pairs, n = pairs with a house number on both sides):

| word | n | s (sibling offset) | e (same hn) | reading |
|---|---|---|---|---|
| **développement** | 4,735 | **0.018** | **0.974** | noise word; the lens takes it |
| groupe | 5,544 | 0.084 | 0.887 | taken by `g` only. US analogue cell 87% true. As an **added** word it is a sibling signal (P20: 51% SIB3), but the rule takes swaps only |
| fils / associes / cie (R1 fillers) | 11,018 / 5,647 / 6,529 | 0.079 / 0.088 / 0.109 | 0.860 / 0.849 / 0.818 | handled by R1 (in #31's LB gain); excluded here |
| club / école / comité / amicale / amis / union / … | 1–6k each | **0.28–0.39** | 0.50–0.65 | decoy words: siblings plus India-style same-address swaps; correctly rejected |
| international / participations / distribution / holding | ~34k SX each | — | — | sibling branch words (added); 99.5% unassigned |

**Why the model misses them:** in the US a swapped corporate word is a decoy (group s 0.786, 0% true; holdings 0.797, 0%). The one US noise swap word, "partners", is in the training data (accepted 94%). "Développement" is not, so its swaps are most likely scored like a decoy's. Samples (all at the identical address):
- `Rapiere Sportive SAS` → `Rapiere SAS Développement` (p 0.57)
- `Val Dentraide Comite` → `Val Dentraide Développement` (0.36)
- `Coeur Culture SAS` → `COEUR SAS-DÉVELOPPEMENT` (0.02)
- `Tribunal Comite EURL` → `Tribunal EURL Développement` (0.21)

**France data catalogue** (whole France split, label-free: token document frequency in 1.43M S2/S3 vs 259k S1, plus raw rows). France is 3 régions / 4 départements (Nord, Pas-de-Calais, Gironde, Loire-Atlantique), 23.8k streets, median house number 24 (US 1,352), so 16.7% of France S1 share an exact address with another S1 (US 8.6%).

| generator operation on France S2/S3 | size | handled |
|---|---|---|
| région → département (the learned P21 map) or région dropped (~⅔ of SX) | ~150k per département | yes |
| legal form dotted / bracketed (`[[Sarl]]`), **SNC swapped in** (never in S1), all-caps (13%), "(France)" toggled, "&" → "et" / "+" | SNC 16k | yes |
| alias around a synthetic brand (`Tavojax formerly: …`, `… trading as …`), websites (4.2%), bare brand names | 12k / 60k | yes (#41 for brands) |
| "& Fils / Cie / Associés / Frères / Frs" fillers | — | yes (R1) |
| **"Développement" (and "Groupe") swapped in after the legal form** | 3.2k (+2.1k) pairs lost | **P44** |
| street abbreviations `Ch` (chemin), `Q` (quai), `Psg` / `Pass` (passage); avenue typos (aveue / aenue / avnue …) | 7k / 4.2k / 1.6k / ~3k addresses | **no** (small; needs a France rescore) |
| house-number labels (`N°`, `NO`, `#`), zero padding (`005`), `(n)` prefix | 94k / 71k / 47k | yes |

## What failed      (approaches tried and dropped, with the number that killed them)
- **P33's s(w) denominator** (all swap pairs, address-less included).
  - India's address-less pairs dilute decoy words below 0.05 (ventures, technologies, management). The holdout 0.05–0.08 bin is **1.7% true** (596 pairs).
  - My first LOCO reading (US→India +0.00033, 196 flips at 0.985) used that definition. The flipped words (partners / jay / laxmi / jai / lakshmi) are true noise words, but the definition cannot be trusted. Superseded.
- **LOCO as a gain measure.** The corrected lens needs e ≥ 0.8. India's true pairs carry house-number noise, so its noise words fall below that in the LOCO frame and the lens flips 0–24 pairs there. LOCO confirms precision only; the gain estimate is arithmetic on the holdout precision.
- **Other France-wide checks from the same session, all label-free:**
  - Address-collision feature for India/US: accepted precision 0.9975 at shared vs 0.9983 at unique S1 addresses. Nothing.
  - Unique-address disjoint names: LOCO +0.00012 / +0.00013 at precision 0.81–0.84. #41's brand rule covers it.
  - Same address up to a street typo: 3,575 France pairs at p ≥ 0.9; LOCO tiny, and the loose version fails US→India (precision 0.64).
  - Two same-name S1 splitting an SX against its address: 35 cases in all France.
  - Same-address org-word swaps (club / école …): India-like decoy profile, correctly rejected.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `p44.py selftest` checks `merge` (caps, owned SX, input untouched) and `name_rel` on the two template forms.
- `ho US India`: ~15 s (plus loading), 1.6 GB.
- `loco 'India>US' 'US>India'`: ~1.5 min, 2.5 GB.
- `fr`: ~2.5 min, 2.6 GB. Writes `data/work/p44/flips_{lens,lens_g}.parquet` (s1, sx, p, w) and `lens_France_top.csv`.
- `build BASE OUT [lens|lens_g]`: 10 s, 1.0 GB. It asserts that every base France pair is an argmax pair of the France file, and that every added pair is a candidate.
- `lens_table(pairs, xs, c)` → per word: n (hn on both sides), n_all, s, e. `lens_masks(d, split, c, t, am, learn, s_max, best)` → P33 masks + `lens_rule`. `merge(pred, add)` → (new pred, added rows).

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **The precision transfer is assumed.** The ~0.96 figure uses the holdout lens cell (94.7% / 96.9%) and France's extreme profile: développement's sibling share is 0.018, 17× below France's decoy words (0.28–0.39). LOCO cannot measure the gain, so the LB is the arbiter. Break-even precision is ~0.73.
- **groupe (`g`)**: s 0.084 is the edge of India's decoy band; its US analogue cell is 87% true. France is US-like on house numbers (P27 / P37), which argues for it, but it stays the riskier half.
- **Rule risk: none.** No labels, no model, no external data. It uses only unlabelled test statistics (ruling 2) plus P33 / P28 guards.
- **Not in run.py yet.** P43 owns the final package. A best-public P44 file must be regenerable by the frozen code.

## Open questions   (for the master; each with your best guess)
- **Which base?** #43_r90 (the queue's recommendation) → `sub44_on43`; if #43_safe is uploaded instead → `sub44_on43safe`. For any other best-public base, `build` makes the file in 10 s.
- **Also fix the France street abbreviations (Ch / Q / Psg / Pass)?** Best guess: no, not before the freeze. It needs a France rescore; its main effect is on P28's exact-street same-address rule, and the rough size is ≤ +0.00005.
