# P40-generator — RESULT (final, 27 Sep ~01:15 IST; addendum ~01:35 for the master's sub41 follow-up)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **India/US: one generator op, the address-less SX, carries 59% of the remaining loss (0.00492 of 0.00828 LB), and
  0.00373 LB (45% of all India/US loss) is a Bayes floor.** Those SX belong to an S1 whose full name ≥ 2 train S1 share, and
  the generator leaves no tie-breaker. No India/US op rule passes the gate (best +0.00003 LB).
- **France under-accepts two generator-variant classes:** gib brand names at the S1's address, and address-less SX whose name
  core is unique among France S1. Label-free, France holds 3–6× more of them per S1 below its threshold than either
  LOCO target. `xu_noaddr|gib` at France p ≥ 0.9 is positive in both LOCO directions: +0.00043 / +0.00004 France-F0.5,
  or ≈ +0.00026 / +0.00006 LB scaled to France's volume. It misses the +0.0005 both-ways gate.
- **Update (after P41):** on top of `output/sub41`, stage **`output/sub40x`**. It adds only the address-less part, with a
  full-name key: +2,441 France pairs, LOCO +0.000077 / +0.000024, expected LB +0.00002 to +0.00010. See the Addendum.
- Staged before P41: `output/sub40` = #37_gc70 + that rule (+6,521 France pairs, validator PASS). Expected LB +0.00006 to +0.00026,
  downside ≈ 0. Looser `output/sub40_t70` (+9,624 pairs): +0.0004 / −0.00002.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- **Now that #41 is being uploaded: queue `output/sub40x` after #41** (then `sub40x_t70`). Addendum at the end;
  both rows are in SUBMISSIONS.md.
- Without #41: **upload `output/sub40`** after any higher-priority France upload. If it gains, `output/sub40_t70` is the next step
  (same rule at FR 0.7).
  - Both folders: India/US lines, `candidate_pairs.tsv` and `code.zip` are #37_gc70's (hard links). Every added pair comes
    from the France pair file, so it is already a candidate (checked: 0 lines with a match outside the candidates).
- **run.py** (France decision loop, right after P37's `gc_add` line that makes #37_gc70):
```python
import rule40 as R40                                   # promote probes/P40-generator/rule40.py (numpy + gib.py)
pool = am & ~acc & (d["p"] >= 0.90)                    # argmax, unaccepted, p >= 0.9
cls = np.zeros(len(acc), bool)
cls[pool] = R40.op_class(n1[pool], nx[pool], ax[pool], core1[pool], corex[pool], fr_core_count)
acc |= R40.add(s1c, sxc, is2, d["p"], am, acc, sib, cls, 0.90)          # = output/sub40 (0.70 = sub40_t70)
```
  - `n1` / `nx` are the raw S1 / SX names and `ax` the raw SX address. `core1` / `corex` are the rf `core` columns
    (`normalize.name_core`).
  - `fr_core_count` = `{core: #France S1 with that core}`, from the test France S1 cores (label-free).
  - `sib` is the P15 SIB3 mask, the same one P37 uses.
  - `op_class` needs P36's `gib.py` plus `data/work/p36/morph_train.json` (the 44 syllables, learned label-free).
- **India/US: adopt nothing.** Stop spending probes on India/US residuals. Of the 0.0083 LB, 0.0037 is the floor. The rest
  is spread thin: every op class is ≥ 97% precise among accepted pairs, and a10's p' is calibrated inside each op class.

## Evidence         (table: ledger name | world | key metrics | verdict)
### 1. Op → share of the remaining #34 India/US loss (main deliverable)
- Setup: holdout, #34 decision out of fold (P36 `state`), P36 error pairs (misses + candidate FN + FP).
- Loss weight per error pair e of S1 s: w_e = (1 − F0.5_s) · g_e / Σg, where g_e is the F0.5 gain of fixing e alone. Rows
  therefore add up to the loss.
- Each FN pair goes to ONE primary op: the op on it with the highest FN rate among holdout true pairs (last column).
- FP pairs are split by what the SX is.
- LB-eq = F0.5 × test share (India 0.468, US 0.383).

| primary op | error pairs India / US | LB-eq India / US | LB-eq | share | FN rate of true pairs with op |
|---|---|---|---|---|---|
| **A_EMPTY** (SX address empty) | 5,220 / 9,725 | 0.00239 / 0.00253 | **0.00492** | **0.594** | 0.446 |
| ↳ true S1's full name shared by ≥ 2 train S1 (**Bayes floor**) | 11,450 FN | | 0.00373 | 0.451 | |
| ↳ unique full name | 3,495 FN | | 0.00119 | 0.143 | |
| N_GIB_NAME (synthetic brand, 44 syllables) | 1,047 / 1,305 | 0.00051 / 0.00034 | **0.00085** | 0.103 | 0.203 |
| FP: orphan SX (decoy, owned by nobody) | 269 / 395 | 0.00039 / 0.00032 | **0.00070** | 0.085 | |
| FP: SX owned by an S1 of another fold | 160 / 285 | 0.00020 / 0.00021 | **0.00041** | 0.050 | |
| N_GENERIC_ADD (branch word: center/services/group…) | 487 / 290 | 0.00024 / 0.00009 | **0.00034** | 0.041 | 0.064 |
| N_TOKDROP (core token dropped) | 172 / 143 | 0.00009 / 0.00003 | 0.00013 | 0.016 | 0.039 |
| N_LEGAL_ADD | 27 / 178 | 0.00001 / 0.00010 | 0.00011 | 0.013 | 0.034 |
| A_HN_OTHER (hn changed, not ±1/2, not SIB3) | 108 / 152 | 0.00007 / 0.00004 | 0.00011 | 0.013 | 0.030 |
| A_HN_SMALL (hn ±1/±2) | 55 / 151 | 0.00003 / 0.00006 | 0.00009 | 0.011 | 0.048 |
| A_SEG_DROP | 173 / 2 | 0.00008 / 0.00000 | 0.00008 | 0.010 | 0.012 |
| N_LEGAL_DROP | 110 / 26 | 0.00007 / 0.00001 | 0.00008 | 0.009 | 0.037 |
| A_HN_SIB (true pair with hn + SIB3 offset) | 16 / 128 | 0.00001 / 0.00006 | 0.00006 | 0.008 | 0.180 |
| A_SEG_ADD | 66 / 13 | 0.00004 / 0.00000 | 0.00004 | 0.005 | 0.014 |
| FP: SX owned by another holdout S1 | 16 / 28 | 0.00002 / 0.00002 | 0.00004 | 0.004 | |
| FP: sibling orphan (hn + SIB3) | 20 / 14 | 0.00002 / 0.00001 | 0.00003 | 0.004 | |
| A_HN_LEADDROP (hn loses its first digit) | 24 / 7 | 0.00003 / 0.00000 | 0.00003 | 0.004 | 0.023 |
| 40 more ops, each < 0.00003 | | | 0.00026 | 0.031 | |
| **total** | 8,309 / 13,052 | 0.00439 / 0.00389 | **0.00828** | 1 | |

- By pipeline stage, the A_EMPTY loss splits into block miss 0.00117, pruner 0.00079, low-p 0.00147 and stolen 0.00150.
- Address-less FN rate by the true S1's core-name twin count k (train S1 of the country sharing its core):

| k | FN rate India | FN rate US |
|---|---|---|
| 1 | 8.7% | 9.9% |
| 2 | 52% | 70% |
| 3 | 66% | 78% |
| ≥ 4 | 91% | 92% |

- With an address, the FN rate is 0.5–1.0% at every k.
- 56% (India) and 49% (US) of train S1 share their name core with another S1.

### 2. Generator catalogue (holdout true pairs, fold 0; rates per true pair; `catalogue_rates.csv`)
**Cluster sizes** (identical in US and India):

| per S1 | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| S2 | 13% | 35% | 30% | 15% | 5% | 1% | — |
| S3 | 12% | 32% | 30% | 17% | 7% | 1.6% | 0.1% |

- corr(n2, n3) = 0.12; mean 3.46–3.48 variants per S1; 5.5% singletons.
- **Every op is decided independently per record.** Co-occurrence lift ≈ 1. P(empty address) is 0.043–0.047 at every
  cluster size. (S1, source) groups are all-empty 0.1% vs mixed 10.9%. Case, spacing and accents agree between two copies of
  one entity no more often than between random copies.
- **S1 names are drawn with replacement.** Twin S1s have the same singleton rate (5.6%) and the same mean size (3.46) as
  unique ones. There is no row-order or ID leak: same-entity copies sit at random row and ID distances, corr 0.001.

**Name ops** (US / India, S2–S3 range):

| group | ops |
|---|---|
| exact / format | exact 3–6%; case 2–13% (S2 upper 23–38%); double space 11%; accent inserted 5–7%; Indic script 0 / 13–23% |
| name replaced by a form | website 4%; handle 0.4%; `name \| site` 0.3%; phone 0.3%; "formerly/aka…" 3–4% (S3 only) |
| affixes | honorific 1–8%; brackets 3%; hyphen join 5–6% |
| legal form | add 3–9%; drop 14–21%; swap 1–9% |
| tokens | branch word 7–9%; core-token drop 10–20%; reorder 8%; duplication 3–4%; random token 5–13% |
| typos | digit-for-letter 2–3%; deletion 1%; insertion 1%; substitution 1%; transposition 0.3%; multi-edit 3% |
| other | gib brand 1–2.3%; acronym ≤ 0.5%; unrelated name 0.7% |

**Address ops:**

| group | ops |
|---|---|
| whole address | exact 0 (S2) / 4% (S3); empty 3.8–4.9%; NULL segment 2–3%; N/A segment 1%; S2 upper-case (US 89%, India 24%) |
| house number | same 48–79%; dropped 6–9%; added 0 / 4%; first digit dropped 1–2%; extra number prefixed 7–8% (India); ±1/2 0.8%; + SIB3 0.1%; other change 3–5%; label (#, No., H.No) 5–16%; letter suffix 2% (US) |
| street / segments | street abbreviation 21–52%; segment drop 19–65%; reorder 26–39%; typo 17–29%; segment added 5–11%; filler (REGION, CITY…) 5–14% |
| state | form swap (S3) 67–89%; Indic state name 22% (India) |

**Distractors** (argmax pairs with y = 0; `catalogue_distractors.csv`):
- **Orphans.** US orphans are hn siblings: hn + SIB3 65%, hn ±1/2 16%. India orphans are spread out: hn same 27%, hn
  missing on one side 26%, no hn 22%, SIB3 12%. A branch word from my short list marks ≥ 11% (US) / 28% (India) of them;
  P12's wider list says 66% / 82%.
- **Decoys at the S1's own house number** do one of four things:
  - change one short token (DO → DOW, IT → ITY);
  - glue a syllable onto a word (Delta → Deltalum);
  - rename the SX to a gib brand;
  - change a unit or second number.
  The model already rejects them: short-edit / syllable-append pairs are rare, and 0.89–0.97 precise when accepted.
- **SX owned by another S1** are 93% (US) / 74% (India) address-less (the name-twin case).

| ledger name | world | key metrics | verdict |
|---|---|---|---|
| op-loss-table-34-holdout | holdout (#34 oof) | table 1 | keep |
| addrless-structure-and-leaks | holdout + full train | independence, no leak, twin FN curve (above) | drop (no tie-breaker) |
| iu-lacks-source-true-rate | holdout (#34 oof) | rejected argmax pairs whose S1 group lacks the SX's source: 0.67–0.79 true at p' 0.5–0.71, vs 0.49–0.62 with the source | drop |
| iu-add-rules-gc-noaddruniq-gib | holdout, half split | best arm gc_lacks@0.65: LB **+0.000021 ± 0.000009 (h0) / +0.000030 ± 0.000010 (h1)**; noaddr_uniq@0.65 +0.000002 / +0.000034; gib@0.6 −0.000014 / −0.000054; union@0.65 +0.000005 / +0.000037 | drop (gate +0.00015 at 3 SE) |
| france-labelfree-classrates | train + test | see §3 | keep |
| loco-add-{gib,noaddr,noaddr_unique,…} (loco40) | LOCO, gc70 base | noaddr@0.3: +0.00116 (prec 0.95) / **−0.00385 (0.40)** | drop |
| loco41-add-{xu_noaddr,xu_any,gib,xu_noaddr\|gib} | LOCO + France counts | table §3 | xu_noaddr\|gib keep; the rest drop |
| loco_allops (`loco_allops.csv`) | LOCO + France counts | every op as an add-class at FR 0.7 / 0.9: nothing beats gib. Acronym (+0.00009) and exact address (+0.00006) exist only in US→India | drop |
| stage-sub40 / stage-sub40_t70 | France test | +6,521 pairs / 6,341 S1; +9,624 / 9,306; validator PASS | keep / inconclusive |

### 3. France (label-free op mix + LOCO)
**Label-free class rates.** Shares of SX, from a row-group sample; the train owned-share comes from train truth.

| | train India | train US | test India | test US | **test France** |
|---|---|---|---|---|---|
| gib-named SX share (train: share owned) | 2.30% (99.2%) | 3.10% (88.1%) | 1.83% | 2.87% | **2.08%** |
| gib SX per S1 (share × SX/S1) | 0.108 | 0.146 | 0.107 | 0.165 | **0.115** (India-like: no gib decoys) |
| address-less SX share (train: share owned) | 2.98% (97.9%) | 3.55% (97.8%) | 2.35% | 2.90% | **3.01%** |
| address-less SX per S1 | 0.140 | 0.167 | 0.137 | 0.167 | **0.166** |
| digit-for-letter names | 1.9% | 2.5% | 1.7% | 2.3% | **0.3%** (France's name noise is lighter) |

**Recall on each class.** France: accepted per S1 / (class SX per S1 × 0.98).

| class | France | LOCO US→India base | LOCO India→US base | India/US in-domain |
|---|---|---|---|---|
| address-less | **≈ 0.32** | 0.38 | 0.52 | 0.57 |
| gib | **≈ 0.32** | 0.62 | 0.67 | — |

- France under-accepts both classes more than either LOCO base.
- It holds 3–6× more of them rejected per S1: gib at FR ≥ 0.9 is 0.015 per S1, against 0.0025 (US→India) and 0.0019
  (India→US).
- US→India is the France-like direction for these classes on both counts: strictness (t_base 0.97 vs France's 0.9868) and
  the gib decoy rate.

**France op mix** (per S1, pairs carrying the op):

| op | India truth | US truth | US accepted | France accepted | France rejected p ≥ 0.5 |
|---|---|---|---|---|---|
| all pairs | 3.475 | 3.460 | 3.372 | 3.137 | 0.241 |
| A_EMPTY | 0.136 | 0.163 | 0.092 | **0.052** | 0.038 |
| N_GIB_NAME | 0.059 | 0.048 | 0.039 | **0.019** | **0.030** |
| A_HN_OTHER / A_HN_LEADDROP | 0.096 / 0.032 | 0.188 / 0.072 | 0.183 / 0.071 | 0.028 / 0.001 | 0.009 / 0.000 |
| N_T_DIGIT / N_T_MULTI | 0.078 / 0.092 | 0.100 / 0.118 | 0.098 / 0.115 | 0.013 / 0.038 | 0.001 / 0.003 |
| N_EXACT | 0.094 | 0.205 | 0.202 | 0.506 | 0.014 |

- France's hn-change and typo ops are rare in all its argmax pairs, not only in the accepted ones. So the generator's
  France rates differ there (lighter noise), and those ops are not a France FN source.
- Estimated France truth per S1: 5.53 SX/S1 × the India/US owned share 0.60 = 3.32. The implied France deficit is about
  0.28 true pairs per S1, and address-less (≈ 0.11) plus gib (≈ 0.04) are about half of it.

**LOCO of the France rules** (P27 M2, gc70 base, density-mapped t):
- Cell format: dF ± SE(h1), then (pairs added, precision).
- "Scaled LB" = LOCO dF per added pair × France pairs per S1 × 0.1497.

| rule @ FR t | US→India dF ± SE (added, prec) | India→US dF ± SE (added, prec) | France pairs | scaled LB US→India / India→US |
|---|---|---|---|---|
| xu_noaddr@0.9 | +0.000262 ± 0.000025 (318, 0.987) | −0.000015 ± 0.000010 (67, 0.552) | 2,577 | +0.00011 / −0.00004 |
| xu_noaddr@0.5 | +0.000559 ± 0.000055 (579, 0.981) | −0.000039 ± 0.000014 (118, 0.542) | 3,757 | +0.00018 / −0.00009 |
| xu_any@0.9 | +0.000860 ± 0.000063 (1,318, 0.923) | −0.000114 ± 0.000062 (1,756, 0.695) | 3,744 | +0.00012 / −0.00002 |
| xu_any@0.3 | +0.002531 ± 0.000171 (3,249, 0.919) | −0.000985 ± 0.000147 (3,927, 0.623) | 8,505 | +0.00034 / −0.00016 |
| gib@0.9 | +0.000163 ± 0.000019 (222, 0.964) | +0.000053 ± 0.000020 (257, 0.813) | 3,929 | +0.00015 / +0.00006 |
| gib@0.7 | +0.000244 ± 0.000044 (339, 0.932) | +0.000025 ± 0.000040 (461, 0.748) | 6,205 | +0.00023 / +0.00003 |
| gib@0.5 | +0.000326 ± 0.000059 (463, 0.918) | −0.000006 ± 0.000054 (678, 0.718) | 7,778 | +0.00028 / −0.00001 |
| **xu_noaddr\|gib@0.9 (= sub40)** | **+0.000425 ± 0.000032 (540, 0.978)** | **+0.000039 ± 0.000022 (324, 0.759)** | 6,505 | **+0.00026 / +0.00006** |
| xu_noaddr\|gib@0.7 (= sub40_t70) | +0.000677 ± 0.000062 (808, 0.963) | −0.000016 ± 0.000043 (564, 0.707) | 9,591 | +0.00041 / −0.00002 |
| xu_noaddr\|gib@0.5 | +0.000885 ± 0.000080 (1,042, 0.953) | −0.000045 ± 0.000056 (796, 0.692) | 11,534 | +0.00050 / −0.00005 |

- The staged files use `rule40` (the run.py path). It adds 16 more pairs than the tagger-based class (6,521 vs 6,505),
  and it contains all of the tagger's pairs.
- Eyeballed France adds look like generator true variants:
  - an exact or case-changed copy of a unique S1 name with an empty address;
  - a gib brand at the S1's exact house number, street and city.

## What failed      (approaches tried and dropped, with the number that killed them)
- **India/US op rules.** Every op class is 97–99.9% precise among accepted pairs. The best rejected classes are only 24–35%
  true (address-less, case-only, gib), and a10's p' is calibrated inside each of them. The half-split gate fails for every
  arm; the best is +0.00003 LB.
- **In-domain group completion:** +0.00002 / +0.00003 LB.
- **Tie-breakers for address-less name twins.** Per-(entity, source) consistency of name traits: none. Row order or ID
  proximity: none. Singleton / size law by twin status: flat.
- **Decoy removal:** decoy-style edits are rare among accepted pairs and 0.89–0.97 precise. Removal pays only below 0.7.
- **France `noaddr` (any address-less pair) and `xu_any`:** strongly negative in India→US (precision 0.40–0.70).
- **France hn-change / typo ops as an FN source:** they are rare in France's whole argmax pool (generator difference), so
  there is nothing to recover.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `rule40.py` (for run.py, numpy + P36 `gib.py`):
  - `op_class(n1, nx, ax, c1, cx, core_count) -> bool[]`
  - `add(s1, sx, is2, p, am, pred, sib, cls, t) -> bool[]`
  - Self-check: `.venv/bin/python probes/P40-generator/rule40.py`.
  - France: pool ≈ 30k pairs at p ≥ 0.9, < 5 s.
- `build40.py [RULE@T] [OUT]` (`P37_BASE=output/sub37_gc70`): writes `output/sub40*`. 15 s, 1.2 GB.
- `ops.py`: the generator-op tagger.
  - `tag(n1, a1, nx, ax, h1, hx, country) -> int` bitmask over 59 `OPS`; `tag_frame(...)` vectorised (≈ 8k pairs/s);
    `names(mask)`.
  - Self-check: `.venv/bin/python probes/P40-generator/ops.py`.
- `cat.py tag`: holdout true pairs + ops → `data/work/p40/tp.parquet` (12 MB, 90 s).
- `loss.py tag|table`: error pairs, per-pair weights, primary-op table → `err_{c}.parquet`, `err_w.parquet`,
  `loss_primary.csv`, `loss_oracle.csv`, `op_rates.csv`. Runs in 3 min, 2 GB.
- `cand.py tag|rates`: every #34 holdout argmax pair + ops → `am_{c}.parquet`, `am_rates.csv`. Runs in 4 min.
- `france.py tag|cmp`: France accepted + rejected argmax pairs + ops → `fr_pairs.parquet`, `fr_opmix.csv`. Runs in 2 min.
- LOCO scripts:
  - `loco40.py` (class rules);
  - `loco41.py loco|france` (near-exact classes + France counts);
  - `loco_allops.py` (every op).
  - Each needs `data/work/p27/p6/S_holdout_{US,India}.parquet`, fetched back from S3. Runs in 10–20 min, 2.4 GB.
- `rules_iu.py`: India/US half-split gate.
- Aggregate scripts behind the numbers above: `catalogue.py`, `empty.py`, `empty2.py`, `twins.py`, `floor.py`, `leak.py`,
  `copytraits.py`, `gcrate.py`, `fpdiag.py`, `uniq.py`, `frhn.py`, `decoyrate.py`, `locorecall.py`.
- Disk: `data/work/p40/` ≈ 60 MB. `data/work/p27/p6/S_holdout_*` is 102 MB and can be deleted again: it is on S3.
  `output/sub40*` are 97 MB each; the other files in those folders are hard links.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Licences / rules:** no new dependency; no external data. The 44 syllables are learned from the data (P36). The France
  core counts come from test S1 (label-free, allowed by the organisers' ruling 2).
- **France is unvalidated.** The rule passes both LOCO directions only at FR 0.9, and only by +0.00004 in India→US. The
  France-likeness argument (US→India for these classes) rests on label-free evidence: France's class recall ≈ 0.32, 3–6×
  denser rejected pools, India-like gib counts.
  - If France behaves like India→US, the LB is ≈ +0.00006; if it behaves like US→India, ≈ +0.00026.
  - `sub40_t70` risks ≈ −0.00002.
- **Detector caveats:** the op detectors are heuristic, and rates are descriptive ("segment typo" includes normalisation
  mismatches). N_SHORT_EDIT / N_SYLL_APPEND / A_NUM_SUB are missing from `tp.parquet` (they were added after that tagging
  run); they are present in `am_*` / `fr_pairs`.
- **The Bayes floor assumes nothing outside the record separates twin S1s.** The group-size prior is the only residual
  signal and it is weak: P(n+1)/P(n) is 0.5–3, and the one-sided test gave +0.00003.

## Open questions   (for the master; each with your best guess)
- **Is #1 ahead on India/US?** Our India/US loss is 0.0083 LB, and 0.0037 of it is the floor. #1 can gain at most ≈ 0.0046
  there, and realistically ≤ 0.001. Best guess: #1's lead is France (≈ 0.99 vs our ≈ 0.964). France's remaining deficit is
  ≈ 0.28 true pairs per S1, and about half of it is address-less + gib variants.
- **Is a larger France address-less push worth it?** France accepts ~0.052 address-less pairs per S1 of ~0.163 true.
  Best guess: only via unique-name matches (sub40's xu part). Ambiguous names are coin flips in France as well.
- **Add acronym / exact-address classes to rule40?** +0.00006 to +0.00009 LB, with evidence in US→India only. Best guess:
  not worth an upload on their own; fold them in only if sub40 gains.

## Addendum (27 Sep ~01:35): rule40's non-gib part on top of P41's `output/sub41`
The master asked for this follow-up: P41's gib rule (`sub41`) overlaps rule40's gib half. As pair sets, 3,206 of sub40's
3,944 gib pairs are in sub41's 5,742 adds. The question was whether the address-less part still adds something on top of
sub41 (`loco42.py`).

- **The LOCO base is faithful.** It is P37's gc70 base + P41's `key1_bare` rule at FR 0.3, and it reproduces P41's own
  numbers exactly: +0.000078 (83 pairs, precision 1.00) and +0.000102 (182 pairs, 0.989).
- **Result table.** Cell format: dF ± SE(h1), then (added, precision).

| class, on top of sub41 | FR t | US→India | India→US | France pairs (overlap with sub41 adds) | scaled LB |
|---|---|---|---|---|---|
| `xu_noaddr` (core == S1 core, core unique) | 0.9 | +0.000262 ± 0.000025 (318, 0.987) | **−0.000015 ± 0.000010 (67, 0.552)** | 2,577 (0) | +0.00011 / −0.00004 |
| `xu_noaddr` | 0.7 | +0.000433 ± 0.000044 (469, 0.985) | −0.000041 ± 0.000013 (103, 0.524) | 3,387 (0) | |
| **`xf_noaddr`** (full name, legal form kept, == S1's and unique) | **0.9** | **+0.000077 ± 0.000015 (98, 0.969)** | **+0.000024 ± 0.000022 (292, 0.764)** | 2,441 (0; 56 shared S1) | **+0.000098 / +0.000015** |
| `xf_noaddr` | 0.7 | +0.000114 ± 0.000020 (145, 0.959) | +0.000034 ± 0.000048 (456, 0.735) | 4,692 (0; fills 77 empty S1) | +0.00019 / +0.000027 |
| `xf_noaddr` | 0.5 | +0.000137 ± 0.000022 (195, 0.954) | +0.000028 ± 0.000052 (522, 0.713) | 5,596 (0) | |

- **xu_noaddr fails the master's condition** (India→US ≥ 0). Its false adds are SX that dropped a token and landed on
  another S1 with a unique core. The full-name key (legal form kept) removes most of them, so **xf_noaddr passes at
  0.9 and at 0.7**.
- The two classes are disjoint from P41's rule (gib SX need an address; these SX have none). The LOCO shows no interaction:
  on the gc70 base, xu_noaddr gives exactly the same dF.
- **Staged on sub41** (`loco42.py build`, `P40_XU=full P41_BASE=output/sub41`):
  - `output/sub40x`: +2,441 pairs / 2,360 S1;
  - `output/sub40x_t70`: +4,692 / 4,514.
  - Both: validator PASS, only France lines differ from sub41, 0 matches outside the candidates. Candidates and code.zip
    are hard links to sub41's.
  - Both rows are in SUBMISSIONS.md.
- **run.py:** after P41's gib line, same loop:
  `acc |= R40.add(s1c, sxc, is2, d["p"], am, acc, sib, cls, 0.90)` with
  `cls[pool] = R40.xf_noaddr(n1[pool], nx[pool], ax[pool], fr_full_count)`.
  - `pool = am & ~acc & (p >= 0.9)`.
  - `fr_full_count` = `{rule40.fkey(name): #France S1}` over the test France S1 names.
- **`output/sub40` / `sub40_t70` are superseded** once sub41 is uploaded: their gib half is mostly inside sub41, and their
  address-less half is the failing `xu` class.
