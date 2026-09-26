# P37-fr-complete — RESULT (final, 27 Sep ~01:00; follow-ups after #37_gc70's LB)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **Group completion is uploaded and is the new best.** Rule: a France S1 whose predicted group has only S2 or only S3 gets every free, argmax, non-sibling candidate from the missing source with p ≥ 0.70. `sub37_gc70` scored **LB 0.986304 (+0.00034 vs #34v_r90)**. LOCO had predicted +0.0008 / +0.0020 France-F0.5 (US→India / India→US); the LB delta equals +0.0023.
- **Next upload: `output/sub37_gc70_hn90`.** It adds extra members to *two-sided* France groups when the SX has p ≥ 0.90 and the same house number as the S1: +18,180 pairs. It passes LOCO in the France-like direction (India→US +0.0021, precision 0.93) and fails in US→India (−0.0014).
  - I call India→US France-like for three reasons. It is the direction that predicted both France LB gains so far. France's addresses behave like US ones. And train truth shows that same-address SX with invented names are true matches (7,763 true vs 159 false).
  - Expected LB ≈ +0.0002 to +0.0003; downside ≈ −0.0002.
- **Dropped:**
  - Lower gc bars (0.6 / 0.5): they fail US→India (−0.0001 / −0.0003) and are ≈ +0.0002 in India→US.
  - The Task 3 retrieval pass: the ceiling is ≤ 1.7k S1, and the pool it would retrieve has junk-like precision.
  - `gc70 + rescue70`: staged as asked, but it is a coin flip.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
**Queue for 27 Sep (base = #37_gc70, LB 0.986304):**

| rank | folder | rule on top of #37_gc70's France | France Δ vs gc70 | per S1 | LOCO vs gc70: US→India / India→US | expected LB |
|---|---|---|---|---|---|---|
| — | `output/sub37_gc70` (uploaded) | `oneall` 0.70 on #34v_r90 | — | 3.137 | (vs r90: +0.00082 / +0.00204) | **0.986304** |
| **1** | **`output/sub37_gc70_hn90`** | + two-sided extras: free argmax non-SIB3 SX, p ≥ 0.90, hn(SX) = hn(S1), caps 5/6 | +18,180 pairs, 17,222 S1 | 3.207 | −0.00137 (prec 0.51) / **+0.00206 (prec 0.93)** | +0.0002 to +0.0003 (risk −0.0002) |
| 2 | `output/sub37_gc70_r70` | gc70 rebuilt on #34v_r70, so the empty-S1 rescue runs at p ≥ 0.70 | +753 S1 (rescues) | 3.140 | −0.00127 (rescued prec 0.33) / +0.00002 (0.51) | ≈ ±0.0001 |

- Both staged folders: validator PASS; only France lines differ from `sub37_gc70` (17,222 and 753 lines).
- `candidate_pairs.tsv` and `code.zip` are hard links to #34v_r90's: every added pair is in the France pair file, so candidates are unchanged.
- **If #1 gains:** `P37_BASE=output/sub37_gc70 … build allhn 0.7` (LOCO India→US +0.0027 / US→India −0.0024) is the next, looser step.
- **Deleted:** `sub37_gc80` and the old `sub37_hn90` (on r90), both superseded. Rebuild either in 15 s if needed.

**run.py integration** (`run()` decision loop, right after `if CAP: acc = cap(d, acc)`, France only):
```python
if d["c"] == "France":                                  # #34v_r90 rescue, then P37 group completion (+ hn extras)
    s1c, sxc = d["K"] >> 32, d["K"] & 0xFFFFFFFF
    am = M.argmax_sx(sxc, d["p"])
    h1, hx = d["hn"]("s1"), d["hn"]("sx")
    sib = S.sib_mask(h1, hx)                             # same call drop_sib_cands uses
    is2 = pc.starts_with(d["idx"].take(pa.array(sxc)), "S2-").to_numpy(zero_copy_only=False)
    acc |= FR.rescue(s1c, d["p"], am, acc, sib, 0.90)
    acc |= FR.gc_add(s1c, sxc, is2, d["p"], am, acc, sib, 0.70, "oneall")                 # = #37_gc70
    acc |= FR.gc_add(s1c, sxc, is2, d["p"], am, acc, sib, 0.90, "allhn", (h1 != "") & (h1 == hx))   # = #37_gc70_hn90
```
- **Move into `france.py`:** `best_per`, `one_sided`, `gc_add`, `rescue` from `probes/P37-fr-complete/p37.py` (≈ 40 lines, numpy only).
- **`d["p"]`** is the post-rules p (tier / bxneg / veto / cap), i.e. the p in `fr31v0.1_t9868.parquet`.
- **Order matters:** rescue, then gc, then extras. The staged files were built in that order.

## Evidence         (table: ledger name | world | key metrics | verdict)
**LOCO protocol:**
- Data: P27's M2 trained on one country, scoring the other's fold-0 pairs (argmax over all train S1).
- Base: the target's t_profile (0.97 US→India, 0.81 India→US) plus the empty-S1 rescue.
- Threshold mapping: each France t maps to the target t with the same density of argmax pairs per S1 in [t, t_base). France densities: 0.138 at 0.90, 0.222 at 0.70, 0.253 at 0.60, 0.285 at 0.50.
- Scoring: dF = paired F0.5 change over all fold-0 target S1 (France-F0.5 units; LB ≈ 0.1497 × dF). h1 = report half 1 ± SE.

| ledger name | world | key metrics | verdict |
|---|---|---|---|
| p37-loco-gc-oneall-fr0.7 | LOCO | vs r90: US→India **+0.00082** (h1 +0.00047 ± 0.00020, prec 0.713); India→US **+0.00204** (h1 +0.00206 ± 0.00011, prec 0.850). France +10,084 pairs | **keep → LB 0.986304 (+0.00034)** |
| p37-loco-gc-oneall-fr0.8 / 0.9 / 0.95 | LOCO | +0.00079 / +0.00185; +0.00053 / +0.00154; +0.00031 / +0.00113 | 0.8 / 0.9 keep, 0.95 drop (superseded by gc70) |
| p37-loco-gc-one-fr* | LOCO | best candidate only: ≤ `oneall` at every France-mapped t | drop |
| p37-loco-gc-all-fr* | LOCO | any non-empty group, no guard: US→India −0.0038 to −0.0105 (prec 0.50–0.54); India→US +0.0020 to +0.0025 | drop |
| p37-loco-lb-calibration | LOCO vs LB | rescue90: LOCO −0.00167 / +0.00088 vs LB +0.00207. gc70: LOCO +0.00082 / +0.00204 vs LB +0.00227. **The LB tracks India→US or does better** | keep (used to rank) |
| p37-loco-gc60/50/40/30-vs-gc70 | LOCO | US→India −0.00012 / −0.00026 / −0.00034 / −0.00054 (new-pair prec 0.61); India→US +0.00013 / +0.00023 / +0.00024 / +0.00014 (prec 0.70–0.76). France +1,082 / +2,137 / +3,325 / +4,616 pairs | drop (fails both-ways gate; ≈ +0.00003 LB in the France-like direction) |
| p37-loco-gc70-r70-vs-gc70 | LOCO | US→India −0.00127 (h1 −0.00109 ± 0.00028; rescued prec 0.33); India→US +0.00002 (prec 0.51). France +753 rescued S1 (my rescue() reproduces #34v_r70's 1,958 − 1,205 = 753 to within 1) | inconclusive, staged as asked |
| p37-loco-extras-guards-on-gc70 | LOCO | two-sided extras at FR 0.9 by guard (US→India / India→US): hn_eq −0.00137 / **+0.00206**; name-token overlap −0.0064 / +0.0008; hn_eq & overlap −0.0014 / +0.0020; core-equal **+0.0019** / −0.0005. At FR 0.7: hn_eq −0.0024 / +0.0027. No guard passes both directions: hn and name behave oppositely in India | hn_eq staged (France-like direction) |
| p37-namerel-train-truth | train truth | hn-equal + same-city pairs whose names are unrelated (no token, containment, acronym or 5-letter common substring): US p < 0.05 54,732 pairs at prec 0.002; p 0.9–0.97 prec 0.975; p > 0.97 prec 0.996. Unrelated-name true pairs are 2.5% (US) and 3.6% (India) of all true pairs. Eyeballed: invented brand names ("…xylo", "…vio") at the same address are TRUE variants | the decoy theory is wrong |
| p37-fr-namerel-lens | France | unrelated-name share: accepted p ≥ 0.999 0.8%, [0.995, 0.999) 5.0%, [0.9868, 0.995) 7.2%; gc70 adds 19.7% (hn-equal adds 32%); hn90 extras 29.6%. Consistent with invented-name true variants being scored lower | inconclusive |
| p37-extra-by-size | LOCO | no group-size bin makes extras pass US→India (best prec 0.51) | drop |
| p37-task3-retrieval-ceiling | LOCO + France | see Task 3 below | drop |

**Shape check.** France moves toward India's shape, not past it:

| world | both | S3-only | S2-only | empty | per S1 |
|---|---|---|---|---|---|
| France #34v_r90 | 0.732 | 0.111 | 0.097 | 0.061 | 3.10 |
| France gc70 (LB best) | 0.768 | 0.092 | 0.080 | 0.061 | 3.14 |
| France gc70_hn90 | 0.768 | 0.092 | 0.080 | 0.061 | 3.21 |
| India predictions / train truth | 0.789 / 0.805 | 0.079 / 0.075 | 0.074 / 0.065 | — / 0.055 | 3.38 / 3.46 |
| LOCO US→India base / truth | 0.738 / 0.806 | 0.108 / 0.075 | 0.111 / 0.065 | 0.044 / 0.055 | 3.32 / 3.48 |
| LOCO India→US base / truth | 0.764 / 0.806 | 0.103 / 0.075 | 0.088 / 0.064 | 0.046 / 0.055 | 3.36 / 3.46 |

**Label-free lens on France pairs:**

| pairs | n | hn equal | hn one-side missing | name-token overlap | core equal | same city |
|---|---|---|---|---|---|---|
| accepted (#34v_r90) | 803,789 | 0.919 | 0.065 | 0.904 | 0.795 | 0.957 |
| gc70 adds | 10,084 | 0.586 | 0.320 | 0.722 | 0.431 | 0.736 |
| gc70_hn90 extras | 18,180 | 1.0 | 0 | 0.574 | 0.197 | 0.995 |

- **Why the extras have low name overlap:** they are acronyms ("EM" for an EHPAD, "ÉPP" for an École Primaire), handles and websites, plus invented brand names at the same address.
- **Train says both kinds are true variants.** Same-address pairs with invented names are 98% true at p > 0.6. The decoys of that kind sit at p < 0.05, and the model separates them.

**Task 3 (retrieval for one-sided S1 with no candidate in the missing source):**
- **LOCO, on the unpruned P3 list:** only 130 (US→India) and 198 (India→US) of 16–21k one-sided S1 lack a missing-source candidate, and 127 and 191 of them are truly one-sided. Perfect retrieval ceiling: **+0.000003 / +0.000005**.
- **France: the gap comes from the P10 pruner.** 16,212 one-sided France S1 have no missing-source pair in the pruned file. 16,154 have such pairs in the full P3 list: 93.6k pairs, 77% hn-different, 13.7k SIB3.
  - 1,963 pairs (1,693 S1) are free, non-sibling, hn-equal and same-city.
  - The pruner scored them q < 0.03. The train analogue is hn-equal same-city pairs at low p, which are 0.2% true (unrelated names) or 29% true (shared token).
  - Upper bound at perfect precision: ≈ +0.00005 LB.
  - The chain pass (c_hn features at 2.65 GB plus an AWS bge top-up) is not worth it. The hn + street + city key pass was not built.

## What failed      (approaches tried and dropped, with the number that killed them)
- **`all`** (every free argmax member into any non-empty group, no guard): US→India −0.0038 to −0.0105, precision ≈ 0.5.
- **Guards on extras** that were meant to pass both directions. None does:
  - hn-equal: US→India −0.0014;
  - name overlap: −0.0064 / +0.0008;
  - core-equal: India→US −0.0005;
  - group size: best precision 0.51.
  - The reason: in India, hn-equal extras are 0.50 precise and hn-different ones 0.93, the reverse of the US.
- **Lower gc bars 0.6 / 0.5 / 0.4:** fail US→India (−0.00012 / −0.00026 / −0.00034 vs gc70), and India→US gives only +0.0001 to +0.0002. Not staged, per the "LOCO passes both ways" condition.
- **"Unrelated name = decoy"** (my first read of the lens): refuted by train truth, where invented-name same-address pairs at p > 0.6 are 98% true.
- **Task 3 retrieval:** ceiling ≈ 0 on LOCO; France pool ≤ 1.7k S1 at junk-like prior precision.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
All code is in `probes/P37-fr-complete/p37.py` (numpy only). It reuses P15 `sib_rule.sib_mask`, `model.argmax_sx` and `features.ids`.

**Functions:**
- `gc_add(s1, sx, is2, p, am, pred, sib, t_gc, mode, guard=None) -> bool mask`
  - Returns the pairs to add; it never removes any.
  - `mode`:
    - `one`: best missing-source candidate;
    - `oneall`: every missing-source candidate;
    - `all`: any non-empty group;
    - `allhn`: `oneall`, plus two-sided extras where `guard` holds.
  - Caps: 5 S2 / 6 S3 per S1, counting existing members.
- `rescue(s1, p, am, pred, sib, t_r)`: #34v_r90's empty-S1 rescue, reconstructed. It reproduces r70's +753 over r90.
- `fr_load()`: the France pair file plus the France rows of `$P37_BASE` (default `output/sub34v_rescue90`). It asserts that every predicted pair is in the file and is argmax.

**Commands** (toy asserts run first):
- `fr` (8 s, 1.1 GB);
- `loco` (~2 min, 1.6 GB);
- `hnscan`;
- `loco2` (lower bars and r70);
- `build MODE T OUT` (15 s, 1.2 GB; base from `$P37_BASE`).

**Helper scripts** (run from the repo root):
- `ceiling.py` and `fr_nocand.py`: task 3;
- `calib.py`: LOCO vs LB;
- `ovscan.py`: extras guards;
- `namerel.py`: `rel()` name relation plus the train-truth scan;
- `fr_namerel.py`;
- `lens.py`;
- `size_scan.py`.

**Rebuild the queue:**
```
.venv/bin/python probes/P37-fr-complete/p37.py build oneall 0.7 output/sub37_gc70
P37_BASE=output/sub37_gc70 .venv/bin/python probes/P37-fr-complete/p37.py build allhn 0.9 output/sub37_gc70_hn90
P37_BASE=output/sub34v_rescue70 .venv/bin/python probes/P37-fr-complete/p37.py build oneall 0.7 output/sub37_gc70_r70
cd data/student_resource && python3 utils/validate_submission.py --matching ../../output/<dir>/matching_results.tsv --candidate ../../output/<dir>/candidate_pairs.tsv --test-dir dataset/test
```

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **gc70_hn90 fails the usual both-directions gate.** It is ranked #1 only on the calibration argument:
  - two LB points, both at or above India→US;
  - France's hn is informative like the US's (92% of accepted pairs hn-equal; US 78%, India 50%);
  - the train-truth check on invented names.
  - If France behaves like India, expect about −0.0002 LB. The LB is the arbiter, and #37_gc70 stays the fallback best.
- **The LOCO base is M2 only** (no hn features, no bge, no France rules), and the threshold map by argmax density is an assumption. LOCO has under-predicted France twice, so read its absolute numbers as conservative.
- **`rescue` is reconstructed**, and it reproduces #34v_r70's count to within 1. The France bases are the actual staged files.
- **Sibling hn:** S1 from `data/work/rf_test_s1_France.parquet`, SX from `p29/w_fix` (the fixed '(n)' parse, as in P31).
- **Compliance:** no country filter; `build` rewrites only changed lines, so every S1 row stays. No new model or data; label-free rules on test-side data only.
- **Disk:** three P37 folders × 97 MB (the matching file only; the rest are hard links). About 1.2 GB free.

## Open questions   (for the master; each with your best guess)
- **What if gc70_hn90 loses on the LB?** Then France's extras behave like India's. Best guess: stop adding extras and keep gc70. Do not try the core-equal guard instead: it passes US→India but fails India→US.
- **What if gc70_hn90 gains?** Try `allhn 0.7` on the same base (+~9k more France pairs; India→US +0.0027 vs gc70). Then re-derive the rescue threshold on that base.
- **gc70_r70?** Best guess ≈ 0. Upload it only if a slot is spare. LOCO calls it negative or 0, but LOCO under-called rescue90 by +0.0012 to +0.0037.
- **A later France model (new chain or t) should re-derive t_gc.** Keep the density rule: argmax pairs per S1 in [t_gc, t) ≈ 0.22 for gc, and 0.138 for the extras.
