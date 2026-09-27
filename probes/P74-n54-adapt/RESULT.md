# P74-n54-adapt — RESULT (final, 27 Sep ~16:30 IST)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **N54's decoy-word removal is ported to main (`france.decoy_mask` + run.py `DECOY` hook, recipe `sub66_final_n54`).** On sub66_final it removes **1,666 France pairs on 1,658 S1 (144 S1 emptied)**. The removed pairs are the generator's org-word swap template ("X Sportive" → "X Comite", "X College" → "X Ecole", "X Amicale" → "X Amis"). 97% keep the house number. None come from P62 acro, P54 dup or P66 xfc.
- **LOCO on P69's chain analogue is positive in all 8 readings** (2 directions × 2 analogues × 2 halves). The removed pairs are 0.3–3.6% true.
  - US→India chain: **+0.00567 / +0.00570 ± 0.00028** (h0 / h1).
  - India→US chain: **+0.000027 / +0.000084**. That misses the brief's +0.0001 bar on size alone: only 28 removals, 3.6% true.
  - India→US M2: **+0.00013 / +0.00019**, which passes.
  - Expected LB **+0.00017 to +0.00029** (at 70% / 97% false) → sub66_final_n54 ≈ **0.98756–0.98768**.
- **Recommendation: upload N54 on whatever France base goes up.** Both `output/sub66_final_n54` (on #66_final) and `output/sub75a_n54` (on #75a: −2,120 pairs) are validator PASS `--check-ids`.
  - P71's noise-word adds conflict with N54 on 0 pairs.
  - sub75a's adds carry **454 decoy-word pairs** that N54 drops: P67q 200, t975ce 231, P70a 51.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- **src (committed to main):**
  - `france.decoy_mask(s1, sx, am, acc, s_min=.25, n_min=20) -> bool mask`, cherry-picked unchanged from origin/night/aditya (no other branch file).
  - run.py `DECOY` global and recipe `RECIPES["sub66_final_n54"] = dict(RECIPES["sub66_final"], decoy=0.25)`.
  - `run.decoy_drop(d, ka, s_min)` applies it to the **final lines km**. The lens comes from the argmax pairs `ka = K[am]`, taken before acro and dup extend K. Dropped pairs also leave `d["tail"]`; candidates are untouched.
  - The hook sits **after P64 lf_veto and P66 xfc_fit, before dup_last** (so after every France add rule, acro included). dup_last then adds 0 pairs and re-adds none of the removed ones.
  - Default `RECIPE` is still `sub66_final`. To ship N54 through run.py, set `RECIPE = "sub66_final_n54"` and rebuild the zip with package.py.
- **Upload order (my view):** `sub75a_n54` if #75a is the chosen base, else `sub66_final_n54`. Each is a France-only removal on its base, so one upload measures N54 cleanly against that base's LB.

**SUBMISSIONS rows** (added to SUBMISSIONS.md):

| # | status | files | India/US | what | expected LB |
|---|---|---|---|---|---|
| 66_final_n54 | staged, validator PASS --check-ids (p74-val-n54-161413, md5 80fc7f46…) | `output/sub66_final_n54/` (candidates = #66_final's, hardlink; code.zip: package.py from committed src with RECIPE sub66_final_n54) | = #66_final (byte-identical) | #66_final − 1,666 France decoy-word pairs (1,658 S1, 144 emptied): `run.py --recipe sub66_final_n54` France lines (France-only run, spliced) | 0.98756–0.98768 (+0.00017 to +0.00029) |
| 75a_n54 | staged, validator PASS --check-ids (p74-val-75a-n54-161938, md5 275a075a…) | `output/sub75a_n54/` (candidates + code.zip = #75a's, hardlinks) | = #75a | #75a − 2,120 France decoy-word pairs: sub66_final_n54's 1,666 + 454 of #75a's adds (P67q 200 / t975ce 231 / P70a 51). build74.py = run.py's hook, verified byte-identical on #66_final | #75a + ≈ 0.0002–0.0003 |

## Evidence         (table: ledger name | world | key metrics | verdict)
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| p74-sub66_final_n54-build | test France | 818,630 → 816,964 France pairs. Removed 1,666 (sub1 1,450 / add1 216; hn equal 1,619). Stage: chain (decision + post-rules) 981, gc70 588, rescue90 97, acro 0, dup 0, **P66 xfc 0**. dup_last 0. Port check vs p54.py's cell rule (P46 lens_table / swap_words, test_port.py's comparison on these lines): **1,666 / 1,666 both, 0 only-src, 0 only-p54**. Validator PASS | keep |
| p74-loco-n54-on-p69-chain | LOCO, P42 frames with ids restored | US→India chain: 1,394 removed, 0.8% true, **+0.00567 / +0.00570 ± 0.00028**; M2: 3,136, 0.3% true, +0.0109 / +0.0112. India→US chain: 28, 3.6% true, **+0.000027 / +0.000084 ± 0.00001 / 0.00003**; M2: 79, 1.3% true, **+0.00013 / +0.00019**. Flagged cell (all argmax rows): 0.05% / 0.06% true | keep (gate: ≥ 0 everywhere; India→US +0.0001 met on M2, missed on chain by size) |
| p74-overlap-p70-p71-sub75a | test France | P71 adds_a / b / bi and sub71a: **0** conflicts. P70a adds: 51 / 1,991. sub75a adds: **454 / 8,121** (P67q 200 / 2,793; t975ce 231 / 5,270). No file removes any of N54's 1,666 pairs, so N54 is disjoint from their vetoes | keep |
| p74-sub75a_n54-build | test France | −2,120 pairs, 2,110 S1, 164 emptied. Validator PASS | keep |

**Qwen r of the removed pairs (the brief's "low r" test): it is not low.**
- Qwen: coverage 65%, median .94, ≥ .9 54%, < .5 30%.
- bge: median .99, ≥ .9 64%.
- p: median .989.

I read this as the CE's known blind spot rather than as a sign the pairs are true. Two checks support that reading:
1. **In LOCO the source-only bge CE scores the same cell low:** median r of removed pairs .06 (US→India chain), .005 (India→US M2). Those removals are 97–99% false. There the CE was trained on English org words. In France it scores French org-word swaps (Comite vs Sportive) at .99. The swap sits in low-IDF French words it never saw as decoys, which is what N54 identified.
2. **The label-free decoy signature is flat across Qwen bands.** The dropped S1 word is itself a decoy word in 93% of sub1 removals. That share is .91–.94 in every Qwen band (≤ .3 … > .99), so Qwen does not separate them.

**Pessimistic bound (mix74.py).** Treat each word's France lens (s, e) as a mix of pure decoys (top-s words: s .42, e .47) and true noise (s .02, e .96). The removed pairs are then **69% (from s) / 71% (from e) decoy**. Break-even is ~23–29% false, so even this read is positive: +0.00017 LB. The LOCO precision (97–99% false) gives +0.00029.

## What failed      (approaches tried and dropped, with the number that killed them)
- The branch's hook position (inside the acc stage, before acro / P64 / xfc) was not kept. On main the France lines change after acc, so the hook moved onto km after P66. Either position removes the same pairs here, since acro and xfc contribute 0 hits.
- test_port.py as written needs `data/work/p31/pairs_France.parquet` and `p54/cells.parquet`, which are not on this box. The same comparison (src mask vs p54.py's rule) runs inside ana74.py on sub66_final's own lines instead.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `p74.py OUT`: France-only `sub66_final_n54` run (p69.py's pattern, wraps complete / p55 / dup / lf / xfc / decoy_drop). Writes France lines plus `OUT/p74_frame.npz` (stage keys, lens pairs, decoy in/out, p, Qwen r, bge r). 15 min, 1.82 GB. `data/work/p74/n54` (= `probes/P74-n54-adapt/work/n54`).
- `splice.py` (P69) pastes those lines into sub66_final → `output/sub66_final_n54`.
- `build74.py BASE DST`: N54 on any staged file's France lines, with the lens from p74's frame. Byte-identical to the run.py build on #66_final. 2 min, ~1 GB.
- `ana74.py OUT` → `ana74.json` + `OUT/removed.parquet`: removal stats, stage attribution, port check. 3 min.
- `loco74.py` → `loco74.json`: ids restored from P6's S_holdout (prep.py's row order, asserted); `decoy_mask` itself runs with rf_train standing in for rf_test. 20 s.
- `ov74.py OUT [output/<sub> …]` → `ov74.json` (P70 / P71 / sub75a / sub71a / sub70a) and `ov74_attrib.json` (sub67q / t975ce).
- `mix74.py` → `mix74.json`. The expected-LB table is `elb74.json`.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Main risk: France might use org-word swaps as a TRUE noise op.** The CE says .94 median, and the lens mix says up to ~30% true. The break-even margin covers that, but the gain would shrink to +0.00017. An LB below #base − 0.0001 would mean France org swaps are largely true. In that case also drop N54's cousins (P70's decoy-side vetoes).
- 144 S1 are emptied (164 on #75a). Those need ≥ 50% false to pay. The mix bound says 69–71%.
- Keys on the unseen-country set (`unseen()`), never on a country name. No new model, library or data. Candidates are unchanged, so the validator is PASS on both files.
- `sub75a_n54` is not reproducible by run.py alone: #75a is combine.py's union. Its N54 step is build74.py, which equals the hook on #66_final.

## Open questions   (for the master; each with your best guess)
- Should N54 go in the final zip's default RECIPE? My guess: yes, if the N54 upload holds or beats its base. Set `RECIPE = "sub66_final_n54"`, or add `decoy=0.25` to the recipe that reproduces the final base.
- Should the 454 decoy-word adds come out of #75a's add sets (P67q / t975ce / P70a) at source? My guess: yes. They are the same template N54 removes from the base, and sub75a_n54 already does this.
