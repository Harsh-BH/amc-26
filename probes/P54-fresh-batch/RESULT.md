# P54-fresh-batch — RESULT (final, 27 Sep ~05:50 IST)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **Idea 1 (France in-band CE rescue) passes by a wide margin.** It adds free, argmax, non-sibling band pairs (0.02 < g < 0.98, not accepted) whose bge r ≥ 0.99, best r first, up to the caps. LOCO on top of P48's veto, France-F0.5:
  - chain: US→India **+0.00225 (h1 +0.00228 ± 0.00012)**, India→US **+0.00243 (h1 +0.00242 ± 0.00015)**; added pairs 94% / 89% true;
  - M2-only analogue: +0.0123 / +0.0063.
  - France's add rate per recipe pair (0.92%) sits between LOCO's two directions (0.93% / 1.23%).
- **Idea 3 (duplicate consistency) passes.** Train truth: 100% of 60,319 case-folded content-identical groups share one owner. On test it adds 46 pairs (France 32, India 8, US 6). **Idea 2 (India/US empty-S1 fill) and 3b (fully unassigned dup groups) fail:** out of fold, every arm is ≤ +0.00001 or negative.
- **Upload `output/sub54`** = sub52_max + 7,631 France rescue pairs + 46 dup pairs. Nothing is removed. Validator `--check-ids` PASS. **Expected LB ≈ +0.00035 over sub52_max** (0.1498 × India→US +0.00243).
- **It stacks with P55's France combiner:** on P55's LOCO decision it still adds +0.0011 / +0.0010 at u .995. If P55 is adopted, apply P55 first, then this rescue at **u .995**, then dup.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
**Upload:** `output/sub54` (matching md5 ed4cc1a4…, candidates md5 2d03c892…). Validator `--check-ids` PASS (SageMaker p54-val-0927-0526; the job's md5s match). Row #54 is in SUBMISSIONS.md.

**If P55's France combiner is adopted too** (for P50's sub_all), the order is:
1. `p55.py build BASE X v=0.6,scope=band,u=0.95`;
2. `fr54.py X 0.995`;
3. `build54.py X OUT data/work/p54/fr_rescue_X_u0.995.parquet`.

fr54 never re-adds recipe pairs (so P55's vetoes stick), skips SX P55 already placed, and counts caps on X's lines. LOCO for the P54 step on top of P55: u .995 **+0.00105 (h1 ± 0.00010) / +0.00097 (h1 +0.00115 ± 0.00012)**; u .99 +0.00114 / +0.00090 (precision drops to .80 India→US). On top of P55, u .995 is the better bar.

**Rebuild on any base** (in this order; the dup step must stay last):
1. `probes/P54-fresh-batch/py.sh probes/P54-fresh-batch/fr54.py BASE 0.99`. Writes `data/work/p54/fr_rescue_BASE_u0.99.parquet` (s1, sx, g, r) in 10 s.
2. `probes/P54-fresh-batch/py.sh probes/P54-fresh-batch/build54.py BASE OUT data/work/p54/fr_rescue_BASE_u0.99.parquet`. It appends the rescue pairs, then the dup copies; line-level, ~2 min, 1.2 GB.
3. `probes/P54-fresh-batch/py.sh probes/P54-fresh-batch/check54.py BASE OUT`. Asserts: pairs are only added, every SX sits at ≤ 1 S1, and matches ⊆ candidates.

**Rule 1: France rescue (unseen-country recipe).** Runs after the P48 veto, on `france.npz` with the P42 recipe t .99 / r .9 / gc .7 / hn .8, τ .1, caps 5/6. Add a pair when all of these hold:
- it is free (argmax, non-sibling, not capped out; P42 `base`) and not accepted by the recipe;
- 0.02 < g < 0.98, and p0 equals the OOD blend (no tier / bxneg rule moved it);
- CE r (ce_bge_pool, already cached for every band pair; 0 NaN) ≥ 0.99;
- its SX is in no line of the base file.

Pairs go in best r first, up to 5 S2 / 6 S3, counted on the base file's own line, so tier pairs count toward the cap.

**src port** (≈ 8 lines in `france.complete`, after the P48 veto): `cand = free & ~acc & band & (abs(p - blend) <= 1e-4) & (r >= 0.99)`, then `add(stage="any", …)` with the cap fill ordered by r. This needs no new GPU work.

**Rule 3: dup consistency.** Key = lower/strip(name) | lower/strip(address) | country, S2 and S3 pooled. If a group's accepted copies all sit at one S1, append every unassigned copy to that S1, past the cap. A copy missing from that S1's candidate list is appended there too (7 lines on sub54). Code: `build54.dup_adds(rows)` plus `build54.apply`.

## Evidence         (table: ledger name | world | key metrics | verdict)
LOCO: P42/P48 frames (P6 source-only M2 g, P30 source-only bge r, OOD blend). dF = mean per-S1 France-F0.5 change over all target fold-0 S1; h1 = report half ± SE. India/US holdout: sub44r's p'' out of fold on both halves (P49 `load`); LB-eq = 0.468 India + 0.383 US; the lever is picked on one half and reported on the other.

| ledger name | world | key metrics | verdict |
|---|---|---|---|
| loco-inband-rescue-grid | LOCO, recipe alone | chain band / any, u picked on h0: US→India u .95 **+0.00243 (h1 ± 0.00016)**, India→US u .99 **+0.00240 (h1 ± 0.00015)**. By scope at .99: nonempty +0.0019 / +0.0018, onesided +0.00035 / +0.00034, empty +0.00034 / +0.00064. M2: +0.0120 / +0.0065. The region g ≥ 0.98 & p < t has 0 pairs with r ≥ 0.8 (nothing to rescue there) | keep |
| loco-inband-rescue-on-p48-veto | LOCO, recipe − P48 veto (= sub48's France) | u .99: **+0.00225 (h1 +0.00228 ± 0.00012) / +0.00243 (h1 +0.00242 ± 0.00015)**, 2,657 / 5,377 adds, precision .944 / .887, rate 0.93% / 1.23% per accepted pair. u .995: +0.00210 / +0.00244 (prec .954 / .922). M2 u .99: +0.0123 / +0.0063 | **keep → sub54 (u .99)** |
| (hn54.py) | LOCO | adds by hn class (chain, u .99): hn-equal share 1.6% / 28.5%. The not-hn-equal adds are **.947 / .864 precise** and carry most of the dF (+0.00225 / +0.00130) | France's 0.37–0.41 hn-equal share is not a red flag |
| fr-rescue-test-counts | test France, label-free | on sub52_max, u .99: **7,631 adds, 7,271 S1, 0.92% per recipe pair** (LOCO .93% / 1.23%). By scope, per recipe pair: empty 476 (0.057%; LOCO 0.017% / 0.035%), onesided 469 (0.056%; LOCO 0.09%), nonempty two-sided 6,686 (0.80%; LOCO 0.91% / 1.19%). Median g .24, median p .66. On sub48 there are 8,509 adds (1.02%); the difference is P47's cls adds already taking those SX | keep |
| dup-consistency-scan | train truth + test sub48 | train: **60,319 case-folded groups (43,910 raw), 100% one owner, 0 all-unmatched**. Test sub48: 50 mixed groups (France 35 / India 10 / US 5; raw key 25), 0 split across S1, 51 adds (43 already candidates) | keep |
| dup-consistency-build-sub54dup | test | on sub48: 51 pairs, 50 lines; 8 candidate adds on 7 lines | keep |
| fill-empty-s1-best-cand-b | holdout | empty S1 under sub44r: India 4,972 (**4,810 true singletons**), US 7,531 (7,259). Best-candidate calibration at p'' .3–.5: precision .50 / .28, singleton rate .43 / .69. Out of fold: b = .6 → **−0.000033 ± 0.000036**; b = .45 → **−0.000029 ± 0.000044** | **drop** |
| fill-empty-s1-s1-model-gate | holdout | LGBM regression of the fill gain (best / 2nd p'', #argmax, #cand, name freq, noaddr, hn class, …): θ .2 → −0.000012 ± 0.000008; θ .05 → −0.000096 ± 0.000045 | **drop** |
| dupgrp-no-accepted-copy-assign | holdout | only 13 India + 78 US holdout groups have no accepted copy; their best row is 46% / 26% true. b = .6: **+0.000006 ± 0.000006** on both halves | **drop** (under the gate) |
| build-sub54 | test | on sub52_max: France +7,663 (7,631 rescue + 32 dup), India +8, US +6 (dup only). **0 removed**, 7,315 S1 lines, 0 SX at > 1 S1, 0 matches outside candidates, 7 candidate adds. Lines over 5/6: 37 → 39 (dup copies) | keep |
| loco-rescue-on-p55-decision | LOCO chain, P55's saved target scores (band veto c < .6 + adds c > .95) | P55 alone vs ref: +0.00464 / +0.00527. **P54 on top of P55:** u .99 +0.00114 (h1 ± 0.00010) / +0.00090 (h1 +0.00096 ± 0.00014), 1,258 / 2,941 adds, precision .91 / .80; **u .995 +0.00105 / +0.00097 (h1 +0.00115 ± 0.00012)**, precision .93 / .85. Overlap at u .99: 1,399 of 2,657 / 2,436 of 5,377 P54 adds are also P55 adds | keep (u .995 when stacked on P55) |
| validator-sub54 | test | **PASS --check-ids** (SageMaker p54-val-0927-0526, spot m5.xlarge). md5 ed4cc1a4… / 2d03c892… match the local files. 1,732,544 rows; 100,314 empty matching lines | keep |

**Flip profile (India/US):** India/US change only by 14 dup copies. Each is a content-identical copy of a pair the base already accepts, and every one of the 60,319 train groups shares one owner. The India/US profile is therefore sub52_max's, unchanged.

## What failed      (approaches tried and dropped, with the number that killed them)
- **Idea 2, empty-S1 fill (brief's b grid 0.35–0.7, plus 0.3).** The break-even argument misses who the empty S1 are: **97% of them are true singletons.** A wrong fill there costs a full 1.0. Among the best candidates, P(correct) ≈ 1 − P(singleton), so the break-even is p ≈ 1/(1 + F(1 of k)) ≈ 0.58. p'' is well calibrated in this slice (it is 0.5 precise at .3–.5), so 0.71 is already near-optimal. The S1-level gain model finds nothing either. The oracle is small: 162 India / 272 US non-singleton empty S1, and only 86 / 106 of them have any free argmax candidate.
- **3b, fully unassigned dup groups.** In train, a group with a copy is never unowned. But on the holdout, the uncertainty is *which* S1 owns it, not *whether* anyone does: the best row is 26–46% true. Only 91 holdout groups; ≈ 0.
- **Rescue in the region g ≥ 0.98, p < t:** it has no pairs with r ≥ 0.8, so there is nothing to add.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
All scripts run from the repo root via `probes/P54-fresh-batch/py.sh` (1.9 GB cap). Work files go to `data/work/p54/`.
- `dup54.py scan SUB [norm]` / `train [norm]`: test counts and the train-truth check. `keys(split, norm)` is batched and uses int64 ids. 35 s / 2 min, 1.2 GB.
- `fill54.py`: ideas 2 and 3b on the holdout → `fill.json`. 1 min, 1 GB.
- `loco54.py` (grid → `loco54.json`); `loco54.py stack` (on the P48 veto → `loco54_stack.json`); `hn54.py` (hn split). About 1 min each, 0.9 GB. They reuse P48's `loco48` and P42's `solution_p42`.
- `on55.py`: the P54 rescue on top of P55's LOCO decision (from `data/work/p55/cb_plain_*.npy`) → `on55.json`. 1 min.
- `fr54.py BASE U…`: France adds lists plus label-free counts → `fr_rescue_BASE[_uU].{parquet,json}`. 10 s, 0.6 GB.
- `build54.py BASE OUT [adds.parquet…] [--nodup]`: line-level builder, dup step last, writes `OUT/p54_info.json`. ~2 min, 1.2 GB.
- `check54.py BASE OUT`: streamed asserts. 1 min, 0.7 GB.
- AWS: the validator only (spot m5.xlarge, ≈ $0.01).

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **No new model or data.** bge-reranker-v2-m3 (Apache-2.0) is already shipped, and its France r is cached. The dup key is exact text; nothing filters on country (the rescue keys on the unseen-country recipe's frame).
- **CE mismatch:** LOCO uses P30's source-only CE; France uses the pooled CE (India + US). If the pooled CE is overconfident on France, u .99 lets in more false pairs than in LOCO. The matching add rate (0.92% vs 0.93–1.23%) says it is not grossly off. u .995 is the conservative alternative: 5,365 adds; LOCO +0.00210 / +0.00244, precision .95 / .92.
- **Empty-S1 adds** are 3× LOCO's rate in France (476 pairs) and are the least precise slice in LOCO (.81–.90). At worst this costs ≈ 476 × 0.2 false / 260k France S1 ≈ −0.0004 France-F, and it is already inside the LOCO number. Excluding them (onesided + nonempty only) is a one-line change if the master wants it.
- **Interaction with other France probes:** P55's combiner uses the same band CE signal. It is measured: about half of P54's adds are P55's too, and the rest still adds about +0.001 France-F on top (see Evidence). P52 (calibration) is not measured. If it ships a stronger CE weight, re-run `fr54.py` and the LOCO on top of it.
- The expected LB uses India→US chain, as P48 did. LOCO has overstated France's gain before (see P37 / P42), so treat +0.00035 as an upper-middle estimate.

## Open questions   (for the master; each with your best guess)
- **u .99 or .995?** Best guess: .99. It is India→US's h0 pick, and France's add density matches LOCO's there. If the LB gain disappoints, try .995 next.
- **Rescue empty S1 at all?** Yes. LOCO counts them and they are net positive in both directions (+0.00034 / +0.00064 alone).
- **3b for France (302 case-folded test groups with no accepted copy)?** No. The holdout shows no gain, and France has no labels to check it against.
