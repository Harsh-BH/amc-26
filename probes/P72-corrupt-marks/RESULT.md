# P72-corrupt-marks — RESULT (final, 27 Sep 16:05)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **France: keep.** Three raw-text number-change marks (`num_trunc`, `num_step`, `num_other`) as extra columns of P55's combiner
  pass both LOCO directions: India→US **+0.001281 ± 0.000129**, US→India **+0.001458 ± 0.000111**, both halves > 0.
  At production's P55 stage (veto only, no adds), their real lever is a precise add tier. At c > 0.99 they give 957 / 1,429 LOCO
  adds at 0.9875 / 0.9874 precision. The plain combiner gives 495 / 1,042 at that bar.
- **Staged `output/sub72`** = sub66_final + **1,060 France adds** on 1,034 S1 (0 removed, 0 legal-form conflicts). Pooled Qwen r ≥ .5 on
  93.6% of them, against 80% for sub66_final's own accepted band pairs. Plain validator PASS and SageMaker `--check-ids` PASS (both files). **Expected LB +0.00005** (0.987391 → ≈ 0.98744). The aggressive option is `sub72_u98` (+2,251 adds, ≈ +0.00010), but its
  US→India LOCO precision is 0.96.
- **But it overlaps almost fully with P67 / P70:** 932 of the 1,060 adds are in P70's c > .98 set and 1,044 are in P67's, with 0 conflicts.
  **Only 3 pairs are outside P67 ∪ P70** (sub72_u98: 180 of 2,251). Inside the master's combined final (sub66_final + P67 + P70), sub72
  is worth ≈ 0. Its value is independent corroboration: a third signal (number marks) picks the same high-c pairs.
- **India/US: drop everything.** The 24 marks in sub54's combiner are null (−0.000017 ± 0.000024 / +0.000032 ± 0.000027). The 3 num
  marks alone are null too. Unique-key recovery (B) is +0.000015 ± 0.000020 / +0.000039 ± 0.000019, below the +0.00015 gate on every grid point.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- **Staged file: `output/sub72`** (only an upload candidate if the final does NOT carry P67 / P70; md5 a61203d9…; candidates = sub66_final's, md5 a8b425b0…, hardlinked; code.zip hardlinked).
  India/US lines are byte-identical to sub66_final (checked: only France lines differ, 1,034 of them). The France change is adds only.
  `--check-ids` PASS on SageMaker (p72-val-0927-1554, validator_exit=0 for both files).
  **If the final already carries P67 + P70, sub72 adds nothing (3 pairs); do not spend a slot.** Use it as corroboration of their adds.
  If a union is wanted, sub72_u98's 180 pairs outside P67 ∪ P70 are ≈ +0.000008 LB, which is not worth the risk.
- **Row for SUBMISSIONS.md** (lead to paste):
  `| 72 | staged, validator PASS plain + --check-ids (SageMaker p72-val-0927-1554) (P72) — ⊂ P67 ∪ P70 but 3 pairs | output/sub72/… | = #66_final (India/US byte-identical) | #66_final + **1,060 France adds (1,034 S1, 15 empties filled)**: P55's combiner refit with 3 raw-text number-change marks (SX number truncated / arithmetic step / other change vs S1's numbers), free argmax pairs with c > 0.99, caps 5/6, SX unused, built by p55.py build sub66_final sub72 mode=p72num,v=0,scope=band,u=0.99. LOCO combiner gate +0.00128 ± 0.00013 / +0.00146 ± 0.00011; LOCO c > .99 adds 0.9875 / 0.9874 precise (plain combiner: 495 / 1,042 adds, num: 957 / 1,429). 0 P64 conflicts; pooled Qwen r ≥ .5 on 93.6% (num-only adds 88%, LOCO num-only adds 75–84% at 98.3–98.6% precision). Expected LB ≈ +0.00005 |`
- **Optional `output/sub72_u98`** (md5 a6644bfb…): same build at u = 0.98, minus 17 P64 legal-form conflicts. It adds 2,251 pairs on 2,165 S1.
  LOCO add precision is 0.982 / 0.961; pooled Qwen ≥ .5 is 88%. Expected ≈ +0.00010. Take it only if the lead accepts sub-0.98 US→India precision.
- **Port (final package):** in `france.py`'s P55 stage, append the 3 NUM columns to `p55_feats`' X, from `marks.num_cls` on the raw
  (name + address) digit runs of S1 vs SX. Then use boosters refit with NUM (`fr72.py frscore` fits them; saving the 3 boosters is a 2-line
  change) and add free own argmax pairs with c > 0.99 after the P64 veto. The reference build is `p55.py build … mode=p72num,v=0,…,u=0.99`.
  Branch it on `c in UNSEEN` only.
- Do **not** swap P55's veto to the num combiner: +0.00037 / +0.00003 in LOCO, and on France it vetoes 1,319 pairs vs production's
  3,771, a shift too large for a +0.00005 lever.

## Evidence         (table: ledger name | world | key metrics | verdict)
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| p72-iu-refit-sw-mk24 | holdout sw (P57), half-split, paired | h0→h1 −0.000017 ± 0.000024, h1→h0 +0.000032 ± 0.000027. Gain share 5.5–6.2%. Band (0.2 < p'' < 0.97) residual y − p'' per mark ≤ 0.05 (≤ 3 SE) | drop |
| p72-iu-refit-sw-num3 | holdout sw | −0.000016 ± 0.000020 / −0.000000 ± 0.000015 | drop |
| p72-iu-unique-key-recovery | holdout sw, out-of-fold sub54-recipe p'' | brief's rule (0.30 / 0.10 / < 0.70): report h0 +0.000015 ± 0.000020, h1 +0.000039 ± 0.000019. Best grid point (0.5 / 0.3 / < 0.71): +0.000029 ± 0.000016 / +0.000046 ± 0.000016. 11–31 India / 39–50 US adds per half, 0.73–1.0 precise | drop (below +0.00015) |
| p72-fr-loco-mk | LOCO (P42/P55 frames, P65 ev65 protocol) | all 24 marks, ship cell: India→US +0.000159 ± 0.000136 (h0 −0.00004), US→India +0.000784 ± 0.000188 | drop (gate fail) |
| p72-fr-loco-mk_inv | LOCO | 15 binary / relative marks: +0.000302 ± 0.000136 (h0 +0.00012) / +0.002439 ± 0.000159 | weak pass, not used |
| p72-fr-loco-mk_num | LOCO | **3 num marks: +0.001281 ± 0.000129 (h0 +0.00124, h1 +0.00132) / +0.001458 ± 0.000111 (h0 +0.00150, h1 +0.00142)**. veto06: +0.000371 ± 0.000113 / +0.000030 ± 0.000062 | **keep** |
| p72-fr-loco-cells-num | LOCO, vs production P55 stage (base veto c < .6, no adds) | adds c > .99: base +0.000343 (495, prec 0.992) / +0.001021 (1,042, 0.985). **num +0.000641 (957, 0.9875) / +0.001379 (1,429, 0.9874)**. c > .98: num +0.001395 (2,187, 0.982) / +0.002746 (3,210, 0.961). Every cell has both halves > 0 | keep |
| p72-fr-qwen-calib-loco | LOCO + test France | LOCO num-only c > .99 adds: precision 0.9857 / 0.9828, out-of-domain Qwen ≥ .5 on 0.75 / 0.84. France sub72 num-only adds: Qwen ≥ .5 0.882, all adds 0.936, sub66_final accepted band pairs 0.804 | keep |
| (staging) sub72 | test | +1,060 / −0 pairs, 1,034 S1, 15 empties filled, 0 lf-class-4. 94% band (Qwen coverage 94%), Qwen r quantiles .1/.25/.5 = 0.78 / 0.985 / 0.998. 649 of the adds are also > .99 under the plain combiner. Plain validator PASS | keep |
| p72-stage-sub72 (overlap) | test | sub72 adds vs P70 changes.parquet (1,991 adds) 932 identical; vs output/sub67 (3,302 adds) 1,044 identical; 0 same-SX-other-S1 conflicts; outside P67 ∪ P70: 3. sub72_u98: 1,560 / 1,975 / 180 | corroboration only |
| (staging) sub72_u98 | test | +2,251 pairs (17 legal-form conflicts removed), 2,165 S1, 46 empties filled, Qwen ≥ .5 0.882 | optional |

Expected LB = LOCO gain per add (dF × n target S1 / n adds: 0.0885 India→US, 0.0851 US→India) × 1,060 / 259,452 France S1 × 0.1498
= **+0.000052 to +0.000054** (sub72); sub72_u98 +0.000098 to +0.000110.

Mechanism (France adds): 97% of the 411 num-only adds have no SX number changed vs S1 (step > 0 on 0.2%, vs 13.4% of France's
frame). The marks demote the generator's distractor signature, a number moved by a small arithmetic step (12590 → 12594). That lets
number-clean pairs clear the 0.99 bar. True copies truncate (4712 → 471) instead.

## What failed      (approaches tried and dropped, with the number that killed them)
- **India/US combiner + 24 marks:** ±0.00003, 0 SE both ways. sub54's combiner is already calibrated against every mark within the
  uncertain band (largest residual 0.05 on 318 bracket pairs). The upstream M2 / CE / zoo have priced in case, accents, DBA and domain.
- **India/US unique-key recovery:** the rescued band is tiny (40–80 adds per half on 220k S1). It is positive on all 8 half × grid cells but
  ≤ +0.00005, a third of the gate.
- **France, all 24 marks:** they fail India→US (+0.00016, h0 < 0). Absolute levels (name token count, case of each side, digit counts,
  vocabulary-token counts) differ by country (India names are longer and carry more numbers). Case, accents, DBA and domain get ~0
  France importance.
- **Swapping P55's veto to the num combiner:** +0.00037 / +0.00003. Not worth the shift on France (1,319 vetoes vs 3,771).

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `marks.py`: `marks(n1, a1, nx, ax, vocab) -> DataFrame[MK]` on raw strings. `num_cls(nums1, numsx) -> (same, trunc, step, other,
  drop)`: SX digit runs vs S1's (name + address). Truncation = a length-changing prefix; step = same length, 0 < |Δ| ≤ 20.
  Self-check: `.venv/bin/python probes/P72-corrupt-marks/marks.py`. About 16 s per 250k pairs, pure Python.
- `attach72.py <world> <c>`: MK for P57's A frames from the raw sources (streamed row groups, id-filtered) → `data/work/p72/mk_*`
  (45 s, 1.13 GB).
- `ev72.py [world]` (`P72_DROP`, `P72_TAG`): ev64's gate with MK. 220–240 s, 1.33 GB.
- `b_unique.py`: `unique(pool, queries)` and `keys(...)`. `ev` is the holdout gate (42 s, 1.07 GB). Self-check with no args.
- `fr72.py marks | loco | cells | calib | frscore`:
  - `marks`: LOCO marks from prep65 frames.
  - `loco`: ev65 protocol (60–73 s, 1.0 GB).
  - `cells`: veto / add decomposition.
  - `calib`: Qwen calibration.
  - `frscore`: pooled bag ± NUM → `data/work/p55/cf_p72{base,num}.npy`. p72base reproduces cf_plain exactly (max |d| 0). 156 s, 1.35 GB.
- `audit72.py BASE OUT [--drop-conflicts]`: the label-free add audit → `data/work/p72/audit_<out>.json`.
- `val72.sh SUB...`: the SageMaker `--check-ids` job (ml.m5.xlarge).

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **No new model, library or data.** Python `re` / `unicodedata`, plus rapidfuzz (already used). The marks read only the provided
  records. The India/US vocabulary counts come from the train S1 pool (on test: the test S1 pool, which the organiser allows).
- **France transfer:** France numbers are mostly house numbers (truncations 0.5% vs 4–7% in LOCO), and "no number changed" includes SX
  records with no number at all. Mitigation: both LOCO directions agree, and pooled Qwen rates the adds cleaner than the base's
  accepted pairs. The residual risk is a France precision below LOCO's 0.98. Break-even for an add is ≈ 0.67–0.75 (P68).
- Adds skip every SX already listed in sub66_final and respect caps 5 / 6. P54's dup pass is not re-run, so content-identical
  duplicates of an added SX are not added (≤ a few pairs).
- The India/US B gate used the full train S1 pool of each country as "the country": 883k India / 1.32M US, vs about 815k / 660k on
  test. That is conservative for US.

## Open questions   (for the master; each with your best guess)
- sub72 or sub72_u98? Best guess: **sub72**, since the lesson is that only ≥ 0.98-precision France adds transferred, and u98's US→India
  LOCO precision is 0.961.
- Worth an upload by itself? No. On its own it is +0.00005, and inside sub66_final + P67 + P70 it is 3 pairs.
