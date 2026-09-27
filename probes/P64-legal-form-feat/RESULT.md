# P64-legal-form-feat — RESULT (final, 13:20)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **India/US: drop.** The combiner already prices legal-form agreement. A sub54 refit with 5 lf columns on P57's honest sw
  frame scores −0.000018 ± 0.000025 (h0→h1) and +0.000012 ± 0.000023 (h1→h0) LB-eq. That is about 40 seed-noise flips per half.
- **France: keep a legal-form conflict veto.** It removes a France pair when S1 and SX both carry a legal form and the forms are
  disjoint (SARL vs SAS, SAS vs SA, SCI vs SNC).
  - LOCO, gc70, chain analogue: **US→India +0.00300 ± 0.00013, India→US +0.00074 ± 0.00007**. Both halves are positive in both
    directions, and the m2 analogue agrees.
  - Qwen independently rejects these conflicts.
- **Port it as the last France stage of the final.** On the default final `sub_iu54_frgc70` it removes 571 pairs, expected
  **+0.00005 to +0.00009 LB**. Staged as `output/sub64_frgc70`, plus `output/sub64` on sub_iu54_best (−344 pairs, +0.00003–0.00006).
  Too small to spend an upload on by itself.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- **Final package (P43's job, P43 has no live session):** add a France post-stage after every other France stage:
  - `lf.sig(names) -> Series[str]`: the legal-form signature. It uses normalize's own `_is_legal` (LEGAL + typos of the long
    forms), canonicalised: private→pvt, limited→ltd, incorporated→inc, corporation→corp, company→co, établissement(s)→ets.
  - A name made only of legal tokens gets `''`, because it is an acronym ("PC" = the initials of a two-word core).
  - `lf.pair_class(sig_s1, sig_sx) == 4` means drop the pair.
  - Code: `probes/P64-legal-form-feat/lf.py` (45 lines, no new dependency; rapidfuzz is already used by normalize).
  - Reference build: `frtest.py build <BASE> <OUT>`. The run.py flag should reproduce `output/sub64_frgc70` byte for byte (md5 b5227e3d…).
- It applies to `c in UNSEEN` only (the organiser allows stricter unseen-country rules). Do **not** apply it to India/US.
  Accepted class-4 pairs there are 0.89 (India) / 0.99 (US) precise, because the combiner already learned them.
- Staged, both with the plain validator PASS:
  - `output/sub64_frgc70` = sub_iu54_frgc70 − 571 France pairs (556 S1, 39 emptied).
  - `output/sub64` = sub_iu54_best − 344 pairs (336 S1, 9 emptied).
  - Candidates are hardlinked from each base. Stream diffs: only France lines changed, and each changed list is a strict subset.
  - SUBMISSIONS.md has rows #64_frgc70 / #64.

## Evidence         (table: ledger name | world | key metrics | verdict)
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| p64-lf-class-calibration | holdout sw (P57), argmax pairs | differ-form (cls 4) P(true) India 0.013 / US 0.082, and #34 pa has the same mean (0.016 / 0.082). In the 0.2–0.95 band and inside core-twin groups, pa tracks y per class within ≈ 0.03 | drop |
| p64-iu-refit-sw-lf5 | holdout sw, half-split, paired | h0→h1 −0.000018 ± 0.000025 (India −0.000062, US +0.000027), h1→h0 +0.000012 ± 0.000023; lf gain share 0.6–0.8% | drop |
| p64-fr-loco-conflict-veto | P42 LOCO frames, gc70, chain + m2 | chain post-veto: US→India +0.002996 ± 0.00013 (h0 +0.00292, h1 +0.00307; 951 removed, 0.4% true); India→US +0.000744 ± 0.00007 (h0 +0.00063, h1 +0.00085; 697 removed, 36% true). m2: +0.00294 / +0.00903. Pre-decision veto (p := 0) is the same ±0.00001 | **keep** |
| p64-qwen-audit-conflicts | LOCO band (P38 Qwen LOCO r) + test France | LOCO: conflicts' Qwen r mean 0.03; false conflicts with r ≥ .5 1–3% vs 20–28% for other false pairs. Test: sub_iu54_best's stack (Qwen CE + P55) drops **45%** of frgc70's conflicts vs ≤ 1.7% of any other class. Conflicts' blend p < .5 on 17% (others 0.1%) | keep |
| p64-fr-test-conflicts | test France, sub_iu54_best | 817,090 pairs; cls 0/1/2/3/4/5 = 201,716 / 415,696 / 149,445 / 43,335 / **344** / 6,554. Top: sas-sa 43, sarl-sa 32, sarl-sas 31, sas-sasu 23, sci-snc 22, sasu-sas 20 | keep |
| p64-stage-sub64_frgc70 | test | −571 pairs, 556 S1 (39 emptied), md5 b5227e3d…, validator PASS (plain) | keep |
| p64-stage-sub64 | test | −344 pairs, 336 S1 (9 emptied), md5 ec77e5f2…, validator PASS (plain) | keep |

Per-class table (P57 sw frame, #34's pa):

| class | India n | India P(true) | India mean pa | US n | US P(true) | US mean pa |
|---|---|---|---|---|---|---|
| 0 neither has a form | 41,221 | 0.887 | 0.887 | 169,650 | 0.934 | 0.935 |
| 1 same | 210,607 | 0.881 | 0.881 | 185,504 | 0.938 | 0.938 |
| 2 S1 only | 40,617 | 0.917 | 0.917 | 83,255 | 0.937 | 0.938 |
| 3 SX only | 14,162 | 0.731 | 0.736 | 59,377 | 0.601 | 0.602 |
| 4 differ | 5,319 | 0.013 | 0.016 | 24,735 | 0.082 | 0.082 |
| 5 overlap | 41,304 | 0.735 | 0.735 | 1,714 | 0.407 | 0.413 |

(Counts are after the acronym fix. Rates were computed before it; the fix moved ≤ 50 rows per country.)

- **What the generator does:** conflicts between two valid forms are ~0% true. Examples: US llc-ltd 0/3,086, llc-inc 0/3,040,
  pllc-llc 0/168; India pvt ltd-llp 0/3,241.
  - Exceptions are words that double as forms: US pc/lp → llc/inc ≈ 0.7 true, India ltd → co ≈ 0.45.
  - US llp-llc (a substitution) is 25/32 true.
  - So "same name, other legal form" is a hard negative the generator makes, and true variants keep or abbreviate the form.
- **LOCO → LB:** per removed pair, LOCO gains +0.28 F per S1 (US→India) / +0.14 (India→US).
  - 571 pairs / 1,732,544 S1 → +0.000046 to +0.000092 LB (frgc70).
  - 344 pairs → +0.000028 to +0.000055 (best).

## What failed      (approaches tried and dropped, with the number that killed them)
- **lf columns in the India/US combiner:** ±0.00002, 0 SE. The raw signal is large, but #34's p already carries it per class
  and within twin groups.
- **Vetoing class 5 (the forms overlap but differ)** is harmful in LOCO: −0.0267 US→India (the removed pairs are 91% true).
  Vetoing class 3 (SX-only form) is also harmful: −0.0117 / −0.0153.
- **The first extractor read acronym SX names as forms** ("PC", "PA", "LP", "SA" = initials of the core). It flagged 814
  conflicts on sub_iu54_best, most of them acronym matches that are probably true. The fix is the all-legal-tokens → `''` rule. Now 344.
  LOCO did not move (+0.00299 → +0.00300).
- **`--check-ids` validator locally:** killed by my 900 s timeout under the 2 GB cap. It was not a failure. The plain validator
  passes, and every ID is in the base file, which passed `--check-ids`.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `lf.py`: `sig(names: Series) -> Series[str]` and `pair_class(sig1, sigx) -> int8[]`.
  - Classes: 0 none, 1 same, 2 S1 only, 3 SX only, 4 differ, 5 overlap.
  - Self-check: `.venv/bin/python probes/P64-legal-form-feat/lf.py`.
- `attach.py <world> <c>`: lf columns for P57's A frame → `data/work/p64/lf_<world>_<c>.parquet` (5–8 s).
  It also exposes `names(split, c, q1, qx)` over P3's text caches and `sigs()`.
- `ev64.py [world]`: the sub54 recipe refit with vs without lf columns, both directions, paired SE → `data/work/p64/ev64_<world>.json`.
  100 s, 1.3 GB.
- `fr64.py lf SRC TGT | eval`: lf classes on P42's LOCO rows (P27 S_holdout order), then veto variants on gc70 → `fr64.json`. 10 s each.
- `frtest.py stats BASE | build BASE OUT [strict]`: the France test class mix → `frtest_<BASE>.json`, and the veto build.
  About 60 s, < 1 GB. `strict` also ignores co/pc/lp/pa. It is moot after the acronym fix (304 vs 344) and not recommended.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Licence / data:** rapidfuzz (MIT, already a dependency) plus normalize's hand-written LEGAL set. No new data.
- **France precision is unmeasured.**
  - Of sub_iu54_best's 344, Qwen scored 159: 64 at r ≥ .5, 59 at r < .1. The r ≥ .5 ones are mostly one-letter form pairs
    (sas-sa 16, sasu-sas 8, sas-sasu 6), which Qwen reads as typos.
  - I veto them anyway. Both sides are valid French forms, which is the US pllc-llc shape (0/168 true). And in LOCO even Qwen
    r ≥ .5 conflicts are only ≈ 3% (US→India) / ≈ 34% (India→US) true, at or below the break-even precision for removing a pair.
- **The LOCO analogues are India and US forms, not French ones.** The mechanism transfers: the GBDT is blind to forms that
  name_core strips. French form-swap behaviour is assumed to follow the generator's US/India behaviour.
- Size: ±0.0001 LB at most either way.

## Open questions   (for the master; each with your best guess)
- Spare the one-letter conflicts (sas-sa, sasu-sas, sas-sasu; ≈ 94 of 344, ≈ 30 of them Qwen r ≥ .5)?
  Best guess no, i.e. veto them (pllc-llc analogue). Sparing only the Qwen r ≥ .5 conflicts is the conservative middle.
- Veto before the rescue / gc stages (pre-decision) instead of after? In LOCO pre and post are equal within 0.00001.
  Post-stage is simpler to port.
