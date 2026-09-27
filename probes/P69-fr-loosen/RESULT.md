# P69-fr-loosen — RESULT (final, 27 Sep 15:30)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **No: France's strict t is not dropping net-genuine pairs.** The pairs just below t (p .95–.9868, 15.2k) look CE-healthy (Qwen median .96), but 99% of them would be an **extra member of a source the S1 already holds** (only 1.1% fill a missing source, 0.1% an empty S1). LOCO puts that class at **50–62% true**, below the ~73% break-even for an extra member. The LB-proven gc70 / rescue90 adds are the opposite case: they fill a missing source or an empty S1.
- LOCO France-F0.5, t .9868 → .975: **chain −0.0012 / −0.0023, M2 −0.0024 / −0.0028** (US→India / India→US, both halves agree). t .95: −0.0028 to −0.0048. **The LB already measured this direction.** sub54 → sub66_final changed France only: −40,888 looser pairs, +6,245 others, **LB +0.000706**. That removed set holds **85% of sub66_t975's adds** (and 85% of t95's).
- **Recommendation: no LB slot for sub66_t975 or sub66_t95** (expected −0.0002 to −0.0004 and −0.0004 to −0.0007). The one variant with positive LOCO in all four readings is `output/sub66_t975ce`: t .975's adds with bge r ≥ .99, +5,270 pairs, expected **+0.0001** (−0.00015 to +0.0002). It is worth a slot only after higher-EV candidates, and it is the fairest single LB test of the user's hypothesis. Retrieval is confirmed not to be France's bottleneck: 59.1% of unassigned France SX are candidates, and 17.6% of France SX are never candidates.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- **Keep `sub66_final` as final.** Do not upload `sub66_t975` / `sub66_t95`. A t975 upload is not worth a slot as a "direct France measurement". 85% of its 9,498 adds were in sub54's France lines, and the LB rejected that set as part of the +0.0007 sub54 → sub66_final step. The upload would re-measure a known direction at an expected cost of −0.0002 to −0.0004.
- **Optional slot, low priority: `output/sub66_t975ce`.** It is sub66_final plus 5,270 France pairs on 5,141 S1, additions only; India/US lines are byte-identical. The rule: a pair accepted by run.py `--recipe sub66_final --t-country France=0.975` (full chain at .975, P30 / P55N / P64 vetoes and caps on), not in sub66_final, with **bge r ≥ .99** (P30 pool, then cache: the veto's r), SX owned by no sub66_final line, caps 5 S2 / 6 S3 on the line's counts. Upload it only if a slot is spare after anything with a stronger expected gain.
  - Its LB is the direct test of "strict t drops CE-confirmed genuine pairs".
  - ≥ 0.98745 → keep it as the new best. ≤ 0.98735 → the hypothesis is dead even in its strongest form.
  - Rebuild: `p69.py 0.975 …/t975` (8 min, 1.65 GB) → `build69ce.py 0.99 …/t975ce` → `splice.py …/t975ce output/sub66_final output/sub66_t975ce`.
- Nothing needs porting to src/. If t975ce wins the LB, the port is ~15 lines in run.py after `xfc_fit`: the same adds, from a second France decision at .975.

**SUBMISSIONS rows** (base sub66_final, LB 0.987391; LB-eq = 0.15 × France dF):

| # | status | files | India/US | what | expected LB |
|---|---|---|---|---|---|
| 66_t975ce | staged, validator PASS --check-ids (P69, SageMaker p69-val-t975ce-151656, md5 add3db4f…) — optional | `output/sub66_t975ce/` (candidates = sub66_final's, hardlink) | = sub66_final (byte-identical) | sub66_final + 5,270 France pairs (5,141 S1): the t .975 chain's adds with bge r ≥ .99, SX free, caps 5/6. Qwen median .998 on the 35% it scored. LOCO (the same slice): chain +0.00041 / +0.00066 (prec .95 / .90), M2 +0.00082 / +0.00105 (prec .985 / .94). France rate 0.020 / S1 vs LOCO 0.006 / 0.015. 89% were in sub54's France lines | ≈ 0.9875 (+0.0001; range −0.00015 to +0.0002) |
| 66_t975 | staged, validator PASS --check-ids (p69-val-t975-151644, md5 f6068a35…) — **do not upload** | `output/sub66_t975/` | = sub66_final | France chain t .9868 → .975 (post-rules + decision), every veto / cap kept: +9,498 / −223 pairs, 9,214 S1. Adds: 0.35% fill a missing source; bge 72% ≥ .9 / 18% < .5. LOCO −0.0012 / −0.0023 chain, −0.0024 / −0.0028 M2, added precision .50–.62. 85% inside sub54's LB-rejected extras | ≈ 0.98700–0.98722 (−0.0002 to −0.0004) |
| 66_t95 | staged, validator PASS --check-ids (p69-val-t95-152411, md5 725af7f7…) — **do not upload** | `output/sub66_t95/` | = sub66_final | France t → .95: +14,452 / −412 pairs, 13,836 S1. 85% inside sub54's extras. LOCO −0.0028 / −0.0042 chain, −0.0037 / −0.0048 M2 | ≈ 0.98667–0.98697 (−0.0004 to −0.0007) |

## Evidence         (table: ledger name | world | key metrics | verdict)
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| p69-funnel-sub66_final-france | test France | p69.py at .9868 = sub66_final's France matching lines **byte for byte** (259,452, 0 differ; candidates: P43's known 1-line #54 dup candidate). Funnel below | keep |
| p69-loco-t975-t95 | LOCO (P42 frames) | t .975: chain −0.00117 / −0.00231, M2 −0.00237 / −0.00280. t .95: chain −0.00282 / −0.00419, M2 −0.00368 / −0.00478. Halves within ±0.0001. Added precision .50–.62 | drop |
| p69-sub66_t975 | test France | +9,498 / −223. Qwen med .995 on 32% coverage; bge ≥ .9 72%. To S1 lacking the source 0.35%, empty 0.09%. 85.4% in sub54's extras. Validator PASS | drop |
| p69-sub66_t975ce | test France + LOCO | +5,270 (bge r ≥ .99). LOCO r ≥ .99 slice: chain +0.00041 / +0.00066, M2 +0.00082 / +0.00105. The r ≥ .9 slice fails chain India→US (−0.00016). Validator PASS | inconclusive (optional slot) |
| p69-sub66_t95 | test France | +14,452 / −412, 13,836 S1. Qwen med .97 on 55%. 85% in sub54's extras. Validator PASS | drop |

**A. Funnel of sub66_final's France lines.** Qwen = pooled Qwen3-Reranker r (band pairs only; "cov" = share scored); m2 = c_hn's p (g).

| stage | pairs | S1 | Qwen cov / median / ≥.9 / <.5 | bge ≥.9 / <.5 | m2 p deciles 10/50/90 |
|---|---|---|---|---|---|
| blend decision (argmax & pbl ≥ .9868) | 793,087 | 242,040 | .03 / 1.00 / 1.00 / 0 | .95 / .04 | .996 / 1.0 / 1.0 |
| post: tier / bxneg raise into acc | +16,994 | 15,571 | .63 / .70 / .33 / .38 | .77 / .13 | .74 / .971 / .985 |
| post: **P30 veto** (g ≥ .98, bge r < .1) | −6,142 (7,753 vetoed in all) | 5,995 | 0 (no Qwen score exists for any of them; 1 of 7,753 in the diag file) | 0 / 1.0 by construction | .989 / .996 / 1.0 |
| post: cap | −10 (+14 SX zeroed) | 10 | .4 / 1.0 | .7 / .2 | — |
| = P31 chain decision (t .9868) | 803,929 | 242,622 | | | |
| rescue90 (LB-proven) | +1,168 | 1,043 | .76 / .88 / .48 / .25 | .57 / .32 | .70 / .953 / .985 |
| gc70 (LB-proven) | +9,030 | 8,415 | .79 / .61 / .32 / .38 | .44 / .42 | .66 / .940 / .984 |
| **P55N veto** | −619 | 618 | 1.0 / .62 / .39 / .36 | .23 / .56 | .80 / .947 / .977 |
| P62 acronym adds | +3,538 (2,638 new candidates) | 3,445 | .27 / .996 / .88 / .01 | .08 / .83 | .58 / .998 / 1.0 |
| P54 dup | +44 | 44 | — | — | — |
| **P64 legal-form veto** | −344 | 336 | .46 / .31 / .21 / .60 | .40 / .50 | .15 / .978 / .997 |
| P66 xfc adds | +1,884 | 1,810 | .69 / .44 / 0 / .76 | 0 / .99 | .91 / .973 / .985 |
| dup_last | 0 | | | | |
| **final** | **818,630** | 243,738 | | | |
| rejected argmax pairs p [.95, .9868) | 15,233 | 14,578 | .57 / .96 / .60 / .11 | .64 / .24 | .80 / .976 / .985 |
| rejected argmax pairs p [.9, .95) | 7,403 | 7,236 | 1.0 / .59 / .28 / .37 | .36 / .48 | .64 / .937 / .968 |
| rejected argmax pairs p [.5, .9) | 29,630 | 27,721 | 1.0 / .42 / .15 / .60 | .26 / .61 | .43 / .823 / .957 |

- **Not in sub66_final's France lines:** P48 low-CE veto, P54 in-band CE rescue, P47 cls80, P41 brand, hn-equal extras. run.py computes them only inside P62's acronym reference (#clean's chain), and they never touch the final lines. The brief's stage list comes from sub54 / #clean.
- **Vetoes-off diagnostic** (no file): P30 −6,142 accepted pairs (Qwen n/a, bge r < .1, m2 ≥ .989); P55N −619 (Qwen median .62, 39% ≥ .9); P64 −344 (Qwen median .31); cap −10. Turning all of them off adds ≈ 7.1k pairs. Only P55N's has a Qwen profile comparable to gc70's adds, and P55N passed LOCO both ways when it was built.

**B. Unassigned France SX that are candidates** (by their argmax pair's post-rules p), with LOCO precision of the same density-mapped bins for unassigned argmax pairs (chain US→India / India→US):

| argmax p bin | n | Qwen cov / median / ≥.9 | argmax S1 lacks that source | S1 empty | LOCO precision, chain U→I / I→U · M2 U→I / I→U | LOCO precision when the S1 lacks that source (same order) |
|---|---|---|---|---|---|---|
| < .3 | 294,156 | .47 / .001 / .01 | 15.5% | 6.1% | .12 / .05 · .19 / .07 | .16 / .07 · .27 / .09 |
| .3–.5 | 17,220 | 1.0 / .20 / .06 | 17.3% | 4.6% | .18 / .31 · .39 / .35 | .25 / .40 · .48 / .44 |
| .5–.7 | 13,614 | 1.0 / .32 / .11 | 17.6% | 4.2% | .19 / .37 · .42 / .42 | .27 / .45 · .54 / .51 |
| .7–.9 | 15,972 | 1.0 / .48 / .18 | 5.8% | 4.3% | .19 / .40 · .41 / .43 | .15 / .18 · .36 / .17 |
| .9–.95 | 7,398 | 1.0 / .59 / .28 | 2.0% | 0.1% | .23 / .48 · .46 / .47 | .13 / .02 · .57 / .02 |
| .95–.9868 | 15,231 | .57 / .96 / .60 | 1.1% | 0.1% | **.56 / .58 · .49 / .54** | .37 / .01 · .73 / .01 |
| ≥ t (blocked by caps / owner) | 410 | | | | | |

Every bin is below the ~.73 break-even for an extra member on every LOCO reading. The bins that lack a source (where gc70 lives) are tiny near t.

- **Retrieval, one line (confirmed):** France has 1,434,993 SX; 616,363 are unassigned, of which **59.06% are candidates**; **17.59%** of all France SX are never candidates. The decision is the lever, not blocking. But the gap sits in low-p bins (< .5: 311k) whose Qwen median is ≤ .20.
- **Why the CE profile misleads here:** the t975 adds' Qwen profile (median .995) beats the LB-proven gc70 adds' (.61). But gc70 fills a missing source, which the train prior favours: 81% of S1 hold both sources. The t975 adds are extra same-source members. LOCO's own CE-gated slice shows the split: bge r ≥ .99 is 90–98% true (passes), r ≥ .9 is 75–96% (fails chain India→US), and all adds are 50–62% (fails).

**LB evidence (the decisive row).** sub54 (LB 0.986685) → sub66_final (0.987391) kept India/US identical (#54's lines in both recipes). France went −40,888 / +6,245 pairs. The removed pairs: Qwen median .54 (29% ≥ .9), 2.2% to empty S1, 7.8% to S1 lacking that source. The kept-new pairs (acro 2,638 new candidates, P66, Qwen swap) carry maybe +0.0004–0.0005 expected, so removing the looser 40.9k was worth about +0.0002–0.0003. **Looser France already lost on the LB**, and 85% of any t-loosening file's adds sit inside that removed set.

## What failed      (approaches tried and dropped, with the number that killed them)
- France t .9868 → .975 (sub66_t975): LOCO −0.0012 / −0.0023 chain, −0.0024 / −0.0028 M2; 85% of its adds are inside the LB-rejected sub54 extras.
- France t → .95 (sub66_t95): LOCO −0.0028 / −0.0042 chain, −0.0037 / −0.0048 M2.
- The CE gate at bge r ≥ .9 instead of .99: chain India→US −0.00016 (prec .75).
- hn-equal gating of the t975 adds is not a transferable gate: India→US prec .95, but US→India .54–.64 (India's hn misleads, as in P37).

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `p69.py T OUT` runs run.py's `sub66_final` France stage only (`model_parts(only=France)`) at `T_COUNTRY["France"] = T`. france31 / post / complete / p55_veto / dup_add / lf_veto / xfc_fit are wrapped. Output: France matching + candidate lines, and `OUT/p69_frame.npz` (keys, post p, blend pbl, g, Qwen r, bge r, per-stage keys). 8–11 min, 1.65 GB peak. The run ends on run()'s expected all-countries row assert, which it catches.
- `funnel69.py OUT` → tables A + B in `OUT/funnel.json` (copy: `results/funnel_t9868.json`). 30 s, 0.6 GB.
- `diff69.py A B` → adds / removes between two dumps, or a dump vs a France-lines TSV → `A/diff_<B>.json` (copies in `results/`).
- `loco69.py` → `loco69.json`: P42 decide at t .9868 / .975 / .95, density map re-measured on sub66_final's frame, plus the bin table. 40 s. The CE-gated slice is in `loco69_gated.json`, from an inline run of the same machinery.
- `build69ce.py GATE OUT` builds the additive CE-gated France lines. `splice.py SRC BASE DST|-` pastes France lines into a base, keeps India / US byte-identical, and hard-links candidates when they are covered. `fr_lines.py` filters a matching TSV on stdin to its France lines.
- Big dumps live in `data/work/p69/` (`probes/P69-fr-loosen/work` is a symlink to it), ~430 MB. Delete them for disk after the master is done.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- Assumption: "France chain t" means run.py's `--t-country France=T`. That moves both the post-rules (tier / bxneg / veto / cap) and the decision to T; every veto and cap stays on. At .975 this also shifts P55N (−751 vs −619) and P66 (1,297 vs 1,884 adds) slightly, because both are computed on frames that read t.
- The sub54 → sub66_final attribution is confounded: the Qwen swap, acro, P64 and P66 moved at the same time. The +0.0002–0.0003 for "removing the looser pairs" is a residual, not a measurement. Still, a looser France was not rewarded.
- t975ce: France carries 1.3–3× LOCO's rate of these pairs. A higher target rate than LOCO has in the past meant lower precision (P47, P54's shortfall). 89% of the pairs were in sub54. Downside at LOCO's ungated precision ≈ −0.00015 LB.
- The LOCO analogue reads P42's #37-style chain, not P55N / acro / P64 / P66. Those stages touch < 1% of the pairs a t change moves.
- No licence or external-data change. Every file keeps every France S1 row (validator --check-ids).

## Open questions   (for the master; each with your best guess)
- Is a spare slot better spent on t975ce than on any other staged France / India-US candidate? Best guess: only if nothing else has ≥ +0.0001 expected. t975ce's EV is ≈ +0.0001 with a symmetric-ish risk.
- Should P54's in-band CE rescue (r ≥ .99 on band pairs, LOCO +0.0023 / +0.0024 when built) return to the final chain? It is absent from sub66_final's France lines. Best guess: it was part of sub54's LB-rejected looser set, so no LB slot without an isolated test. t975ce is the nearest isolated test of "CE ≥ .99 extra members".
