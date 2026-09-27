# P42-fr-joint-tune — RESULT (final, 27 Sep 01:35)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **Some cells beat gc70 in both LOCO directions, on both analogues.** The knee is `sub42_both` = t 0.99, rescue 0.97, gc 0.7, hn-equal extras p ≥ 0.8 (τ 0.1, caps 5/6 unchanged). vs gc70, US→India / India→US:
  - chain analogue (P39): **+0.00003 ± 0.00026 / +0.00321 ± 0.00015**;
  - M2-only analogue (P37): **+0.00061 / +0.00440**;
  - on the held-out half, India→US replicates at +0.00315.
- (b) collapses onto (a). Allowing US→India down to −0.0005 buys no more India→US on the chain analogue (+0.00321 at both constraints). Past the knee, each +0.0001 of India→US costs about −0.0005 of US→India.
- **Recommendation: upload `output/sub42_both`.** Expected LB ≈ **0.9868** (+0.00043 to +0.00061; downside ≈ 0 if France behaves like India). It dominates hn90 on all four LOCO readings. `output/sub42_both_on41` (sub42_both plus P41's gib pairs) is the stacked alternative (≈ +0.00005 more).

## Recommendation   (exactly what the master should adopt: module, function signature, config)
**Recipe** (France only; the order is P37's):
- p(τ 0.1), then accept SX-argmax pairs with p ≥ **0.99** (was 0.9868);
- cap 5 S2 / 6 S3;
- rescue empty S1 at p ≥ **0.97** (was 0.90);
- gc `oneall` at p ≥ 0.70 (unchanged);
- **hn extras**: non-empty S1 get every free, argmax, non-SIB3, hn-equal pair with p ≥ **0.80**, up to the caps.

One function holds it all: `probes/P42-fr-joint-tune/solution_p42.py: decide(s1, is2, p, am, sib, hn_eq, n, t, cap, t_r, t_gc, t_hn)`. It is composed of `base()` and `add(stage, …)`. Stage order: base, then rescue, then gc, then hn.

**run.py integration** replaces P37's block, i.e. the current `rescue(0.90)` and `gc_add(0.70, "oneall")` calls, with:
```python
if d["c"] == "France":                        # P42 joint recipe; d["p"] = post-rules p (tier / bxneg / veto 0.1 / cap)
    import solution_p42 as J                  # move decide/base/add/rank_in/counts into france.py (~45 lines, numpy)
    am = M.argmax_sx(qx, d["p"])
    acc = J.decide(q1, is2, d["p"], am, sib, (h1 != "") & (h1 == hx), n_s1, 0.99, (5, 6), 0.97, 0.70, 0.80)
```
Here `sib` is P15's `sib_mask(h1, hx)`, and `h1` / `hx` are the fixed '(n)' house numbers (P37's).

**Queue** (base = #37_gc70, LB 0.986304; LB-eq = 0.1498 × France dF):

| rank | folder | France vs gc70 | LOCO chain: US→India / India→US | LOCO m2: US→India / India→US | expected LB |
|---|---|---|---|---|---|
| **1** | **`output/sub42_both`** | +23,509 / −2,570 pairs, 24,331 S1, 3.137 → 3.218 per S1 | **+0.00003 / +0.00321** | +0.00061 / +0.00440 | **≈ 0.98680** (+0.00043 to +0.00061). Risk ≈ 0 |
| 1-alt | `output/sub42_both_on41` | sub42_both + the 1,441 of sub41's 5,742 gib pairs it lacks (4,301 already inside); 0 S1 over the caps | (P41: gib pairs pass both ways, precision 0.98–0.99) | | ≈ 0.98685. Upload instead of #41 → #42 if one slot must carry both |
| 2 | `output/sub42_r90` | the same, with rescue 0.90 kept: +23,537 / −1,878 | −0.00323 / +0.00336 | −0.00066 / +0.00474 | +0.0005 to +0.0007; risk −0.0005. The LB has rewarded rescue90 before (LB +0.0021 vs LOCO +0.0010) |
| 3 | `output/sub42_ius` | (b) literally: gc 0.6 instead of 0.7, +1,109 more pairs | −0.00044 / +0.00321 | +0.00048 / +0.00454 | ≈ sub42_both. Dominated on chain; skip unless a slot is spare |
| — | `output/sub37_gc70_hn90` (P37) | +18,180 extras (p ≥ 0.9); **18,162 of them are inside sub42_both** | −0.00180 / +0.00175 | −0.00138 / +0.00203 | +0.00026 to +0.0003; risk −0.00027. **Dominated by sub42_both on every reading** |
| — | `output/sub39_ind80` (P39) | +15,226 / −1,530; 59% of its adds are inside sub42_both | P39: −0.06 / +0.0045 | — | P39: +0.0003 if India>US-like |
| — | `output/sub41` (P41) | +5,742 gib pairs | P41: +0.0001 / +0.0001 (plain LOCO) | — | P41: +0.00016 to +0.00024; 75% of its pairs are already in sub42_both |

All four P42 folders:
- **validator PASS**: official script on SageMaker job p42-val-0127 (the laptop cannot hold the candidate set under the cap), with a sub37_gc70 control that also PASSes. A local streaming check agrees: all 24,950 added pairs are in candidate_pairs.tsv;
- India/US lines byte-identical to sub37_gc70 (streamed check: 0 non-France lines differ);
- `candidate_pairs.tsv` and `code.zip` are hard links to gc70's. Every France pair is a pair of the France chain file, so the candidates are unchanged.

**Why sub42_both over sub42_r90.** They differ only in rescue 0.97 vs 0.90, and the evidence splits:

| lever | chain US→India / India→US | m2 US→India / India→US | LB evidence |
|---|---|---|---|
| rescue 0.97 instead of 0.9 (on gc70) | +0.00326 / −0.00015 | +0.00127 / −0.00033 | rescue90 LB +0.0021 France F (LOCO had said −0.0037 / +0.0010), so India→US under-reads rescue by ~2× |

- sub42_both passes the gate, and its worst case is ≈ 0.
- sub42_r90 has about +0.00005 LB more expected gain, and a −0.0005 worst case.
- Upload sub42_r90 only after sub42_both gains, as a pure rescue-bar test.

## Evidence         (table: ledger name | world | key metrics | verdict)
**Protocol.**
- Frames: `prep.py` builds one row per SX-argmax pair of the target's fold 0 (P6 am_all, fixed), with raw M2 g and P30's source-only bge r. The veto τ can therefore move: the chain p is P39's `blend_veto` with τ free.
- Two analogues:
  - chain (P39): M2 + bge band blend + veto, t_base 0.915 / 0.92;
  - m2 (P37): M2 only, t_base 0.97 / 0.81.
- Threshold map: every France threshold x maps to the LOCO threshold with the same **signed** argmax density per S1 between x and the anchor (France 0.9868 ↔ t_base). The map is recomputed at each τ on both sides. It is P37's rule, extended above the anchor.
- Scoring: dF = paired mean per-S1 F0.5 over all fold-0 target S1 vs the gc70 cell. The SE is of that paired difference. h0 / h1 are the tune / report halves.

| ledger name | world | key metrics | verdict |
|---|---|---|---|
| p42-repro | LOCO + France | chain: rescue90 −0.00374 / +0.00098; gc70 −0.00097 (prec 0.617) / +0.00221 (0.871); hn90 −0.0018 / +0.00175. All equal P39's ontop. m2: gc70 +0.00079 / +0.00204, hn90 −0.00138 / +0.00203, equal P37's. France: `build` of the gc70 cell is **byte-identical** to sub37_gc70, and the hn90 cell is byte-identical to sub37_gc70_hn90. Stage adds 1,205 / 10,084 / 18,180, equal to #34v_r90 / gc70 / hn90 | keep (machinery) |
| p42-grid-joint | LOCO | 49,392 cells × 2 analogues × 2 directions = 197,568 evaluations on 4 × ml.m5.4xlarge spot, ≈ 5 min each. All six jobs (smoke, 4 grid shards, validator) came to ≈ 340 billable spot-seconds, about $0.1 of the $25 budget. Buildable cells (τ ≤ 0.1, t ≤ 0.99): 28,812. Cells dominating gc70: 1,316 (chain), 2,176 (m2) | keep |
| p42-sub42_both | LOCO + test | see the queue. Selected on h0 as well (the same cell); h1: −0.00021 / +0.00315 | **keep → upload** |
| p42-sub42_ius / r90 / both_on41 | test | validator PASS (all four, plus the gc70 control, on SageMaker). hn90's extras: 18,162 of 18,180 are inside sub42_both; sub41: 4,301 of 5,742; sub39_ind80: 9,025 of 15,226 | staged |

**One lever at a time from gc70** (×1e-4, US→India / India→US; **chain** | m2):

| lever | values |
|---|---|
| τ | 0: −274/−56 · 0.02: +13/−12 · 0.05: +7/−3 · **0.1: 0/0** · 0.2: −10/−1 · 0.3: −20/−4. Same shape on m2: **0.1 stays** |
| t | 0.97: −18/−32 · 0.98: −11/−22 · 0.985: −1/−5 · **0.9868: 0/0** · 0.988: −2/+3 · 0.99: −5/+7 · 0.9925: −39/−5 ‖ m2: 0.988: +6/+6 · 0.99: **+15/+16** · 0.9925: +23/+52 |
| cap | **5/6: 0/0** · 4/5: −2/−2 · 3/4: −24/−23 (m2 alike): **5/6 stays** |
| rescue | off: +37/−11 · 0.97: +33/−2 · 0.95: +20/0 · **0.9: 0/0** · 0.8: −23/−1 · 0.5: −65/−9 ‖ m2: 0.97: +13/−3 |
| gc | off: +10/−22 · 0.9: +12/−5 · 0.8: +5/−2 · **0.7: 0/0** · 0.6: −5/0 · 0.5: −10/+1 ‖ m2: 0.5: −3/+2 |
| hn | **off: 0/0** · 0.97: −3/+10 · 0.95: −7/+14 · 0.9: −18/+18 · 0.8: −29/+20 · 0.7: −39/+21 · 0.5: −57/+21 ‖ m2: 0.8: −19/+24 |

- **Where the current recipe sits:** four of its six levers are already at their one-lever optimum (τ 0.1, cap 5/6, gc 0.7, and rescue 0.9 in India→US).
- The two moves are t → 0.99 and the hn extras. They combine super-additively in India→US:
  - chain: +0.00065 and +0.00203 alone, **+0.00336** together;
  - why: t 0.99 drops the non-hn-equal pairs in [0.9868, 0.99), about 1.9k on France (only 71% of that band is hn-equal, vs 92% of accepted pairs), and the hn stage adds the hn-equal ones back.
- Rescue 0.97 then buys back the US→India cost of the hn extras.

**Decomposition** of the picks (×1e-4; chain | m2):

| cell | US→India | India→US | US→India m2 | India→US m2 |
|---|---|---|---|---|
| t99 only | −4.6 ± 0.7 | +6.5 ± 0.9 | +15.4 | +16.1 |
| hn80 only | −29.1 ± 1.0 | +20.3 ± 0.5 | −19.1 | +24.4 |
| r97 only | +32.6 ± 2.2 | −1.5 ± 1.2 | +12.7 | −3.3 |
| t99 + hn80 (= sub42_r90) | −32.3 ± 1.2 | +33.6 ± 1.0 | −6.6 | +47.4 |
| t99 + hn80 + r97 (= **sub42_both**) | **+0.3 ± 2.6** | **+32.1 ± 1.5** | +6.1 | +44.0 |
| hn90 (staged, P37) | −18.0 | +17.5 | −13.8 | +20.3 |

**Frontier** (chain Pareto set, buildable cells; the cells toward the high-India→US end, m2 alongside):

| τ | t | cap | rescue | gc | hn | chain US→India | chain India→US | m2 US→India | m2 India→US |
|---|---|---|---|---|---|---|---|---|---|
| 0.02 | 0.99 | 5/6 | 0.97 | 0.9 | 0.97 | +0.00503 | +0.00064 | +0.00325 | +0.00164 |
| 0.05 | 0.99 | 5/6 | 0.97 | 0.9 | 0.95 | +0.00404 | +0.00180 | +0.00222 | +0.00281 |
| 0.1 | 0.99 | 5/6 | 0.97 | 0.9 | 0.95 | +0.00346 | +0.00212 | +0.00138 | +0.00308 |
| 0.1 | 0.99 | 5/6 | 0.97 | 0.8 | 0.95 | +0.00275 | +0.00243 | +0.00161 | +0.00336 |
| 0.1 | 0.99 | 5/6 | 0.97 | 0.7 | 0.9 | +0.00113 | +0.00294 | +0.00114 | +0.00400 |
| 0.1 | 0.99 | 5/6 | 0.97 | 0.8 | 0.8 | +0.00054 | +0.00308 | +0.00058 | +0.00423 |
| **0.1** | **0.99** | **5/6** | **0.97** | **0.7** | **0.8** | **+0.00003** | **+0.00321** | **+0.00061** | **+0.00440** |
| 0.1 | 0.99 | 5/6 | 0.97 | 0.6 | 0.8 | −0.00044 | +0.00321 | +0.00048 | +0.00454 |
| 0.1 | 0.99 | 5/6 | 0.97 | 0.8 | 0.7 | −0.00053 | +0.00326 | +0.00001 | +0.00463 |
| 0.1 | 0.99 | 5/6 | 0.95 | 0.7 | 0.8 | −0.00121 | +0.00337 | +0.00000 | +0.00460 |
| 0.1 | 0.99 | 5/6 | 0.95 | 0.5 | 0.7 | −0.00321 | +0.00346 | −0.00081 | +0.00504 |

The full 36-cell set is in `report.json`. Every cell on the frontier has τ ≤ 0.1 and t = 0.99, the top of the buildable range.

**Picks** (brief's definitions):
- (a) best India→US with both directions ≥ 0: **sub42_both** on the chain analogue alone. It is also the pick when both analogues must pass, and when the pick is made on the tune half.
- The pick that holds on every half of both analogues is gc 0.8 instead of 0.7: +0.00054 / +0.00308. It is not built, because the difference is inside 1 SE.
- (b) best India→US with US→India ≥ −0.0005: `sub42_ius` (gc 0.6): −0.00044 / +0.00321, tied with (a) to 1e-5.

## What failed      (approaches tried and dropped, with the number that killed them)
- **τ ≠ 0.1.**
  - Lower τ trades India→US for US→India (0.02: +0.0013 / −0.0012). No veto costs −0.027 / −0.006.
  - Higher τ loses both ways (0.2: −0.0010 / −0.0001).
- **Tighter caps:** 4/5 gives −0.0002 / −0.0002, 3/4 gives −0.0024 / −0.0023.
- **Lower France t:** 0.985 and below loses both ways on both analogues. Stricter is better, as D2 / D4 found.
- **Lower gc / rescue bars:** on the chain analogue they lose US→India and are flat in India→US (P37 found the same). On m2, gc 0.4–0.5 gives +0.0002 India→US at −0.0003 US→India.
- **t > 0.99 (0.9925) is not buildable honestly.** 13,472 France argmax pairs sit at exactly p = 0.99, P28 bxneg's boost value. A t above 0.99 would cut a label-free rule the LOCO frames do not have. m2 likes t 0.9925 (+0.0023 / +0.0052); chain does not (−0.0039 / −0.0005).
- **τ > 0.1 on France is inexact.** It needs the tier / bxneg exemption for newly vetoable pairs, which is only a proxy here. It is moot: τ > 0.1 loses on LOCO anyway.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `prep.py loco SRC TGT | france` → `data/work/p42/{loco_US_India, loco_India_US, france}.npz` (5 / 8 / 26 MB; 2–10 s; ≤ 1.0 GB).
  - loco: s1, is2, g, r, y, hn_rel, ntrue, hv. It asserts row alignment with P39's frames.
  - france: all 1.33M chain pairs as codes, with:
    - p0 / p05 / p10 (veto off / 0.05 / 0.1 files);
    - g, r (P30 pool, then the CE cache);
    - sib (P15), hn_eq (w_fix SX).
- `solution_p42.py` (numpy only; the SageMaker entry point, and the local import):
  - `decide(...)`, `base(...)`, `add(stage, ...)`;
  - `loco_p(g, r, tau, analogue)`;
  - `sdens` / `inv_sdens`: the signed density map;
  - `run_grid(data, G, shard, nshard, out)`;
  - `main()`: channel `train` = the npz frames + `grid.json`; params `shard` / `nshard` only, read from hyperparameters.json;
  - `_selfcheck()` runs on `python solution_p42.py`.
- `p42.py`:
  - `grid`: France densities → `data/work/p42/sm/grid.json`;
  - `repro`;
  - `local`;
  - `build SPEC OUT`: 4 s, 0.55 GB. SPEC is e.g. `t=0.99,r=0.97,gc=0.7,hn=0.8`; unset keys are the current recipe; `off` turns a stage off;
  - `stack TOP EXTRA OUT`.
- `report.py <cells dirs>` → `report.json` (picks, Pareto, per-lever) and `data/work/p42/wide.parquet`.
- Grid results: `out/p42-grid-s{0..3}-*/cells_*.jsonl` (44 MB). Upload prefix: `s3://sagemaker-114872709824-ap-south-1/amc26-p42/`.
- **Rebuild the queue:**
  ```
  .venv/bin/python probes/P42-fr-joint-tune/p42.py build t=0.99,r=0.97,gc=0.7,hn=0.8 sub42_both
  .venv/bin/python probes/P42-fr-joint-tune/p42.py build t=0.99,r=0.97,gc=0.6,hn=0.8 sub42_ius
  .venv/bin/python probes/P42-fr-joint-tune/p42.py build t=0.99,r=0.9,gc=0.7,hn=0.8 sub42_r90
  .venv/bin/python probes/P42-fr-joint-tune/p42.py stack sub42_both sub41 sub42_both_on41
  ```

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **The gate is passed by offsetting, not by every lever passing.**
  - sub42_both's hn-extras part is US→India −0.0029 (chain), as India's hn-equal extras are ~50% precise.
  - Its rescue part is +0.0033. Both are real LOCO effects, but they come from different mechanisms. If France is India-like on hn and US-like on rescue, the LB could go negative.
  - France's label-free lens is US-like on hn: 92% of accepted pairs are hn-equal (US 80%, India 50%; P37 / P39).
- **Rescue calibration.** The LB rewarded rescue90 at 2× LOCO India→US, while US→India called it −0.0037. So sub42_both's rescue 0.97 probably gives back ~0.0003 France F that LOCO does not see. That cost is already in the expected-LB range.
- **The t move rests on the density map:** France's [0.9868, 0.99) band behaves like LOCO's band of equal argmax density above t_base.
  - Chain and m2 disagree on t alone (−5/+7 vs +15/+16 ×1e-4).
  - They agree on t combined with hn extras.
  - t 0.99 is the edge of the buildable grid, so the optimum may sit higher, but it cannot be reached without touching bxneg's pairs (see What failed).
- **Winner's curse:** 28,812 buildable cells were scanned. The pick replicates on the held-out half in India→US (+0.00315 vs +0.00326 on the tune half), and US→India h1 is −0.00021 (inside 1 SE).
- **No new model or data.** Only label-free thresholds and the house-number / group rules already used. Nothing is keyed on a country: France S1 are identified by the S1 file, as before. Every S1 row stays in the file.
- **Disk:** four new folders × 97 MB (matching file only; the rest are hard links). About 2.2 GB free.
- **Worktree:** I worked in the main checkout's `probes/P42-fr-joint-tune/`, where the brief was placed. With < 3 GB of disk free I made no separate worktree. Nothing under `src/` was touched, and nothing is committed.
- **P42 rows in SUBMISSIONS.md** were added as the brief asked: four `42_*` rows plus a "P42 ranking" paragraph above "Upload next".

## Open questions   (for the master; each with your best guess)
- **sub42_both vs sub41 first?** sub42_both subsumes 75% of sub41's pairs and has ~2.5× its expected gain. Best guess: upload `sub42_both_on41` if only one France slot is left today. With two slots, upload sub42_both first, then on41 as the +1,441-pair increment.
- **If sub42_both gains:** test sub42_r90 (rescue 0.9 back). hn 0.7 is not worth it: chain −0.00096 / +0.00327, i.e. +0.00006 India→US for −0.001 US→India.
- **If sub42_both loses:** France is India-like on hn extras. Drop the hn stage and keep t 0.99 + r 0.97 only. That cell passes both ways on both analogues: chain **+0.00280 / +0.00050**, m2 +0.00281 / +0.00128. Not built; rebuild in 4 s with `build t=0.99,r=0.97 sub42_t99r97`.
- **Raise t above 0.99 while exempting rule pairs?** It would need P31's tier / bxneg flags per pair; `france.post` does not return them. m2 says +0.005 India→US at 0.9925. Best guess: worth one probe hour only if sub42_both gains.
