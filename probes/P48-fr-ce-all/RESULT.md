# P48-fr-ce-all — RESULT (final, 27 Sep 04:35)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **The CE's win for France is inside the band, not outside it.** France already has bge r on every accepted pair: the band, plus P30's pool on g ≥ 0.98. The arm that passes is a **veto of band pairs added by the rescue and gc stages when bge r < 0.2**. In the band blend the CE has weight 0.26, too little for its negative calls. LOCO France-F0.5 vs the current recipe:
  - chain analogue: **US→India +0.0071 (h1 +0.0070 ± 0.0004), India→US +0.00041 (h1 +0.00035 ± 0.00012)**. Gate passed as "both ≥ +0.0003".
  - M2-only analogue: +0.0084 / +0.00105.
- **Every out-of-band arm fails:**
  - harder veto on g ≥ 0.98: negative both ways (τ 0.1 is already optimal);
  - full-range blend: −0.0126 / −0.0042;
  - low-side rescue with the honest source-only CE (GPU): India→US negative in every scope except one-sided at u ≥ 0.99, where it reaches only +0.0001.
- **Upload `output/sub48`** = output/sub44r minus 1,138 France pairs. India/US are byte-identical. Validator `--check-ids` PASS. **Expected LB ≈ +0.00006 over sub44r** (up to +0.00016 on the M2-only reading).

## Recommendation   (exactly what the master should adopt: module, function signature, config)
**Upload:** `output/sub48` (validator PASS, SageMaker p48-valb-0927-0417, md5 8a820166…). Alternative: `output/sub48_all`, which also vetoes hn extras (PASS, p48-val-0927-0415, md5 38e254eb…). Both have rows in SUBMISSIONS.md.

**Rule** (unseen country only; applied after `france.complete`'s rescue / gc stages, and never to tier pairs from P41 gib / P40 xf / P46 swap). Drop a pair when all four hold:
- the rescue or gc stage added it;
- 0.02 < p_gbdt < 0.98;
- p equals the OOD blend σ(0.9245·logit g + 0.2595·logit r + 0.0515), i.e. no tier / bxneg rule moved it;
- bge r (ce_bge_pool) < 0.2.

**src port** (≈ 6 lines in `france.complete`): after `gc_add`, compute `veto = added_by_rescue_or_gc & band & (abs(p - blend) <= 1e-4) & (r < 0.2)` and do `acc &= ~veto`. Every such pair already has r in the France CE cache, so there is **no new GPU work**.

**Build from staged files:** `.venv/bin/python probes/P48-fr-ce-all/build48.py sub44r sub48 veto=0.2,stages=rescue+gc` (3 s, 0.6 GB). Then `check48.py sub44r sub48`: only France lines differ, and pairs are only removed.

## Evidence         (France-F0.5 units; dF = mean paired per-S1 change over all target fold-0 S1 vs the current recipe t .99 / r .9 / gc .7 / hn .8, τ 0.1, caps 5/6; h1 = report half ± SE; v picked on h0)
| ledger name | world | key metrics (US→India / India→US) | verdict |
|---|---|---|---|
| p48-loco-veto-stage | LOCO chain | **rescue+gc, r < 0.2: +0.00712 (h1 +0.00700 ± 0.00039) / +0.00041 (h1 +0.00035 ± 0.00012)**; flipped pairs 75% / 40% false; 2,024 / 712 pairs | **keep → sub48** |
| p48-loco-veto-stage | LOCO m2 | rescue+gc: +0.00839 / +0.00105 (h1 +0.00106 ± 0.00013); 86% / 64% false | keep |
| p48-loco-veto-stage | LOCO chain | by stage at 0.2: rescue +0.0040 / +0.00032; gc +0.0027 / +0.00008; **hn +0.0031 / −0.00003 ± 0.00004 (76% of India→US hn vetoes are true)** | hn → alternative only |
| p48-loco-veto-band-added | LOCO chain | rescue+gc+hn (= sub48_all): v 0.1 +0.0103 / +0.00033; **0.2 +0.0103 / +0.00037 (h1 +0.00034)**; 0.3 +0.0102 / +0.00034; 0.5 +0.0099 / +0.00006 | alternative (sub48_all) |
| p48-loco-veto-band-all | LOCO chain | adds the base stage: 0.2 +0.0103 / +0.00055. Pre-decision (p := 0) ≈ post. France base-stage band pairs with r < 0.2 (1,030) are **all** tier / bxneg-promoted (p0 0.99–1.0), and LOCO has no rules | not applicable to France |
| p48-loco-veto-hi | LOCO chain | post-decision veto of g ≥ 0.98 pairs at r < 0.2: −0.00095 / −0.00008; 0.3: −0.0021 / −0.00045 (P42's τ grid agrees) | drop |
| p48-loco-blend-hi | LOCO chain | OOD blend on g ≥ 0.98 too, France bars re-mapped by density: −0.0126 / −0.0042. On the low side the blend can never reach a bar: max σ(0.9245·logit 0.02 + 0.26·9.2) ≈ 0.23 | drop |
| p48-loco-rescue-dom-bound | LOCO chain | in-domain r_all (P45 OOF; the CE saw the target country; P45-pruned pairs only) > 0.9: any +0.0012 / +0.0013; empty +0.0003 / +0.0003; one-sided +0.0003 / +0.0003. **This is the ceiling** | inconclusive → honest test |
| p48-loco-rescue-src | LOCO chain | source-only CE (P30 ce_bge_loco*.pt, GPU): any u .98 +0.00065 / **−0.0021** (prec .92 / .46); empty u .995 +0.00017 / −0.00023; one-sided u .995 +0.00019 / +0.00010 (prec .99 / .81) | **drop** (fails the gate everywhere) |
| p48-sub48 | test | 1,138 pairs removed (rescue 41, gc 1,097), 1,112 S1, 37 made empty; validator PASS; India/US byte-identical to sub44r | **keep → upload** |

**France vs LOCO veto rate** (vetoed pairs per accepted pair):
| | France | LOCO US→India | LOCO India→US |
|---|---|---|---|
| rescue+gc | 0.14% | 0.70% | 0.16% |
| rescue+gc+hn | 0.33% | 1.31% | 0.32% |

- France looks like India→US, the direction that has predicted every France LB gain so far (P37 / P42). That is why the expected LB uses India→US.
- Label-free check: vetoed France gc pairs are 9% of the gc stage. Their same-house-number share is 0.57, vs 0.60 for the stage, so the veto does not target same-address pairs.

## What failed      (approaches tried and dropped, with the number that killed them)
- **Out-of-band veto (g ≥ 0.98, r between 0.1 and v):** those pairs are 73–86% true in LOCO. τ 0.1 already takes the veto gain; −0.0009 / −0.0001 at 0.2.
- **Full-range OOD blend:** it shrinks every g ≥ 0.98 pair and reorders the base stage; −0.0126 / −0.0042. On the low side it cannot reach any bar.
- **Low-side rescue (p < 0.02, CE > u):**
  - The in-domain ceiling is only +0.0012 / +0.0013, even at 99% precision.
  - With the honest source-only CE, India→US precision is 0.36–0.61 in the "any" scope and 0.16–0.36 for empty S1, i.e. negative. The best cell (one-sided, u 0.995) gives +0.00019 / +0.00010, under the gate.
  - The France low-side scores (118,927 pairs, `data/work/p48/r_lo_France.parquet`) are fetched but unused.
- **hn-extras veto:** ≈ 0 in chain India→US (76% of the vetoed pairs are true). This fits P37: same-address pairs with invented names are true, and a text CE underrates them.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `prep48.py loco`: P45's r_all joined onto P42's LOCO frames, with ids → `data/work/p48/loco_*_ids.parquet` (4 s).
- `prep48.py sm`: the GPU channel. `prep48.py fetch JOB`: `r_lo_{US_India,India_US,France}.parquet`.
- `loco48.py [veto resc blend vdec vfine vstage]`: `loco_<arms>.json` (3 s, < 0.6 GB). It reuses P42's `solution_p42` (decide, density map, fbeta).
- `frcount.py BASE`: France veto counts by stage. `build48.py BASE OUT veto=V[,stages=a+b][,rescue=U,scope=S]`. `check48.py BASE OUT`.
- GPU: `solution_p48.py` (P13's bge_ckpt path, run once per ckpt) + `launch48.py NAME INSTANCE`.
  - Job p48-ce-0927-0412 on ml.g5.2xlarge: 399 billable s (≈ $0.17), 1,596 pairs/s.
  - Validators p48-val / p48-valb on spot m5.xlarge: 39 + 45 s.
  - **Total AWS ≈ $0.25.**

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **No new model or data.** bge-reranker-v2-m3 (Apache-2.0) was already shipped. No country filter: the rule keys on the unseen-country recipe stages.
- **Assumption:** France = the P42 part of sub46_on43's recipe (t 0.99 / rescue 0.90 / gc 0.70 / hn 0.80), rebuilt from `france.npz`. All 835,531 recipe pairs are in sub44r's France lines; the other 9,292 are tier pairs and are never touched.
- **CE mismatch:** LOCO uses P30's source-only CE and France uses the pooled CE (trained on India + US). The pooled CE may be sharper on France than the source-only CE is on the target, which would favour the veto; the raw threshold r < 0.2 is shared.
- **Size of the gain:** India→US is only 3 SE on h1 (+0.00035 ± 0.00012). The US→India reading is 20× larger because that analogue's low bars let in many false pairs.
- **Empty S1:** 37 S1 become empty (all are rescue-stage pairs whose r < 0.2). In LOCO those rescue vetoes are 70–88% false.

## Open questions   (for the master; each with your best guess)
- **sub48 or sub48_all?** sub48 is the safer one: it is ≥ sub48_all on chain India→US. sub48_all wins on the other three readings. If sub48 gains on the LB, sub48_all is the next single-lever test.
- **Should the veto cover tier pairs (gib / xf / swap; 3,279 of them are band pairs with r < 0.2)?** No. Those tiers exist to add pairs the models underrate, and LOCO cannot test them.
- **Should the veto cover rule-promoted base pairs (1,030)?** Not without an analogue that has the rules; P42's τ grid treated rule-exempt pairs the same way.
