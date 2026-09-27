# P56-thresh-grain — RESULT (final, 27 Sep 05:35 IST)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **Finer threshold grain does not pay, in India/US or in France.** The best India/US cell is t by greedy group state
  (0 / 1 / 2+ accepted higher pairs, per country): **+0.000028 ± 0.000057 (h0→h1) / +0.000053 ± 0.000053 (h1→h0)**.
  That is ~1 SE, against a gate of +0.0001 at 3 SE. Per-source t is negative both ways, and the fit-free F0.5 break-even
  bars are +0.000034 / +0.000008.
- **France per-source bars (182 LOCO cells) and group-size bars for gc / hn (84 cells):** no cell has chain US→India ≥ 0
  and India→US ≥ +0.0003. The best admissible cells read India→US **+0.00003** (rescue S2 0.90 / S3 0.95) and
  +0.00001 (gc 0.5 for 1-member groups, 0.9 for 2+).
- **Recommendation: adopt nothing. No sub56.** sub44r's p'' is calibrated in every source × group-state cell, so a cell bar
  can only reproduce t 0.71.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- **Nothing.** No output/sub56, no validator run, no SUBMISSIONS.md row, nothing sent to P50.
- Keep India/US at one t 0.71 on the combiner p''. Keep France at P42's pooled bars (t .99 / rescue .9 / gc .7 / hn .8).

## Evidence         (table: ledger name | world | key metrics | verdict)
**India/US frame.** P44's holdout A frames (#8c argmax pairs). p'' comes from sub44r's combiner and is out of fold on both
halves: half-1 S1 use P44's half-0 fit (ho_p44r), half-0 S1 use P49's swapfit (h1fit). The base is p'' ≥ 0.71 everywhere,
and every arm reproduces it with all cells at 0.71 (asserted).
- Deltas are per-S1 F0.5 over the report half's S1, with paired SE. LB-eq = 0.468 India + 0.383 US.
- Bars are fit by coordinate descent on the other half, on a grid of 0.30–0.98 in steps of 0.01.
- Greedy state = the # of the S1's higher-p'' pairs already accepted: same source (rk_same) or both sources (rk_any).
  A pair below its cell's bar blocks the lower pairs of its group.

| ledger name | world | key metrics (LB-eq, fit h0 → report h1 / fit h1 → report h0) | verdict |
|---|---|---|---|
| arm1-src_pooled | holdout, sub44r OOF | t (S2, S3): h0 fit (0.77, 0.71) **−0.000027 ± 0.000019**; h1 fit (0.68, 0.70) −0.000014 ± 0.000018 | drop |
| arm1-src_percountry | holdout | −0.000096 ± 0.000035 / −0.000016 ± 0.000016 | drop |
| arm1-rk_same_pooled | holdout | t (0, 1, 2+) = (0.72, 0.81, 0.80) / (0.68, 0.75, 0.75): −0.000016 ± 0.000019 / +0.000002 ± 0.000023 | drop |
| arm1-rk_same_percountry | holdout | −0.000058 ± 0.000026 / +0.000007 ± 0.000017 | drop |
| arm1-rk_any_pooled | holdout | (0.61, 0.72, 0.80) / (0.37, 0.70, 0.75): −0.000032 ± 0.000042 / −0.000052 ± 0.000057 | drop |
| arm1-rk_any_percountry | holdout | India (0.35, 0.68, 0.80) / (0.31, 0.66, 0.74), US (0.58, 0.78, 0.78) / (0.58, 0.68, 0.75): **+0.000028 ± 0.000057 / +0.000053 ± 0.000053** (≈ 1 SE) | drop |
| arm1-src_rk_same_pooled | holdout | 6 bars: −0.000091 ± 0.000037 / +0.000018 ± 0.000021 | drop |
| arm1-src_rk_same_percountry | holdout | −0.000031 ± 0.000045 / −0.000025 ± 0.000026 | drop |
| arm1b-theory | holdout (no fit) | bars q*(k) = 0.5, 0.727, 0.759, 0.771, 0.778, … (the exact F0.5 break-even with k true, complete members): h0 +0.000034 ± 0.000045, h1 +0.000008 ± 0.000050 | drop |
| arm1b-theory_hi | holdout (no fit) | q*(k) for k ≥ 1, k = 0 stays 0.71 (raise only): +0.000030 ± 0.000017 / −0.000001 ± 0.000017 | drop |
| arm1b-theory_lo | holdout (no fit) | k = 0 at 0.5 only (empty-S1 fill): +0.000004 ± 0.000041 / +0.000009 ± 0.000047 | drop |
| (diag.py) | holdout | precision by p'' band × state × source tracks p'' in every cell (India [0.71, 0.76): S2 0.75, S3 0.79; US 0.74 / 0.70; k = 2+: 0.75 / 0.72). The band is thin: [0.66, 0.76) holds 340 India and 719 US pairs out of 354k / 525k. State-0 (empty-S1) pairs below 0.71 number 52 India / 121 US | — |
| (ceiling.py) | holdout | label oracle inside the band, base elsewhere: [0.3, 0.97) +0.00185 / +0.00184; [0.5, 0.9) +0.00112 / +0.00103; [0.66, 0.76) +0.00027 / +0.00024 LB-eq | — |
| arm2-fr-per-source-bars | LOCO (P47 frames, sub43_r90 recipe analogue) | 182 cells (t, rescue, gc, hn; S2 ≠ S3 bar). h0-admissible (chain both ways ≥ 0): 5, all rescue splits. Best: rescue (S2 0.90, S3 0.95), chain **+0.00117 / +0.00003** (h1 +0.00114 / 0.0000), m2 +0.00024 / +0.00001. Best India→US chain: t (0.99, 0.9925) +0.00077, but US→India −0.00155. gc and hn splits: India→US ≤ +0.00007 | drop |
| arm2b-fr-group-state-bars | LOCO (same frames) | gc / hn bar by group size before the stage (1 vs 2+), 84 cells. h0-admissible: 2. Best: gc (1: 0.5, 2+: 0.9), chain −0.00006 / +0.00001 (h1 −0.00022 / +0.00001), m2 −0.00014 / +0.00017. Max-min chain: gc (0.7, 0.8) +0.00023 / −0.000003; hn (0.9, 0.8) +0.00008 / −0.00005 | drop |

## What failed      (approaches tried and dropped, with the number that killed them)
- **India/US per-source t.** The fitted S2 bar flips between halves (0.77 vs 0.68), and the result is negative both ways.
  S2 and S3 pairs in the same p'' band are equally precise (the diagnostic).
- **India/US group-state t, fitted.** Every variant is within ±1.5 SE of 0, and the 6-bar source × state variant loses
  −0.00009 at h0→h1. The fitted bars do lean the way the theory says: they rise with k (0.68 → 0.80), and India's k = 0 bar
  drops to 0.31–0.35. That is the empty-S1 fill (P54's arm), on 52 pairs.
- **The exact F0.5 break-evens (no fit).** +0.00003 / +0.00001. The combiner already sees the group through its columns
  (n_same_src_acc, s1_psum, s1_rank, s1_gap_*). Its p'' is calibrated per state, so the extra bar margin moves only a few
  hundred pairs whose precision already sits at their break-even.
- **France per-source bars.** Base-t splits trade chain US→India against India→US (−0.0016 for +0.0008). The pooled
  0.9925 that P42 rejected dominates every split on m2. Rescue splits are positive only US→India, and gc / hn splits are
  ≤ +0.00007 India→US.
- **France group-size bars (gc / hn by 1 vs 2+ members).** The 2+ bar can only trade US→India for India→US
  (gc 0.7 / 0.8: +0.00023 / −0.000003). No cell reaches +0.0001 India→US with US→India ≥ 0.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
All scripts run from the repo root with `probes/P56-thresh-grain/py.sh` (a 1.2 GB cap).
- `arm1.py [src|rk_same|rk_any|src_rk_same ...]`:
  - `load(c)`: the OOF frame, sorted by S1 and p'';
  - `cells(d, kind)`, `accept(d, thr, key)` (greedy, with a cached sort), `score`, `fit` (coordinate descent), `report`.
  - Takes 15 s (src) to 2.5 min (6 bars) per kind and ≤ 0.3 GB. A self-test asserts that all bars at 0.71 give the base,
    plus a greedy toy.
- `arm1b.py`: `qstar(k)` (asserted) and the three fit-free variants. 20 s.
- `diag.py`, `ceiling.py`: the calibration table and the oracle ceiling. 20 s each.
- `arm2.py`: France per-source cells. `bars(fr, cell)` turns France bars into per-row LOCO bar arrays;
  `solution_p42.decide` broadcasts them unchanged. It asserts the recipe reproduces P47's pred. 96 s for 182 cells × 4 frames,
  ≤ 0.5 GB.
- `arm2b.py`: France gc / hn bars by group size m before the stage (1 vs 2+), on the same frames. It asserts the recipe
  reproduces. 48 s for 84 cells.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- Nothing ships. numpy only; no external data.
- **Ledger:** arm1 rows at 05:08–05:18 come from a first run that was killed and relaunched with a cached sort. They are
  identical duplicates of the 05:20–05:2x rows (deterministic).
- **Gate reading:** "India+US" is taken as LB-eq with P44's weights (0.468 / 0.383).
- **Base:** the test ships sub50e (the full-holdout refit), while the holdout reads the two half-fit combiners. A bar that
  failed on half fits would not start passing on a refit.

## Open questions   (for the master; each with your best guess)
- Is any decision-layer lever left on India/US? Best guess: not by threshold grain. The +0.0018 in-band oracle is out of
  reach for any rule that sees only calibrated p'' and group counts. It needs new evidence per pair (a better model).
