# A07 — France: add free argmax SX whose normalised key equals an ACCEPTED copy's key (sibling-copy key) — RESULT

## TL;DR
- **Rule (R0)**: Add candidate SX that share a normalised key `(core, hn, city)` (with non-empty `core`) with an accepted SX belonging to the same S1 `ow` (where the key group has accepted members exclusively belonging to `ow`), provided the candidate is free, `argmax`, and `p >= 0.5`.
- **STRICTER GATE Check**: The rule operates at a high precision but falls short of the `0.95` threshold for a "first tier" shippable lever in both directions. The `p >= 0.5 resid` tier achieves **.932** in target India and **.972** in target US, qualifying it as a strong **"second tier"** (reportable as .90–.95). 
- **LOCO Delta (p>=0.5 resid)**: US→India (target India) dF **+0.001938**; India→US (target US) dF **+0.001103**.
- **Free-share diagnostic**: France free pool is 6,491 (0.025/S1). LOCO free pools are 5,690 (0.065/S1) for target India and 3,805 (0.029/S1) for target US.
- **Expected LB**: Mean dF (0.00152) × 0.149 × Scaling (mean of 0.025/0.065 and 0.025/0.029 = 0.627) = **+0.00014**.
- **Vetoes & hn_cls split**: The France pool (6,491) is overwhelmingly `hn_cls` 0 (hn-equal): {0: 6219, 1: 100, 4: 95, 3: 59, 2: 18}. (No local N54 decoy files were available).

## Recommendation
This is a **second tier** lever. It adds 3,558 pairs over 3,352 S1s in France using the `R0_p0.5` tier.
Staged in `output/sub76_n69_a07`.
Run:
```bash
.venv/bin/python probes/A07-fr-sibling-copy-key/a07.py build output/sub76_n69 output/sub76_n69_a07 R0_p0.5
```
Validator: PASS (plain), 1,732,544 rows.

## Evidence
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| A07-r0-p0.5-resid | N63 chains | US→India prec 0.932 (pool WLB 0.923) dF +0.001938; India→US prec 0.972 (pool WLB 0.964) dF +0.001103 | second tier |
| A07-r0-km=1 | N63 chains | US→India prec 0.941 (pool WLB 0.933); India→US prec 0.923 (pool WLB 0.911) | second tier |
| A07-build | test-France | 3,558 pairs added over 3,352 S1s for `R0_p0.5`. Validator plain PASS. | staged |

## What failed
- `p >= 0.8` achieves 1.000 in target US but drops the candidate pool to a microscopic 36 pairs (and prec 0.938 on 2,761 pairs in target India, missing the >0.95 mark).
- `R1` (using K2 = `(core, city)` for groups where all members have empty hn) achieved poor precisions: 0.737 in target India, 0.896 in target US.

## MASTER ADDENDUM (27 Sep ~20:45, supersedes the recommendation above)
- **Bug in gate.py tiers:** `~hn_empty` ran on an object Series, so "hn non-empty" returned the whole pool, and "src same/diff" is broken
  too. Re-run with the fix: the hn-EMPTY key part is the poison (India target .667, dF < 0).
- **Ship tier `R0_hneq`** = R0 + key hn non-empty + SX hn == S1 hn (97% of the France pool; the P73/N62 hn-diff veto classes are
  excluded by construction). LOCO (N76 template, caps, halves):
  | cell | India→US (target US) | US→India (target India) |
  |---|---|---|
  | R0 hn non-empty, no sibling | 3,475 adds, prec .970, dF h0 +.00162 / h1 +.00179 | 4,199 adds, prec .984, dF +.00341 / +.00330 |
  | **R0 hn non-empty, hn == S1** | **2,600 adds, prec .977, dF +.00127 / +.00134** | **339 adds, prec .917, dF +.00022 / +.00019** |
  | R0 hn empty (dropped) | 183, .951 | 1,237, .667, dF −.00007 / −.00024 |
- France: 5,103 pool pairs → **+4,991 pairs on 4,314 S1** (caps, candidates) on sub76_n69. N54 decoy-word hits 0. Pool CE r is low
  (33.5% r < .2), matching LOCO's low-p slice, which stayed ~.97 precise.
- Expected: ~.06 France-F per added pair (LOCO: US .066, India .053) → +4,991 × .06 / 259,452 ≈ **+.0012 France-F ≈ +0.00018 LB**.
- Build: `.venv/bin/python probes/A07-fr-sibling-copy-key/a07.py build <base> <out> R0_hneq` (owners come from the base's own lines,
  so run it on sub84 directly). `output/sub76_n69_a07h`: validator PASS, France-only, additions only, 0 double assignments.
- Open risk: vetoed pairs sitting free in the base (P48/P55/P73/R3 removals) can be re-added. Subtract those lists before shipping.
- `gate_master.py` = gate.py with the hn-empty fix and the combined tiers (the table above).
- **Not guaranteed.** Honest range ≈ 0 to +0.0002 LB, most likely ~+0.0001. Risks: the LOCO→LB record (sub84 came in below its
  estimate; sub77_hnst lost), the India-like cell is thin (339 adds, .917), France has ~1.7× the decoys per S1, and the build was
  on sub76_n69, not sub84. Break-even France precision ≈ .75–.80 (a true add ≈ +.06 F on its S1, a false one ≈ −.15–.2).
