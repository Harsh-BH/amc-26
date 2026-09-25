# P17-hn-stack — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **Round 2 (bge base, #8c):**
  - Stacking both countries on P13's bge blend does **not** pass: +0.00028 ± 0.00006, and India is
    **−0.00013 ± 0.00006**. On the stronger CE, India has nothing left to gain (half-swap India +0.00002).
  - **US-only passes**: India+US **+0.00033 ± 0.00005** on half 1 and +0.00046 ± 0.00005 on the swap. US alone:
    +0.00055 ± 0.00009 / +0.00076 ± 0.00009. Over #7 that is ≈ +0.00088.
- **#8c is staged and the validator passes**: `data/work/p17/sub8c/`.
  - US = bge + stack; India and France = #8a's.
  - It differs from sub8a on 11,230 US S1 only, and the candidates are byte-identical.
  - Expected LB over #8a ≈ 0.85 × 0.45 × 0.00055…0.00076 ≈ **+0.0002–0.0003**.
- **France: do not stack it now.**
  - Leave-one-country-out (LOCO) in France's no-CE regime transfers (US→India +0.00073 ± 0.00021, India→US
    +0.00203 ± 0.00015).
  - On France test, though, it reshuffles 4.7–8.1% of S1 and barely filters siblings.
  - Round 1 (#7 base, e5 CE): +0.00067 ± 0.00007, merged as #8p.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
**#8c = bge + US stack.** It is already built in `data/work/p17/sub8c/` (validator PASS). To rebuild it:
```
.venv/bin/python code/business_entity_resolution/src/submit_probs.py --probs 'data/work/p17/prob_test_{country}_bge.parquet' \
    --out output/sub8c --t 0.71 --t-country France=0.95
```
The three input files:
- `prob_test_US_bge.parquet`: s1, sx, p = the stacked p over #8a's US p. Same pruned rows as #7 / #8a, in #7's
  m2-cache order.
- `prob_test_India_bge.parquet` → `../p13/prob_test_India_blend.parquet`: #8a's India, **unstacked**.
- `prob_test_France_bge.parquet` → `../p10/prob_test_France.parquet`: #8a's France, #6c / #7 at 0.95.

Stacker details:
- Booster: `data/work/p17/stack_sup-bge-l21.0_US.txt`. Same recipe as round 1: 200 × 31, lr 0.05,
  min_data_in_leaf 200, lambda_l2 1, init_score = logit p, fit on US holdout S1 of `run.half == 0`.
- Rebuild: `stack.py holdout sup base=bge t=0.71 l2=1`, then `stack.py test sup base=bge l2=1 t=0.71 only=US`.
- t stays 0.71. On the report half the US curve is flat from 0.61 to 0.81 (+0.00050 to +0.00057).

**Why US only.** India's stack on the bge base is −0.00013 ± 0.00006 on the report half and +0.00002 ± 0.00005 on the
swap half. Neither half gains, and the report half fails the brief's India ≥ −0.0001 bar.
- This choice uses both halves. The US numbers are unaffected, because India contributes exactly 0 whether or not it
  is stacked. So +0.00033 is the report-half value of the shipped config.

**Upload order.** #8c should follow #8a, since it is #8a plus one US-only change: the LB delta #8c − #8a measures the
stack directly. If #8b (bge France) wins its A/B, rebuild #8c with `--probs` France pointed at P13's France blend at
0.98: the stack does not touch France.

**run.py integration (unchanged proposal from round 1).** Add a stage after `rerank()` for the countries listed in
the recipe's `stack` key, which is now `{"US": "p17/stack_sup-bge-l21.0_US.txt"}` with the bge CE. Support columns are
computed on the shipped (pruned) list.

## Evidence         (table: ledger name | world | key metrics | verdict)
**Round-2 yardstick.** `stack.frame` with `base=bge` starts from #7's frame (m2 p on the τ 0.03 pruned list; #7's p for
every row), then:
- **Holdout-S1 band pairs** are re-blended with P13's bge r (`r_bge_poolck_holdout`, w [0.5901, 0.6319, 0.2973]).
  All of them are covered: India 59,094 of 59,094, US 95,494 of 95,494.
- **Competing train-S1 band pairs** (India 218,740, US 351,325) keep #7's e5 p, because P13 did not score them.
  On test every band pair is bge.
- Same scoring as round 1: per-S1 F0.5 over all holdout S1, `run.half`, paired SE, t 0.71.

The bge base gains over #7 **+0.00055 ± 0.00007** on half 1 (P13's own yardstick: +0.00059). check7 confirms the test
bge base reproduces sub8a exactly (India 2,733,906 / US 2,241,658 matches).

| ledger name (`P17-hn-stack/…`) | world | key metrics | verdict |
|---|---|---|---|
| hn-stack-sup-on8a-bge-l21.0-t0.71 | half 1, base bge, both countries | vs bge base **+0.00028 ± 0.00006**; **India −0.00013 ± 0.00006** (148 up / 87 down S1; negative at every t 0.61–0.85); US +0.00055 ± 0.00009 (726 / 258). vs #7: base +0.00055 ± 0.00007, stack +0.00082 ± 0.00009 | fail (both) |
| hn-stack-sup-on8a-bge-l21.0-swap-t0.71 | half 0 (fit half 1), base bge | vs bge base +0.00046 ± 0.00006; India +0.00002 ± 0.00005; US +0.00076 ± 0.00009. vs #7: base +0.00068, stack +0.00115 ± 0.00009 | India ≈ 0 |
| **hn-stack-sup-on8a-bge-USonly** | both halves, US stacked, India = bge base | **half 1 +0.00033 ± 0.00005; half 0 +0.00046 ± 0.00005**; India 0 by construction. Test: 11,230 US S1 change vs sub8a (US accepted 2,241,658 → 2,245,739). sub8c validator PASS, candidates = sub8a's | **ship as #8c** |
| (check7.py base=bge) | test | bge base rebuild = output/sub8a matching_results, India and US identical | yardstick valid |
| hn-stack-loco-m2base | holdout half 1, m2 base (France's no-CE regime), target t = its own base optimum, kept | India→India +0.00148 ± 0.00019, **US→India +0.00073 ± 0.00021**; US→US +0.00273 ± 0.00016, **India→US +0.00203 ± 0.00015** | transfers |
| hn-stack-loco-m2base (France test) | France test, #7's France p (m2), label-free | India model at t .95: accepted 776,184 → 779,896, **13,790 S1 (5.3%) change**; +SIB 2,788 in / 3,176 out. US model at .95: → 787,996, **21,015 S1 (8.1%)**, +SIB 3,785 / 3,806, none/eq +10,369 / −2,441. Count-matched t (keeps #7's 776,184 accepts): India model t .9557, 12,288 S1 (4.7%), +SIB net −806; US model t .9641, 16,927 S1 (6.5%), +SIB net −881, far +1,481 / −17 | not for #8c |

**Round 1 (#7 base, e5 CE; merged as #8p):**

| ledger name (`P17-hn-stack/…`) | world | key metrics | verdict |
|---|---|---|---|
| hn-stack-sup-on7 | half 1, brief A exactly (t tuned on half 0) | +0.00057 ± 0.00008; India +0.00003, US +0.00092 (t .77) | pass |
| hn-stack-sup-on7-swap | half 0 | +0.00078 ± 0.00008 | confirms |
| hn-stack-hn-on7 | half 1, hn-only | +0.00051 ± 0.00008; India +0.00013, US +0.00076 | fallback |
| hn-stack-brc-on7 | half 1, + brief C branch words (P12 train list) | at equal t ≈ sup; column gain ~1% | drop |
| hn-stack-tcurve | both halves | US tune-half curve flat 0.61–0.79 | t 0.71 kept |
| hn-stack-sup-on7-l21.0-t0.71 | half 1, t .71, lambda_l2 1 | **+0.00067 ± 0.00007**; India +0.00008, US +0.00107 | shipped (#8p) |
| hn-stack-sup-on7-l21.0-swap-t0.71 | half 0 | +0.00082 ± 0.00008 | confirms |
| hn-stack-l2-resid-shift | holdout-S1 rows vs test | flip rate 0.57% / 0.70% (US); +SIB share 9.8% → 17.4% | in range |
| hn-stack-sub8-dryrun | test | 19,715 S1 vs #7, validator PASS (→ output/sub8p) | staged |

## What failed      (approaches tried and dropped, with the number that killed them)
- **India stack on the bge base:** −0.00013 ± 0.00006 (half 1), +0.00002 (half 0). bge's India band AUC is 0.971 (#5's
  e5: 0.955, P13), so the hn/support residual has nothing left to fix. Even on e5, India was only +0.00003…+0.00013.
- **The stack on France, as a drop-in at t 0.95:**
  - It *loosens* France: +3.7k (India model) / +11.8k (US model) accepts. The LB showed France already over-matches
    (D2/D4: stricter is better).
  - Its +SIB effect is near zero: new accepts ≈ new rejects.
  - Count-matched, it still reorders 4.7–6.5% of France S1, 3–4× the in-domain US churn (1.7% of S1 on #8c).
- **Round 1:**
  - Brief C branch words: no signal.
  - The half-0 t retune: a noisy pick on a plateau.
  - Unregularised residual boosting: max |leaf| up to 48,320 → lambda_l2 = 1.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `stack.py holdout <arm> [base=bge] [swap] [t=0.71] [l2=1]`
  - Writes `holdout_<tag>.json`, where `<tag>` = arm[-bge][-l2…][-swap][-t…]. With `base=bge` the json adds
    `<c>_vs7` / `all_vs7` (the bge base and the stack vs #7).
  - Writes the booster `stack_<model tag>_<c>.txt` and a ledger row.
  - 3–7 min, peak 2.1 GB.
- `stack.py test <arm> [base=bge] [l2=1] [t=0.71] [only=US]`
  - Writes `prob_test_<c>[_bge].parquet`. 5 min per country, peak 1.6 GB.
- `stack.frame(split, c)`:
  - `BASE` = `e5` (#7), `bge` (#8a; adds `p7`, `has_bge`, `band`), or `m2` (no CE, France's regime).
  - A country without a CE cache (France) gets p = m2 p.
- `loco.py`:
  - LOCO in the m2 regime plus the France test diagnostic (at t 0.95 and count-matched).
  - Saves the no-CE boosters `stack_sup-m2-l21.0_{India,US}.txt`, which are the France candidates.
  - Writes `loco.json`. 6 min, peak 2.1 GB.
- `check7.py <sub dir> [base=bge]`: rebuild vs a staged matching_results.
- Unchanged from round 1: `tcurve.py`, `resid.py`, `branch.py`.
- Scratch in `data/work/p17/`: prob files (85 + 65 + 65 MB), `sub8c/` (227 MB; delete after copying to output/), and boosters.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Competing rows are mixed in the bge holdout.** Competing train-S1 band rows carry e5 p while holdout-S1 rows carry
  bge p. The argmax competition therefore mixes two CEs. Test is all-bge.
  - The effect is the same in both arms, and P13's yardstick (no competitors at all) gives the same base delta
    (+0.00059 vs my +0.00055). So the mixing is not moving the comparison.
- **Holdout spend:** the US booster is fit on half 0; India was gated using both halves (see Recommendation).
- **Rules:** LightGBM (MIT) only, with no new dependency and no external data. bge-reranker-v2-m3 is Apache-2.0 and
  568M, P13's.
- **France:** the transfer evidence (LOCO) is at the target's own base-optimal t near 0.70. France's t 0.95 sits where
  the stack shifts calibration (round-1 US median residual +1.4 to +2.0 logits). France is also further out of domain than the LOCO pairs
  (P13: CE–GBDT disagreement 31% vs 20–26%).

## Open questions   (for the master; each with your best guess)
- **Could the stack be applied to France?** Label-free argument: not now, and at most one late LB slot after #7s.
  - *For:* in France's no-CE regime the stack transfers across countries at a kept, base-optimal t (US→India
    +0.00073 ± 0.00021, India→US +0.00203 ± 0.00015). The India-trained model generalises better (74% of in-domain
    vs 49%) and is the gentler one on France.
  - *Against:*
    - At France's t 0.95 it loosens France, which the LB says over-matches.
    - Count-matched, it still reorders 4.7–6.5% of France S1, with a net +SIB effect of only −0.8k pairs. P15's rule
      removes 4.4k. So it is not the sibling filter France may need, and its large reordering is unvalidated.
  - *Decision rule once #7s (P15's France sibling rule) is scored:*
    - **#7s gains** → France's siblings are FPs and the rule captures that. The stack does not: keep the rule, no
      stack.
    - **#7s loses** → France is India-like. The only defensible variant is the India-trained no-CE booster
      (`stack_sup-m2-l21.0_India.txt`) at the count-matched t 0.9557, one LB slot, expected ±0.0002 LB.
      `loco.py` + `stack.stacked` build it in 5 min.
- **Stack India again if a later India CE changes?** Only if it is re-measured. The India gain went from about +0.0001
  (e5) to ≤ 0 (bge), so it is CE-dependent.
- **Combine with #8b (bge France)?** Yes. #8c touches only US p, so pointing its France at P13's France blend at 0.98
  gives #8b + US stack.
