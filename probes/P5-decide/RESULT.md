# P5-decide — RESULT
## TL;DR
**MERGE-2 (blended p from P4 M2 + the P4b cross-encoder): keep the threshold, do not adopt the stack.** On the blended
p, the D7 stack refit scores 0.9882 against 0.9881 for the threshold fallback. That is +0.0001 ± 0.0001, below the
0.001 bar. Use `{"excl": True, "t": 0.70}` for India/US; `decide.FALLBACK` gives the same F0.5 to 4 decimals. France
stays on P6's threshold of 0.82. See "Blend refit" below.

The earlier result, on P4's M2 holdout probabilities: the best decision layer scores dense F0.5 **0.9824**, against 0.9814 for P4's
argmax + global threshold 0.71 (+0.0010 ± 0.0001 paired SE, report half, 110k S1). It chains four steps: a small
learned second stage on each pair's standing among its S1's candidates, exclusivity on P4's all-S1 argmax, then an
expected-F0.5 top-k (k ≥ 1) behind an empty-answer gate.
Every threshold refinement (D1, D3–D5) lands within ±0.0002 of the plain threshold. Pure EF (D2) and sibling
expansion (D6) both hurt. Adopt `decide.CFG`, and fall back to `decide.FALLBACK` (0.9815) whenever the matcher changes
and the stack has not been refitted yet.

## Recommendation
**Current matcher (MERGE-2 blend):** answer = exclusivity on `am` + global threshold 0.70 on the blended p, i.e.
`decide(pairs, {"excl": True, "t": 0.70})`, or `decide.FALLBACK`, which scores the same. Nothing new needs to be
promoted. The refit stack is kept as `stack_blend_s1only.txt` / `cfg_blend_s1only.json` but is not adopted.
The stack recommendation below applies only if the matcher reverts to M2 alone:
```python
sys.path.insert(0, "probes/P5-decide"); from decide import decide, CFG, FALLBACK
sets = decide(pairs, CFG)    # pairs: DataFrame[s1, sx, p, am] -> {s1: set(sx)}; S1 absent from pairs -> empty
```
- `CFG = {"excl": True, "mode": "ef", "t0": 0.55, "cap": 24, "stack": "probes/P5-decide/stack_p4_s1only.txt",
  "stack_cols": S1_CTX}`. The same cfg is in `cfg_p4_s1only.json`.
  1. D7: LightGBM, 300 trees × 31 leaves, on 15 S1-side context columns computed from (s1, sx, p) alone:
     p, rank, candidate count, max p, p/max, gap, sum p, count ≥ 0.5 and ≥ 0.2, neighbouring p's, 2nd–5th p.
  2. D1 exclusivity, taken on the matcher's p.
  3. Gate: an S1 whose best stacked p is below 0.55 gets an empty answer.
  4. Otherwise, the top-k set by stacked p that maximises expected F0.5, with k ≥ 1.
- **`am` column**: pass P4's argmax flag. With it, every S1 is decided from its own rows, so test can be run in S1
  chunks (`test_diag.py` does this) and the result is exact. Without it, `decide` takes the argmax inside the frame,
  so give it all pairs of a country in one call.
- **The stack is tied to M2's p.** If P4 ships a new model (e.g. its stage 2), rerun
  `bench_stack.py p4 D7-stack-s1only` (~25 min). It rewrites `stack_p4_s1only.txt` and `cfg_p4_s1only.json`; update
  `t0` in `decide.CFG` from the json. Until then use `FALLBACK = {"excl": True, "t": 0.71, "t0": 0.72}`.
- No per-country thresholds. France gets the same cfg: D4 picks t 0.71 for both countries anyway.

## Evidence
Setup for every row:
- Report = hash half 1 of the holdout S1 (110,362 S1). Tuned on half 0.
- Full train corpus of S2/S3 (10.3M), and every train S1 competes. This is dense, full-density scoring.
- True pairs missing from the candidates count as misses.
- F0.5 comes from `bench.py`'s vectorised scorer. `metric.score` on `decide()` output matches it exactly
  (checked for the proxy's 0.9229 and for P4's 0.9824 and 0.9815).

### Blend refit (MERGE-2, `data/work/p4b/prob_holdout_blend.parquet`, 15:05)
Same protocol as below: tune on half 0, report on half 1 (110,362 S1), exclusivity on `am`. Paired SE is taken
against the best threshold rule on the tune half (the strictest baseline). Script: `blend.py`; the stack was refit
with `bench_stack.py blend D7-stack-s1only`.
| ledger name | report F0.5 | vs (a) retuned | singleton acc | pair P / R | India / US |
|---|---|---|---|---|---|
| blend-a-t0.70 (P4b's threshold) | 0.9881 | +0.0000 ± 0.0000 | 0.9876 | 0.9975 / 0.9695 | 0.9888 / 0.9877 |
| blend-a-FALLBACK (t 0.71, t0 0.72) | 0.9881 | −0.0000 ± 0.0001 | 0.9877 | 0.9976 / 0.9693 | 0.9888 / 0.9876 |
| blend-a-retuned (t 0.68, t0 0.65) | 0.9881 | — | 0.9851 | 0.9974 / 0.9699 | 0.9888 / 0.9877 |
| **blend-b-CFG-refit** (stack + excl + EF + gate t0 0.40) | 0.9882 | **+0.0001 ± 0.0001** | 0.9814 | 0.9978 / 0.9692 | 0.9887 / 0.9879 |
| blend-b-M2-stack-unrefit (for contrast) | 0.9882 | +0.0001 ± 0.0001 | 0.9879 | 0.9980 / 0.9686 | 0.9888 / 0.9878 |
| (bench_stack) D7-stack-s1only + threshold | 0.9883 | | 0.9831 | 0.9978 / 0.9696 | 0.9888 / 0.9879 |
| blend-verdict | | b − a = +0.0001 < 0.001 bar | | | drop the stack, keep the threshold |

The stack's gain shrinks from +0.0010 on M2's p to +0.0001 on the blend. My reading, untested: the cross-encoder
already fixes the uncertain band where the stack helped.
Where the remaining loss sits (t 0.70, report half):

| source of loss | F0.5 lost |
|---|---|
| partial answers (93% of these S1s miss a true pair; 7% carry a wrong one) | 0.0089 |
| empty answers on non-singletons | 0.0023 |
| answered singletons | 0.0007 |

So the headroom is matcher recall, not the decision rule. The blended file's `am` differs from P4's on 1.7% of pairs,
i.e. it was recomputed on the blended p, and it was used as given.

### P4 M2 probabilities (`data/work/p4/prob_holdout.parquet`, 13:47) — the M2-only stack recommendation rests on these
| ledger name | report F0.5 | singleton acc | pair P / R | India / US | notes |
|---|---|---|---|---|---|
| p4-D0-global-t | 0.9814 | 0.9747 | 0.9939 / 0.9589 | 0.9802 / 0.9822 | t 0.71 (= P4's own result) |
| p4-D1-excl+t | 0.9815 | 0.9755 | 0.9941 / 0.9588 | 0.9803 / 0.9823 | exclusivity on P4's all-S1 `am` |
| p4-D3-excl+t+gate | 0.9815 | 0.9763 | 0.9941 / 0.9588 | 0.9803 / 0.9822 | t 0.71, t0 0.72 = FALLBACK |
| p4-D3r / D4 / D5 | 0.9815 | 0.9763 | | | rel 0, per-country t 0.71 / 0.71, cap 10 (inactive) |
| p4-D2-ef-* (6 variants) | 0.9806–0.9811 | 0.946–0.956 | | | EF without a gate over-answers singletons |
| p4-D2g-ef-gate-raw | 0.9816 | 0.9743 | 0.9948 / 0.9569 | 0.9805 / 0.9823 | t0 0.70; best without a stack |
| p4-D7-stack-s1only | 0.9823 | 0.9775 | 0.9953 / 0.9585 | 0.9812 / 0.9830 | stack + threshold t 0.75, t0 0.55 |
| **p4-D7-stack-s1only+ef-gate** | **0.9824** | 0.9775 | 0.9954 / 0.9585 | 0.9813 / 0.9831 | **CFG**; vs D0 +0.0010 ± 0.0001 |
| p4-D7-stack-all+ef-gate | 0.9826 | 0.9771 | 0.9954 / 0.9589 | 0.9815 / 0.9834 | not adopted (see What failed) |
| p4-final-CFG | 0.9824 | 0.9775 | | 0.9813 / 0.9831 | `decide(pairs, CFG)` + `metric.score`; FALLBACK 0.9815 |

Where the remaining loss sits (CFG, report half):

| source of loss | F0.5 lost |
|---|---|
| partial answers on non-singletons (mostly missed true pairs) | 0.0130 |
| empty answer on a non-singleton | 0.0033 |
| answered singletons | 0.0012 |

The oracle ceiling is 0.9932: the best k per S1, chosen with the labels, over P4's own ranking.

Leave-one-country-out, as a France proxy. The decision layer is fitted on country A's tune half and scored on
country B's report half:
| ledger name | fit -> score | D3 threshold | D7 S1-only + EF gate | in-country CFG |
|---|---|---|---|---|
| p4-loco-India-to-US-s1only | India -> US | 0.9822 | 0.9829 | 0.9831 |
| p4-loco-US-to-India-s1only | US -> India | 0.9803 | 0.9803 | 0.9813 |

Label-free test check (`p4-test-answer-profile`, `test_diag.py`). Each cell is empty-answer share / mean answer size
under CFG:
| | holdout India | holdout US | test India | test US | test France |
|---|---|---|---|---|---|
| CFG | 5.66% / 3.35 | 5.73% / 3.33 | 5.67% / 3.37 | 5.73% / 3.36 | 5.22% / 3.33 |

France's answer profile sits with US and India. Train's singleton share is 5.6%.

### Proxy probabilities (development only, before P4 landed)
The proxy is `proxy-lgbm-crossfit`: LightGBM on block, rule and name-similarity features, 2-fold cross-fitted over
the holdout S1 halves. AUC 0.9914, AP 0.9770.
| ledger name | report F0.5 | singleton acc | notes |
|---|---|---|---|
| (bench line) R-rules | 0.7712 | 0.717 | rules.py accepted pairs, same pairs and truth; pair P 0.916 |
| proxy-D0 / D1 / D3 / D4 / D5 | 0.9156 / 0.9156 / 0.9157 / 0.9158 / 0.9157 | 0.83–0.85 | thresholds are saturated here too |
| proxy-D2-ef-* | 0.9135–0.9138 | 0.67 | independence over-answers singletons |
| proxy-D2g-ef-gate-raw | 0.9170 | 0.829 | |
| proxy-D6-sib-core / core\|city | 0.9151 / 0.9155 | | sibling expansion hurts |
| proxy-D7-stack-s1only+ef-gate | 0.9212 | 0.829 | |
| proxy-D7-stack-all+ef-gate | 0.9229 | 0.828 | vs D4 +0.0061 ± 0.0003 (threshold version) |
| proxy-loco-*-to-* | India->US D3 0.9278 / D7 0.9283; US->India 0.8969 / 0.9004 | | never hurts out of country |
| proxy-D7-capacity | 300×31 0.9229, 600×63 0.9228, 150×15 0.9222 | | keep 300×31 |

## What failed
- **D2, EF without a gate** (Ye et al. / Jansche exact DP). On P4 it scores 0.9806–0.9811 against 0.9814, and
  singleton accuracy falls to 0.95 from 0.975. Under independence, P(no match) = Π(1−p) looks small for any S1 with
  several middling p's, but an S1's candidates are correlated. EF picks k ≥ 1 well once a gate makes the empty
  decision (D2g).
- **Isotonic calibration**: no gain over raw p, with or without the gate.
- **D4 per-country thresholds**: the tuned t is 0.71 for both India and US, so D4 is identical to a global threshold.
  France therefore needs no default of its own.
- **D5 top-m cap and D3r relative threshold**: the sweep chooses an inactive setting (m = 10, rel = 0).
- **D6 sibling expansion** (admit an SX sharing a `core` or `core|city` key with an answered SX of the same S1):
  proxy 0.9151 / 0.9155 against 0.9157. It admits more wrong look-alikes than missed duplicates.
- **D7 with SX-side context** (`stack-all`: the SX's rank among competing S1s, their count, the best competitor's p)
  is +0.0002 over S1-only on P4, but it is not adopted:
  - On holdout, a frame holds only holdout S1, so the SX side sees just 10% of the competitors. On test it sees all
    of them, so the feature distribution shifts.
  - It cannot be computed in S1 chunks.
  - P4's `sx_margin` already carries full-pool SX competition inside p.
- **Exclusivity D1** adds only +0.0001: P4's `sx_margin` feature already encodes it. On the proxy, whose p had no
  full-pool competition, the in-frame argmax could not help, because only about 14% of SX have more than one
  holdout S1.

## Interfaces
- `blend.py`: the MERGE-2 comparison, (a) threshold fallbacks vs (b) the refit CFG, with paired SE (~4 min, 1.1 GB).
  `bench.py` / `bench_stack.py` accept `blend` as the source.
- `decide.py`:
  - `decide(pairs, cfg) -> {s1: set}` and `select(pairs, cfg) -> row mask`.
  - `ctx(p, g, x) -> float32 [n, 18]` builds the context columns.
  - `ef_select(g, p, cap, miss, empty)` / `ef_k(P, miss, empty)` implement the exact expected-F0.5 DP. It is
    vectorised over S1s bucketed by candidate count and checked against brute force in `_check()`.
  - Also exports `CFG`, `FALLBACK`, `CTX`, `S1_CTX`.
  - Input columns: `s1, sx, p`, plus optional `am` (argmax flag), `country` (only used by `t_country`) and `sib` (D6).
  - `.venv/bin/python probes/P5-decide/decide.py` runs the self-check.
- **Runtime and RAM**:
  - `decide(CFG)` takes 28 s on 3.13M pairs, dict build included. Process peaks for load + select in this probe
    stayed at or below 1.8 GB.
  - Cost is linear in pairs, so test's 27.5M pairs take about 4 min.
  - Memory: the context matrix is 72 bytes/pair (18 float32), plus 60 for the S1-only column copy, plus a few float64
    temporaries of 8 bytes/pair each.
  - With `am` present, run in S1 chunks for flat memory (`test_diag.profile`).
- `bench.py proxy|p4` runs D0–D5, D2 and D2g (~28 min). `bench_stack.py proxy|p4 [variant]` runs D7 and writes
  `stack_*.txt` and `cfg_*.json` (~25 min for two variants). `bench_d6.py` runs D6, `loco.py` the
  leave-one-country-out check and `test_diag.py <cfg>` the label-free test profile.
- `proxy.py sxfeat|feats <country>` and `proxy.py fit` build the proxy probabilities. They are for development only
  and are not needed downstream.
- Outputs:
  - `data/work/p5/`: proxy features and probabilities, bench JSON, logs.
  - `probes/P5-decide/`: `stack_*.txt` (~1.1 MB each) and `cfg_*.json`.

## Risks
- **Licences / external data**: LightGBM (MIT), numpy, pandas and scikit-learn (BSD). No external data. The stack
  sees only (s1, sx, p) and never country, which keeps it France-safe.
- **Stack/matcher coupling**: `stack_p4_s1only.txt` is fitted to M2's p distribution from 13:47. A new P4 model
  without a refit could do worse than FALLBACK. Refit, or use FALLBACK.
- **France has no labels**:
  - Leave-one-country-out shows D7 never hurts out of country (+0.0007 / +0.0000).
  - Thresholds transfer almost losslessly between countries.
  - France's test answer profile (5.2% empty, 3.33 per S1) matches US and India.
  - This is still unvalidated for 15% of test.
- **Stack training data**: it was fitted on half 0 of the holdout only (1.56M pairs), so that half 1 stays a clean
  report set. Refitting on the whole holdout would add data but leave t0 unchecked.
- **Baseline comparability**: the 0.8201 rules number comes from the mini-world, which samples only about 12% of
  the look-alike S2/S3. On the same full-corpus setup, rules score 0.7712 (P4 independently: 0.7708).
- **Holdout hashing**: `half()` uses `hash >> 7`. The low hash bits are constant for these ids, and a plain `% 2`
  put every row in one bucket (caught in `test_diag.py`). Use a shift for any new hash split.

## Open questions
1. **Refit the stack for P4's stage-2 model?** Superseded by MERGE-2: on the blend it gains only +0.0001, so don't. Original answer: Best guess: yes, if its `prob_holdout` lands. It costs about 25 min
   and the expected gain is again about +0.001. Until the refit, use FALLBACK.
2. **Retrain the stack on the whole holdout for the final submission?** Best guess: no. The gain is small and t0
   would lose its out-of-fold check.
3. **Use the label-free profile as an upload gate?** Best guess: yes. Rerun `test_diag.py` on every new test
   probability file and refuse an upload if France's empty share or mean answer size moves more than about 1 pt
   away from US and India.
