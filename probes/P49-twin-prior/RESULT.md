# P49-twin-prior — RESULT (final, 27 Sep ~04:45 IST)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **The floor is closed: all five levers sit at ~0 on top of sub44r.** Perfect handling of the ambiguous twin SX would add
  +0.0022 LB-eq. The best real decode, a joint exact EF decode with the size prior, recovers **+0.000026 ± 0.000024 (h0→h1) and
  +0.000002 ± 0.000021 (h1→h0)**, far under the +0.00015 at 3 SE gate.
- The in-set signals exist but are diluted. 38–43% of these SX belong to a twin outside the candidate set. The group-size
  prior is calibrated but weak (+0.01–0.02 nats per SX). The raw-name channel is 0.82–0.89 precise given the owner is in the set,
  but only 0.73 / 0.48 unconditionally. Script → state decides 67 holdout SX.
- **Recommendation: adopt nothing and build no sub49.** Stop spending on the address-less twin bucket.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- **Nothing.** No output/sub49, no validator run, no SUBMISSIONS.md row: the gate failed in both directions.
- For the approach doc (the literature point, now measured):
  - With exchangeable records, the correct group-size likelihood ratio is **r(k) = f(k+1)(k+1)/f(k)**. P(z) ∝ Π f(n_i) n_i!, so
    Poisson sizes carry no information.
  - The brief's f(k+1)/f(k) is overconfident. It predicts 0.54–0.57 top-twin ownership where 0.37–0.39 is observed.
  - The train law is under-dispersed (var/mean 0.74, caps 5 for S2 and 6 for S3), so the smaller twin is mildly favoured.

## Evidence         (table: ledger name | world | key metrics | verdict)
**Frame.** The #8c dense holdout (P23 dump). "Ambiguous" means an address-less SX (P35's placeholder set) whose candidate list holds
≥ 2 S1 sharing its #8c argmax S1's full-name key (P40 floor.key). There are 32,421 such SX in India and 53,407 in the US; 14,142 /
20,606 of them have a holdout twin.

**Tests 1 and 3–5** are conditional on the owner being in the set (India 8,045 SX, US 12,707).

**Decode rows** are per-S1 F0.5 deltas vs **sub44r out of fold on both halves**, with paired SE. `swapfit.py` refits P44's VAR=r
combiner on half 1. It reproduces P44's saved half-0 p'' exactly (max |diff| 0), and its swap numbers match P44's
(+0.000596 / +0.000590). LB-eq = 0.468 India + 0.383 US.

| ledger name | world | key metrics | verdict |
|---|---|---|---|
| t1-group-size-lr | holdout | nset=2, the smaller-k twin owns the SX: India **0.583 ± 0.014** (r-model 0.576), US **0.564 ± 0.010** (0.577). All nset: Δ log-lik vs uniform +0.012 / +0.021 nats per SX. Brief's h form: model 0.54 / 0.57 vs observed 0.365 / 0.393 (worse than uniform). #8c argmax is the owner 0.394 / 0.360 (1/nset 0.246 / 0.288) | drop |
| t2-joint-ef-decode | holdout, sub44r OOF | oracle **+0.002216 / +0.002059**. Per-SX argmax +0.000006 / −0.000004. **Joint EF + size prior +0.000026 ± 0.000024 / +0.000002 ± 0.000021**. Without the size factor +0.000022 / −0.000008. Exact on components of ≤ 4 twins × ≤ 4 SX (14% / 25% of rows) | drop |
| t3-raw-closest-twin | holdout | the twins' raw names differ in 4.8% / 17% of SX. Unique raw-closest twin is the owner **0.889 ± 0.030 (n 108) / 0.820 ± 0.017 (n 484)** vs uniform 0.38 / 0.33, given the owner is in the set. **Unconditional: 0.727 (n 132) / 0.479 (n 829)** | drop |
| t4-india-script-state | holdout + non-holdout train | P(script \| state) is deterministic: MI 1.520 of 1.522 nats. Best-state twin is the owner 0.672 ± 0.057 on **67** decided SX (uniform 0.203) | drop |
| t5-within-group-identical | full train truth + holdout | identical raw address-less member pairs: within a group 3.4% (India 56 of 1,617; US 119 of 3,487); across same-raw-name S1 **0 of 1,005 / 0 of 3,635**. In the frame, an identical-raw address-less sibling shares x's owner 28/28 and 98/98. An addressed identical-name sibling at one twin: 0.243 / 0.275 (uniform 0.224 / 0.273) | drop |
| dec-gbdt-twin-rows-on-sub44r | holdout, sub44r OOF | LightGBM on twin rows, re-deciding every ambiguous SX (best twin iff score ≥ τ, τ from the other half). base −0.000005 / −0.000026; +counts −0.000026 / −0.000010; +raw **+0.000016 ± 0.000020 / 0.000000**; all +0.000006 / −0.000004 | drop |

**Context from `why.py`:** sub44r accepts 1.1% of the ambiguous argmax pairs at a holdout twin (precision 0.78 India / 0.68 US). The
argmax p'' median is 0.15 / 0.19. The loss is recall that no in-set signal can buy back above break-even.

## What failed      (approaches tried and dropped, with the number that killed them)
- **Group-size prior (T1).** It is real and calibrated, but the likelihood ratios are 1.0–1.8 (r = 2.74, 1.66, 1.54, 1.43, 1.02,
  0 for k = 0…5). They cannot lift a 0.3–0.5 posterior past the 0.67–0.75 break-even. As a decode feature: −0.000026 / −0.000010.
- **Joint exact decode (T2).** Its extra is the second-member correction r(k+m)/r(k). That adds +0.000004 / +0.00001 over the same
  decode without it. The oracle's +0.0022 is out of reach.
- **Raw-string channel (T3).**
  - The slice is small (raw names differ in 4.8% / 17% of SX).
  - The owner is outside the set in 38–43% of cases, so the precision a decision sees is 0.73 (India) and 0.48 (US).
  - Result: +0.000016 / 0.000000.
- **Script → region (T4).** It is exact, but only 128 ambiguous India SX carry an Indic name with the owner in the set. Of those,
  67 have twins in different script regions.
- **Within-group distinctness (T5).** The hypothesis is inverted. Identical address-less records always share an owner (they are
  copies of one generated record), and they never occur across twins. A copy adds no evidence about who owns the pair. Addressed
  identical-name siblings carry none either.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
All scripts run from the repo root with `probes/P49-twin-prior/py.sh` (1.9 GB cap). Work files go to `data/work/p49/`.
- `frame.py [c]` builds tw_{c}.parquet (twin rows: p, am, y_full, raw names, kd/kt counts, frame coverage) and s1_{c}.parquet.
  22–30 s per country.
- `law.py` builds law.json (f per country × source). `ratios(f)` returns (h, r); the `__main__` Poisson self-check passes.
- `swapfit.py` (P44_VARIANT=r) builds ho_p44r_h1fit_{c}.npy: sub44r's p'' out of fold for half-0 S1. It asserts the half-0 path
  equals P44's file. About 2 min.
- `dec.py [arm]`:
  - `load(c)`: the sub44r OOF state;
  - `feats(c, tw)`: the twin-row columns;
  - `apply(d, tw, s, tau)`: per-S1 F0.5 after re-deciding. About 5 min for 4 arms.
- `t2.py` runs the joint EF decode and the oracle (`decode(comp, base_np, rtab, prior)`; asserts in `__main__`). About 4 min.
- `test1.py`, `test35.py`, `test5b.py`, `test4.py` (writes script_state.parquet), `why.py`. Logs in `logs/`, aggregates only.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- Nothing ships. rapidfuzz and LightGBM are MIT; no external data.
- **Competitor-count proxy.** The frame holds only 16% of competing train S1's addressed truth, so their count is a truth-addressed
  stand-in (kt). On holdout twins the label-free #8c count equals it 98.1% (India) / 98.8% (US) of the time, and is within 1 at
  99.9%. The proxy mildly favours the prior, and the prior still adds nothing.
- **τ tuning.** τ was tuned on the fit half with that half's in-sample GBDT scores. That is optimistic, and the arms are still ≈ 0.
- **Definitions.** "Ambiguous" is an exact full-name key match to the argmax S1. Fuzzy near-twins were not included; P35 measured
  0.85–0.95 core similarity with the same outcome.
- **France / unseen countries:** untouched.

## Open questions   (for the master; each with your best guess)
- Is any address-less lever left? Best guess: no. The +0.0022 oracle needs knowing which twin owns the SX, including owners outside
  the candidate set (38–43%). The generator leaves no label-free trace of that in the name, the script, the copies or the group
  sizes.
- Could twin-set blocking (adding every same-key S1 as a candidate) help? Best guess: no. It would raise "owner in set", but also
  nset, and P35 found such propagation 1–4% precise.
