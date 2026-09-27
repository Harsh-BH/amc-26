# P55-fr-combiner — RESULT (final, 27 Sep 05:35)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **The combiner transfers to France's stack.** It is a LightGBM over country-invariant inputs, fit on one country's out-of-domain LOCO frame and applied to the other country's. It re-scores the recipe (+ P48 veto) and passes the gate in both directions:
  - arm A (veto band pairs with c < 0.6, add free pairs with c > 0.95): h1 **India→US +0.0052 ± 0.0002, US→India +0.0046 ± 0.0003**;
  - arm B (veto only): h1 +0.0029 / +0.0019.
- France is restricted to the stage × region cells LOCO has. Reweighted to France's own cell mix, the gain is **+0.0022 / +0.0029** France-F0.5, so **expected LB ≈ +0.0003 to +0.0004 over #48** (A) and +0.0001 to +0.0003 (B).
- **Upload `output/sub55`** (= #48 − 3,198 France pairs + 4,830 France pairs; India / US byte-identical). Conservative fallback: `output/sub55b` (the veto only).

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- **Files** (both on `output/sub48`, since sub52_max does not exist):
  - `output/sub55`: md5 f258ae32…, 7,732 France S1 changed, 156 made empty.
  - `output/sub55b`: md5 460f3d6c…, 3,160 S1 changed, 156 made empty.
  - `check48.py sub48 sub55[b]`: only France lines differ.
  - Validator `--check-ids` on SageMaker: **both PASS** (p55-val2-55-0927-052339, p55-val2-55b-0927-052555; md5 match; 100,830 / 100,924 empty rows).
  - SUBMISSIONS.md rows #55 / #55b.
- **Rebuild on another base** (≈ 15 s, 1.3 GB):
  - `.venv/bin/python probes/P55-fr-combiner/p55.py build <BASE> <OUT> v=0.6,scope=band,u=0.95` (drop `,u=0.95` for the veto only).
  - Needs `data/work/p55/cf_plain.npy`, the France combiner scores. Regenerate it with `p55.py scores plain` (≈ 1.5 min).
- **Rule** (unseen country, applied after the recipe and P48's veto, on recipe pairs only; tier pairs from P41 / P40 / P46 are never touched):
  - c = σ(mean over 3 seeds of the logit of LightGBM p). Params: 200 rounds, 31 leaves, lr 0.05, min_data_in_leaf 200, λ2 1, bagging 0.8, feature_fraction 0.8, deterministic.
  - Fit on both LOCO frames pooled (India pairs under US-trained M2 / CE, and US pairs under India-trained ones).
  - Boosters: `data/work/p55/fr_plain_s{0,1,2}.txt`.
  - **Veto:** an accepted recipe pair, 0.02 < g < 0.98, p not moved by key tier (p := 1) or bxneg (p := 0.99), c < 0.6.
  - **Add:** a free argmax pair (non-sibling, not capped out, own p), c > 0.95, best c first.
    - Caps 5 / 6 are counted on the base's France line; an SX already listed for any France S1 is skipped.
    - A g ≥ 0.98 pair is added only if the τ veto zeroed it (p 0), because LOCO has no other free g ≥ 0.98 pairs.
- **Features** (`p55.feats`, 20 columns):
  - dens_p and dens_g: the frame's argmax pairs per S1 with a value ≥ the pair's (the recipe's density map, used as a feature);
  - logit CE r;
  - is2, hn_rel (P39's 6 classes), stage (base / rescue / gc / hn / none), region (lo / band / hi);
  - S1 same-source rank and gaps;
  - n / accepted count per same and other source, max p of the other source, s1_psum;
  - SX candidate count, count with g ≥ 0.5, competitor g mass, and the logit gap to the best competitor, all over every candidate pair.
- **For P50 (sub_all):**
  - The France change set = `p55.py build <sub_all base> <out> v=0.6,scope=band,u=0.95`.
  - It reads only the base's France lines (caps, used SX) and the recipe state from france.npz, and asserts that every add is in the base's candidate_pairs.tsv.
  - Apply it after any other France change, so the caps see the final lines.

## Evidence         (France-F0.5 dF vs the current recipe analogue t .99 / r .9 / gc .7 / hn .8, caps 5/6, τ 0.1, + P48's veto; "A→B" = fit on A's labels, test on B; h1 = report half ± SE)
| ledger name | world | key metrics (India→US / US→India) | verdict |
|---|---|---|---|
| p55-loco-plain-srcpick | LOCO chain, 1 seed | bars picked on the source by 5-fold cross-fit (honest): **B v 0.4 +0.00353 ± 0.00012 / +0.00085 ± 0.00013** (vetoed 85% / 50% false); A (0.4, u 0.8) +0.0080 / **−0.0080** (US-fit adds only 60% precise on India) | B keep; A with the source bar drop |
| p55-loco-bag3-B:all:0.5 | LOCO, 3 seeds, v picked on h0 (max of min) | h1 +0.00373 ± 0.00020 / +0.00111 ± 0.00022. The base g ≥ 0.98 cell alone: +0.0030 / **−0.0008** (India: 13.5% false) | inconclusive (sign flip in half of France's vetoes) |
| p55-loco-bag3-B:nobasehi:0.6 (= band on LOCO) | LOCO, 3 seeds | **h1 +0.00287 ± 0.00021 / +0.00189 ± 0.00021**; the France-reweighted value is +0.00073 / +0.00183 | keep → sub55b |
| p55-loco-bag3-add:0.95 | LOCO, 3 seeds | h1 +0.00228 ± 0.00010 / +0.00276 ± 0.00021, precision 0.96 / 0.84. Band adds 0.98 / 0.93 precise, τ-vetoed g ≥ 0.98 re-adds 0.95 / 0.80 | keep |
| p55-loco-A-nobasehi0.6-add0.95 | LOCO, 3 seeds, (v, u) on h0 | **h1 +0.00517 ± 0.00023 / +0.00463 ± 0.00029** (+4,116 / +6,898 pairs, −2,591 / −1,739). France-reweighted: veto +0.00073 / +0.00183, adds +0.00147 / +0.00104, **total +0.0022 / +0.0029** | **keep → sub55** |
| p55-loco-bag3-resid | LOCO, 3 seeds | P44-style residual (init = logit p): veto ≈ plain (all 0.6 +0.0051 / +0.0007), **adds −0.012 / −0.013** | drop |
| label-free profile (prof.json) | France vs LOCO | vetoes per accepted pair: France 0.0038 (band), LOCO 0.0059 / 0.0061. Adds per accepted pair: 0.0072 / 0.0094 / 0.0241. France's r median: vetoes logit −0.13, adds +7.2 (LOCO adds +7.4 / −3.5) | PASS: France is India→US-like and milder |
| p55-sub55 | test | −3,198 France pairs (rescue 154, gc 593, hn 2,402, base band 49), +4,830 (sub55 only); India / US byte-identical (check48); every add is a candidate (asserted). Validator `--check-ids` PASS on both | **keep → upload** |
| agree.json | France | single-direction fits vs the pooled fit: India-fit vetoes 3,616 (77% in pooled) / adds 4,260 (74%); US-fit 2,439 (90%) / 7,450 (62%). 58% of the pooled vetoes and 63% of its adds are in both single sets | the pooled model sits between the two |

## What failed      (approaches tried and dropped, with the number that killed them)
- **Add bar picked on the source (u 0.8):** US→India −0.0080. The source's cross-fit precision (0.95) does not transfer (0.60), so u comes from LOCO h0 (0.95), like P42's France bars.
- **The residual form (init_score = logit p, P44's):** adds −0.012 / −0.013 at every u. The residual trusts the blend's p on free pairs.
- **Vetoing base-stage g ≥ 0.98 pairs:** +0.0030 on US, −0.0008 on India (86% of those India vetoes are true). This cell is excluded, and it is half of France's raw veto list.
- **France cells that LOCO has no data for:**
  - In LOCO the France base bar 0.99 density-maps to ≈ 0.92, so the rescue / gc / hn stages never hold g ≥ 0.98 pairs, and no free g ≥ 0.98 pair has p > 0.
  - France has 12,354 such accepted pairs (793 would be vetoed) and 1,134 such add candidates.
  - All of them are excluded (the `scope=band` rule and the add guard); an unseen stage × region combination is an extrapolation.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `p55.py loco [plain|resid]`: the gate, 1 seed, source-picked bars → `loco_<mode>.json` (48 s, 1.0 GB).
- `p55.py scores <mode>`: bagged scores → `data/work/p55/cb_<mode>_<dir>.npy` + `cf_<mode>.npy` + France boosters (≈ 1.5 min, 1.3 GB).
- `p55.py ana2 <mode>`: scope × v / add grids, the France counts and the reweighted estimate → `ana2_<mode>.json` (25 s).
- `ana3.py`: adds by region and the A combinations. `prof.py`: the label-free profile. `pick.py`: the h0 picks.
- `p55.py build BASE OUT v=..,scope=all|band|nobasehi|added|hn[,u=..][,mode=plain]` → output/OUT + `p55_info.json` (15 s).
- Core functions: `sx_feats(sx, g)`, `feats(...)`, `france_frame()`, `loco_frame(dr)`, `decide(fr, c, v, u)`, `fr_own(z)`.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Licence / rules:** LightGBM (MIT). No new models or data. The inputs are the shipped M2 / CE scores plus label-free counts. No country filter: the rule keys on the unseen-country recipe stages.
- **Model mismatch:** France is scored by the pooled M2 / CE, LOCO by single-source ones.
  - The features are densities / ranks where possible. But CE r is raw, and it is the top feature (India→US gain share 0.49).
  - If the pooled CE is sharper on France, the veto should do better; if it is worse on French text, both arms lose.
- **Bar selection:** v and u were picked on LOCO h0 across both directions (h1 reported). The scope (band) was chosen after seeing that the base-hi cell flips sign; that choice is in-sample. The fully source-picked veto (v 0.4, all scope) still passes: +0.0035 / +0.00085.
- **Size:** the gain is 5–10× earlier France levers (P48: +0.0004). That fits a re-scorer that sees both CE and context, where earlier levers were one threshold on the blend.
  - The LOCO recipe analogue is weaker than France's full stack (no tiers).
  - The France-reweighted estimate (+0.0022 / +0.0029) is the number used for the expected LB.
- **Empty S1:** 156 France S1 become empty. All of them are band vetoes in S1 whose only pairs were vetoed.

## Open questions   (for the master; each with your best guess)
- **sub55 or sub55b first?** sub55: it is ≥ sub55b on every reading (the adds alone are +0.0023 / +0.0028, 0.96 / 0.84 precise). If sub55 loses on the LB, try sub55b to separate the two levers.
- **Port to src?** The builder is self-contained (france.npz + CE caches + 3 boosters). A src port = `p55.feats` + `p55.decide` after `france.complete`'s P48 veto (≈ 60 lines). Best guess: ship the staged file, and port only if the LB confirms.
- **Extend to the excluded cells** (France g ≥ 0.98 pairs in later stages)? Not testable on LOCO as built. It would need an analogue with the France bar kept above 0.98 (e.g. P37's M2-only analogue). Skip in the remaining time.
