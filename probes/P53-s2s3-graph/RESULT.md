# P53-s2s3-graph — RESULT (final, 27 Sep ~05:00 IST)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **Drop: S2↔S3 triangle evidence adds nothing on top of sub44r, for India/US or for France.** The best combiner arm has a
  cross-fit of **−0.000001 ± 0.000013 LB-eq** (tri_comp); all four arms land between −0.000039 and −0.000001, and none is
  positive in both halves. That misses the +0.00010 at 3 SE gate by a wide margin.
- France LOCO: the triangle-extras acceptance rule loses in US→India at every cell (−0.0006 to −0.011; added pairs only 24–40%
  precise). The competitor veto cannot be tested, because the LOCO frames have one S1 per SX.
- **Recommendation: adopt nothing. No output/sub53 was built and no SUBMISSIONS.md row was written.** The signal is already
  inside the combiner's p. Where it is not, near-identical same-block twins belong to *another* S1 (P35's finding, now on
  every SX).

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- **Nothing.** The India/US and France gates failed in both directions, so there is no staged file and no validator run.
- Current best stays **output/sub48** (sub52_max if P52 stages one). P53 changes no bytes.
- For the approach doc, one line: "Graph/triangle post-processing (Foursquare/Shopee style): SX–SX similarity inside S1
  blocks, p-weighted. It was measured as combiner columns, a competitor veto, a re-assignment rule and a France extras stage.
  All ≈ 0 or negative. The generator's same-name different-business records make twin evidence ambiguous."

## Evidence         (table: ledger name | world | key metrics | verdict)
**Base.** P44's sub44r model: te44.FINAL (VAR=r), 5-seed bag, t 0.71. It is refit on each half; the h0 fit reproduces
`ho_p44r_*.npy` (asserted). Deltas are per-S1 F0.5 vs that bag on the report half, paired. LB-eq = 0.468 India + 0.383 US.
Cross-fit = the mean of both directions.

**Triangle columns** (`tri.py`), per candidate row (a, y). The similarity is token_set_ratio on the P3 normalized name
and address, s = (sn + sa) / 2, over the other SX x of a's #6-pruned block.
- t_max, t_max_n, t_max_a = max_x min(p(x,a), s | sn | sa).
- t_ncs = #x with p ≥ 0.71 and s ≥ 0.8.
- t_best = s(y, the block's best other SX); t_hq = hn agreement with that SX.
- c_max, c_pmax, c_ncs = the same evidence for y's other candidate S1; d_max = t_max − c_max.

| ledger name | world | key metrics | verdict |
|---|---|---|---|
| p53-tri_all | holdout | h0→h1 −0.000028 ± 0.000028 (India −0.000020, US −0.000047); h1→h0 +0.000018 ± 0.000022; **cross-fit −0.000005 ± 0.000018** | drop |
| p53-tri_own | holdout | −0.000064 ± 0.000027 / −0.000014 ± 0.000019; cross-fit **−0.000039 ± 0.000016 (−2.4 SE)** | drop |
| p53-tri_comp | holdout | −0.000008 ± 0.000016 (India +0.000039, US −0.000069) / +0.000007 ± 0.000021; cross-fit −0.000001 ± 0.000013 | drop |
| p53-tri_core | holdout | −0.000037 ± 0.000020 / −0.000037 ± 0.000024; cross-fit −0.000037 ± 0.000016 | drop |
| p53-triangle-headroom | holdout (#8c dump, sub44r p'') | AUC for y inside #34's 0.3–0.95 band, within p' quintiles: **0.47–0.53 for every column** (p' itself 0.56). Accepted h1 pairs whose SX has a competitor with stronger triangle support: **5 India / 3 US** (of 149k / 223k). Rejected pairs with p'' ≥ 0.4: 0.546 / 0.503 true; adding t_best ≥ 0.9 gives 0.622 (45 pairs) / **0.462** (65). Re-assignment (runner-up b has triangle support ≥ 0.2 over a, p_b ≥ 0.3): b right 0.29 / 0.26, a right 0.01 | drop |
| p53-loco-accept | LOCO chain (P48 frames) | extras on non-empty S1 (free, under cap, p ≥ mapped bar b ∈ {.5,.7,.8,.9}, t_best ≥ u ∈ {.9,.95,.99}, t_ncs ≥ 1). **US→India −0.0006 … −0.0108 at every cell** (add precision 0.24–0.40). India→US −0.0011 … +0.00017 (u .99: +0.00010 to +0.00017, precision 0.92–0.93) | drop |
| p53-loco-veto | LOCO chain | c_pmax ≥ v and c_pmax − t_max ≥ m: **0 pairs in both directions**. Each LOCO SX has one candidate S1 (`loco_*_ids.sx` unique), so the rule cannot be tested there | untestable → drop |

## What failed      (approaches tried and dropped, with the number that killed them)
- **Triangle columns in the combiner** (4 arms): the best cross-fit is −0.000001. The own-support columns overfit (−2.4 SE).
  - Unconditionally they look strong (t_max AUC 0.73–0.76).
  - All of that is already in p, because the matcher scored y against a directly.
- **Competitor-evidence veto** (India/US): nothing to veto. sub44r already rejects the argmax pairs whose SX has a
  triangle-stronger rival (8 accepted pairs across both h1 sets).
- **Re-assignment to the triangle-supported runner-up:** switching gives a pair that is right only 26–29% of the time. Most
  of those SX belong to neither S1 (distractors, or a third S1).
- **Triangle recall rule** (rejected argmax pairs with a near-identical confident twin): precision is **no better** than
  the band's own (India 0.62 vs 0.55 on 45 pairs; US 0.46 vs 0.50). This is P35's "the twin belongs to a different same-name
  S1", now shown for every SX and not only address-less ones.
- **France triangle-extras stage:** −0.0006 to −0.011 in US→India. India→US is positive only at u .99, with +0.00017 at most.
- **Not run: a stronger SX–SX similarity (e5 cosine / bge CE).** US is single-script, where token-set similarity is
  reliable, and US is as null as India (tri_all US −0.000047 / +0.000045). So the bottleneck is conditional independence
  given p, not similarity quality. P45 also measured frozen e5 cosines at AUC 0.55–0.70 on pruned pairs.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
All scripts run from the repo root via `probes/P53-s2s3-graph/py.sh` (2 GB cap).
- `tri.py <split> <c>`: triangle features for every row of the pairs file → `data/work/p53/tri_<split>_<c>.parquet`.
  - train = P23 holdout dump; test = P34 te_pairs.
  - 35–60 s and < 1.2 GB per country on the holdout (1.1–1.6 M rows, 3–4 M block pairs).
  - Test would be about 5× larger, but was never needed.
  - `tri.feats(split, c, k1, qx, p)` is the reusable core. `tri.py selftest` checks the block and top-2 helpers.
- `ev53.py [arm ...]`: the sub44r bag ± triangle columns, both halves, paired → `data/work/p53/ev53.json` and
  `d_<arm>_<c>_h<fit>.npy` (per-S1 deltas). About 1 min per arm.
- `loco53.py`: France LOCO veto/accept grid on P48's frames → `data/work/p53/loco53.json`. About 2 min.
- `diag.py`, `veto_diag.py`, `reassign_diag.py`: the headroom diagnostics above (label-based, holdout only).
- Disk: `data/work/p53` = 81 MB (it can be deleted).

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- rapidfuzz (MIT) only. No model, no external data, and no test-side artefact was produced.
- **Gate assumed:** cross-fit LB-eq ≥ +0.00010 at ≥ 3 SE and both directions > 0. Every arm fails every reading, so the
  exact form does not matter.
- **The holdout competitor rows carry in-sample #8c p** (same caveat as P44's sx_psum). That would inflate c_* columns,
  not hide them, so the null is conservative.
- **LOCO frames are argmax-only**, with one S1 per SX. The competitor half of the triangle has no France analogue. France
  test has competitors, but with no way to validate them no rule is recommended.
- Worked in the main checkout (data/ lives there) rather than a separate worktree, as P44 did.

## Open questions   (for the master; each with your best guess)
- **Is a GNN or learned SX–SX matcher worth it?** Best guess: no. The label diagnostics show that twin evidence *lowers*
  precision in the US and is flat in India. A better similarity cannot fix records that the generator made deliberately
  ambiguous.
- **Does the triangle help on the S1 side (S1–S1 pairs sharing SX)?** Not tested: S1 is deduplicated by construction, so
  there is no S1–S1 edge to exploit.
