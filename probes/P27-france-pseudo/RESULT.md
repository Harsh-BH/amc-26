# P27-france-pseudo — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
**No pseudo-label arm passes the LOCO gate (≥ +1 pt in both directions).** Across 27 arm variants, each arm helps one direction and hurts the other.
- The house-number lesson ("a mismatch means no match") gives India→US up to **+2.5 / +2.7 pt**, but costs US→India **−3 to −8 pt**: India's hn mismatches on true pairs are noise, US's are generator siblings.
- The hn-blind version helps US→India (**+1.21 pt** at w 0.35) and slightly hurts India→US (**−0.20**). With a base that has already seen the country, every arm loses 0.1–0.5 pt. Seed noise is ±0.2 pt.

**Do not adopt for France as validated.** Two France LB-test files are staged (validator PASS, India/US = #9ab byte for byte). Only worth a spare slot. $1.00 AWS.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
**Nothing in src/.** P6's verdict "don't self-train on France" stands for plain, filtered, down-weighted, negative-only, soft, sibling-negative and recalibration arms.

The only open door is an LB test. Uploads that score below the best cost nothing (organiser ruling 1), and France is 14.97% of the S1, so ΔLB ≈ 0.15 × ΔF0.5(France). Two staged folders (matching + candidate TSVs; **add code.zip**), both = #9ab with France swapped:

| folder (this worktree) | France p | t | France profile | churn vs #9ab | LOCO evidence (lf Δ, US→India / India→US) |
|---|---|---|---|---|---|
| **`output/sub27n35`** (first choice) | σ(0.65·logit p₉ₐᵦ + 0.35·logit p_fr), p_fr = hn-blind France-only model | 0.98 | 6.66% empty, 3.03 matches/S1 (#9ab 6.71% / 3.02) | 23,807 France S1 (9.2%), +14,467 / −10,952 pairs | +1.21 / −0.20; pooled-base bound −0.15 / −0.13 |
| `output/sub27f20` | same with the full-feature France-only model, w 0.2 | 0.98 | 6.69% empty, 3.04 matches/S1 | 17,625 France S1 (6.8%), +12,796 / −6,431 pairs (4,798 of the removed are hn-mismatched) | −1.19 / +1.22; pooled-base bound −0.18 / −0.01 |

- **t choice.** t is set where France's empty-S1 rate equals #9ab's at t 0.985, the LB-best France operating point. P6's rule (5.75% empty) would say 0.755 / 0.73, which would add ~70k pairs; the LB already showed that rule is too loose.
- **How to read the LB.** Against #9ab's 0.984072:
  - sub27n35 ≥ +0.0003 → France behaves like the LOCO India target. Keep it, and try w 0.5 (`p_nohn_w0.5` is in `fr_blend.parquet`).
  - ≤ −0.0003 → drop pseudo-labelling for France for good.
  - sub27f20 is the other regime's variant. Upload it only if sub27n35 loses and a slot is free.
- **Rebuild:**
  ```
  .venv/bin/python probes/P27-france-pseudo/p27.py prep_fr && .venv/bin/python probes/P27-france-pseudo/p27.py upload_fr
  .venv/bin/python probes/P27-france-pseudo/p27.py launch_fr          # ~2.5 min billable on ml.m7i.4xlarge, ~$0.04
  .venv/bin/python probes/P27-france-pseudo/p27.py fetch <job>        # -> data/work/p27/jobs/<job>/fr_blend.parquet
  # France prob file = data/work/p19/prob_test_France_idf144_bge.parquet [s1, sx] + p := fr_blend.p_nohn_w0.35
  #   (already written: data/work/p27/prob_test_France_{nohn_w0.35,nohn_w0.2,full_w0.2}.parquet)
  .venv/bin/python probes/P14-france-shift/combine.py /home/harsh1/github-repos/amc-26/output/sub8c \
      data/work/p27/prob_test_France_nohn_w0.35.parquet 0.98 output/sub27n35
  cd data/student_resource && python3 utils/validate_submission.py --matching <out>/matching_results.tsv \
      --candidate <out>/candidate_pairs.tsv --test-dir dataset/test          # PASS (both folders)
  ```

## Evidence         (table: ledger name | world | key metrics | verdict)
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| phaseA-pseudo-precision | holdout-dense-full (P6 LOCO scores, target fold 0) | Phase A table | inconclusive: the precision targets are reachable only India→US |
| loco-selftrain-r1 | holdout-dense-full LOCO (P6 protocol, SageMaker) | base / plain / plain_cls0 / filt / filt_w03 / neg_only / neg_hard / sibneg / recal | drop |
| loco-selftrain-r2-seeds | same | source-only seeds, arm seeds, soft, tgtonly | drop |
| loco-selftrain-r3-tgtonly-nohn | same | hn-blind / plain-label target-only models | drop (US→India +1.2, India→US −0.2) |
| loco-tgtonly-pooledbase-bound | same, base = P6 pooled (in-domain) | tgtonly −0.01…−1.0 | drop |
| france-tgtonly-labelfree | test-France | m2_big on the IDF-fixed X = #9ab p_gbdt (max \|Δ\| 3e-8); pseudo-labels 789k pos / 215k neg; churn and profile per w | inconclusive (LB-test only) |

**Phase A.** The source-only model (P6) scores the unseen country's fold 0, with every train S1 competing.

| | US model → India | India model → US |
|---|---|---|
| pos: argmax & p ≥ .98 (P6) | 283,156 @ **94.37%** | 380,310 @ **97.90%** |
| pos: p ≥ .99 / .995 / .999 | 96.08 / 97.56 / 99.54% | 98.45 / 98.92 / 99.70% |
| wrong p ≥ .98 positives by hn class: none/eq, +SIB, digit, small, far | 98.5 / 0.7 / 0.1 / 0.1 / 0.6% | 5.0 / **87.9** / 1.3 / 0.4 / 5.4% |
| pos p ≥ .98 & hn eq/none | 94.05% (93% vol) | **99.88%** (91% vol) |
| pos p ≥ .995 & hn eq/none & (bge r ≥ .5 or no r) | 98.25% (83% vol) | **99.96%** (80% vol) |
| pool (target fold 2 ×0.5), full candidate lists: pos p ≥ .995 & hn eq/none & margin ≥ .5 | 116,911 @ 97.39% | 152,011 @ **99.95%** |
| same at p ≥ .999 | 90,907 @ 99.48% | 121,150 @ 99.98% |
| pos p ≥ .98 with margin to the 2nd candidate 0.9–0.99 | 8,513 @ 67.9% | 12,268 @ 89.7% |
| chain size (S1 sharing the core) | flat 93.9–94.9% across bins | flat 97.8–98.2% |
| bge LOCO CE r (only 5% of the positives have r: m2_big band) | r < .5 → 27% precise; r ≥ .9 → 94% | r < .5 → 31%; r ≥ .9 → 78% |
| neg: p ≤ .02 / .005 / .001 | 99.73 / 99.92 / 99.99% | 99.75 / 99.90 / 99.98% |
| neg: losers of a confident argmax (1-to-many) | 506k @ 99.99%, but 99.6% already have p < .02 | 718k @ 100.0%, same |

**Phase B: LOCO, retrained arms.** Setup, P6's protocol:
- Train = source fold 2 ×0.5 plus pseudo-labels on the target's fold 2 ×0.5 (619,812 / 949,277 pool pairs, the same as P6).
- Early stopping on the source's fold 1 ×0.2.
- Scored on fold 0 with every train S1 competing.
- lf = label-free t: the target's empty-S1 rate matched to the source's at its own t*. The Δ column is vs base, in pt.
- In-country = the source country's own fold 0.

| arm | pseudo-labels US→India / India→US (n @ precision) | US→India lf (Δ) | oracle | India→US lf (Δ) | oracle | in-country US / India |
|---|---|---|---|---|---|---|
| base (P6 model) | – | 0.9054 t .98 | 0.9066 | 0.9293 t .91 | 0.9296 | 0.9808 / 0.9792 |
| source-only refit, seeds 1/2/3 | – | 0.9035 / 0.9079 / 0.9073 | ≤ 0.9088 | 0.9313 / 0.9287 / 0.9298 | ≤ 0.9315 | 0.9807–0.9808 / 0.9791–0.9792 |
| plain (P6: argmax&p≥.98 / p≤.02) | 140k@94.3% + 417k@99.74% / 190k@97.9% + 668k@99.74% | 0.8922 (−1.32) t .998 | 0.8922 | 0.9286 (−0.07) | 0.9311 | 0.9806 / 0.9784 |
| plain_cls0 (hn eq/none positives), 3 seeds | 130k@94.0% / 173k@**99.89%** | 0.8531 / 0.8592 / 0.8520 (**−5**) | ≤ 0.8644 | 0.9398 / 0.9387 / 0.9411 (**+1.1**) | ≤ 0.9413 | 0.9806–0.9808 / 0.9787–0.9788 |
| filt (p≥.995 & cls0 & margin≥.5 / p≤.005) | 117k@97.4% + 391k@99.92% / 152k@99.95% + 638k@99.89% | 0.8795 (−2.59) | 0.8839 | 0.9358 (+0.65) | 0.9367 | 0.9807 / 0.9786 |
| filt ×0.3 weight, 2 seeds | same | 0.9019 / 0.8987 (−0.4/−0.7) | 0.9019 | 0.9338 / 0.9344 (+0.5) | 0.9353 | 0.9807 / 0.9789–0.9791 |
| neg_only (p≤.005) | 391k@99.92% / 638k@99.89% | 0.8789 (**−2.65**) | 0.8795 | 0.9326 (+0.33) | 0.9335 | 0.9807 / 0.9791 |
| neg_hard (look-alike negatives + losers) | 338k@99.92% / 510k@99.87% | 0.8811 (−2.43) | 0.8824 | 0.9342 (+0.50) | 0.9344 | 0.9808 / 0.9788 |
| sibneg (filt + confident sib-offset → 0), 3 seeds | +339 / +3,019 negatives | 0.8214–0.8217 (**−8.4**) | ≤ 0.8570 | 0.9540–0.9542 (**+2.5**) t .11 | 0.9553 | 0.9806–0.9808 / 0.9784–0.9786 |
| soft (all pool pairs, σ(logit p / 0.5)) | 169k / 234k with y ≥ .5 | 0.9029 (−0.25) | 0.9029 | 0.9286 (−0.07) | 0.9298 | 0.9807 / 0.9786 |
| recal (target-only residual GBDT on logit p) | filt + losers | 0.8654 (−4.00) t .999 | 0.8654 | 0.9260 (−0.33) | 0.9260 | unchanged |

**Target-only model blended with the base**, p′ = σ((1−w) logit p_base + w logit p_tgt). The target model is fit on the pool's pseudo-labels only and early-stopped on source val (21–41 trees). In-country is unchanged by construction.

| arm | US→India lf (Δ) | oracle | India→US lf (Δ) | oracle |
|---|---|---|---|---|
| tgtonly (plain_cls0 labels, all features) w .2 / .35 / .5 | 0.8935 / 0.8761 / 0.8468 (−1.2 / −2.9 / −5.9) | ≤ 0.8966 | 0.9415 / 0.9506 / 0.9567 (+1.2 / +2.1 / **+2.7**) | ≤ 0.9577 |
| tgtonly_nohn (plain_cls0 labels, no hn_eq/hn_both/hn_diff/addr_rule/rule_acc) w .2 / .35 / .5 | 0.9128 / 0.9175 / 0.9206 (+0.74 / **+1.21** / +1.51) | ≤ 0.9212 | 0.9283 / 0.9273 / 0.9257 (−0.10 / −0.20 / −0.36) | ≤ 0.9287 |
| tgtonly_plain (P6 labels, all features) w .2 / .35 / .5 | 0.9043 / 0.9035 / 0.9025 (−0.1 / −0.2 / −0.3) | ≤ 0.9059 | 0.9301 / 0.9304 / 0.9296 (+0.1 / +0.1 / 0.0) | ≤ 0.9310 |
| tgtonly_nohn_plain w .2 / .35 / .5 | 0.9125 / 0.9171 / 0.9199 (+0.7 / +1.2 / +1.45) | ≤ 0.9209 | 0.9276 / 0.9255 / 0.9228 (−0.2 / −0.4 / −0.65) | ≤ 0.9277 |
| **pooled base** (P6 pooled, has seen the target) tgtonly w .2 / .35 / .5 | 0.9782 → 0.9764 / 0.9731 / 0.9679 | | 0.9807 → 0.9806 / 0.9796 / 0.9777 | |
| pooled base, tgtonly_nohn w .2 / .35 / .5 | 0.9782 → 0.9776 / 0.9767 / 0.9751 | | 0.9807 → 0.9805 / 0.9794 / 0.9772 | |

**France, label-free.** Which LOCO target does France resemble? Argmax pairs per 1k by hn class, and the share of confident accepts:

| | eq/none | +SIB | digit | small | far | SIB / eq | confident: hn eq/none share | confident hn-mismatched: SIB share |
|---|---|---|---|---|---|---|---|---|
| LOCO India target (US model, train fold 0) | 792 | 76 | 30 | 18 | 85 | 0.096 | 93.2% | 8.8% |
| LOCO US target (India model) | 689 | 203 | 35 | 9 | 64 | 0.295 | 91.1% | 25.7% |
| test India (#8c p) | 834 | 50 | 28 | 15 | 73 | 0.059 | 87.8% (p ≥ .98) | – |
| test US (#8c p) | 735 | 184 | 37 | 8 | 36 | 0.25 | 91.6% | – |
| **test France (#9ab p)** | 810 | 157 | **3** | 11 | 20 | **0.19** | **98.8%** | 15.5% |

France sits between the two regimes on every statistic. So a label-free regime switch, which would pass both LOCO directions (hn-blind for India-like targets, hn-full for US-like ones), cannot place France.

France is extreme in one respect: almost no true-pair hn noise (digit class 3/1k vs 28–37). The hn-full lesson can therefore cost France at most the 1.2% of its confident pairs that carry an hn mismatch (9,674 of 798,871 at p ≥ .98). Its uncertain band is mostly eq/none: 118/1k argmax pairs at p ∈ [.02, .985) vs US 24, India 29.

**AWS spend.** 10 CPU jobs, 2,560 s on ml.m7i.4xlarge ($1.018/h) + 952 s on ml.m5.4xlarge. ≈ **$1.00** in total. S3: ~250 MB.

## What failed      (approaches tried and dropped, with the number that killed them)
- **P6's plain self-training, reproduced exactly.** US+psIndia→India is 0.87009 on P6's grid (P6: 0.8701). Base 0.90541 = P6's 0.9054, and 0.8636 at the source t = P6's.
  - P6's threshold grid stops at 0.98, while the self-trained model's optimum is t 0.998.
  - On the extended grid the real loss is **−1.3 pt** (0.9066 → 0.8922), not −3.5.
- **Filtering for precision does not fix it.**
  - filt pushes India→US positives to 99.95% and negatives to 99.89%, and still loses −2.6 pt US→India.
  - neg_only (99.92%-precise easy negatives, no positives) loses −2.65 pt US→India.
  - The damage is selection bias, not label noise. Pseudo-labels exist only where the source model is sure, so target-only feature regions get one-sided labels (e.g. India's Indic / hn-noisy pairs → "negative"). The retrained model then pushes the whole region down.
- **House-number filters and sibling negatives.**
  - plain_cls0 / sibneg encode "hn mismatch = no match". That is right for US siblings (+1.1 / +2.5 pt) and wrong for India, where 6.8% of the confident (p ≥ .98) positives carry an hn mismatch and ~99% of those are true (−5 / −8.4 pt, stable over 3 seeds).
  - 339 relabelled India pairs cost 5.8 pt more than filt. The lesson generalises from sibling offsets to every hn mismatch.
- **Down-weighting (×0.3) and soft labels (T 0.5)** shrink both effects to within ±0.7 pt: nothing gained.
- **Recalibration GBDT on pseudo-labels** (−4.0 / −0.3 pt). It is fit on the extremes only, so its residuals extrapolate badly into the uncertain middle.
- **Chain-size guard.** Precision is flat across chain bins (93.9–94.9% / 97.8–98.2%).
- **bge CE agreement filter.** It separates well (r < .5 → 27–31% precise) but covers only 5% of the confident positives, because P13's LOCO r exists only on m2_big's band. Not used in Phase B.
- **Target-only models with an in-domain (pooled) base** lose everywhere (−0.01 to −1.0 pt). Once the base has seen a country, its own pseudo-labels add nothing.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- **`phase_a.py`**: 35 s, 1.85 GB. Reads `data/work/p27/p6/S_holdout_{US,India}.parquet` (P6's offload, fetched from `s3://…/amc26-offload/data/work/p6/`) and writes `data/work/p27/phase_a.json`.
- **`fr_profile.py`**: label-free hn-class profile of the production test p per country. ~1 min, 1.5 GB.
- **`p27.py`**: the laptop side.
  - `prep` / `upload`: aux channel `s3://…/amc26-p27/data/aux/` with fold-0 truth counts, train house numbers, and P6's US / India / pooled models.
  - `tiny` / `rehearse`: local code-path check on fake data.
  - `launch SRC TGT [go '{"arms": [...], "base_model": "pooled"}']`, `status JOB`, `fetch JOB` (→ `data/work/p27/jobs/<job>/`).
  - `prep_fr` / `upload_fr` / `launch_fr`: the France job.
- **`solution_p27.py`**: the SageMaker job. It loads all 31.4M train pairs once (15 GB RAM), scores the base, builds the target pool, then fits and scores every arm. Arms are named by `labels()`; `_sK` = seed K; `tgtonly[_nohn][_plain]` = blended target-only models.
  - Output: `result.json` with per-arm summaries and full t-curves.
  - Timing on ml.m7i.4xlarge: ~40 s per retrained arm, 3–8 min per job.
- **`solution_p27fr.py`**: the France job. Two S1-hash-half cross-fitted France-only LightGBMs per variant (full / nohn), early-stopped on US+India fold 1 ×0.2, blended at w ∈ {.1, .2, .35, .5}, with profiles.
  - Output: `fr_blend.parquet` (`p_fr_{v}`, `p_{v}_w{w}`) in #9ab's row order.
  - Timing: 50 s compute.
- **`ledger_rows.py NAME VERDICT JOB…`**: one ledger row per job set.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Licences / data.** LightGBM (MIT), numpy / pyarrow / pandas. No external data. France pseudo-labels use test France's own unlabelled rows, which organiser ruling 2 allows.
- **The staged France files are not validated.** The LOCO sign depends on the target's hn regime, and France's label-free statistics fall between the two regimes.
  - The pooled-base bound (−0.15 pt) is the likely outcome if the pooled m2_big already "knows" France's regime.
  - Expected France effect for sub27n35: somewhere in [−0.2, +1.2] pt, i.e. LB ≈ −0.0003 … +0.0018.
- **Proxy mismatch.**
  - The LOCO base is a weak single-country model (fold 2 ×0.5, no CE). France's base is pooled m2_big (folds 2–5) + IDF fix + bge blend.
  - The France target model trains on #9ab's pruned rows: 79% positives, vs 25% in the LOCO pool. Thresholds are re-derived by empty-rate matching, but the blend's calibration differs from the proxy's.
- **Model size.** The France hn-blind model early-stopped at 61 / 112 trees (half 0 / 1), vs 34–37 in LOCO; the full-feature one at 33 / 33 (LOCO 21–29). Same recipe, larger pool.
- **t.** 0.98 is where #9ab's LB-best empty rate is matched. With a different p the LB-optimal t may move. `fr_blend.parquet` holds every w, so re-thresholding is a one-liner.

## Open questions   (for the master; each with your best guess)
- **Spend a slot on sub27n35?** Best guess: only as the last slot of 27 Sep, after the P26 / other France candidates. Its downside is bounded (≈ −0.0003 LB) and it costs nothing if it loses, but its expected gain is small.
- **Is France US-like (siblings) or India-like (hn noise)?** Best guess: sibling-rich like the US (SIB/eq 0.19), with almost no true-pair hn noise. That would make the hn lesson harmless on France, but also nearly spent: the high France t (0.985) already rejects most siblings, and #9as vs #9a changed only 858 S1.
- **Does the France gap live in the eq/none uncertain band (118/1k vs ~25/1k for US/India)?** Best guess: yes. It is name/address ambiguity among France's chain-heavy S1 (30% share core+city), which neither pseudo-labels nor hn features resolve.
