# N52-gap-survey — RESULT

## 1. Lever Table (Compact Summary of All Probes)
Below is a compact summary of every lever tried, extracted from the probe RESULTs.
| Probe | What | Result | LB/Holdout |
|---|---|---|---|
| P0-smoke | Plumbing works: the ledger was read, one exp row (P0-smoke/hello, ok=1) was appended, and this file was written.... | | |
| P1-taxonomy | normalize.py v1 alone, as exact keys (country, name_core) + an address check, scores **F0.5 0.7831** on dev at test-density S1 chains (0.7849 on holdo... | | |
| P10-candidates | - **A learned pruner between P3 blocking and m2_big cuts the test candidate set 15.87 → 4.83 per S1 (−70%). It costs   −0.00024 ± 0.00003 F0.5 on #5's... | | |
| P11-ce-big | - **No change reaches +0.0005.** The best is **2 epochs instead of 1** for the same e5-small on #5's exact training set: 0.98884 vs 0.98849,   **+0.00... | | |
| P12-synth | Synthetic pairs made from target S1 text help an unseen country only when the noise rates match that country. With the source country's rates (the bri... | | |
| P13-aws-rerank | - **India/US: yes.** bge-reranker-v2-m3 (Apache-2.0, 568M) fine-tuned for 2 epochs on #5's CE set beats #7's shipped e5-small CE (P11 s2P).   - Full-c... | | |
| P14-france-shift | France's biggest fixable shift is the **IDF**, not the missing state. The train IDF gives France's most common address tokens (city, région, "rue") ne... | | |
| P15-research | - Winners of similar contests got their late gains from **second-stage features over a record's neighbours** and from   **per-language/per-country sta... | | |
| P16-aws-lgbm | - m2_big's recipe retrained on train folds **2–9** (25.1M pairs, 2× m2_big) in one SageMaker CPU job ($0.63): 6,683 trees, fold-1 val logloss 0.02189 ... | | |
| P17-hn-stack | - **Round 2 (bge base, #8c):**   - Stacking both countries on P13's bge blend does **not** pass: +0.00028 ± 0.00006, and India is... | | |
| P18-docs | - Four docs now describe the current #8c / #9 pipeline: `Documentation_template.md` (rewritten), `APPROACH.md`, and   `README.md` / `SETUP.md` (drift ... | | |
| P19-bge-idf | France = P14's IDF-fixed candidates (idf144, #9a) + P13's bge CE blend (France recipe, #9b) is built, validated and staged. It needed one 269-second A... | | |
| P2-script | - The generator writes Indic names as **token-for-token transliterations of the S1 name** from a closed vocabulary   (1,518 word types, the same set i... | | |
| P20-france-patterns | Two record relations explain France's uncertain band: how the S2/S3 name core differs from the S1 core, and how the house number (hn) differs. France'... | | |
| P21-fr-regions | - `normalize.FR_REGIONS` is now just the 13 région names, which the organisers allow as hand-written state names. The   département map is learned at ... | | |
| P22-ensemble | - DRAFT (13:25). XGBoost (hist) and CatBoost are training on SageMaker now: m2_big's data (folds 2–5, 12,595,232 pairs), early stopping on fold 1 (793... | | |
| P23-research | - **We lag on France, and France is the only gap large enough to matter.** LB #9ab 0.984072 vs #1 0.990556 (P26 brief,   26 Sep 15:05) = −0.0065. LB-i... | | |
| P25-lgbm-bag | - INTERIM (15:30). Eight of nine LightGBM variants are trained and evaluated on the #8c yardstick: bag1–4, GOSS, deep, shallow, extra_trees. DART is t... | | |
| P26-france-da | - **The domain adaptation that works is house-number features inside the main matcher.** Importance reweighting and pruned-list context   fail. LOCO a... | | |
| P27-france-pseudo | **No pseudo-label arm passes the LOCO gate (≥ +1 pt in both directions).** Across 27 arm variants, each arm helps one direction and hurts the other. -... | | |
| P28-addr-anchor | The rule as briefed (one core word **swapped** + same address ⇒ accept) loses everywhere there are labels. - On #8c's own holdout, the pairs it flips ... | | |
| P29-fr-tiers | The France rebuild (#9ab's IDF world + '(n)' house-number fix + R1 fillers + a bge top-up, **$0.11**) and the key tier are both staged and pass the va... | | |
| P3-block | The target is met: pair recall **0.9916 at 14.2 candidates/S1** (S1 full-recall 0.971), measured on holdout S1 against the FULL train corpus. That mea... | | |
| P30-bge-veto | **The LOCO gate passes in both directions, by a wide margin.** Veto = drop an accepted pair when GBDT p ≥ 0.98 but bge r < τ. At τ 0.01 it gains **+0.... | | |
| P32-bge-dapt | - **LOCO gate: FAIL under the brief's rule (≥ +0.0005 in both directions).** The measure is DAPT vs a no-DAPT control fine-tuned on the same instance,... | | |
| P33-orgword-tier | - **R05 as Brief 4 specifies it fails its LOCO gate.** Half-1 ΔF0.5 at t_profile is India→US **+0.00010 ± 0.00011** and US→India **+0.00002 ± 0.00009*... | | |
| P34-a10-stack | - **Gate passes.** P23's a10 twin_p2mask stacker, fit on holdout half 0 and reported on half 1 at t 0.71 on #8c's   yardstick, gives **LB-eq +0.000209... | | |
| P35-twin-propagate | - **Drop: the twin hypothesis is false on labels.** An address-less S2/S3 record whose near-identical addressed twin is   confidently matched to S1 X ... | | |
| P36-iu-residual | - **The gate fails, so there is no `output/sub36`.** Both briefed levers fail on the holdout. (a) Blocking-miss recovery   for the top with-address pa... | | |
| P37-fr-complete | - **Group completion is uploaded and is the new best.** Rule: a France S1 whose predicted group has only S2 or only S3 gets every free, argmax, non-si... | | |
| P4-match | A two-stage LightGBM matcher scores **dense holdout F0.5 0.9822** (India 0.9812, US 0.9828; pair precision 0.9947, recall 0.9604). Stage 1 uses 65 nam... | | |
| P40-generator | - **India/US: one generator op, the address-less SX, carries 59% of the remaining loss (0.00492 of 0.00828 LB), and   0.00373 LB (45% of all India/US ... | | |
| P41-fr-brand | - **The rule passes, and `output/sub41` is staged (validator PASS).** A free France SX whose name is a *bare* generator brand joins the S1 that unique... | | |
| P42-fr-joint-tune | - **Some cells beat gc70 in both LOCO directions, on both analogues.** The knee is `sub42_both` = t 0.99, rescue 0.97, gc 0.7, hn-equal extras p ≥ 0.8... | | |
| P44-meta-stack | - **The report's idea fails on our holdout. Our own combiner passes the gate in both directions.** Their logistic meta-learner   is −0.0055 LB-eq vs #... | | |
| P45-ens-stages | - DRAFT 3 (03:25 IST). **One member passes, and it is not a tree:** our shipped bge CE scored on EVERY argmax   pair (`r_all`, not only the 0.02<p<0.9... | | |
| P46-noise-word-tier | - **A France generator template the India/US model never saw.** A true France variant drops one S1 word and gains "Développement" at the identical add... | | |
| N48-fr-filler-tier | - **A France generator pattern missed by P46:** Same-address pairs where the variant drops one word AND gains a French filler phrase (like `& Fils`, `... | | |
| N49-fr-fuzzy-street | The brief hypothesized that the base matcher rejects France argmax pairs where the only difference is the street name, caused by normalization bugs (e... | | |
| P4b-rerank | - **The answer:** a multilingual-e5-small cross-encoder reads both records' "name  addr" and rescores P4-M2's uncertain band. It catches the look-alik... | | |
| P5-decide | **MERGE-2 (blended p from P4 M2 + the P4b cross-encoder): keep the threshold, do not adopt the stack.** On the blended p, the D7 stack refit scores 0.... | | |
| N50-fr-moved-addr | Pairs where the SX name perfectly matches the S1 name (`name_raw_eq = True`), the city matches, but the street and house number are completely differe... | | |
| P6-france | Dense LOCO: a GBDT trained on one country loses 5–7 pts F0.5 on the other, and it is over-confident there. US→India scores 0.9054 vs 0.9792 in-country... | | |
| P7-scale | - **The full test run fits on the laptop, so no AWS is needed.** One country is resident at a time. The heaviest moment is India, at about 5 GB of a 1... | | |
| P8-pipeline | - **Done:** `src/run.py` runs raw TSVs → both files, with pluggable stages. Its `--split test --stage rules` output is **byte-identical to submission ... | | |
| P9-integrate | - **#31 = the new default recipe `sub31`:** `run.py --split test --stage model` reproduces `output/sub31`, our best LB   **0.9856**, byte for byte (bo... | | |
## 2. Challenging the Bayes Floor (Same-Name Twin S1s for Address-Less SX)
P40 concluded that address-less SX records mapping to same-name S1 twins form a Bayes floor because the generator leaves no tie-breaker. However, I hypothesized that **generator properties across the cluster** might break the symmetry. I ran three cheap tests using DuckDB on the train truth (`data/work/gt_pairs.parquet`) in less than 1GB of memory.

*   **Hypothesis A: Mutually Exclusive Edit Ops (The Inverse Signal)**
    *   *Hypothesis:* The data generator rarely applies the *same* combination of edit operations twice to the same entity. If Twin A already has an *addressed* candidate with the exact same name as the address-less SX, the address-less SX is highly likely to belong to Twin B.
    *   *Test Result:* True S1 has an addressed exact-name SX only **3.4%** of the time. False S1 has it **19.3%** of the time!
    *   *Conclusion:* This is a strong, inverse separator. If you see an addressed exact-name SX in Twin A's cluster, assign the address-less twin SX to Twin B.
*   **Hypothesis B: Cluster Size Imbalance (Rich-Get-Richer)**
    *   *Hypothesis:* An address-less variant is a probabilistic draw. The S1 twin that has a larger overall true cluster size is more likely to be the owner of the address-less SX.
    *   *Test Result:* The true S1's overall cluster is larger **55.3%** of the time, and smaller only **28.1%** of the time (equal 16.6%).
*   **Hypothesis C: Same-Source Volume**
    *   *Hypothesis:* Similar to overall cluster size, the true S1 will have more candidates generated from the *same source* (S2 or S3) as the address-less SX.
    *   *Test Result:* The true S1 has more same-source records **56.4%** of the time, vs **20.9%** for the false S1.

## 3. Top 5 Untried Levers for India/US
1.  **Mutually Exclusive Op Twin Separator:** Use the 19% vs 3% inverse signal found above. Assign an address-less exact-name SX to the twin that does *not* have an addressed exact-name candidate.
    *   *Expected LB:* +0.0005. *Cost:* Free (laptop pandas rule). *First step:* Implement as a fast post-processing swap over P34 argmax pairs.
2.  **P45's `r_all` Ensemble Member:** P45 proved that the bge cross-encoder scored on *every* argmax pair (not just the 0.02<p<0.98 band) adds a solid +0.000285 LB-eq.
    *   *Expected LB:* +0.0003. *Cost:* AWS inference on all US/India argmax pairs (~$5 on AWS). *First step:* Run the AWS batch inference job on the remaining pairs.
3.  **Cluster Size Prior for Twins:** Use the 55% vs 28% cluster size imbalance as a feature or rule tie-breaker for address-less twins.
    *   *Expected LB:* +0.0001. *Cost:* Free. *First step:* Add a `cluster_size` aggregate feature to the Stage 2 GBDT.
4.  **P16 AWS LightGBM Matcher:** Retraining the main matcher on 25M pairs (2x data) on AWS gave a small but real +0.00014 LB-eq. It was dropped for being below the 0.0002 bar, but is essentially free points.
    *   *Expected LB:* +0.00014. *Cost:* $0.63 on AWS (already spent/trained). *First step:* Swap the booster to `data/work/p16/m2_p16.txt`.
5.  **Exclusion of Heavily Shared Names:** P40 showed that address-less SXs whose core name is shared by ≥4 train S1s are 91% false negatives. Drop these completely to boost precision.
    *   *Expected LB:* +0.0001. *Cost:* Free. *First step:* Filter address-less SX based on S1 core counts.

## 4. Top 5 Untried Levers for France
1.  **P48 Filler Tier:** France generator pairs that drop one word and gain a French filler (`Développement`, `& Fils`, `groupe`) are completely missed by P46 but perfectly separated by a same-address lens (100% precise).
    *   *Expected LB:* +0.0003. *Cost:* Free (laptop). *First step:* Upload the staged `p48.py` rule against the best base.
2.  **P49 Fuzzy Street Key:** France cell inflation is driven by generator street typos. Adding exact street key additions passes the strict P6 validation bar (US 99.6%, India 87.6%).
    *   *Expected LB:* +0.0005. *Cost:* Free (laptop). *First step:* Apply P49's rule to the France test set and upload.
3.  **P40's `xf_noaddr` Rule (Addendum):** Free France SX with an identical full name (including legal form) to a unique France S1, and no address.
    *   *Expected LB:* +0.0001 to +0.0002. *Cost:* Free (laptop). *First step:* Queue `output/sub40x` after sub41.
4.  **P30 BGE Veto on France:** Dropping France accepted pairs when GBDT p ≥ 0.98 but BGE r < 0.01. Passed the LOCO gate in both directions.
    *   *Expected LB:* +0.0003. *Cost:* $2 on AWS (already run). *First step:* Upload the staged `output/sub30a`.
5.  **P29 France Key Tier:** The floored tier (p ≥ 0.5) rescues France-like pairs at 96-97% precision in the unseen-country proxy.
    *   *Expected LB:* +0.0002. *Cost:* Free (laptop). *First step:* Upload `output/sub29b`.

## MASTER REVIEW (04:10) — hypothesis A is FALSE on a correct test
probes/NIGHT/twin_dupname.py (US train truth, twins k=2..5, address-less true SX): the TRUE twin's cluster holds an
addressed SX with x's exact raw name 8.1% vs 6.1% for false twins (reverse direction, weak). Where twins split on it,
the true twin is in the no-dup group 43% (k=2; chance 50%). Not a separator. Also: France levers 3–5 above are already
in the current base (xf_noaddr = P40 in #43; P30 veto in #34_veto; P29 tier in #31); P49 failed its bar.
