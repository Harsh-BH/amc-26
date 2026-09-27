# P61-fr-cotrain — RESULT (final, 27 Sep 07:10)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **Co-training works on LOCO; on France only the adds carry over.** P55's combiner, refit on source labels plus target pairs where P55's c and Qwen agree (c, q ≥ .7 → 1; c, q ≤ .3 → 0), beats P55 in both directions: **India→US +0.00108 (h1 +0.00113 ± 0.00014), US→India +0.00075 (h1 +0.00064 ± 0.00024)**, both against 9-seed bags. Reseeding P55 alone moves US→India by +0.0004, so that direction is only ~2× the seed noise. The self-training control splits the directions the way P27 did (+0.0005 / −0.0027).
- **France's change set is a different mix than LOCO's.** Most of it is veto re-admits, and on LOCO those hurt India→US. Reweighted to France's own component counts, the full swap is −0.00018 / +0.00013 (drop). The **hybrid** keeps P55's vetoes and takes only the co-trained adds: **+0.00041 / +0.00006 France-F0.5**. That passes the gate (IU ≥ +0.0003, both ≥ 0), but it is ≈ **+0.00001 to +0.00006 LB**.
- **Staged `output/sub61h9`** = sub_clean + 1,613 / − 73 France pairs. Change set `data/work/p61/sub61h9_changes.parquet`. Validator `--check-ids` on SageMaker: **PASS** (p61-val9-0927-0653). Low priority: fold it into the final France stack, and don't spend an upload slot on it alone.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- **Adopt the hybrid only**, as a France change set on `sub_clean`: `output/sub61h9`, md5 5b3eada5….
  - vs sub_clean: France only, 1,665 S1 changed, +1,613 / −73 pairs, 0 S1 emptied, 0 changed lines over caps 5 / 6.
  - Every add is asserted to be in sub_clean's candidate_pairs.tsv. candidate_pairs.tsv and code.zip are hard links; India / US lines are byte-identical by construction.
- **Change set for P50:** `data/work/p61/sub61h9_changes.parquet`, columns `s1, sx, op` (add / rm), 1,686 rows. Apply it to sub_clean's France lines.
  - It swaps P55's add step only, so it composes with anything that leaves P55's France vetoes alone.
- **Rule** (France; applied where P55 runs, after cls80 and before dup; only P55's add step changes):
  1. **Pseudo-labels** on France argmax band pairs, where P38's pooled Qwen r exists (all 241,776 band pairs):
     - 1 if P55's pooled c (`data/work/p55/cf_plain.npy`) ≥ 0.7 and Qwen r ≥ 0.7;
     - 0 if c ≤ 0.3 and Qwen r ≤ 0.3.
     - Result: 41,544 positives and 122,879 negatives.
  2. **Class weights:** each class's total weight is a fixed share of the source rows, pos 4.1% and neg 3.05%. These are the mean of LOCO's two directions at w = 1. On France that gives w_pos = 1.02 and w_neg = 0.26.
  3. **Refit** P55's LightGBM (same 20 features, same params) on both LOCO frames' labels plus the France pseudo-labels.
     - 9 seeds (0–8), c = σ(mean logit). Boosters: `data/work/p61/fr_c70q70_c30q30_share{,_s345,_s678}_{0,1,2}.txt`. Scores: `data/work/p61/cf_c70q70_c30q30_share9.npy`.
  4. **P55's decision:** keep the veto on P55's c (band, own, c < 0.6), then add free own argmax pairs with the co-trained c > 0.95, best first. Caps 5 / 6 are counted on the base's France lines, and SX already used anywhere in France are skipped, exactly as in `p55.build`.
- **Rebuild** (≈ 5 min, ≤ 1.1 GB):
  - `p61.py cache US_India; cache India_US; frcache`
  - `frfit c70q70_c30q30_share 0,1,2; frfit … 3,4,5 _s345; frfit … 6,7,8 _s678`
  - bag the three → `cf_c70q70_c30q30_share9.npy` (the one-liner in the Interfaces section)
  - `frbuild c70q70_c30q30_share9 sub61h9 hybrid`
- **Don't adopt the full swap** (`frbuild … full`: +2,879 / −89, 1,268 re-admits). It fails the France-reweighted gate on India→US (−0.00018).

## Evidence         (table: ledger name | world | key metrics | verdict)
Setup:
- **LOCO:** P55's frames (P42 chain, P48 veto).
  - "A→B" fits on A's labels and tests on B. IU = India→US, UI = US→India.
  - dF is France-F0.5 on the target against P55's arm A (v .6 band, u .95), with P55's own 3-seed scores `cb_plain`, unless a row says otherwise. h1 = report half ± SE.
- **Qwen:** P38's source-trained LOCO scores. They cover 28% / 37% of the target band pairs, because P38 scored P13's m2_big band, not the P6 band.

| ledger name | world | key metrics (IU / UI) | verdict |
|---|---|---|---|
| p61-acc-pseudolabels | LOCO targets, Qwen-covered pairs (69,840 / 41,640) | Positives, precision: c ≥ .95 & q ≥ .97 **0.960 / 0.982** vs c alone 0.916 / 0.908. p (chain blend) ≥ .95 & q ≥ .97 0.906 / 0.980 vs p alone 0.715 / 0.909. Negatives, NPV: c ≤ .05 & q ≤ .03 0.984 / 0.956 vs c alone 0.981 / 0.897. The c ≥ .7 & q ≥ .7 / ≤ .3 set is 0.928 / 0.937 accurate | keep: agreement beats either view |
| p61-g1-self_c95_c05_w1 | LOCO | self-training (c alone on the same pairs): +0.00051 / **−0.00274** | drop (reproduces P27's split) |
| p61-g1-p95q97_p05q03_w1 | LOCO | chain blend p as view 1: −0.00192 / −0.00138 | drop |
| p61-g1/g2/g3 (w = 1, view c) | LOCO | c95q97 +0.00057 / +0.00100; c90q90 +0.00083 / +0.00067; c80q80 +0.00119 / +0.00100; **c70q70 +0.00132 / +0.00167**; c50q50 +0.00132 / +0.00263; c98q99 +0.00030 / +0.00046. Pos-only +0.00064 / +0.00031; neg-only −0.00022 / +0.00043. Half the pseudo-labels (sub50): about half the gain | keep: looser agreement is better |
| p61-g2-q97_q03_w1 | LOCO | Qwen alone labels: +0.00128 / +0.00190. Mostly covered-pair fusion: UI covered-band AUC goes to Qwen's own 0.909. IU removes 1,573 pairs, 37% false | not used (it copies Qwen, and France Qwen calibration is unknown) |
| p61-g4 (weights, c70q70) | LOCO | w 0.5 +0.00099 / +0.00212; w 1.5 +0.00145 / +0.00127; **w 3 +0.00147 / +0.00015**. UI degrades at high weight | – |
| p61-g5-c70q70_c30q30_share (the France rule) | LOCO | **+0.00118 (h1 +0.00124 ± 0.00014) / +0.00101 (h1 +0.00085 ± 0.00025)**. c50q50-share +0.00097 / +0.00255 | keep |
| p61-g6-c80q80_c20q20_share | LOCO | +0.00101 / **−0.00030** (seeds 3–5: −0.00007). c95q97-share +0.00055 / +0.00129 | UI is fragile at neighbouring bars |
| p61-g7-null_s345 / null_s678 | LOCO | P55 reseeded, no pseudo-labels: −0.00001 / +0.00042 and +0.00015 / +0.00073. **This is the noise floor.** c70q70-share at seeds 3–5 / 6–8: +0.00109 / +0.00116 and +0.00115 / +0.00122 | control |
| **p61-bag9-c70q70-share** | LOCO, 9 seeds on both sides | vs P55 9-seed: **full +0.00108 (h1 +0.00113 ± 0.00014) / +0.00075 (h1 +0.00064 ± 0.00024)**; **hybrid +0.00120 (h1 +0.00122 ± 0.00010) / +0.00053 (h1 +0.00041 ± 0.00022)**. vs shipped P55 (3 seeds): full +0.00112 / +0.00117, hybrid +0.00128 / +0.00092 | keep |
| (decomp.py, c70q70-share) | LOCO | Gain on pairs without Qwen scores (no pseudo-label): +0.00040 / +0.00068 of the +0.00118 / +0.00101. The combiner learns target patterns, not just Qwen | – |
| (comp.py, 9 seeds) | LOCO components vs P55 | IU: re-admits 301 (35% true) **−0.00028**; new vetoes 222 +0.00011; adds gained 2,909 (92% true) **+0.00151**; adds lost 494 −0.00023. UI: re-admits 201 (62%) +0.00003; new vetoes 117 (2%) +0.00022; adds gained 4,697 (72.5%) +0.00002 (h1 −0.00014); adds lost 447 (13% true) **+0.00094** | – |
| (comp.py, France, label-free) | test France | Re-admits 1,268 = 0.15% of accepted pairs (LOCO 0.07%), Qwen ≥ .5 61%. New vetoes 16. Adds gained 1,613 realized (3,097 candidates), Qwen ≥ .5 63%; P38's audit had the LB-proven gc70 adds at 62% and P54 at 64%. Adds lost 73, none Qwen-scored. France's changes are mostly the components that were weak on LOCO | – |
| p61-sub61f9-fullswap | test France + LOCO per-pair reweighting | +2,879 / −89. Estimate **−0.00018 / +0.00013** (h1 −0.00016 / +0.00005) | drop |
| **p61-sub61h9** | test France + LOCO per-pair reweighting | +1,613 / −73, 1,665 S1. Estimate **+0.00041 / +0.00006** (h1 +0.00042 / +0.00004). The 3-seed build had the same estimate (+0.00041 / +0.00005); its add set overlaps 1,532 of 1,613 | **keep → staged** |
| (fr_build repro) | test France | P55's step rebuilt on sub_clean_fr = sub_clean with **0 pairs missing**. The only extras are sub_clean's 49 dup copies | pass |
| validator p61-val-0927-0645 | test | 3-seed sub61 (fbc88709…) and sub61h (4f814938…): **both PASS** `--check-ids` | pass |
| validator p61-val9-0927-0653 | test | sub61h9 (md5 5b3eada5…): **PASS** `--check-ids` (9,969,589 valid IDs; 47 billable s, spot) | pass |

AWS: two ml.m5.xlarge spot validator jobs, 110 billable s in total (≈ $0.01).

## What failed      (approaches tried and dropped, with the number that killed them)
- **Self-training** (c alone, same pairs, same weight): UI −0.00274. It is P27's failure reproduced on this frame, and Qwen's agreement is what fixes it.
- **Chain blend p as the co-training view:** −0.0019 / −0.0014. p alone at ≥ .95 is only 0.715 precise on the US target.
- **The full swap on France:** the France-reweighted IU estimate is −0.00018, driven by the re-admits:
  - France re-admits 1,268 of P55's 3,198 vetoes, twice LOCO's rate;
  - on LOCO, IU re-admits are 35% true and cost −0.00028.
- **High pseudo-label weight:** c70q70 at w 3 gives UI +0.00015, and c80q80 with share weights (≈ 2 per label on UI) gives −0.00030.
- **Not tried:** Qwen scores as a combiner feature. That is a different experiment, and it needs LOCO Qwen on the whole band, while P38's LOCO scores cover only 28–37% of it.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- **`p61.py`** (self-check runs on import-as-main):
  - `cache <dr>`: P55's LOCO frame + Qwen r → `data/work/p61/f_<dr>.npz`. 1 min, 0.8 GB.
  - `acc` → `acc.json`.
  - `grid <g1..g7>` → `cot_<g>.json`, `data/work/p61/c_<spec>_<tgt>.npy` + ledger rows. ≈ 45 s per spec, 0.6 GB.
  - `frcache` → `data/work/p61/france.npz`. 1 min, 1.3 GB (ru_maxrss).
  - `frfit <spec> [seeds] [tag]` → `cf_<spec><tag>.npy` + boosters. 1 min, 0.6 GB.
  - `frbuild <spec> <out> [full|hybrid]` → `output/<out>` + `p61_info.json` (component counts). 20 s, 1.1 GB.
  - Core functions: `plab(fr, v1, pos, neg)`, `pl_weight(spec, lab, n_source)`, `cotrain(trn, tst, cb, spec, seeds)`, `arm_a(fr, c, v, u)`, `p55_sets(c, cur, a1, ax, fz, rg, v, u, cv)`.
- **Analysis scripts:**
  - `decomp.py <spec…>`: coverage split and the hybrid on LOCO.
  - `comp.py <spec>`: swap components on LOCO + the France label-free profile → `comp_<spec>.json`.
  - `est.py <spec> <out…>`: France-reweighted estimate → `est_<spec>.json`.
  - `bag9.py <spec>`: the 9-seed bags.
- **9-seed France bag:** the logit mean of `cf_<spec>.npy`, `cf_<spec>_s345.npy` and `cf_<spec>_s678.npy` → `cf_<spec>9.npy`.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Licence / rules:** LightGBM (MIT). The only extra input is P38's Qwen3-Reranker-4B scores (Apache-2.0, 4.0B). No external data, no country filter: the rule keys on P55's stages and Qwen coverage.
- **The LOCO Qwen coverage is partial and selected in-domain.** P38 scored P13's m2_big band, and m2_big saw the LOCO target's other folds. So *which* target pairs got pseudo-labels was chosen by an in-domain model; the labels themselves are source-only.
  - I treat this as second-order.
  - France's Qwen covers the whole band, selected by France's own OOD model, so France has more pseudo-labels (164k = 16% of source rows vs LOCO's 4–10%) and more negatives (75% vs ~40%).
  - The share-matched class weights (pos 1.02 / neg 0.26) are my fix for that. They are an assumption, validated only as "the LOCO mean share", which passes both directions.
- **US→India is fragile.** Seed noise alone is +0.0004 to +0.0007, and one neighbouring bar (c80q80-share) is negative. India→US is solid in every configuration (≥ +0.0010, noise ≤ +0.00015).
- **The hybrid was chosen after seeing the components.** That is an in-sample structural choice between two options. It is positive on LOCO in 5 of 6 configurations (the exception is c80q80-share on UI).
- **Pooled Qwen calibration on France is unknown.** The France add_gain pairs are 63% Qwen ≥ .5, like the LB-proven gc70 adds. #54 under-delivered its LOCO estimate 4–5×, so the +0.00001–0.00006 LB is an upper bound.
- **Dup interplay:** the change set is applied after sub_clean's dup step. No dup copies are made for the 1,613 new pairs, and none of sub_clean's 49 dup copies are touched.

## Open questions   (for the master; each with your best guess)
- **Upload sub61h9 on its own?** Best guess: no. The expected +0.00001–0.00006 LB is below the LB's resolution. Fold the change set into the final France stack only if P50 is rebuilding anyway.
- **Re-seed P55 itself?** On LOCO, P55's shipped 3-seed bag is an unlucky draw on UI: a 9-seed bag is +0.00005 / +0.00041 over it. France's P55 would move a few hundred pairs. Best guess: ±0 on the LB, not worth a slot.
- **Co-train other France stages** (P54 rescue / P48 veto thresholds on Qwen + bge agreement)? Best guess: the same pattern holds (adds help, vetoes don't transfer), but it is untested and small.
