# P15-research — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
- Winners of similar contests got their late gains from **second-stage features over a record's neighbours** and from
  **per-language/per-country statistics**, not from bigger models. Foursquare 1st: +0.0056 from merging via common
  matches. Foursquare 13th: GNN on the 2-hop subgraph, +0.02. Shopee 2nd: neighbour-context LightGBM, +0.012. LECR 1st:
  TF-IDF per language. Clustering and transitive closure do not apply to us: S1 is deduplicated, so the graph is a star.
- **Measured here: our 65 features cannot tell a generated sibling from a noisy true pair on the house number.** The
  generator's orphan siblings always take hn + {1,2,3,4,5,7,9,11,13,21}; noise on true pairs is a digit drop or ±1/±2.
  A residual LightGBM on #6's blended p with a signed hn delta + SX↔SX same-hn support gives **US +0.00127 ± 0.00011
  (11 SE), India +0.00000 ± 0.0001**. On m2_big p (France's regime: no CE) it gives US +0.0031, India +0.0014.
- **France is where it bites (label-free).** France accepts **22 sibling-offset pairs per 1k at t 0.78 vs US 2.6 / India 3.9 per 1k**.
  Raising t to 0.95 (+0.0011 LB) removes mostly those (→ 5.6/1k), and it also drops true pairs with house-number noise:
  France's accepted hn-mismatch share is 1.7% vs US 8.8%. Recommendation: spend **one LB slot** on the ready France-only
  sibling rule. Build the stacker for India/US (brief A) now.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
Three briefs, ranked. The full idea table is in Evidence. Everything here uses only train labels, generator facts
measured on train (P12), and MIT/BSD libraries.

**Brief A — P17-hn-stack: generator-aware house-number + SX↔SX support stage for India/US (fits the freeze: 3–4 h CPU, no GPU).**
- *Inputs:* the current India/US test probability files (the #6/P11 blended p over P10's kept pairs) and #5's full-list
  test p `data/work/p4/prob_test_{c}_big*.parquet` (for support counts, see risk below). Record hn from
  `data/work/rf_test_{s1,sx}_{c}.parquet`. Holdout rows from `probes/P15-research/hn_check.py <c>` (13 s India / 11 s US)
  plus the CE r cache `data/work/p11/r_s2P_holdout.parquet`.
- *Change:* a new stage between the CE blend and the argmax (`submit_probs` / run.py decide step): `p' = sigmoid(logit p + f(X))`.
  - `f` = LightGBM with 200 trees × 31 leaves, lr 0.05, min_data_in_leaf 200, `init_score = logit p`.
  - X = `[lp, cls, d, eq, n_same, p_same, n_s1hn]`, built by `hn_stack.py`'s `hn_cls` / `support`:
    - `cls`: none/eq, +SIB (d in the offset set), digit drop/add, small other, far;
    - `d`: signed hx − h1, clipped to ±100;
    - `eq`: hn equal;
    - `n_same`: other SX of this S1 with the same hn;
    - `p_same`: best p among those;
    - `n_s1hn`: other SX of this S1 carrying the S1's own hn.
  - Train on holdout fold-0 S1-hash half 0 only (out-of-sample for m2_big folds 2–5 and the CE folds 4–9). Retune t on half 0 (US 0.73 measured).
  - One model per labelled country, or pooled (see open questions). France stays untouched unless brief B says otherwise.
- *Validate:* re-measure on the #6 yardstick itself (P10 `evalp.Bench` / P4 `big._blend_eval`, paired SE on half 1).
  Mine excludes blocking misses from n_true, so its level is higher (0.991 vs 0.988); the delta should carry over.
  Also recompute `n_same` / `p_same` on the **pruned** holdout list (P10 hold_q: q ≥ 0.03 | rule_acc) and confirm the gain.
  - The hn-only arm (no list dependence): US +0.00079 ± 0.00010.
- *Acceptance:* India+US report half ≥ +0.0003 at ≥ 3 SE, and India ≥ −0.0001.
  - Expected LB: 0.85 × (0.45 × 0.00127 + 0.55 × 0) ≈ **+0.0005**.

**Brief B — one LB slot now: France-only sibling rule (15 min; file ready, validator not yet run).**
- *File:* `data/work/p15/prob_test_France_sib.parquet`. It is `data/work/p10/prob_test_France.parquet` (#6c's France p)
  with p := 0 on the 170,752 pairs whose hx − h1 ∈ {3,4,5,7,9,11,13,21}. Same rows and order as the source (asserted).
  - At t 0.95, 4,367 accepted pairs go (776,184 → 771,817); 293 France S1 lose their only answer.
  - The argmax fallback never re-accepts (pair and SX variants are identical on holdout).
- *Submit:* the last LB-scored India/US files unchanged, with France = that file at t 0.95. Compare with the same config
  without the rule (#6c = 0.980954 if India/US are #6's).
- *Read the result:*
  - **≥ +0.0002:** France's siblings are FPs, as in the US holdout (+0.00097 on m2_big p). Keep the rule, and use a
    second slot for France t 0.88 + rule: the threshold was doing the sibling filtering, so a lower t should recover recall.
  - **≤ −0.0002:** France's house-number noise is India-like (rule −0.0017 on the India holdout). Drop the rule, and keep
    brief A's stacker off France.
- *Why a rule and not the stacker for France:* the stacker's hn effect differs by country (US +, India ≈ 0). A pooled
  stacker would average the two on an unseen country; the rule is a clean one-bit test of which regime France is in.
- It composes with P14's IDF refit (the rule zeroes pairs by hn, independent of p).

**Brief C — branch-word feature (2–3 h; validate on US/India, France by LB).**
- *Label-free finding:* the token an SX core **adds** to its argmax S1's core is a short, flat list — the generator's
  sibling vocabulary.
  - US test (p ≥ 0.5): partners 2,999; then federation / council / foundation / district / association / board / trust /
    commission / society / authority at 865–939 each.
  - France: fils 2,604, associes 1,607, cie 1,499, groupe 1,269, developpement 1,158. That top-5 holds 69% of 12,478
    added-token pairs.
  - The GBDT sees only similarity scores, never word identity, so it cannot learn "the extra word is a branch word".
  - A list mined from US/India train (group/holdings/partners/ventures…) never fires on France's groupe/developpement.
- *Change:* add columns `add1` (the SX adds exactly one core token) and `add_in_list` (that token is in the country's
  top-k added tokens) to brief A's stacker.
  - Train-only list for US/India: P12's `orph_sib_added`.
  - France list, two options:
    - (i) label-free counts on test France, which is open question 4 (test text);
    - (ii) a hand list from domain knowledge (groupe, développement, holding, participations), the same footing as
      P6's france.py.
  - Do NOT include fils / associés / cie blindly: P6 showed real France S1 names carry "& Fils", and dropping them cost coverage.
- *Validate:* US/India holdout with the same residual yardstick (≥ +0.0002 at 3 SE). France: an LB slot with the rule
  "p := 0 when add_in_list", France only, as in brief B.

**Not recommended** (numbers in Evidence):
- EF / rank-dependent thresholds / min2 / relative-gap acceptance: P5 already covered these (D2, D3r, D7+EF, all ≤ +0.0002).
- Clustering, transitive closure, Hungarian, one-to-one assignment: they are for dirty or one-to-one graphs; ours is a star.
- Self-training, synthetic target pairs: P6 −3.5 pt, P12 −0.8 pt.
- A new label-free threshold estimator (ATC/BBSE/EM): the LB already calibrates France's t.
- Hard sibling rule for India/US: India −0.0017.

## Evidence         (table: ledger name | world | key metrics | verdict)
Yardstick for this probe's own runs (`hn_stack.py`, `sib_rule.py`):
- Rows: P10's holdout rows `data/work/p10/hold_{c}.parquet`, i.e. fold-0 S1 plus every train pair that competes for their SX.
- p: m2_big p, or #6's p (P11 2-epoch CE blended on the band, w [0.617, 0.666, 0.261], via the r cache).
- Decision: argmax per SX over all rows + threshold. Tuned on S1-hash half 0 and reported on half 1, paired per-S1 SE.
- n_true = true rows in the file, so blocking misses are excluded: levels are ~0.003 above the #6 numbers, and deltas are fair.
- **Valid ledger rows:** from 00:25 on (residual boosting, `init_score = logit p`). The earlier `hn-signed-delta-stack-*`
  rows (00:13–00:24) used a plain GBDT stacker on lp, which loses threshold resolution in the CE world (flat plateaus,
  −0.0005 to −0.0008 vs raw p). They are superseded.

**Measured (this probe):**

| ledger name (`P15-research/…`) | world | key metrics | verdict |
|---|---|---|---|
| (hn_check India / US) | holdout fold 0, m2_big p, t .69 | hn mismatch in missed true pairs: India 21% vs 12% of all true, **US 36% vs 8.7%**. In false-accepted pairs: India 22%, **US 46%**. US +SIB false-accepted 742 of 903 false hn-mismatch accepts. True share of +SIB rows by d: US d=1/2 0.07, **d≥3 ≤ 0.005**; India d=1/2 0.11, d≥3 0.01–0.05 | signal exists |
| hn-signed-delta-stack-India / -US (00:25 / 00:27) | holdout-dense-half1, m2_big p | residual [lp, cls, d, eq]: India **+0.00077 ± 0.00013**, US **+0.00215 ± 0.00018** | keep (France regime) |
| hn-signed-delta-stack-India-onCE / -US-onCE (00:26 / 00:28) | holdout-dense-half1, #6 p | India +0.00007 ± 0.00006, US **+0.00079 ± 0.00010** (t 0.64 → 0.72) | keep US |
| hn-signed-delta+support-stack-India / -US | holdout-dense-half1, m2_big p | + n_same, p_same, n_s1hn: India **+0.0014 ± 0.0002**, US **+0.00310 ± 0.00018** | keep (France regime) |
| hn-signed-delta+support-stack-India-onCE / -US-onCE | holdout-dense-half1, #6 p | India +0.0000 ± 0.0001, **US +0.00127 ± 0.00011** (US t 0.73). US runtime 18 min under load, 1.7 GB | **keep (brief A)** |
| sib-rule-India / -US (+ -onCE) | holdout-dense-half1 | hard rule p := 0 if hx − h1 ∈ {3,…,21} set: India **−0.00168 ± 0.00017** (m2_big) / −0.00190 (#6 p); US **+0.00097 ± 0.00012** / +0.00035 ± 0.00009. Pair-level and SX-level variants identical | US-like regime + / India − |
| hn-delta-class-mix-test-accepted | test, accepted pairs | France (t .95): hn both 0.906, hn mismatch 1.74%, **+SIB d≥3 5.63/1k**, digit drop 1.4/1k. US (t .71): 0.889, 8.77%, 2.56/1k, 42.9/1k. India (t .71): 0.615, 12.3%, 3.90/1k, 31.2/1k | France over-accepts siblings, under-accepts noisy hn |
| (inline, France t .78 / .88) | test-France | +SIB d≥3: **22.45 / 13.31 / 5.63 per 1k** at t .78 / .88 / .95. Other classes barely move: far 5.9k → 5.2k → 4.3k pairs, digit drop 1.3k → 1.2k → 1.1k | the France threshold acts as a sibling filter |
| sib-rule-France-test-file | test-France | 1,332,296 pairs, 170,752 zeroed. Accepted: t .78 840,019 → 821,175; .88 811,100 → 800,311; .95 776,184 → 771,817 | LB slot (brief B) |
| (branch_words.py France / US) | test, argmax p ≥ .5 | share that add exactly one token: France 1.4%, US 1.0%. Top-10 hold 69% / 50%. Lists quoted in brief C | brief C |

**Literature and competition map, ranked for OUR pipeline.** Gains: "LB" = expected leaderboard delta. Effort in hours.
"Freeze?" = fits before 27 Sep 14:00.

| # | idea | source | what changes in OUR pipeline | expected gain (India/US · France) | effort | rule risk | freeze? |
|---|---|---|---|---|---|---|---|
| 1 | Generator-aware hn delta + SX↔SX same-hn support, 2nd-stage residual GBDT | P12 generator facts; lieu (hn must match exactly), Comber & Arribas-Bel 2019 (street + hn are the top features); Foursquare 1st/13th, Shopee 2nd 2nd-stage context models | new stage after the CE blend (brief A) | **measured US +0.0013, India 0 → LB +0.0005**. France: unknown sign until brief B | 3–4 | none | yes |
| 2 | France-only sibling rule (LB diagnostic) | this probe (US holdout +0.001; France excess 6–9×) | France p := 0 on hn + SIB3 pairs (brief B) | France ±0.003 → **LB ±0.0005**, sign unknown | 0.25 | none | yes |
| 3 | Per-country IDF / frequency statistics (target text) | LECR 1st (TF-IDF per language); Foursquare 2nd (per-country candidate mix); P14 proxy: −1.23 pt, 96% won back | block.py / features TF-IDF fit incl. France S1+S2/S3 | India/US 0 · **France largest known lever** (P14) | P14 | **open Q4: fits on test text** | P14 |
| 4 | France CE with out-of-domain blend weights | DADER (gain grows as MMD shrinks; KD/MMD stable, reweighting worse); P13 LOCO: bge +0.0027 / +0.0013 both ways | CE band for France with w_ood | France +0.001–0.003 (P13 LOCO) → LB +0.0002–0.0004 | P13 | none (bge-m3 MIT) | P13 |
| 5 | Branch-word feature (country's added-token list) | this probe; POI review (chains / branch names cause name-only FPs); Splink TF-adjusted comparisons | stacker columns add1 / add_in_list (brief C) | India/US small (the CE reads words) · France possibly large (GBDT has no word identity) | 2–3 | France list from test text = open Q4; a hand list avoids it | yes |
| 6 | SX↔SX name-similarity support / conflict ("merge if >50% common", 2-hop GNN) | Foursquare 1st (+0.0056 CV), 13th (+0.02), Shopee 2nd GAT (+0.002) | extend brief A's support key from hn to (core, hn) or a rapidfuzz SX–SX score | hn-key support already adds US +0.0005 over hn-only on #6 p, India 0. Name key: ≲ +0.0003 | 4–6 | none | tight |
| 7 | Adversarial-validation feature audit (France vs US/India) | adversarial validation; TableShift (robustness methods trade in-domain accuracy) | drop / relativise the most-shifted features | France ?, India/US −0.0001–0.0003 | P14 has the SMD table | none | P14 |
| 8 | Collapse identical SX before scoring | SIGMOD 2022 winner (WBSG) | block/features | ~0: argmax is already per SX | 1 | none | skip |
| 9 | Expected-F0.5 set decoding, rank-dependent t, min2, relative gap | Ye et al. 2012, Dembczynski 2011, Lipton 2014; Shopee 1st min2 (+0.019); LECR gap rule | decide | **P5 measured: D2 EF hurt, D3r ±0.0002, stack+EF +0.0001** | — | none | done, drop |
| 10 | Clustering / transitive closure / 1-to-1 assignment (MERGE-CENTER, CLIP, UMC, Hungarian) | Hassanzadeh VLDB 2009; Saeedi (CLIP, MSCD-AP); Papadakis 2021 | decide | ≤ 0: S1 is clean, S1 clusters hold several S2 and S3, and per-SX argmax is already exact for "≤ 1 S1" | — | none | drop |
| 11 | Label-free threshold estimators (ATC, BBSE, Saerens EM, Auto-FuzzyJoin 2d-ball) | Garg ICLR 2022; Lipton 2018; Saerens 2002; Auto-FuzzyJoin 2021 | France t | ~0: covariate shift breaks label-shift assumptions, and the LB already calibrates t | 2 | none | drop |
| 12 | Self-training with ρ-balanced, neighbourhood-filtered pseudo-labels | Sudowoodo, TransER (+9% F1 over naive transfer) | matcher retrain | P6 plain self-training −3.5 pt. Filtered version untested, >1 day | >8 | none | no |
| 13 | DADER MMD / AnyMatch attribute-level pairs for the CE | DADER SIGMOD 2022; AnyMatch 2024 (−3.1 F1 without attribute pairs) | CE retrain | France ?, GPU-heavy, validation only via LOCO | >8 | none | no |
| 14 | Ditto MixDA / synthetic augmentation | Ditto VLDB 2021 | train data | P12: −0.8 pt with source noise rates | — | none | drop |
| 15 | Country token / country multitask in CE input | Foursquare 13th (+0.003), 2nd (country prefix) | CE input | France is an unseen token | — | none | drop |
| 16 | libpostal parser / cleanco legal forms / unidecode | Foursquare 4th / 7th (unidecode); libpostal; cleanco | normalize | ~0 beyond P6's france.py | 2 | **unidecode GPL = banned. libpostal's parser model is trained on OSM = external-data risk (its dictionaries alone are fine). cleanco MIT OK** | skip |

Sources:
- Foursquare 1st: https://www.kaggle.com/competitions/foursquare-location-matching/writeups/re-waiwai-1st-place-solution
- Foursquare 2nd: https://www.kaggle.com/c/foursquare-location-matching/discussion/336062 ·
  https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336090
- Foursquare 4th: https://www.kaggle.com/competitions/foursquare-location-matching/discussion/335810
- Foursquare 7th: https://future-architect.github.io/articles/20220720a/
- Foursquare 13th: https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336124
- Foursquare leak: https://www.kaggle.com/competitions/foursquare-location-matching/discussion/338035
- Shopee 1st: https://www.kaggle.com/c/shopee-product-matching/discussion/238136
- Shopee 2nd: https://www.kaggle.com/competitions/shopee-product-matching/discussion/238022
- LECR 1st: https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/discussion/394812
- SIGMOD 2022 (WBSG): https://www.uni-mannheim.de/en/news/wbsg-wins-sigmod-programming-contest-2022/
- DADER: https://dbgroup.cs.tsinghua.edu.cn/ligl/papers/entity-sigmod-2022.pdf · Ditto: https://arxiv.org/abs/2004.00584 ·
  ZeroER: https://arxiv.org/abs/1908.06049 · Sudowoodo: https://arxiv.org/abs/2207.04122 ·
  AnyMatch: https://arxiv.org/abs/2409.04073 · TransER: https://users.cecs.anu.edu.au/~christen/publications/kirielle2022edbt-transer.pdf ·
  Unicorn: http://www.dit.unitn.it/~pavel/OM/articles/Tu_Sigmod23.pdf
- Label-free thresholds: ATC https://arxiv.org/abs/2201.04234 · BBSE https://arxiv.org/abs/1802.03916 ·
  EM/MLLS https://arxiv.org/abs/1901.06852 · Auto-FuzzyJoin https://arxiv.org/abs/2103.04489 ·
  TableShift https://arxiv.org/abs/2312.07577
- Clustering / assignment: Hassanzadeh http://www.vldb.org/pvldb/vol2/vldb09-1025.pdf ·
  CLIP https://dbs.uni-leipzig.de/files/research/publications/2018-6/pdf/eswc.pdf · Papadakis https://arxiv.org/abs/2112.14030
- F-measure decoding: Ye et al. https://arxiv.org/abs/1206.4625 · Lipton et al. https://arxiv.org/abs/1402.1892
- Name / address: Cohen et al. 2003 https://www.cs.utexas.edu/~ai-lab/pubs/ravikumarKDD03.pdf ·
  lieu https://github.com/openvenues/lieu · Comber & Arribas-Bel https://onlinelibrary.wiley.com/doi/full/10.1111/tgis.12522 ·
  Splink TF https://moj-analytical-services.github.io/splink/topic_guides/comparisons/term-frequency.html ·
  cleanco https://github.com/psolin/cleanco

## What failed      (approaches tried and dropped, with the number that killed them)
- **Hard sibling rule on India:** −0.00168 ± 0.00017 (m2_big p), −0.00190 (#6 p).
  - India's true-pair hn noise is wider (hn_change 0.20 vs US 0.10, P12), so the pairs the model already accepts with a
    +3…+21 delta are mostly true.
  - The rule is a US-like-regime tool. That is exactly why France needs the LB bit (brief B) before anything is applied to it.
- **Plain GBDT stacker on logit p** (first four ledger pairs, 00:13–00:24): monotone or not, it forms flat plateaus
  where the band is sparse (s = 0.697 for p 0.6–0.8). It then cannot place t at 0.69, and loses 0.0005–0.0008 vs raw p
  in the CE world. Fixed by residual boosting from `init_score = logit p`, so that base = raw p exactly.
- **Support features on top of the CE for India:** +0.0000 ± 0.0001. The CE already resolves what hn support adds on
  India (on m2_big p it adds +0.0006).

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `hn_check.py <c>`:
  - `hn_cls(h1, hx) -> int8`: 0 none/eq, 1 +SIB, 2 digit drop/add, 3 small other, 4 far.
  - It writes `data/work/p15/_hn_<c>.parquet` (holdout rows + m2_big p + h1/hx; India 4.46M rows / 65 MB, US 7.08M / 108 MB).
  - 13 s / 11 s, peak 1.1 / 1.7 GB. Self-check included.
- `hn_stack.py <c> [ce] [sup]`:
  - `ce_blend(h, c)`: #6's blended p on holdout band pairs.
  - `support(h, h1, hx) -> DataFrame[n_same, p_same, n_s1hn]`.
  - `evaluate(h, s, half, t, am)`: per-S1 F0.5.
  - It fits the residual model, sweeps t 0.05–0.98 on half 0, reports half 1, and logs a ledger row.
  - 0.5–1 min India, 1–18 min US (load-dependent), peak 1.7 GB.
- `sib_rule.py <c> [ce] | France-test`:
  - `sib_mask(h1, hx)`: holdout rule check.
  - `France-test` writes `data/work/p15/prob_test_France_sib.parquet` (same schema as `p10/prob_test_France.parquet`).
- `hn_test_mix.py <countries>`: hn class mix of accepted test pairs (France t .95; India/US #6 blend t .71).
- `branch_words.py <countries>`: added-token counts on test argmax pairs (aggregates only).

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Rules:**
  - Nothing here uses external data or a new dependency (LightGBM MIT, pandas/numpy BSD).
  - The sibling offset set is measured on train labels (P12 `rates_*.json`).
  - Brief C's France list from test text, and P14's IDF, both touch open question 4.
- **The stacker is fit on holdout half 0.** Shipping it spends the holdout's tuning half, exactly as the CE blend weights
  already do. Keep half 1 untouched for the report number.
- **Pruned vs full candidate lists:** support counts were computed on the full blocking list (P10 hold rows), but #6's test
  files are pruned (4.83/S1). Compute test support on #5's full-list p (`p4/prob_test_{c}_big*.parquet`), or retrain on the
  pruned holdout rows. The hn-only arm has no such dependence (US +0.00079).
- **France:** the stacker learned opposite hn behaviour in the US and India, so its France effect is not predictable
  without brief B's LB bit. The France label-free statistics (sibling rate 6–9× US; hn-mismatch share 1.7% vs 8.8%) are
  consistent with over-confident sibling acceptance. They are also consistent with France simply having fewer hn changes;
  only the LB separates the two.
- **Yardstick:** my F0.5 excludes blocking misses and uses per-country models. Re-measure with P10's `evalp` before adopting.
- **Interrupted run:** the laptop hibernated at 00:31. Both `_hn_*.parquet` files were verified (open; row counts
  4,459,794 / 7,084,014), and the two US support arms that were lost were rerun.

## Open questions   (for the master; each with your best guess)
- **Per-country or pooled stacker for India/US?** Best guess: per country. The hn effect has opposite signs across the
  two, and a model chosen by country for labelled countries is not a country feature. France needs pooled or none; brief B decides.
- **Does brief A compose with P14's IDF refit and P13's France CE?** Best guess: yes for India/US, since P14 does not
  change them. For France, order the LB slots IDF → rule → CE, one lever per slot.
- **Retrain m2_big with the hn columns instead of stacking?** Best guess: not before the freeze. It needs X recomputed
  for 27.5M test pairs + train, a big.py retrain, a pruner retrain and a CE blend refit (> 4 h). The residual stage gets
  the measured gain in < 1 h of compute.
- **Is 0.95 still France's best t once siblings are handled?** Best guess: no, lower (0.85–0.9). Between t .78 and .95,
  the France removals are ~29% +SIB-offset pairs, and the rest includes noisy true pairs that a sibling-aware France
  would keep.
