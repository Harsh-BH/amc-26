# P23-research — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **We lag on France, and France is the only gap large enough to matter.** LB #9ab 0.984072 vs #1 0.990556 (P26 brief,
  26 Sep 15:05) = −0.0065. LB-implied F0.5 is India/US 0.990 (= holdout) and **France ≈ 0.950** (0.938 at D4; IDF + bge
  added +1.2 pt). Our 0.0159 LB loss splits into India 0.0045, US 0.0040 and **France 0.0075**. #1's France is at least 0.976
  even if its India/US sits at the P5 oracle (0.9932), and 0.994 if its India/US equals ours. So France is 57–100% of the gap.
  Inside France (my rebase of the D4 estimates): matcher FN on candidates ≈ 2.8 pt, FP + wrongly matched singletons
  ≈ 1.5 pt, blocking ≈ 0.4 pt. Up to 0.9 pt of that is S1 we leave empty (#9ab: 6.71% empty against an implied
  singleton share of 5.6–5.8%). That share assumes India/US LB = holdout: each −0.001 there adds 0.57 pt to it.
- **The metric sets two bars.** Adding a pair pays off above 0.67–0.75 precision (at 2–4 answers per S1). Removing one pays
  off above 30% false. France's t of 0.985 rejects whole record-relation classes whose label-backed precision is 0.90–0.998,
  so accepting by class is the cheapest France lever left.
- **The moves, in build order.** Each must be in a run.py recipe committed before the **27 Sep 14:00 code freeze**:
  - (1) **Stop pinning France to D4's empty rate:** upload the staged #9ab_t975 with P15's sibling rule (Brief 3).
  - (2) **Key tiers on the #9ab France file, with the '(12)' house number parsed first** (Brief 1: K + M, L gated). K alone
    is 7,770 pairs on 7,380 S1 at label-analog precision 0.97–0.98, and it fills 267 empty S1. LB ≈ +0.0005, no rule risk.
  - (3) **One France rebuild on the bge world: hn parse + P20's R1 filler strip as a GENERIC change** (Brief 5). This
    replaces the staged #20a and Brief 1's old p-floor F. LB +0.0003 to +0.0005.
  - (4) **Let bge veto France's confident GBDT accepts** (Brief 2, R04). The CE never scored 97% of accepted France pairs.
    In a band-selected LOCO subset, confident pairs with bge r < 0.5 are only 27–31% true. One AWS job, ≈ $0.34;
    LB −0.0002 to +0.001.
  - (5) **India/US: ship P23's a10 residual stacker in its p2-masked form** (Brief 7). The gain is +0.00015 to +0.00021 LB
    after reweighting the holdout to test's mix (measured here: +0.000166 / +0.000209). It rides in the combined file.
  - Protect the rank with the per-source caps (86 S1 in #9ab break them, 0 in train GT) and a run.py recipe for every
    file that could win (Brief 6).
  - Adding R05's org-word tier (Brief 4, now gated at s(w) ≤ 0.4) should bring us to about 0.986. The rest needs a
    France-invariant model (P26/P27, running).

## Recommendation   (exactly what the master should adopt: module, function signature, config)

### Where we lag — error budget
India/US figures are exact. They come from the P23 dump of #8c's holdout, split by a per-S1 Shapley value over 7 error
types, so the rows add up. France is label-free: the gap analysis's D4 estimates, rebased to #9ab by subtracting the P14
proxy's IDF split (0.34 pt blocking, 0.47 low-p, 0.47 FP). The rebased France rows add up to 0.048, against 0.050 implied
by the LB. LB-equivalent = F0.5 lost × the country's share of test S1 (India 0.468, US 0.383, France 0.1498).
1 France pt = 0.0015 LB, and 1k France S1 each gaining +1.0 = +0.0006 LB (259,452 France S1).

| bucket | India F0.5 lost | US F0.5 lost | France F0.5 lost (#9ab, est.) | LB-equivalent India / US / France | owner / brief |
|---|---|---|---|---|---|
| **total** | 0.00952 (holdout 0.99048) | 0.01049 (0.98951) | **≈ 0.050** (LB-implied 0.9505; 0.062 at D4) | 0.0045 / 0.0040 / **0.0075** | — |
| (a) blocking miss | 0.00290 oracle; **0.00218** with an address | 0.00231; **0.00101** | ≈ 0.004 (0.007 at D4) | 0.0010 / 0.0004 / 0.0006 (recoverable part) | P24; R15 |
| (b) pruner miss | 0.00136 oracle, ~0.0002 real | 0.00125, ~0.0002 | ≤ 0.0003 | ≤ 0.0002 in total | do not chase (P10 −0.00024) |
| (c) matcher FN on candidates | 0.00397 (stolen 0.00185, low-p 0.00212) | 0.00535 (0.00198, 0.00337) | **≈ 0.028** (0.033 at D4) | 0.0019 / 0.0020 / **0.0042** | India/US (0.0039 LB): **Brief 7** (a10 stacker). France: **Briefs 1, 4, 5**; P26/P27 |
| (d) FP, non-singleton S1 | 0.00092 (same-name 0.00046) | 0.00123 (same-name 0.00064) | ≈ 0.011 (0.016 at D4) | 0.0004 / 0.0005 / 0.0017 | India/US (0.0009 LB): **Brief 7**. France: **Brief 2**; R08 |
| (e) singletons matched | 0.00036 | 0.00035 | ≈ 0.004 (0.002–0.006) | 0.0002 / 0.0001 / 0.0006 | France: Brief 2. India/US: Brief 7 |
| (f) S1 left empty (slice of a–c) | 0.00177 | 0.00216 | **≈ 0.003–0.011** (6.71% empty vs 5.6–5.8% singletons at India/US LB = holdout; 6.2–6.4% if India/US LB is 0.001 below) | 0.0005–0.0016 France | **Briefs 1, 3** |
| address-less S2/S3 (cross-cut) | 0.00561 (59%) | 0.00742 (71%) | n/a | 0.0026 / 0.0028 | Brief 7 (a10 twin columns); R33/R38 for the rest |
| IDF mismatch (cross-cut) | — | — | ≈ 0.012 at D4, mostly recovered by #9ab | — | R15 for the rest |
| **vs #1 (0.990556)** | | | France_top = 0.994 / 0.985 / 0.976 if #1's India/US = 0.990 / 0.9915 / 0.9932 | France share of the 0.0065 gap: 0.0065 / 0.0051 / 0.0037 | — |

Row (f) is conditional. D1 gives one equation, LB_D1 = 0.85025·F_IU + 0.14975·(France singleton share), in two unknowns.
So each −0.001 of India/US LB against holdout raises the implied France singleton share by 0.57 pt (0.85025/0.14975 ×
0.001). At −0.001, only ≈ 0.3–0.5% of France S1 are wrongly empty, not ≥ 0.9%. The India/US rows (c)–(e) have one owner:
Brief 7, the P23 error-analysis stacker (a10 twin arm: +0.00032 to +0.00052 per half direction over #8c, per-S1 mean).

**Bars for every France rule** (computed from the per-S1 F0.5; formula in Evidence):
- *Adding* a pair to an S1 that misses one of k true matches gains 0.167 / 0.091 / 0.062 (k = 2/3/4). A wrong addition
  costs 0.333 / 0.242 / 0.188. Break-even precision is therefore **0.67 / 0.73 / 0.75**.
- *Removing* a pair from an S1 with all k matches right: dropping a TP costs 0.091 and dropping an FP gains 0.211 (k = 3).
  Break-even is **≥ 30% false**, the same as R08's break-even.
- Filling an empty non-singleton S1 is worth +0.7 to +1.0, which is why the empties in row (f) matter.
- Ship a class only if its label-backed precision is ≥ 0.85 on the US analog (France is US-like on house numbers: generated
  siblings keep the hn 2–6% of the time, US 2.7%, India 26%, per P20) and ≥ 0.73 on India (the downside bound).

### Brief 1 — P28-fr-tiers: France key tiers on the #9ab file, hn parsed first (R03 + Brief 5's parser + tier M) — do now
- **What is new.**
  - R03: no key-tier or legal-swap override exists. P20 found the plain core-equal cell 96.5% accepted at t .95. #9ab at
    t .985 still rejects **2.65% of its 513,436 argmax unique-key + city pairs**, against 0.25% of US test's at t .71
    (`sweep/keyrej.log`). The 3.44% quoted before did not reproduce. India test rejects 2.98% at .71, but India holdout key
    pairs below .71 are ≤ 0.4% true, so India does not model the France gap.
  - Brief 5's parser feeds this brief. `normalize.house_number()` reads '(12) RUE' as ''. Applying `lstrip('#(')` as a label
    step, with no rescoring:
    - it makes 13,952 missing→eq argmax pairs K-eligible;
    - it exposes 5,781 missing→sib pairs to `sib_zero`.
    - Among core-equal, city-equal France pairs with an S1 hn, 12,539 get a parsed hn: 10,117 become hn-equal, and
      1,579 (12.6%) become sib3 (`sweep/mtier.log`).
  - R01 (the filler strip) moved to Brief 5 (see there). There is no p-floor F tier any more.
- **Inputs:**
  - `data/work/p19/prob_test_France_idf144_bge.parquet` (#9ab France; columns s1, sx, p, p_gbdt; 1,305,341 rows).
  - `data/work/rf_test_{s1,sx}_France.parquet` (core, hn, city).
  - Holdout analog: `data/work/p23/pairs_{India,US}.parquet` + `data/work/rf_train_{s1,sx}_{c}.parquet`.
  - Starting scripts:
    - `probes/P23-research/sweep/{keytier,keyprec,keyrej,mtier,legal_swap,classes,hnfix}.py` (`hnfix.hn_new` = the parser);
    - `probes/P20-france-patterns/fr20.py` (`sib_zero`);
    - `probes/P14-france-shift/combine.py`.
- **Change.** New `probes/P28-fr-tiers/tiers.py`, a post-model decision hook that leaves `rules.py` alone:
  `tier(p_df, rf1, rfx, t) -> p'`.
  - Step 0: recompute the France hn with `lstrip('#(')`.
  - Step 1: apply `sib_zero` on the parsed hn.
  - Step 2: set `p' = max(p, t + 1e-4)` on the argmax pair of each SX when the pair is in one of these tiers:
    - **K (R03a):** core1 == corex, hn1 == hnx != '', city equal and non-empty, (core, hn) unique among France S1, p ≥ 0.95
      (variant 0.90). Measured before the parse: 7,770 pairs, 7,380 S1 changed, **267 empty S1 filled**
      (`sweep/keytier.log`). The parse adds candidates; re-measure after it.
    - **M (new; a zero-cost clause, not a slot).** Conditions:
      - core-equal;
      - S1 hn non-empty and SX hn '' after the parse;
      - legal relation same, dropadd or none;
      - street tokens equal and city equal;
      - (core, city) unique among France S1, and one kept candidate of the SX with that core;
      - p ≥ 0.90.

      Holdout analog: US 17,641 pairs and India 643, all true, but only 9 and 3 of them sit in [0.5, .985). France: +287
      pairs on 274 S1 and 13 empties filled at p ≥ .90, so ≈ +0.00002 (`sweep/mtier.log`).
    - **L (R03b, gated):** legal-form swap (disjoint legal sets, core equal), hn equal and non-empty, city equal, unique
      core, p ≥ 0.02 (see Candidate size).
      - France: ≤ 2,634 pairs, 0.68% accepted at .985 (`sweep/legal_swap.log`).
      - Missing-hn swaps stay out (India 0.56 true).
  - All three tiers apply the same p floor, so overlaps (a K pair that is also a legal swap) never count twice. M is
    disjoint from K (SX hn missing vs equal).
- **Validation:**
  - (i) *Label analog.* Apply each mask with its exact conditions to #8c's India/US holdout argmax pairs at a France-like t
    of 0.985. Report the precision of the flipped pairs and the paired-SE F0.5 delta per country.
    - Measured for K (`sweep/keyprec.log`): US 99.85% / India 98.15% overall; 0.970 (n 530) / 0.984 (n 436) in
      [0.95, 0.985).
    - The parse is a no-op on US (0 rows) and moves 6,515 India pairs, so re-run India after it.
  - (ii) *LOCO proxy for France:* the same masks on P27's LOCO world (P6 boosters restored at `data/work/p27/p6/`,
    worktree `p27-france-pseudo` `phase_a.py`), at t .98. India→US is the direction with US-like house numbers.
  - (iii) Label-free France profile: pairs/S1, empty rate, hn-class mix of the flips (`sweep/hncls.py`).
- **Acceptance bar:**
  - K ≥ 0.85 US / ≥ 0.73 India flipped-pair precision → one LB slot for **K + M** on the base that wins Brief 3 (#9ab or
    #9ab_t975). From slot 4 on, apply the tiers to Brief 5's rebuild instead.
  - L ships only if US ≥ 0.90 and India ≥ 0.73 on the exact mask (the unfiltered class is US 0.976 on n 5,583, India 0.796
    on n 3,937). Give it its own slot.
- **Expected LB:** K ≈ +0.0005 (7.1k S1 at ~+0.08 each, plus 267 empties at ~+0.7), plus the parse's new K pairs (not yet
  measured); M ≈ +0.00002; L ≈ +0.0001. In total **+0.0005 to +0.0007**.
- **Effort:** 2 h, CPU under 1.5 GB, no AWS. **Rule risk:** none (deterministic rule, precision measured on train labels).

### Brief 2 — P29-fr-veto: extend France's bge band past t so the CE can veto (R04) — maybe; strongest remover
- **What is new.** `rerank.blend` rescores only 0.02 < p_gbdt < 0.98, and France's t is 0.985, so the CE can rescue France
  pairs but never veto them. 97% of the ~783k accepted France pairs were never scored. No probe has tried a veto.
  - New LOCO evidence (`data/work/p27/phase_a.log`): among unseen-country argmax pairs at p ≥ .98 that have a LOCO bge r,
    **r < 0.5 is 27.1% true** (US→India, n 5,913) and **31.3% true** (India→US, n 4,107).
  - A veto there removes ≥ 69% false pairs, far above the 30% bar. r ≥ 0.5 is 92% / 73% true.
  - **Caveat: these rates come from a band-selected subset.** r exists only for the 14,218 / 15,088 LOCO-confident pairs
    (about 5%) that the pooled model placed in its 0.02–0.98 band. That subset has base precision **0.651 / 0.616**, against
    **0.959 / 0.994** for the confident pairs without r (`phase_a.log` l.30 / l.84).
    - France's confident pairs are not band-selected, so their r < 0.5 slice may be less false.
    - The +0.0007 EV below holds only if France's vetoed pairs are as false as that subset's.
- **Inputs:**
  - `data/work/p19/prob_test_France_idf144_bge.parquet` (p_gbdt).
  - `data/work/p13/ckpt/ce_bge_pool.pt`.
  - P19's SageMaker scoring path (worktree `p19-bge-idf`; `data/work/p19/sm_data` builder; job p19-score-0926-1224 scored
    65,393 pairs in 269 billable s for $0.126).
  - `rerank.blend` (`code/business_entity_resolution/src/rerank.py:262`).
- **Change.**
  - Score r for France pairs with p_gbdt ≥ 0.98: ≈ 771k pairs, or [0.98, 0.9995), ≈ 203k, first.
  - Variant A: `rerank.blend(p_gbdt, r, [0.9245, 0.2595, 0.0515], lo=0.02, hi=1.0001)`.
  - Variant B (pure veto): p := 0 where p_gbdt ≥ 0.98 and r < r_cut, with r_cut ∈ {0.05, 0.2, 0.5}. **Start at 0.05**
    and loosen only if the count gate leaves room.
  - Keep the base file's t, so the LB delta is the veto alone, and report the change in empty rate.
  - Add **≥ 100k in-domain control rows** (US/India holdout pairs at p ≥ 0.98) to the same job, using the P19 mechanism.
    At 99.92% true, 2k rows would hold about 2 false pairs, which estimates nothing. 100k extra pairs cost ≈ 60 s on the
    g5 (≈ +$0.03).
- **Validation:**
  - (i) LOCO proxy: the P27 numbers above, recomputed at r_cut 0.05 / 0.2 with `phase_a.py`.
  - (ii) In-domain control: veto rate and truth on the same ≥ 100k US/India holdout rows. The veto rate should be tiny.
  - (iii) France label-free checks: veto count, hn-class mix of the vetoed pairs against all accepted pairs
    (`sweep/hncls.py`), and the R13 disagreement meter (`sweep/disagree{,2}.py`).
- **Acceptance bar:**
  - France veto ≤ 2% of accepted pairs (≤ ~16k).
  - The vetoed pairs' sib3 + far + other share is ≥ 2× that of all accepted pairs.
  - In-domain vetoed truth ≤ 50%.
  - Then one LB slot against the same base.
- **Expected LB:** each 10k vetoed at 70% false ≈ 7k × 0.21 − 3k × 0.09 ≈ +1,200 S1-units ≈ **+0.0007**. That is conditional
  on the band-selected subset's falseness carrying over (see the caveat). Range −0.0002 to +0.001, point +0.0005.
  - The downside: France CE–GBDT disagreement is 33% (P19), so an OOD CE may say no to true pairs. The count gate limits
    that.
- **Effort:** 2 h. **AWS:** ≈ $0.34 (871k / 1,653 pairs/s + overhead on g5 at $1.69/h). **Rule risk:** none (same
  Apache-2.0 568M checkpoint, same candidates).
- Composes with Brief 1: tiers add at p < t, the veto removes at p_gbdt ≥ 0.98. If a K-tier pair has p_gbdt ≥ 0.98 and gets
  vetoed, the veto wins.

### Brief 3 — unpin France's threshold (R02, the unpin half) — do now, 0 build
- **What is new.**
  - Every France variant so far has been pinned to D4's 6.85% empty rate, and #9ab (t 0.985) sits at 6.71%.
  - The LB implies a singleton share of 5.6–5.8% **if India/US LB = holdout**. On that reading ≥ 0.9% of France S1 are
    wrongly empty, and a better ranker should want fewer empties.
  - Sensitivity: D1 gives one equation in two unknowns. Each −0.001 of India/US LB against holdout raises the implied
    share by 0.57 pt (0.85025/0.14975 × 0.001). At −0.001 the share is 6.2–6.4%, so only ≈ 0.3–0.5% of France S1 are
    wrongly empty. The ~6.0% floor below is equally conditional.
  - R02's branch-word veto is dropped: at t .975 it changes 131 accepted pairs, and the four branch words are already
    0–7 accepted each.
- **Input:** `output/sub9ab_t975/` (staged, validator PASS; 3.09 matches/S1, 6.4% empty, against 3.02 / 6.71% at .985).
- **Change:** none. Upload it first today. Also apply P15's sibling rule (`fr20.sib_zero`) to every file with t < 0.985.
  At .975 it moved 858 France S1 on the IDF world (#9as vs #9a). France siblings are US-like, and the US-holdout sign is
  +0.00097.
- **Validation:** the LB is the only arbiter for a threshold. P6's label-free rule gives 0.73 on this file (ledger
  `profile-idf144_bge`), which the D-series showed is too loose.
- **Acceptance bar:**
  - ≥ +0.0001 vs #9ab: build Briefs 1/2/4/5 on t 0.975, and try 0.965 once they land (mind the conditional floor below).
  - ≤ −0.0001: stay at 0.985.
  - Floor: stay above ~6.0% France empty at India/US LB = holdout. That rises to ~6.5% if India/US sits 0.001 below
    holdout. Until an upload below 6.4% wins, treat 6.4% (the t975 file) as the tested edge.
- **Expected LB:** ±0.0003, sign unknown. **Effort:** 0 (plus 10 min for the sibling rule). **Rule risk:** none.

### Brief 4 — P30-fr-orgswap: hn-equal single org-word swap tier, gated by a label-free sibling score (R05) — maybe; the big bet
- **What is new.**
  - This is P20's biggest remaining cell: 74,887 France argmax pairs with one core word swapped and an equal hn,
    34.7% accepted, against US truth 0.890 (India 0.625).
  - P20's blanket fix (R3, sx_margin quantile map) failed: 26.4k added, 34.7k dropped. This brief is selective by word.
- **Inputs:** the Brief 1 files, plus P20's per-pair taxonomy `data/work/p20/pairs_France.parquet` (columns kind, word,
  hrel; 1.44M France pairs), `probes/P20-france-patterns/{patterns,taxonomy_counts}.py`, and `sweep/orgswap_s.py` /
  `sweep/add1_s.py` for s(w).
- **Change.**
  - s(w) = share of France sub1 pairs where w is the swapped word and the hn relation is sib3. It is label-free and
    test-side; require word support ≥ 20.
  - **s(w) is not the decoy share.** Pure branch words show sib3 only 0.62–0.68 of the time (`sweep/branch_fr.txt`:
    participations 0.684, distribution 0.684, holding 0.624, international 0.656; pm12 takes another 0.16–0.18). So the
    decoy share is d(w) ≈ s(w) / 0.68.
  - Mask:
    - argmax pair; kind sub1; R1 fillers excluded (Brief 5 handles them);
    - hn equal and non-empty; city equal;
    - no other kept candidate shares the remaining core;
    - **the SX's exact core is not another France S1's core in the same city** (the Auto-FuzzyJoin negative-rule guard from
      R21: France has 0.87 one-token-apart same-city S1 pairs per core+city key);
    - **s(w) ≤ 0.4** (d ≤ 0.6), and p ≥ 0.02 (see Candidate size).
  - Then p := max(p, t + 1e-4).
  - Measured size before the city, uniqueness and other-S1-core guards (`sweep/orgswap_s.log`):
    - 47,474 org-word sub1/eq argmax pairs on #9ab, 10.3% of them accepted at .985. That is **42,587 flips on 39,381 S1**,
      more than the 30k assumed before.
    - s(w) ≤ 0.4 keeps 41,369 of them. Almost every org word sits at s 0.3–0.4, and only 1,218 flips lie above 0.4.
    - 30,295 of the flips have p < 0.5.
    - Guards: P20 found 93% of sub1/eq SX have no other kept candidate. The city and other-S1-core guards are unmeasured.
  - Conservative variant: flip only p ≥ 0.5. That leaves 12,292 flips, 11,818 of them at s ≤ 0.4.
- **Why it can be true:** P(true | swap, hn eq) ≈ 0.875·(1−d) / (0.875·(1−d) + ε·d). Here ε = P(hn eq | decoy) = 0.045–0.075,
  the pure branch words' eq share.
  - At a typical org word's s = 0.36: d ≈ 0.53 and P ≈ 0.92–0.95.
  - At the old gate s = 0.6: d ≈ 0.88 and P ≈ 0.72. That is below the 0.85 bar and at the 0.73 break-even, hence the new
    gate.
  - Weighted over the measured flips at s ≤ 0.4: mean P 0.90–0.99 by s bin.
- **Extension to add1/eq** (P20: 17,123 pairs, US truth 0.852, India 0.246). On #9ab there are 18,068 such argmax pairs.
  7,578 of them are R1 fillers (Brief 5). The other words give only 1,276 flips at s ≤ 0.4, or 639 at s ≤ 0.3 with p_med
  0.94 (`sweep/add1_s.log`), ≈ +0.00003 to +0.00004. Add it as a zero-cost clause at s ≤ 0.3 if the US analog passes; it is not a slot.
- **Validation:**
  - Label analog: the same mask on US/India holdout, with s(w) computed on each country's own text, at t .985.
  - LOCO proxy: India→US in P27's world.
- **Acceptance bar:** flipped-pair precision ≥ 0.85 in the US analog and in India→US LOCO. The India analog may be lower
  (India siblings keep the hn 26% of the time); report it, but it does not gate. One slot, after Brief 1 has landed.
  It is mutually exclusive per slot with R14 and R32 (same cell).
- **Expected LB** (41.4k flips before the guards, k = 3; each flip gains P·0.091 − (1−P)·0.242):

  | flip precision | LB before guards |
  |---|---|
  | formula, per s bin (0.90–0.99) | +0.0016 to +0.0019 |
  | US cell truth, 0.890 | +0.0013 |
  | break-even, 0.73 | 0 |
  | 0.6 | −0.0010 |
  | India cell truth, 0.625 | −0.0008 |
  | p ≥ 0.5 variant at 0.89 | +0.0004 |

  The guards shrink every figure. EV ≈ +0.0005 once gated; the range is −0.001 to +0.0014.
- **Effort:** 1.5–2 h, CPU, no AWS. **It must be ported to run.py by ~12:00 on 27 Sep or dropped** (freeze). **Rule
  risk:** none (ruling 2: unlabelled test statistics).

### Brief 5 — P31-fr-rebuild: one France rebuild on the bge world, '(12)' hn parse + P20's R1 fillers (R09 + R01 + R20 name-side) — do now
- **What is new.**
  - `normalize.house_number()` cannot match a leading '('. The bracketed form is on 1.6% of France S2/S3 (161.5 per 10k)
    and parses to '' 99.4% of the time. It is absent in US/India (0.0 per 10k, `sweep/fr_checks_*.txt`).
  - P6's `_paren_hn` never reached src. On the #9ab world, 20,567 argmax pairs change hn relation:
    - 13,952 go missing→eq (≈ 2.1k newly acceptable);
    - 5,781 go missing→sib. Those are hidden siblings that bypass P15's rule (1,600 core+city-equal, 20.4% accepted
      at .975).
  - Brief 1 already applies the parse as a label step. This brief adds what only a rescore gives: the model sees the
    corrected hn features on non-key pairs.
  - **R1 (the #20a mechanism) goes into the same rebuild.**
    - P20's filler strip (& Fils / & Cie / Associés / Frères / Frs added to `normalize.GENERIC`) exists only on the
      non-bge IDF world: #20a, staged 15:29, validator PASS. It adds 17.5k and drops 2.0k France pairs vs #9as, and
      92.7% of the added pairs are hn-equal. No bge file has R1.
    - Keep P20's rescoring route. It is a one-line GENERIC change that run.py reproduces before the freeze. Brief 1's old
      p-floor F (8,040 flips before its guard) would need its own post-hoc port.
    - Apply only one of the two mechanisms: stacking F on top of R1 would count the same pairs twice.
- **Inputs:**
  - `data/work/rf_test_sx_France.parquet`.
  - `probes/P6-france/france.py` `_paren_hn` and `sweep/hnfix.py` (`hn_new`).
  - `probes/P20-france-patterns/rescore.py` + `fr20.noise_core`. Rules join with '_': add a `paren` rule to fr20 and run
    `ER_WORK=data/work/p14/w_idfx … rescore.py noise_paren` on P14's IDF world (ledger `idf-world-rebuild-repro`:
    max |Δp| = 0 vs idf144).
  - The p19 r cache `data/work/p19/r` for the bge re-blend. The CE reads raw text, so cached r stays valid. Missing r
    falls back to p_gbdt.
- **Change.**
  - Recompute France hn with `lstrip('#(')` and apply `noise_core` to the France rf frames.
  - Rescore m2_big (`rescore.py noise_paren`), re-blend bge with `rerank.blend(p_gbdt, r, [0.9245, 0.2595, 0.0515])` on
    band 0.02–0.98, and apply `sib_zero`.
  - Count the band pairs that lack an r. If there are > 2k, score them on AWS with Brief 2's job (≤ $0.1).
  - This file becomes the base for Brief 1's tiers and Brief 2's veto from slot 4 on.
  - Keep old F's guard as a report: how many R1-raised SX have another kept candidate carrying the filler word (fils
    6.9%, cie 8.6% of such SX). Also check the `et` / `+` connectors (`sweep/branch_fr.txt`: added on 11,925 / 6,476
    argmax pairs, 81% / 80% hn-equal, 4.1% / 6.8% sib3). If the strip leaves them dangling, add them to the strip set
    only when a filler is also stripped.
  - Fold in R20's name-side items only if they need no re-blocking (P20: blocking reads the P3 texts, not the cores):
    - St/Ste → Saint/Sainte, except a first-token Ste/Sté (Société);
    - dotted d.b.a (0.064% of France SX unresolved).
  - Skip the street-type abbreviations (ch/psg/q), which need re-blocking.
- **Validation:**
  - US/India files are byte-identical: the bracketed form never occurs there, and R1 touches 21 S1 and 189 S2/S3 of
    12.5M train records (P20).
  - Label-free France checks: the accepted sib3 share must not rise, and the missing→eq flips should be ≥ 90% city-equal.
  - R1's label analog is P20's (US add1/eq truth 0.852; the fillers themselves keep the hn 0.82–0.88 of the time and take
    a sib3 offset 0.02–0.05).
  - A LOCO proxy does not apply to a parser.
- **Acceptance bar:** a France-only diff with the sib3 share down. It takes slot 4 as part of the combined file.
- **Expected LB:** hn parse +0.0001 to +0.0002, R1 +0.0002 to +0.0003; together **+0.0003 to +0.0005**.
- **Effort:** 2–2.5 h. rescore.py's 300k-row slices need ~3 GB, so halve the slice or run it as a SageMaker CPU job
  while the laptop is shared. **Rule risk:** none (parser change, hand-written suffix words).
- **#20a: do not upload.** It measures R1 on a world without bge. Its LB delta against #9ab would confound R1 with the
  missing CE, so it tells us nothing this rebuild does not. Keep it staged as a no-bge fallback.

### Brief 6 — master hygiene before the next winning upload (R17 + R06 + R12) — do now, master-owned
- **R17 caps.**
  - In the decide step (`probes/P14-france-shift/combine.py` for probe files, `submit_probs` for run.py): per S1, keep the
    top 5 S2 and top 6 S3 by p, with a one-line `assert` self-check.
  - The caps are a competitor's claim ([Karthik FINDINGS](https://github.com/Karthikatstuffmama/amazon-ml-challenge-2026/blob/main/FINDINGS.md)),
    now **verified on our train GT** (`sweep/caps.log`). Across 2,206,821 S1, the most any S1 has is 5 S2 and 6 S3, so
    there are 0 violations. #9ab breaks the caps on 86 S1: 75 have > 5 S2 and 11 have > 6 S3.
  - +0.00001 LB, free.
- **R06 recipes.** The final entry is the best *public* upload, and `run.py` RECIPES (line 78) stops at sub9d.
  - Add `sub9ab` and every variant that could win: France IDF share 0.144, France bge w [0.9245, 0.2595, 0.0515] on band
    0.02–0.98, France t, the US stack, the India bge. Add Brief 1's tiers, Brief 5's rebuild and Brief 7's stacker as run.py
    stages as each one lands.
  - Regenerate the file, check that the md5 of the sorted TSV matches the uploaded one, and run the validator. Do this
    before uploading any file that can top 0.984072.
  - **Deadline: the 27 Sep 14:00 code freeze.** A file that cannot be regenerated by the frozen run.py must not become the
    best public score.
- **P21 compliance (ruling 6).** P21 is merged on main (74a8e91). `FR_REGIONS` is now the 13 région names, and the
  département map is learned from test France (`normalize.fr_departements`, cached in `data/work/fr_regions.json`).
  - Every staged France file (#9ab, #9ab_t975, #9ab_t990, #20a) was built with the old hand-written table.
  - P21's rebuild with the learned map left the France rows of sub8c/9a/9b/9s unchanged (0 diffs), but moved sub9d by
    1,039 rows. #9ab (IDF + bge) was never rebuilt.
  - So regenerate the final recipe on main, with the learned map, and diff its France rows against the file being uploaded.
    A non-zero diff means the upload is not what the frozen code produces: re-upload the regenerated file.
- **R12 compliance.** Add "excluded" rows to the Models and licences table (`Documentation_template.md:399–411`):
  - Qwen3-8B, 8,190,735,360 params;
  - Qwen3-Reranker-8B, 8,188,548,096;
  - Qwen2.5-3B, qwen-research licence;
  - Jellyfish and jina-reranker-v2, cc-by-nc;
  - Llama-3.1-8B, custom licence;
  - Ministral-8B, MRL.
- **Effort:** 1.5 h. **LB:** 0; it protects the rank. **Rule risk:** it lowers it.

### Brief 7 — P32-iu-stack: ship the a10 residual stacker in its p2-masked form for India/US (R50, gated by R51) — do now
- **What is new.**
  - Rows (c) and (d) of the error budget (India/US FN 0.0039 LB, FP 0.0009 LB) have no owner among Briefs 1–5, which are
    France-only.
  - P23 error-analysis's a10 "twin" stacker is a 200 × 31 LightGBM residual (P17's recipe; init_score = logit p) over #8c's
    argmax pairs. Its columns are SX context (p2, margin, n_cand_sx), S1 context and twin counts.
    - Gain: +0.000429 ± 0.000064 (h0→h1, t .71) and +0.000363 on the swap, as a per-S1 mean over India + US holdout S1.
    - p2 is its top feature (gain importance 0.18). Source: `probes/P23-error-analysis/work/verify-lowp-fn-1/a10.log`.
  - Part of the p2 gain is a holdout artefact.
    - (a) On holdout, a runner-up that belongs to a competing train S1 carries an e5-blended p; on test every row is
      bge-blended. The SX stack applied to asym rows alone (e5 runner-up) gives +0.00032 of the total +0.00048
      (`verify-decision-theory-2/log_v2.txt`).
    - (b) Test is less dense than holdout. US argmax rows in the 0.2–0.5 runner-up bin are 0.0819 of holdout and 0.0511
      of test (`log_v3.txt`).
  - **The a13 `twin_p2mask` arm removes the artefact.** It sets p2/margin to NaN wherever the runner-up is e5-blended.
    - It keeps +0.000308 ± 0.000064 (h0→h1; India +0.000135, US +0.000423) and +0.000227 on the swap (`a13.log`).
    - Dropping p2 altogether (ctx_noP2) gives ≈ 0.
  - **Measured here (R51, `sweep/rw.log`):** the holdout gain reweighted to test's per-S1 stratum mix. Strata = the maximum
    px_2nd bin over each S1's argmax rows with 0.3 < p < 0.999, with test shares from `p17/prob_test_*_bge`; the method is
    importance-weighted validation (Sugiyama et al., JMLR 2007). LB-equivalent (India 0.47, US 0.38):

    | arm | direction | holdout | test-reweighted | S1 without an e5 runner-up |
    |---|---|---|---|---|
    | twin_p2mask | h0→h1 | +0.000224 | **+0.000209** | +0.000226 |
    | twin_p2mask | h1→h0 | +0.000170 | +0.000166 | +0.000175 |
    | twin (full) | h0→h1 | +0.000328 | +0.000298 | +0.000224 |
    | twin (full) | h1→h0 | +0.000311 | +0.000288 | +0.000270 |

    - The stratum shift costs only ≈ 7%. The 0.45–0.85× bin shrinkage does not carry through to the gain.
    - The full twin arm's asym-free floor equals p2mask's gain. p2mask gets that value without leaning on the e5 artefact,
      so ship p2mask.
  - **A contamination to respect.** P17's US stacker was fit on holdout half 0 (`run.py ensure_stack`), and the P23 dump
    applies it to every holdout S1 (`dump.py` → `run.model_parts("holdout")`). So US half-0 p is P17's in-sample output.
    - Only h0→h1 is a clean US report. The h1→h0 US figures (+0.000300 p2mask) are scored on P17-in-sample rows.
    - Fitting on US half 0 is the "prefit" case that scikit-learn's StackingClassifier docs warn carries "a very high
      risk of overfitting". It would attenuate the residual rather than flip its sign.
- **Inputs:**
  - The a10 frames, built by `a10_stack.load` from `amc_/tw_/tw2_{c}.parquet`. **These live in another session's /tmp
    scratchpad** (`cats.SP`), so copy them to `data/work/p23/` now.
  - `data/work/p23/pairs_{c}.parquet` and `s1ctx_{c}.parquet`.
  - Code: `probes/P23-error-analysis/work/lowp-fn/{a10_stack,a13_arms,a6_twins,a7b_twins,cats}.py`.
  - Test p: `data/work/p17/prob_test_{India,US}_bge.parquet` (post-P17, = #8c).
- **Change.** A probe-side script mirroring `a10_stack.arm` with a13's MASK + TWIN columns: CTX minus p2/margin, plus
  p2m/marginm, n_twins, n_ctwins, n_atwins, noaddr, cat and name_sup.
  - Train rows: India, both halves; US, half 1 only (P17 out-of-sample).
  - Params: 200 rounds, 31 leaves, lr .05, min_data_in_leaf 200, λ2 1.
  - Build the same columns on test. The twin/cat builders read train texts, so port them to the test sources. On test the
    runner-up is always bge, so p2/margin are never masked at inference.
  - p' = σ(raw + logit p) on the India/US argmax pairs; then argmax/threshold as now, at t 0.71 (the tuned t moves
    0.65–0.73 on a plateau).
  - France is untouched. The src/run.py port is the master's, and it is due by ~12:00 on 27 Sep.
- **Validation:**
  - Done: a13 in both directions, and R51's reweighting (the gate is ≥ +0.0001 test-reweighted h0→h1; it passed at
    +0.000209).
  - Still to do: a label-free test profile in the style of P17's hn-stack-l2-resid-shift, comparing the share of
    accepted pairs dropped / added and of S1 changed between test and holdout. Holdout changes 1,727 pairs at t .71 (both
    halves). Reject if the test flip rate is > 1.5× the holdout rate.
  - The US-half-1-only fit has no clean US report of its own. Its h1→h0 US number sits on P17-in-sample rows, so treat
    it as indicative only. The clean US check is h0→h1 (+0.000423 holdout, +0.000378 test-reweighted).
- **Acceptance bar:** the profile gate passes → it rides in the slot-5 combined file. At +0.0002 it is inside the LB's
  tie band, so a dedicated slot would not confirm it.
- **Expected LB:** **+0.00015 to +0.00021**: test-reweighted +0.000166 / +0.000209 by direction, shaded for the
  US-half-1 fit. It is positive in both split directions at ≥ 3.5 SE.
- **Effort:** 3–4 h: 1 h to fit, 2–3 h to port the test-side twin/cat features, CPU under 2 GB. **Rule risk:** none
  (LightGBM on label-free features, trained on provided labels).
- The second-tier stacker bundle's R22 / R38 / R33(1) are this brief. R25, R37 and R39 stay optional extra columns.

### Slot map (≈ 4 left on 26 Sep if the counter resets at midnight IST, 5 on 27 Sep, final by 22:00)
Slots 1–4 sit on #8c's India/US, so each of their LB deltas is France alone. Brief 7 changes India/US, so it joins only
at slot 5.

**Code freeze, 27 Sep 14:00.** Any upload that could become the best public score must be reproducible by a run.py recipe
committed before 14:00 (Brief 6, R06).
- 26-Sep uploads built from probe files count only if their recipe is ported by then.
- Every 27-Sep upload (slots 5–9) must come from a committed recipe.
- Brief 4, Brief 7 and the P26/P27 winner must land and be ported by about **12:00 on 27 Sep**, or be dropped.

26 Sep:
1. #9ab_t975 + `sib_zero` (Brief 3).
2. K + M tiers, hn parsed first, on the winner of slot 1 (Brief 1).
3. Veto (Brief 2) on the same base.
4. The combined France file: Brief 5's rebuild (hn parse + R1 on the bge world) + tiers + veto (if the veto is ≥ +0.0001),
   at slot 1's t.

27 Sep, from committed recipes:
5. Slot 4's file regenerated by run.py on main (learned FR map; md5 / diff check), plus Brief 7 for India/US.
6. The org-word tier (Brief 4) on slot 5's file, if its analog gate passed and it was ported by 12:00. Otherwise L, if its
   gate passed.
7. The P26/P27 winner, if it passes its both-direction LOCO gate and was ported by 12:00, with the tiers re-applied.
8. The best combination at a re-picked t.
9. Hold until ~21:00 as the fallback.

#20a is not in the map (Brief 5 supersedes it). Treat |Δ| < 0.0002 as a tie. These changes touch ≈ 8–15k France S1; the
public→private noise is my estimate.

### Second tier — worth trying if time
- **R08 consensus** (`sweep/agree.py`): 2-of-3 over idf/bge/idfbge = 786,523 pairs, intersection 752,967, at a matched
  empty rate. Break-even is 30% false; sign unknown. It is a remover, like Brief 2: test one of them unless both win.
  Agreement-on-the-line holds only for neural nets, so don't use it to rank GBDT variants.
- **R14** (recompute sx/S1 margin and gap over the pruned list, France only): the model-level alternative to Brief 4 for
  the same cell (P20's own suggested next step). 3 h; gate on India/US moving ≤ 0.0001 without a retrain.
- **R32 rule variant:** France p := max(p, t) when hn equal, city equal and dt_unique_name ≥ 1 (a uk_address_matcher-style
  block-local IDF, reimplemented rather than pip-installed). Same cell, so one slot at most.
- **R15:** P14 `variant.py` at France own-text share 0.5 / 1.0 and one all-country test-text IDF. Pick on P14's
  `locoown` proxy and report the pruner keep rate (accepted sx_margin is still 0.306 vs 0.42–0.44). 2 h CPU.
- **R18:** France website keys. Run the 10-minute join first. Corrected ceiling: the "any variant" key hits 38.4% of
  website SX vs 32.2% folded (not 78.8%), ≈ 3.7k France SX at most. LB ≤ +0.0002.
- **R33, France half:** in multi-match S1, flag accepted SX in [0.985, 0.999) that share neither core nor street+hn with
  any co-assigned SX. Fold this into Brief 2's veto rather than a slot.
- **R43 / R44** (SML weights, reverse validation): only if ≥ 2 France variants are still contested on 27 Sep. Rank them
  against LOCO truth first.
- **R19:** turn `sweep/vocab_asym.py` + `branch_fr.py` into per-country learned GENERIC/SIB tables (a human approves the
  top ~30). This is mainly for the doc's "mined, not guessed" claim.
- **India/US stacker extras:** Brief 7 now ships the core of this bundle (R22 / R38 / R33(1) = a10's columns). Add R25,
  R37 and R39 as extra columns only if Brief 7's port is done early, and only if each one survives the same p2-mask and
  both-direction test.
- **R26:** CatBoost `text_features` on the diff tokens. Only if P22's job is trivially re-runnable; P22 found the family
  adds +0.00003.
- **Candidate size** (no LB effect): hand to P24 as one sweep.
  - **An interpretation question for the master, not a kill:** could the candidate file be m2_big p ≥ 0.02 (3.89 / 3.95 per
    S1) instead of the pruner output (4.83)?
    - PROBLEM.md ruling 3 says "a pruned later stage counts", and m2_big's 0.02 cut is the stage that feeds the CE and the
      threshold.
    - Under that reading, every tier needs a p ≥ 0.02 floor. Without one, matches would stop being a subset of
      candidates. Briefs 1 L and 4 now carry the floor, and K / M floor at 0.95 / 0.90.
  - R10: France τ 0.1 takes 5.14 → 4.27/S1 and loses 228 of 728,896 answers.
  - R11 (sib-aware q cap) and R16 (cost-aware keep) together with R10: 4.83 → 4.28/S1 for ≈ −0.0001.
  - R36 (hn columns in the pruner) and R34, R45.
  - R40 (address-key recovery pass) is P24's.

### For the finale / doc
- **R24 narrative:**
  - the progression ladder from the pipe ledger;
  - the per-country Shapley loss decomposition (kind × empty address × chain) from the table above, stating that
    India/US have ≤ ~0.002 left;
  - a France-without-labels chapter: the D1 decomposition, P14 IDF shift, P15 sibling fingerprint, Brief 1's class table
    and this doc's break-even bars;
  - a generator-operator coverage table (GeCo / Febrl / LANCE / EMBench families → fingerprint → normaliser/feature);
  - an "identified but cut" list (this doc's What failed).
  - Do not cite KhadgaA as "2024 winner": that is a different 2024 problem.
- **R23:** PC / PQ / RR and a recall-vs-cands/S1 figure (P10 pareto.csv, τ 0.002 → 0.1: 7.27 → 4.24/S1) with competitor
  points (20.0–32.4 cands/S1 at 95.65–97.51% recall) and the floor line (answers ⊆ candidates: 3.38 India/US,
  2.99 France). Note that mayank's PC 97.51% is on train.
- **R41:** a 20-line CRC λ for the per-S1 candidate miss rate on US/India (n ≈ 220k holdout S1), excluding France
  (exchangeability).

## Evidence         (table: ledger name | world | key metrics | verdict)
**Measured or derived in this synthesis** (no new ledger rows; aggregates only):

| item | world | key numbers | use |
|---|---|---|---|
| France implied at #9ab | LB | 0.984072 = 0.85025·F_IU + 0.14975·F_FR with F_IU 0.9899–0.990 → **F_FR 0.9504–0.9510**. At D4 it was 0.9377–0.9394. P26's brief says 0.952 | TL;DR, budget |
| #1's France | LB | (0.990556 − 0.85025·F_IU) / 0.14975 = **0.994 / 0.985 / 0.976** at F_IU 0.990 / 0.9915 / 0.9932 | lag table |
| France empties lower bound | ledger `p19-sub9ab` france_empty 0.0671 | 17.4k empty S1 vs 14.5–15.0k singletons → ≥ 2.4–2.9k wrongly empty → ≥ 0.009–0.011 F0.5, **if India/US LB = holdout**. At 0.001 below, the singletons are 16.0–16.5k, so only ≈ 0.9–1.4k S1 are wrongly empty (≈ 0.003–0.005) | row (f) |
| F0.5 break-even | per-S1 F0.5 = 1.25·P·R / (0.25·P + R) | add: 0.67 / 0.73 / 0.75 at k = 2/3/4; remove: 30% false (k = 3) | the bars |
| LOCO veto evidence | P27 phase A, `data/work/p27/phase_a.log` | p ≥ .98 LOCO positives with a LOCO bge r: r < 0.5 → 27.1% true (US→India, n 5,913) / 31.3% (India→US, n 4,107); r ≥ 0.5 → 92.2% / 72.9% | Brief 2 |
| LOCO sibling share | same | India→US: p ≥ .98 has precision 0.979; without sib3 0.9936 (5,876 sib3 pairs, ≈ 95% false); hn eq/none 0.9988. US→India: the errors sit in hn none/eq (98.5% of the wrong ones) | Brief 3 sib rule |
| Connectors after the filler strip | `sweep/branch_fr.txt` (#9ab France, t .975) | 'et' 11,925 pairs (hn-eq 0.814, sib3 0.041, accepted 0.224); '+' 6,476 (0.797, 0.068, 0.627) | Brief 5 (R1 check) |
| Legal-swap class | `sweep/legal_swap.log` | France swap & core_eq 64,621 pairs, of which **68% have a sib3 hn** (43,929), 2,634 hn-eq; US hn-eq truth 0.976 (n 5,583), India 0.796 (3,937) | Brief 1 L |
| LOCO veto subset is band-selected | `data/work/p27/phase_a.log` l.30 / l.84 | r exists for 14,218 of 283,156 (US→India) and 15,088 of 380,310 (India→US) p ≥ .98 positives; precision with r **0.651 / 0.616**, without r 0.959 / 0.994 | Brief 2 caveat |
| K tier, re-run and logged | `sweep/keytier.log`, `sweep/keyprec.log` | K at t_key .95: +7,770 pairs, 7,380 S1, 267 empties filled (.90: 9,130 / 8,580 / 306). Holdout key + city: US 188,650 pairs 0.9985 true, India 81,384 0.9815; [.95, .985): US 0.970 (n 530), India 0.984 (n 436); key S1 not argmax: US 0, India 248 (0.4% true) | Brief 1 K |
| Key-pair rejection share | `sweep/keyrej.log` (new `keyrej.py`) | argmax key + city pairs rejected: France #9ab **2.65%** of 513,436 at .985 (1.14% below .95); US test **0.25%** of 949,738 at .71; India test 2.98% of 759,529 at .71. The earlier "3.44%" did not reproduce | Brief 1 |
| Per-source caps | `sweep/caps.log` | train GT: 2,206,821 S1, max 5 S2 and max 6 S3 per S1 (**0 violations**); #9ab: 75 S1 with > 5 S2, 11 with > 6 S3 | Brief 6 R17 |
| Org-word swap resize | `sweep/orgswap_s.log` (P20 `pairs_France.parquet` × #9ab) | 47,474 org-word sub1/eq argmax pairs (support ≥ 20, fillers excluded), 10.3% accepted at .985 → 42,587 flips on 39,381 S1; s(w) ≤ 0.4 keeps 41,369; 30,295 flips at p < 0.5, 12,292 at p ≥ 0.5; formula EV (s ≤ 0.4) 2,649–3,188 S1-units = +0.0016 to +0.0019 LB before guards | Brief 4 |
| add1/eq per-word gate | `sweep/add1_s.log` | 18,068 add1/eq argmax pairs on #9ab, 20.9% accepted; 7,578 fillers; non-filler flips 1,276 at s ≤ 0.4, 639 at s ≤ 0.3 (p_med 0.94), 7,087 ungated | Brief 4 extension |
| Tier M | `sweep/mtier.log` | US 17,641 / India 643 M pairs, all true; in [0.5, .985) n 9 / 3. France M 10,136 pairs (96% at ≥ .985); flips 206 / 287 / 388 at t_key .95 / .90 / .5, filling 11 / 13 / 23 empties. The parse moves 12,539 France pairs (10,117 → eq, 1,579 → sib3) and 6,515 India pairs; US 0 | Brief 1 M |
| a10 test-mix reweighting (R51) | `sweep/rw.log` | twin_p2mask LB-mix h0→h1 +0.000224 → test-rw **+0.000209** (asym-free +0.000226); h1→h0 +0.000170 → +0.000166. twin h0→h1 +0.000328 → +0.000298 (asym-free +0.000224) | Brief 7 gate: pass |

**Full ranked idea table.** Verdict columns: tried / rules / feasible / sources. "partly" = partly-done; "–" = not checked.
Expected gains are LB unless noted. Ranks 1–9 and 9a–9b (added in round 2) are recommended, 10–33 are second tier, 34–36
are doc, and 37–55 were killed.

| # | id | idea | sources | pipeline change | target | expected gain | h | rule risk | verdicts |
|---|---|---|---|---|---|---|---|---|---|
| 1 | R03 | Exact-key tier (unique core+hn+city, p ≥ .95) + legal-swap class | [mayank repo](https://github.com/mayankgoplani431-del/amazon-ml-2026-entity-resolution) (key passes: recall 95.4→97.5%, LB 0.953→0.964); [lieu](https://github.com/openvenues/lieu) (exact tier above 0.9/0.7); [INSEE catégories](https://www.insee.fr/fr/information/2028129); `sweep/classes.csv`; `sweep/keytier.log`, `keyprec.log`, `keyrej.log` | Brief 1 masks K, L (+ M), hn parsed first | France | K +0.0005; L +0.0001; M +0.00003 | 2 | none | partly / ok / do-now / yes (logged in round 2; 3.44% → 2.65%) |
| 2 | R01 | Filler-suffix strip (P20 R1 in GENERIC), rebuilt on the bge world | [LANCE vt3](http://users.ics.forth.gr/~fgeo/files/ISWC15LANCE.pdf); [Febrl generate.py](https://raw.githubusercontent.com/J535D165/FEBRL-fork-v0.4.2/master/dsgen/generate.py); P6/P20 RESULT; `sweep/suffix_rule.log` | Brief 5 rebuild (R1 in GENERIC, P20's mechanism); Brief 1's p-floor F dropped | France | +0.0002–0.0003 (8,040 p-floor flips before the guard; R1 on #9as +17.5k / −2.0k pairs) | 1 | none | partly / ok / do-now / partial ("36.7–40.1%" not in the log: rows 32.2–42.0%, headline 40.1%) |
| 3 | R04 | bge band past France's t (blend or veto) | [Ovadia 1906.02530](https://arxiv.org/abs/1906.02530); [Wald 2102.10395](https://arxiv.org/abs/2102.10395); [GPL 2112.07577](https://arxiv.org/pdf/2112.07577v3); [2310.11244v4](https://arxiv.org/html/2310.11244v4); P13, P6, P27 phase A | Brief 2 | France | −0.0002 to +0.001, point +0.0005 | 2 + $0.3 | none | new / ok / maybe / yes |
| 4 | R02 | Sibling filter + unpinned / lower France t | [EMBench](https://velgias.github.io/docs/IoannouRV13.pdf) (homonymity); P15 RESULT; `sweep/hncls.py` | Brief 3 (unpin + sib rule); branch-word veto dropped | France | ±0.0003 | 1 | none | partly / ok / maybe / partial (hncls.py figures not logged: re-derive before quoting) |
| 5 | R05 | hn-equal org-word swap tier gated by s(w) | [GeCo CIKM13](https://users.cecs.anu.edu.au/~Peter.Christen/publications/christen2013cikm.pdf); LANCE; EMBench; P20 RESULT | Brief 4 (gate s(w) ≤ 0.4, d = s/0.68; `sweep/orgswap_s.log`) | France | −0.001 to +0.0014, EV +0.0005 (41.4k flips before the guards) | 1.5 | none | partly / ok / maybe / yes (formula fixed in round 2) |
| 6 | R09 | Parse bracketed '(12)' hn + audit French hn forms | [AFNOR XP Z10-011](http://ressources.sitilr.fr/document/types-de-voie-norme-afnor-xp-z-10-011/); `probes/P6-france/france.py`; `sweep/fr_checks_France.txt` | Brief 5 | France | +0.0001–0.0002 | 1 | none | partly / ok / do-now / – |
| 7 | R17 | Per-source caps (≤ 5 S2, ≤ 6 S3 per S1) | [Karthik FINDINGS](https://github.com/Karthikatstuffmama/amazon-ml-challenge-2026/blob/main/FINDINGS.md) | decide step (Brief 6) | all | +0.00001 | 0.3 | none | new / ok / do-now / yes (train GT: 0 violations, `sweep/caps.log`) |
| 8 | R06 | run.py recipes for combine.py winners | `code/business_entity_resolution/src/run.py` RECIPES l.78 | run.py (Brief 6) | all | 0 (protects rank) | 1 | lowers | partly / ok / do-now / yes |
| 9 | R12 | Verified licence / param table incl. exclusions | [HF Qwen3-8B](https://huggingface.co/api/models/Qwen/Qwen3-8B); [HF bge-reranker-v2-m3](https://huggingface.co/api/models/BAAI/bge-reranker-v2-m3); [HF Jellyfish-8B](https://huggingface.co/api/models/NECOUDBFM/Jellyfish-8B); [HF jina-v2](https://huggingface.co/api/models/jinaai/jina-reranker-v2-base-multilingual) | doc (Brief 6) | doc | 0 | 0.3 | lowers | partly / ok / do-now / (12 cards checked by the rules lens) |
| 9a | R50 | a10 residual stacker, p2-masked, fit on India + US half 1 (Brief 7) | `probes/P23-error-analysis/work/verify-lowp-fn-1/{a10,a13}.log`; `verify-decision-theory-2/log_v2.txt`, `log_v3.txt`; `verify-stolen-fn-0/shift.log`; [sklearn StackingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html) (prefit = "a very high risk of overfitting") | new India/US decision step after P17 | India/US | +0.00015 to +0.00021 (test-reweighted) | 3–4 | none | partly (a10/a13 arms, no ledger row) / ok / do-now / yes (all 6 claims opened; "5+ SE" is 3.7 SE on the swap) |
| 9b | R51 | Stratum-reweighted (IWCV-style) gain estimate on test's mix (Brief 7 gate) | [Sugiyama et al. JMLR 8 (2007)](https://www.jmlr.org/papers/v8/sugiyama07a.html) (CV loses unbiasedness under covariate shift; IWCV is proven unbiased); `log_v3.txt` | `sweep/rw.py` (probe only) | India/US | 0 (decides Brief 7): **run here, passes at +0.000209** | 0.5 | none | new / ok / done / yes (the "0.7×" haircut claim is refuted: 0.93×) |
| 10 | R08 | France consensus (≥ 2 of 3 variants) | [mayank audit_consensus](https://github.com/mayankgoplani431-del/amazon-ml-2026-entity-resolution/blob/main/docs/audit_consensus.md) (0.964 ∩ 0.956 → 0.969); [AotL 2206.13089](https://arxiv.org/abs/2206.13089); [Jiang 2106.13799](https://arxiv.org/abs/2106.13799) | combine step, `sweep/agree.py` | France | −0.0003 (25% false) … +0.0018 (81%) | 1.5 | none | new / ok / maybe / yes |
| 11 | R14 | Recompute context features over the pruned list | [uk_address_matcher](https://raw.githubusercontent.com/moj-analytical-services/uk_address_matcher/main/docs/site/optimising_accuracy.md); [pyJedAI cc](https://raw.githubusercontent.com/AI-team-UoA/pyJedAI/main/src/pyjedai/comparison_cleaning.py); P20 SHAP (sx_margin −2.08 logit) | `features._group_ctx` on kept rows, France rescore | France | +0.0004–0.0008 | 3 | none | partly / ok / maybe / – |
| 12 | R32 | Block-local distinguishing tokens (rule variant) | [identify_distinguishing_tokens.py](https://raw.githubusercontent.com/moj-analytical-services/uk_address_matcher/main/uk_address_matcher/post_linkage/identify_distinguishing_tokens.py); [signature evidence](https://raw.githubusercontent.com/moj-analytical-services/uk_address_matcher/main/docs/experiments/signature_evidence_overview.md) | France dt_unique rule; stacker later | France (all) | 0 to +0.0008 | 1 rule / 5 stacker | none (reimplement; don't install) | new / ok / maybe / – |
| 13 | R15 | IDF share sweep + pruner recall under France IDF | P14 RESULT ("a larger share was not tried") | P14 `variant.py` | France | 0 to +0.0005 | 2 | none | partly / ok / maybe / – |
| 14 | R33 | SX↔SX co-assignment support / France FP flag | [TransClean 2506.04006](https://arxiv.org/abs/2506.04006); `sweep/noaddr.py` | stack.X_of columns; France flag | all | India/US +0.0001–0.0005; France ±0.0003 | 3 | none | partly (P23 twins) / ok / maybe / – |
| 15 | R18 | France website keys (legal form kept, accents deleted) | `sweep/fr_checks_France.txt`; GeCo | France web key in features/block | France | ≤ +0.0002 (≤ ~3.7k SX) | 2 | none | new / ok / maybe / partial (78.8% → 38.4%; ~28k → ~2.1k per 800k-SX sample) |
| 16 | R21 | Per-country vocab tiers + S1-mined family token | [EMM features_vocabulary.py](https://raw.githubusercontent.com/ing-bank/EntityMatchingModel/main/emm/features/features_vocabulary.py); [Auto-FuzzyJoin 2103.04489](https://arxiv.org/abs/2103.04489) | stack.X_of, 13 columns | France | +0.0003–0.0012 (LB only) | 3 | none | partly (P23 a10 CATS) / ok / maybe / yes |
| 17 | R22 | Group-relative stats per similarity | [Foursquare 1st code](https://www.kaggle.com/code/takoihiraokazu/sub-ex73-74-75-ex104-115-90-101-merge-train3); [Foursquare roundup](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336140); [EMM features_rank.py](https://raw.githubusercontent.com/ing-bank/EntityMatchingModel/main/emm/features/features_rank.py); [Foursquare 16th](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/335860) | stack.X_of | all | India/US +0.0001–0.0004 | 3 | none | partly (P23 sxstack.py) / ok / maybe / partial (feature-name strings unverified) |
| 18 | R25 | legal_add × hn-class in the US stacker | LANCE; Febrl | P17 `hn_stack.py` X | US | +0.00005–0.0001 | 1.5 | none | new / ok / maybe / – |
| 19 | R37 | Soft-TFIDF with the country IDF | [py_stringmatching soft_tfidf](https://raw.githubusercontent.com/anhaidgroup/py_stringmatching/master/py_stringmatching/similarity_measure/soft_tfidf.py); lieu | add-on to R22's run | all | +0.0001–0.0003 | 2 | none | new / ok / maybe / – |
| 20 | R38 | S1–S1 2d-ball density, same-name-S1-in-city | Auto-FuzzyJoin Eq. 8; [Splink training](https://moj-analytical-services.github.io/splink/api_docs/training.html) | stack.X_of | all | CE base +0.0001–0.0003 | 4 | none | partly (P23 a6/a7 twins) / ok / maybe / – |
| 21 | R39 | India number-bag address features | [Foursquare 7th](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336148); [Shopee 6th](https://www.kaggle.com/c/shopee-product-matching/discussion/238010) | stacker columns | India | bge base 0 to +0.0002 | 2 | none | new / ok / maybe / – |
| 22 | R43 | Spectral Meta-Learner France weights, LOCO-gated | [SML 1303.3257](https://arxiv.org/abs/1303.3257); [Snorkel 1711.10160](https://arxiv.org/abs/1711.10160); [GTI 2211.06975](https://arxiv.org/abs/2211.06975) | France blend weights | France | ±0.0003 (trust check) | 3 | none | new / ok / maybe / yes |
| 23 | R44 | Reverse validation, rank-checked on LOCO | [Musgrave 2208.07360](https://arxiv.org/abs/2208.07360); [CST 2103.03571](https://arxiv.org/abs/2103.03571) | selection tool | France | saves 1–2 slots | 3 | none | new / ok / maybe / partial (Musgrave figures unconfirmed) |
| 24 | R19 | Label-free France vocabulary mining | [Deng 1709.10436](https://arxiv.org/abs/1709.10436); `probes/P1-taxonomy/mine.py`; GeCo; LANCE | learned GENERIC_c / SIB_c (`sweep/vocab_asym.py`) | France | indirect | 2 | low (human-approved list) | partly / ok / maybe / yes |
| 25 | R20 | France normaliser bundle (street types, bis/ter, St→Saint, d.b.a) | AFNOR; [Schwartz–Hearst](https://psb.stanford.edu/psb-online/proceedings/psb03/schwartz.pdf); `sweep/fr_rules.log` (address Jaccard 0.593→0.665) | name-side items folded into Brief 5 | France | ≤ +0.0002 | 1.5 | none | partly / ok / maybe / yes |
| 26 | R26 | CatBoost text features on diff tokens | [Foursquare 9th](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336415) | P22 job | India/US | +0.0001–0.0004 (P22 family +0.00003) | 2 | none | partly / ok / maybe / – |
| 27 | R10 | Per-country pruner τ (France 0.1–0.2) | [LCRON 2503.09492](https://arxiv.org/html/2503.09492); [CRC 2208.02814](https://arxiv.org/pdf/2208.02814) | `prune.TAU` dict | cand size | −0.13 to −0.17 cands/S1 for ≤ −0.00004 | 0.5 | none | new / ok / maybe / – |
| 28 | R11 | Sibling-offset q cap in the pruner | [lieu README](https://raw.githubusercontent.com/openvenues/lieu/master/README.md); [GSM 2204.08801](https://arxiv.org/pdf/2204.08801) | prune keep rule | cand size | 4.83 → 4.50/S1 at −0.00004 | 1.5 | none | partly (P24 hn.py) / ok / maybe / – |
| 29 | R16 | F0.5-cost-aware pruning (q·w(m̂) ≥ λ) | [RLT 2404.18185](https://arxiv.org/abs/2404.18185); [Choppy 2004.13012](https://arxiv.org/abs/2004.13012); [Ye 1206.4625](https://arxiv.org/abs/1206.4625) | prune keep rule | cand size | with R10 + R11: 4.83 → 4.28/S1 for ≈ −0.0001 | 2 | none | new / ok / maybe / – |
| 30 | R36 | Pruner + hn_cls, hn_d | GSM; [Foursquare blog](https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/); P10 RESULT | `prune.FEATS` += 2 | cand size | −0.1 to −0.2/S1 | 3 | low (2 columns only) | partly (P24) / ok / maybe / – |
| 31 | R34 | BLAST / RWNP two-sided node pruning | [pyJedAI](https://github.com/AI-team-UoA/pyJedAI); GSM | P24 sweep | cand size | 4.3–4.6/S1 (guess), 0 LB | 1.5 | none | partly (P10 one-sided dropped) / ok / maybe / – |
| 32 | R45 | Cascade-consistent pruner target | LCRON | `prune.train` y' | cand size | −0.1/S1 or +0.0001 | 3.5 | none | partly / ok / maybe / yes |
| 33 | R40 | Exact address-key recovery pass | [Foursquare 7th blog](https://future-architect.github.io/articles/20220720a/); [Papadakis survey 1905.06167](https://arxiv.org/pdf/1905.06167) | block.py pass | India | ≈ +0.0001 | 4 | none | partly (P23, P24) / ok / maybe / – |
| 34 | R24 | Evidence-first finale narrative | [Harpal methodology](https://github.com/HarpalKalankar/business-entity-resolution/blob/main/docs/methodology.md); [arnav10goel 2024](https://github.com/arnav10goel/Amazon-ML-Challenge-24); [aboutamazon finale](https://www.aboutamazon.in/news/job-creation-and-investment/amazon-ml-challenge-a-unique-upskilling-opportunity-to-tackle-a-real-world-problem) | doc + deck | doc | 0 | 3 | none | partly / ok / do-now / partial (KhadgaA = 2024 image task, "winner" unverified) |
| 35 | R23 | Blocking in PC / PQ / RR terms + floor | [Sparkly](https://pages.cs.wisc.edu/~anhai/papers1/sparkly-vldb2023.pdf); Papadakis survey; [mayank audit](https://github.com/mayankgoplani431-del/amazon-ml-2026-entity-resolution/blob/main/docs/audit.md); [Karthik LITERATURE](https://github.com/Karthikatstuffmama/amazon-ml-challenge-2026/blob/main/LITERATURE.md); [jhansi-jjs](https://github.com/jhansi-jjs/business-entity-resolution) | doc | doc | 0 | 1.25 | none | new / ok / maybe / partial (mayank PC 97.51% is train) |
| 36 | R41 | CRC-certified candidate recall (US/India) | CRC 2208.02814 | doc check | doc | 0 | 0.5 | none | new / ok / do-now / yes |
| 37 | R07 | France-first slot plan, stop P25 | [Harpal](https://github.com/HarpalKalankar/business-entity-resolution); [Karthik](https://github.com/Karthikatstuffmama/amazon-ml-challenge-2026); [PLAN_98](https://github.com/mayankgoplani431-del/amazon-ml-2026-entity-resolution/blob/main/docs/PLAN_98.md) | — | France | organisational | 0.25 | none | already-done / ok / do-now / yes |
| 38 | R13 | CE–GBDT disagreement meter | P13 RESULT; [Ruder 1804.09530](https://arxiv.org/abs/1804.09530); Musgrave | gate (`sweep/disagree*.py`) | France | 0 | 0.3 | none | already-done / ok / do-now / – |
| 39 | R27 | Both-OOD proxy (LOCO GBDT + LOCO CE) | P13 frproxy json; P16; [2409.08185v2](https://arxiv.org/html/2409.08185v2); [CORAL 1511.05547](https://arxiv.org/abs/1511.05547) | new probe | France | +0.0002–0.0006 | 4.5 | none | partly / ok / too-late / – |
| 40 | R28 | Per-country rank/CDF-normalised twin matcher | [Shopee 5th](https://www.kaggle.com/c/shopee-product-matching/discussion/238078); [AdaBN 1603.04779](https://arxiv.org/abs/1603.04779); [UDALM 2104.07078](https://arxiv.org/abs/2104.07078) | twin GBDT | France | EV +0.0003 | 5 | none | new / ok / too-late / – |
| 41 | R29 | DG GBDT (monotone, low capacity, seed bag) | [LightGBM params](https://lightgbm.readthedocs.io/en/latest/Parameters.html); [DomainBed 2007.01434](https://arxiv.org/abs/2007.01434); [GML 1810.12125](https://arxiv.org/abs/1810.12125) | LOCO grid | France | ~+0.0007 | 5 | none | new / ok / too-late / – |
| 42 | R30 | France correction layer trained on out-of-country rows | [Gong 2104.00742](https://arxiv.org/abs/2104.00742); [MLDG 1710.03463](https://arxiv.org/abs/1710.03463) | LOCO residual | France | +0.0002–0.0006 | 5 | none | partly (P17 loco) / ok / too-late / – |
| 43 | R31 | FGM + EMA for bge | [Foursquare 1st/15th](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336140) | `bge_train.py` | France | +0.0003–0.0008, uncertain | 4 + $5 | none | new / ok / too-late / – |
| 44 | R35 | Listwise second-pass pruner | [Foursquare 2nd](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336072); 16th; GSM; RLT | prune pass 2 | cand size | −0.05 to −0.3/S1 | 3 | none | partly (P24 scoping) / ok / too-late / – |
| 45 | R42 | Ditto typed [HN] spans + hn-relation token in CE input | [Ditto 2004.00584](https://arxiv.org/pdf/2004.00584); [Ditto knowledge.py](https://raw.githubusercontent.com/megagonlabs/ditto/master/ditto_light/knowledge.py) | CE retrain | all | India/US +0.0002–0.0004 | 6 + $3–4 | none | new / ok / too-late / yes |
| 46 | R47 | TAPT (MLM on France pair text), then fine-tune | [Gururangan 2004.10964](https://arxiv.org/abs/2004.10964); [AdapterEM 2305.18725](https://arxiv.org/html/2305.18725); GPL; UDALM; [TSDAE 2104.06979](https://arxiv.org/abs/2104.06979) | CE pretrain | France | 0 to +0.0005 | 7 + ~$8 | none | new / ok / too-late / yes (AdapterEM low end +2.4, not +2.8) |
| 47 | R48 | Decoder-backbone matcher (Qwen3-Reranker-0.6B / mxbai; generative Qwen3) | [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B); [mxbai-v2](https://huggingface.co/api/models/mixedbread-ai/mxbai-rerank-base-v2); [2607.24688](https://arxiv.org/abs/2607.24688) | CE swap | France | EV +0.0001 | 6 | caution (generative arm = LLM) | partly / caution / too-late / yes |
| 48 | R49 | Pseudo-labels done right (multi-teacher, star negatives, CBST) | [PromptEM 2207.04802](https://arxiv.org/abs/2207.04802); Ruder; [CBST 1810.07911](https://arxiv.org/abs/1810.07911); [Kumar 2002.11361](https://arxiv.org/abs/2002.11361); [DistillER 2602.05452](https://arxiv.org/html/2602.05452v1); [2606.28823](https://arxiv.org/html/2606.28823v1) | → P27 | France | −0.0005 to +0.0015 | 7 | none | partly / ok / not-worth-it / yes ("3–6k pairs suffice" is benchmark-dependent: 1k–16k) |
| 49 | R46 | [NEEDS LLM] zero-shot local LLM / Qwen3-Reranker-4B gate | [2405.16884v3](https://arxiv.org/html/2405.16884v3); [2310.11244v4](https://arxiv.org/html/2310.11244v4); [HF Qwen3-Reranker-4B](https://huggingface.co/api/models/Qwen/Qwen3-Reranker-4B) | — | France | EV +0.00005 | 3.5 | caution: LLM, master sign-off | new / caution / not-worth-it / yes ("74–77" → 73.8–76.3) |
| 50 | G01 | bis/ter/quater hn suffix | Gemini report (no URL) | — | France | 0 | 1.5 | none | already-done / ok / do-now / yes |
| 51 | G02 | NCD (zlib) pair feature | [Avito 1st interview](https://medium.com/kaggle-blog/avito-duplicate-ads-detection-winners-interview-1st-place-team-devil-team-stanislav-semenov-f9421c43a2b7) | features + full retrain | all | ≤ +0.0001 | 4 | none | new / ok / not-worth-it / yes |
| 52 | G03 | Multi-view TTA for the France CE | [DPLOT 2404.10966](https://arxiv.org/abs/2404.10966) (image TTA only) | extra CE pass | France | small, unvalidated | 3 | none | new / ok / not-worth-it / partial |
| 53 | G04 | CORAL moment alignment of France features | DADER (already in P15's table) | transform France X | France | may be negative | 4 | none | new / ok / not-worth-it / yes |
| 54 | G05 | Split-conformal France answer sets | textbook | France t | France | ~0 or negative | 2 | none | refuted / ok / not-worth-it / yes |
| 55 | G06 | GPL-style margin distillation on synthetic France pairs | [GPL 2112.07577](https://arxiv.org/abs/2112.07577) | synthetic set | France | negative prior | 8 | none | refuted / ok / not-worth-it / yes |

## What failed      (approaches tried and dropped, with the number that killed them)
**Killed by verification** (lens: reason). Do not retry these without a new angle.
- **R07** (tried: already-done): the #8c → France variants → combine sequence already produced #9ab. "Stop P25" is moot:
  P25 `eval-six-8c` = drop at 14:57, with each bag alone within ±0.00008 of #8c. This doc's slot map supersedes R07.
- **R13** (tried: already-done): disagreement is already reported for every France file (P13 31%, P19 33.4%). It stays a
  gate inside Brief 2, never a selector: Musgrave finds entropy/confidence-type validators weak.
- **R27** (feasible: too-late, 4.5 h + AWS). A CPU-only 1 h variant exists, which is the master's call:
  - P6's LOCO boosters are restored at `data/work/p27/p6/`, and LOCO bge r exists at
    `data/work/p13/r/r_bge_loco{US,India}_holdout.parquet`;
  - so `p13.py` xw/frproxy can run on a LOCO-GBDT base;
  - but r covers only pooled-band pairs, so it cannot calibrate Brief 2's band above 0.98.
- **R28, R29** (too-late, 5 h each, P(pass) ~1/3 by their own estimate): each is a one-line arm in P26's LOCO harness (a
  `rank(pct=True)` wrapper that leaves sx_margin raw; `monotone_constraints` + low capacity). That is P26's call.
- **R30** (too-late): 5 h and over-correction risk. P17's LOCO transfer (+0.00073 / +0.00203) is the only evidence.
- **R31, R42, R47, R48** (too-late): CE retrains of 4–7 h with no ablation in this setting (R31), an uncertain gain (R42),
  conflicting literature (R47: UDALM +1.8 vs GPL −12.2 for mixed loss), or EV +0.0001 (R48).
- **R35** (too-late): a new model class for candidate size only.
- **R49** (not-worth-it standalone): P27 runs the same programme (filtered pseudo-labels, LOCO-gated in both directions).
  Its phase A already measured R49's one new part: star-constraint hard negatives (losers of a confident argmax) are
  **99.99–100.00% false** in both LOCO directions (506k US→India, 718k India→US), and "p ≥ .98 & hn eq/none" is 0.9988 precise
  (India→US). Pass the CBST throttling and low-capacity residual student ideas to P27.
- **R46** (not-worth-it; **needs an LLM**, master sign-off): EV ≈ +0.00005. ≤ 8B zero-shot pairwise F1 is 45.9–62.1.
  Qwen3-8B / Qwen3-Reranker-8B (8.19B) and Qwen2.5-3B (qwen-research) disqualify.
- **G01** (already-done): the premise is false. `house_number()` takes the leading digit run, so '12bis' → '12'. bis/ter are
  381 per 10k France SX with hn empty 0.000, and the P20 taxonomy lists it as handled.
- **G02 / G03 / G04** (not-worth-it):
  - NCD needs a full m2_big retrain for ≤ +0.0001.
  - DPLOT is image-only TTA.
  - CORAL would align away a structural shift: sx_margin's France shift is same-city twins (30% of France S1 share
    core+city vs US 1.2%).
- **G05 / G06** (refuted): conformal sets reduce to one threshold without exchangeability (P15 #11; P6 FS-EM 0.27), and
  synthetic France pairs are P12's −0.8 pt family.
- **Partial drops inside surviving ideas:**
  - R02's branch-word veto: 131 pairs.
  - R03's reassignment of key pairs that lost the argmax: India holdout 0.4% true, n 248.
  - R18's "78.8% / ~28k": the actual figures are 38.4% / ~2.1k per 800k-SX sample.
  - `suffix_rule.py`'s 'compagnie'.
  - Brief 1's p-floor F tier. It is a mechanism choice, not a data kill: R1 goes into Brief 5's rebuild as a GENERIC change
    that run.py can reproduce, and running both would double-count.
  - #20a as an upload: it has no bge, so its delta against #9ab confounds R1 with the missing CE (Brief 5 supersedes it).
  - Brief 4's s(w) ≤ 0.6 gate: at s = 0.6 the decoy share is ≈ 0.88 and P ≈ 0.72, which is below the bar. Now s ≤ 0.4.

**Round 2 (vetted by the final editor):**
- **Tier M as its own slot** (core-equal, SX hn missing after the parse, street + city equal, unique core in city):
  killed. The label analog is 100% precise (US 17,641, India 643), but only 9 / 3 pairs sit in [0.5, .985). The model
  already accepts the cell.
  - France flips: 388 on 374 S1 (23 empties). Without the street guard: 755 flips, 32 empties.
  - LB ≈ +0.000025, or +0.00004 without the street guard. The brief assumed +0.0003 to +0.0006, 10–20× too much.
  - The "18.7k France pairs missed" premise is mostly the cell's make-up. In France's core-equal, hn-missing argmax cell
    (89,619 pairs), 54,276 are city-unequal. There France accepts 0.61 of the unique-core pairs against US truth 0.673, so
    there is no clean pool of misses.
  - It survives only as Brief 1's zero-cost clause M at p ≥ .90 (`sweep/mtier.log`).
- **Brief 7's "about 0.7× on test" haircut:** refuted by the reweighting that round 2 could not finish. The first run was
  killed by the memory guard (exit 144) at ≈ 1.6 GB free. Re-run alone under a 2 GB cgroup, it completed: the
  stratum-share shift costs ≈ 7%, not 30% (`sweep/rw.log`).
- **v2_impute** (impute a bge r for the 570k e5-blended competitor band rows, then recompute p2): not run. It needs the
  train-id load plus gt_pairs, well over 2 GB. The reweighting passed, so it is unnecessary; it would be an AWS job only
  if Brief 7's test profile fails.
- **Unreproduced number:** Brief 1's "3.44% of unique-key pairs rejected". `keyrej.py` gives 2.65% for #9ab's argmax
  key + city pairs. The US 0.25% reproduced.

**Sweeper dead ends** (measured here or checked; aggregates only):
- **Negative on data:**
  - Hard S1-family rule: India precision 0.989 vs 0.995, on 1.42% of accepted pairs.
  - Hard "address-less & chain ≥ 2 → abstain" (US): −800 TP for −155 FP.
  - FR↔EN translation lexicon: 67,430 confident no-shared-token pairs gave 0 strong associations. France SX cores with no
    S1-vocabulary token: 0.0563 vs US 0.0564.
  - France blocking is not the gap: France S1 with no kept candidate 0.98% vs US 1.42%, India 1.30%.
  - Indic-script SX are not India's weak spot (recall 0.9899 vs Latin 0.9685), so phonetic keys are pointless.
  - SX/S1 frequency-asymmetry decoy rule is unsafe: 'associes' is true-pair noise, and 'frs' = frères.
  - India H.No./D.No. parse fix: only 46 FN / 22 FP rows touched, and the fix would create hn mismatches.
  - France city_key bug (1.96% of SX): accepted 89.3% vs 85.7% for peers, so ≈ 0.
  - Typo-scramble / phonetic signatures: 0.4% of TPs.
  - Address-key union blocking: India keys too coarse (393 S1 per key, p99 3,462), ≈ +0.0001 for 4–6 h.
  - French abbreviation channels (général/gal, maréchal, fg, ZA/ZI): SX/S1 ≈ 0.9, so nothing to map. 'pas' =
    Pas-de-Calais. Elision 13 of 2.5M tokens. CEDEX/BP < 0.1 per 10k. Extra legal forms ≤ 130 records each.
    'groupe' / 'developpement' carry 23–34% sibling offsets: never strip them.
- **Negative on the model side:**
  - Listwise 'select' / LCE: after pruning, 92.1% / 96.4% / 98.1% of SX (France / India / US) have a single candidate.
  - OOD CE ensembles (bge + e5, bge final + mid): ≤ +0.00014 over bge alone.
  - Legal-form canonicalisation of the CE input: disagreement flat (33.2% vs 33.6%).
  - rapidfuzz empty-string artefact: cores are never empty.
  - Brand-at-address pairs: India/US already recover 98%.
  - India/US t for test's orphan density: LB ≈ holdout.
  - France digit-drop scarcity (8.8 vs 101 per 1k S1) is the generator, not a leak.
- **Candidate size:**
  - Reporting m2_big p ≥ 0.02 as the candidate set (3.89 / 3.95 per S1) is **no longer listed as a kill**. PROBLEM.md
    ruling 3 says "a pruned later stage counts", so it is an interpretation question for the master (see Second tier →
    Candidate size and Open questions).
  - 2–3 cands/S1 is infeasible (answers are 3.38 / 2.99 per S1).
  - Supervised BLAST / RCNP collapse to P10's 'rel' and fixed top-k in a star graph.
  - SC-Block needs a GPU, and the misses are not cross-script (P3 cross-script recall 0.997).
  - Sorted-neighbourhood: blocking recall is already 0.992.
- **Tools and licences that disqualify:**
  - Qwen3-8B, Qwen3-Reranker-8B, granite-3.3-8b and gemma-2-9b: over 8B.
  - Qwen2.5-3B: qwen-research. Jellyfish and jina-reranker-v2: cc-by-nc. Llama-3.1-8B and Gemma: custom licences.
    Ministral-8B: MRL. bge-reranker-v2-gemma and camembert-large: unclear licence.
  - deepparse: LGPL, trained on libpostal data. abydos: GPL. DNB name_matching: ships legal_names.csv. Zingg: AGPL.
    sparker: GPL. bharataddress: pincode tables.
- **No new signal:**
  - dedupe / recordlinkage / Splink EM: Fellegi-Sunter was already dropped at 0.27.
  - string_grouper is the TF-IDF we already run; Sparkly is the same family.
  - AnyMatch has no weights on HF (and it is our LOCO CE idea). CollaborEM publishes no pseudo-label rule.
  - MatchGPT evaluates 70B-class models only. DistillER / 2606.28823 teachers are over the cap or hosted.
  - Risk-based ER 2012.03513 needs target labels. The full TransClean needs manual labels. OpenSanctions' best local model
    is 14B.
  - Agreement-on-the-line holds for NNs only. Tent / DANN raise confidence (DAT 87.68 < 89.93). DAME-style MoE cannot be
    validated with two labelled countries.
- **Competitors:** 83 public repos (53 READMEs read). Best self-reported scores: 0.969 (consensus), 0.964, 0.956, 0.945.
  None handles France beyond normalisation, and the Kaggle mirror notebooks are baselines.
- **Access notes:**
  - The Semantic Scholar API returns 429: use the arXiv API.
  - Kaggle discussions are readable through the internal API: `GetForumTopicById` with the XSRF header, ~1 request per
    1.5 s. Notebooks via `kernels/pull`.
  - Medium returns 403. Unstop and Kaggle notebooks are JS-only. The Foursquare 3rd-place writeup is unreachable.
  - LOCO CE checkpoints were never saved (only `ce_bge_pool.pt`), so any new LOCO CE scoring costs ~45 min on an A10G.
  - The g4dn price in ap-south-1 is unverified.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
The sweepers' scripts lived in the session scratchpad (under /tmp, lost on reboot). They are copied to
`probes/P23-research/sweep/` with their aggregate logs, and the scratchpad imports are re-pointed to that folder. All read
`/home/harsh1/github-repos/amc-26/data/work/` and print aggregates only. Per the gap analysis they ran under a 2 GB
cgroup cap (`systemd-run MemoryMax=2000M`; `prlimit --as` breaks pyarrow's thread start).

| file | what | inputs → output |
|---|---|---|
| `keytier.py` | France key-tier sizing (Brief 1 K) | p19 France file + rf_test_{s1,sx}_France → pairs added / S1 changed / empties filled at t_key 0.95 / 0.90 / 0.71 |
| `keyprec.py <c>` | holdout precision of unique key + city pairs, by p band; share that lost the argmax | `p23/pairs_{c}` + rf_train → stdout |
| `suffix_rule.py [prob] [t]`, `suffix_rule.log` | pairs whose cores differ only by filler words (old Brief 1 F, now Brief 5's R1 check; its FR set includes 'compagnie': drop it) | France p file → counts by band |
| `legal_swap.py`, `legal_swap.log`, `classes.py`, `classes.csv` | (core_eq × legal relation × hn relation) truth on the US/India holdout and France acceptance | P23 dump, rf files, p19 file |
| `agree.py` | overlaps of accepted France sets, 2-of-3 vote (R08) | four staged France p files |
| `caps.py <tsv>…` | per-S1 S2/S3 count histograms (R17) | GT or matching_results TSV |
| `disagree.py`, `disagree2.py` | CE-vs-GBDT band disagreement, overall and per class (R13 gate) | country p + r |
| `hncls.py <c> <prob> <t>`, `hncls_ho.py <c>` | hn-class mix / acceptance of argmax pairs on test, and the holdout analog | p file, rf files, P23 dump |
| `budget.py`, `budget2.py` (+ `fbeta.py`), `proxy.py`, `noaddr.py` | the gap analysis: per-S1 Shapley budget, P14 proxy re-split, address-less split | `p23/*`, `p14/prob_world_IN_*` |
| `branch_fr.py <prob> <t>`, `branch_fr.txt` | added-token classes vs sibling offset (Brief 1 connectors, Brief 4 s(w)) | France p file |
| `fr_checks.py`, `fr_checks_{France,US,India}.txt`; `fr_rules.py`, `fr_rules.log`; `hnfix.py` | generator operators vs our normaliser: website keys, alias markers, hn formats; normaliser touch counts; hn-label fix | test sources, rf files |
| `vocab_asym.py <c> <split>` | S1-vs-SX token asymmetry r(t) (R19) | rf files |
| `keytier.log`, `keyprec.log` (round 2) | the logs behind Brief 1's K numbers | — |
| `keyrej.py <c> <prob> <t>`, `keyrej.log` | share of argmax unique-key + city test pairs a country's t rejects | rf_test + p file → stdout |
| `caps.log` | `caps.py` on train GT and #9ab (R17 check) | — |
| `orgswap_s.py`, `orgswap_s.log`; `add1_s.py`, `add1_s.log` | Brief 4: per-word s(w) over P20's sub1 / add1 pairs; flips on #9ab by s bin and p band; formula EV. Run from the repo root (relative paths) | `p20/pairs_France.parquet` + p19 file |
| `mtier.py <c>`, `mtier.log` | tier M precision (US/India holdout) and France flips; the '(12)' parse's hn moves | P23 dump, rf files, p19 file |
| `rw.py`, `rw.log` | R51: a10 twin / twin_p2mask gain reweighted to test's per-S1 px_2nd strata, both directions. Imports `probes/P23-error-analysis/work/lowp-fn/a10_stack.py`, whose inputs sit in another session's /tmp scratchpad (`cats.SP`): copy them to `data/work/p23/` before a reboot. Run from the repo root; 2 GB cgroup, ~10 min | a10 frames + `p23/pairs_{c}` + `p17/prob_test_{c}_bge` |

Proposed probe ids: P28-fr-tiers, P29-fr-veto, P30-fr-orgswap, P31-fr-rebuild, P32-iu-stack.
- The France briefs write a probability file with the schema s1, sx, p (+ p_gbdt), build with
  `probes/P14-france-shift/combine.py output/sub8c <file> <t> <out>`, and run the validator.
- Brief 7 writes India/US p files in the p17 schema (s1, sx, p) on the same pruned rows.
- Briefs 1, 3, 4 and 7 are CPU under 2 GB. Brief 5's rescore needs ~3 GB (halve the slice, or use SageMaker CPU).
  Brief 2 is one g5 job.
- Every winner needs a run.py stage before the 27 Sep 14:00 freeze.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Final = the best public upload, and code freezes at 27 Sep 14:00.** Any file that tops 0.984072 becomes the entry,
  so R06's recipe and the validator come first. A file the frozen run.py cannot regenerate must never top the board.
  Anything not ported by ~12:00 on 27 Sep (Brief 4, Brief 7, the P26/P27 winner) is dropped, not rushed. With |Δ| < 0.0002 on ~8–15k changed France S1, public-vs-private noise can flip the order
  (my estimate; the public subset size is unknown).
- **France precision comes from label analogs.** US/India holdout class truth stands in for France.
  - France is US-like on house numbers (siblings keep the hn 2–6%), which supports the US numbers.
  - It is denser than both on same-name chains (30% of S1 share core+city, vs India 22%, US 1.2%), which is why every tier
    carries a uniqueness guard.
  - The India analog is the downside bound.
- **The rebased France budget is arithmetic, not a measurement**: D4 estimates minus the P14 proxy's IDF split. It sums to
  0.048 against the 0.050 implied by the LB.
- **Stacking adders at a fixed t lowers the empty rate.** Re-pick t after combining, and do not pin it to 6.71%. The
  ~6.0% floor holds only at India/US LB = holdout; at 0.001 below holdout it is ~6.5% (row f).
- **Brief 2 rests on P27's band-selected LOCO subset.** About 5% of confident pairs have r, and that subset has base
  precision 0.65 / 0.62 against 0.959 / 0.994 for the rest. France's CE is further out of domain (disagreement 33%), and
  the count gate is the guard.
- **Brief 4 flips pairs the model scores low.** 30,295 of its 42,587 pre-guard flips have p < 0.5. Only the US analog
  and India→US LOCO gate it, and the p ≥ 0.5 variant (12.3k flips) is the fallback if either is marginal.
- **Brief 7's US fit.** US half-0 p in the P23 dump is P17's in-sample output. Fitting on it attenuates the residual
  (the StackingClassifier 'prefit' case), and the US-half-1 ship config has no clean US report of its own.
- **Compliance regeneration (P21).** Staged France files were built with the old FR_REGIONS table. The final file must be
  regenerated on main with the learned map and diffed against the upload (sub9d moved 1,039 rows; #9ab was never
  rebuilt).
- **Candidate-set reading.** If the master reports m2_big p ≥ 0.02 (3.89 / 3.95 per S1) as the candidate file, every tier
  needs its p ≥ 0.02 floor, or matches stop being a subset of candidates.
- **Rules and compliance:** nothing recommended uses an LLM, external data or a new dependency. R46 and R48's generative
  arm are flagged NEEDS AN LLM and are killed. R32 must be reimplemented, not `pip install`ed (check what
  uk_address_matcher bundles).
- **Source fixes found by the verify pass:** R01 (32.2–42.0%, not 36.7–40.1%), R02 (hncls figures not logged), R18
  (38.4%, not 78.8%), R23 (PC 97.51% is on train), R24 (KhadgaA), R44 (Musgrave figures unconfirmed), R46 (73.8–76.3),
  R47 (AdapterEM +2.4 low end), R49 (1k–16k pairs). Round 2 added: Brief 1's 3.44% (→ 2.65%), Brief 4's s-as-decoy formula
  and the 30k size (→ 42.6k flips), and Brief 7's 0.7× haircut (→ 0.93×). All the other Brief 1 K numbers and R17's caps are now
  logged in `sweep/`. None of the recommended briefs depends on a disputed number.
- **Resources:** Brief 2 is the only new AWS spend (≈ $0.34, plus ≤ $0.1 if Brief 5 has > 2k band pairs without r). P26/P27 hold the AWS training budget. The laptop is shared, so
  every brief is capped at 2 GB.

## Open questions   (for the master; each with your best guess)
- **Tiers first, or the Brief 5 rebuild first?** Best guess: tiers (K + M) in slot 2 on the #9ab-family file, because
  they need no rescore, then the rebuild in slot 4 with the tiers re-applied on it. L gets its own slot only if its gate
  passes and a slot is spare.
- **Which France t after the tiers and the veto?** Best guess: 0.975 if #9ab_t975 ≥ #9ab. The tiers act below t and the veto
  above 0.98, so neither moves the optimum much.
- **Does P20's R1 strip leave a dangling 'et' / '+'?** Best guess: yes, on part of 'et''s 11.9k pairs. Check noise_core
  output on those pairs before extending the strip set.
- **Should the tiers apply to a P26/P27 France file if one passes?** Best guess: yes, re-measured. They are record-relation
  overrides, independent of the model, but their flip counts will shrink if the new model already accepts these classes.
- **Is France's singleton share really 5.6–5.8%?** Best guess: yes, but it is conditional. It is the value that makes
  India/US's implied LB equal their holdout at D1, and D1–D4 were consistent with it. Each −0.001 of India/US LB against
  holdout moves it +0.57 pt. If a combined file drops France below ~6.2% empty and the LB
  falls, that bound was loose.
- **Does Brief 4 conflict with R14?** Best guess: yes, same cell. Run Brief 4 first (1.5 h, rule-level). Use R14 only if
  Brief 4 misses its analog gate, because R14 fixes the cause (a depressed margin) rather than the symptom.
- **Which file is candidate_pairs.tsv: the pruner output (4.83 per S1) or m2_big p ≥ 0.02 (3.89 / 3.95)?** Best guess:
  keep 4.83 for the final zip unless the master reads ruling 3 ("a pruned later stage counts") as covering m2_big's 0.02
  cut. If it does, switch, and keep the p ≥ 0.02 floor on every tier so that matches stay a subset of candidates.
- **Brief 7: its own slot or inside the combined file?** Best guess: inside it (slot 5). Its +0.0002 is inside the tie
  band, so a dedicated slot cannot confirm it. The holdout (both directions) and the test-reweighted estimate carry the
  decision, and the label-free test profile is the guard against a feature-port bug.
- **Is anything from 26 Sep safe to leave unported?** Best guess: no. If a probe-built upload becomes the best public
  score and run.py cannot regenerate it by 14:00 on 27 Sep, the entry is not reproducible. Port the recipe of every
  upload that beats 0.984072, in order.
