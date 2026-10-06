# P39-fr-group-model — RESULT (final, 27 Sep ~01:00)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **The gate fails.** No group-aware GBDT beats the France chain analogue by ≥ +0.0005 dF in both LOCO directions, on
  top of either the chain or the current recipe (rescue90 + gc70).
- **Group columns alone are worth ≈ 0 on top of gc70** (`p_group`: +0.0003 / 0.0000). The signal that does exist
  (hn / name relations) is country-specific:
  - trained on India labels and applied to US: **+0.0045 ± 0.0002** on top of gc70;
  - trained on US labels and applied to India: **−0.06** (it adds 23k India pairs at 20% precision).
- **Recommendation: nothing ships as gate-passing.** `output/sub39_ind80` (validator PASS) is staged only as a
  France-like-direction spare, on the same footing as P37's hn90. It is riskier, so upload it only after hn90.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- **Adopt nothing into src.** The both-directions gate fails for all 7 arms run, in every decode mode: full / add / add +
  remove / gc = model-scored one-sided completion. The best both-ways
  point on top of gc70 is `p_group_gt` add τ 0.96: +0.00046 / +0.00038, below +0.0005 and ≈ +0.00006 LB.
- **Optional spare: `output/sub39_ind80/`.** Built on `sub37_gc70` (LB 0.986304).
  - Validator PASS with `--check-ids`. Only France lines differ: 15,749 S1, +15,226 / −1,530 pairs (3.137 → 3.190
    per S1). candidate_pairs.tsv and code.zip are hard links to gc70's.
  - Model: the `full` arm (23 columns), trained on **India LOCO labels only** (`loco_US_India`). That is exactly the
    configuration behind the France-like India>US direction. Decode: add s ≥ 0.80, drop base pairs with s < 0.3,
    caps 5 S2 / 6 S3 with base pairs kept first.
  - Rebuild (≈ 20 s, 0.8 GB):
    `.venv/bin/python probes/P39-fr-group-model/build.py make full addrm 0.80 sub39_ind80 India gc70`.
    Use `… India hn90` to rebuild on `sub37_gc70_hn90`.
  - **Rank it after hn90.** Its France adds overlap hn90's extras by 56% (10.3k of 18.3k).
  - If hn90 gains, the increment worth testing is the `hn90` base variant: +6,600 / −3,706 France pairs, LOCO India>US
    +0.0038 on top of the hn90 analogue.
  - If hn90 loses, sub39_ind80 is not worth a slot. The shared hn-equal adds are what failed.
- **Expected LB of sub39_ind80 vs gc70.** It depends on which LOCO direction France resembles:
  - **India>US-like (the France analogue; see the lens below): about +0.0003.** LOCO says +0.0045 F. But on France the
    model mostly adds (0.059 adds and 0.006 removals per S1, vs 0.030 and 0.036 in LOCO), and the removal part carried
    +0.003 of the LOCO gain. So the France-side gain is closer to the add-only +0.0015 F plus whatever the 2× add rate
    keeps.
  - If France's below-t pairs are rarer true matches than India's: a loss, of unknown size.

## Evidence         (table: ledger name | world | key metrics | verdict)
**Protocol**
- Base on the target, the "France chain analogue": P6's source-only M2, plus the source-only bge band blend (P13's
  France weights), plus P30's veto τ 0.1, at P30's fixed label-free t (0.915 / 0.92). It reproduces P30 exactly:
  **US>India 0.95523, India>US 0.95616**.
- "US>India" = second stage trained on US labels (the India-first-stage frame), applied to India.
- Decision set: SX-argmax pairs, with the argmax fixed. τ comes from the training frame's 5-fold OOF, or from a τ
  curve where noted.
- dF is paired over all fold-0 target S1. LB-eq = 0.1498 × dF.
- "On top of gc70" maps France t to target t by P37's density rule: the same number of argmax pairs per S1 in
  [t, t_base). France densities: 0.138 at 0.90, 0.222 at 0.70.

Arms at the training τ, vs the chain analogue (dF US>India / India>US):

| ledger name | world | key metrics | verdict |
|---|---|---|---|
| loco-p_only-* | LOCO | p recalibration only. full −0.0183 / −0.0047; add −0.0156 / +0.0000. The training τ adds at 41% precision on India | drop |
| loco-full-* | LOCO | full −0.0689 / +0.0061; add −0.0687 / +0.0032; **addrm −0.0694 / +0.0068 ± 0.0002**. Train-frame OOF: +0.024 / +0.009 | drop |
| loco-p_group-* | LOCO | group columns + p ranks. full −0.0060 / −0.0008; add −0.0050 / +0.0008 | drop |
| loco-p_group_gt-* | LOCO | + similarity to the group's members. add −0.0124 / +0.0007 | drop |
| loco-p_group_nm-*, loco-full_nohn-* | LOCO | name columns, no hn. add −0.039 / −0.002 (both arms) | drop |
| curve-p_group | LOCO, τ grid | best both-ways: add τ 0.85 +0.0012 / +0.0002; addrm τ 0.85 +0.00125 / +0.00046 | drop |
| curve-full | LOCO, τ grid | US>India is negative at every τ: at τ 0.98 it still adds 23k pairs at 20% precision | drop |
| curve-p_group, gc mode (logs/curve_gc.log) | LOCO, τ grid | the brief's idea as a model: the best missing-source pair of a one-sided S1 if s ≥ τ. τ 0.80 **+0.00054 / +0.00039**; τ 0.75 +0.0001 / +0.0006; never both ≥ 0.0005. The rule gc70 on the same base: −0.0010 / +0.0021. `p_group_gt` gc: ≤ 0 in US>India at every τ < 0.94 | drop |

On top of the current France recipe analogue:

| ledger name | world | key metrics (dF US>India / India>US) | verdict |
|---|---|---|---|
| ontop-gc70 base steps | LOCO | rescue90 −0.0037 / +0.0010; gc70 −0.0010 (prec 0.62) / +0.0022 (prec 0.87). P37 measured gc70 at +0.0008 / +0.0020 on its M2-only base | — |
| ontop-gc70-p_group / p_group_mono | LOCO | add τ 0.88 +0.0003 / −0.0000 (mono +0.0004 / +0.0000); addrm τ 0.88 +0.0012 / −0.0002 | drop |
| ontop-gc70-p_group_gt | LOCO | add τ 0.96 **+0.00046 / +0.00038**; τ 0.98 +0.0009 / +0.0001 | drop (closest to the gate) |
| ontop-gc70-full | LOCO | add τ 0.80 −0.0601 / +0.0015; **addrm τ 0.80 −0.0605 / +0.0045 ± 0.0002** (h1 +0.0046) | drop (one direction) → spare sub39_ind80 |
| ontop-gc70+hn90 base step | LOCO | hn90 extras −0.0018 (prec 0.54) / +0.0018 (prec 0.91); P37 measured −0.0014 / +0.0021 | — |
| ontop-gc70+hn90-full | LOCO | add τ 0.80 −0.0586 / +0.0013; **addrm τ 0.80 −0.0590 / +0.0038 ± 0.0002** | drop (one direction) |
| ontop-gc70+hn90-p_group_gt | LOCO | add τ 0.96 +0.0012 / +0.0002 | drop |
| oracle-headroom | LOCO | perfect accept / reject over argmax pairs: add-only +0.024 / +0.021, both +0.038; completing one-sided S1 only +0.0067 / +0.0055 | — |
| lens-france-vs-loco | LOCO + France, label-free | table below | — |
| p39-sub39_ind80 | test | validator PASS; 15,749 France S1 changed; 0 India / US rows differ from sub37_gc70 | inconclusive (spare) |

**One-sided groups are mostly real one-sided groups.** The LOCO targets do show France's excess:

| world | S2 and S3 | S3 only | S2 only | empty |
|---|---|---|---|---|
| chain analogue, US>India | 0.747 | 0.096 | 0.099 | 0.058 |
| truth, India fold 0 | 0.793 | 0.079 | 0.072 | 0.056 |
| chain analogue, India>US | 0.756 | 0.102 | 0.086 | 0.057 |
| France #34v_r90 (argmax, p ≥ t) | 0.732 | 0.109 | 0.094 | 0.065 |

- But only 26–36% of predicted one-sided S1 truly have a member in the missing source.
- Where they do, the true member is an argmax pair below t 76–82% of the time. It is a non-argmax candidate 11–16% of
  the time, and not a candidate at all 6–9%.
- So the excess is a needle in a mostly-correct haystack. P37's gc70 already takes the part a p-threshold can find.

**Why the text columns fail US>India.** Precision of below-t argmax pairs:

| hn relation | US target (India-first-stage) | India target (US-first-stage) |
|---|---|---|
| hn missing on a side | 0.50 | 0.14 |
| hn equal | 0.74 | 0.15 |

The US-labelled model's bad India adds are 72% hn-missing and 62% add-one-word ("branch word") pairs.

**Label-free lens.** France looks US-like on exactly those slices:

| share | India target | US target | France |
|---|---|---|---|
| below-t pairs: hn missing & add-one-word | 0.231 | 0.008 | 0.010 |
| accepted pairs: hn missing | 0.395 | 0.102 | 0.086 |
| accepted pairs: hn equal | 0.503 | 0.803 | 0.904 |
| below-t pairs: hn equal | 0.203 | 0.202 | **0.420** |
| below-t pairs per accepted pair | 0.209 | 0.154 | 0.222 |

France is unlike both on below-t hn-equal pairs, which is where most of sub39's adds sit.

## What failed      (approaches tried and dropped, with the number that killed them)
- **p-rank columns do not carry calibration** (`p_only`).
  - The same point (logit(p) − logit(t), within-country quantile) has different precision in the two targets.
  - The training-frame τ over-adds on India (−0.016) and over-removes on US (−0.005).
  - Even the oracle τ gains only +0.00002.
- **Text columns** (hn relation, name kind, place overlap) carry most of the in-regime gain: train-frame OOF +0.024 /
  +0.009. That gain is regime-specific.
  - US labels applied to India: −0.069, even at τ 0.98.
  - Dropping the hn columns does not fix it: `full_nohn` −0.039 / −0.003.
- **Model-scored one-sided completion (`gc` mode).** +0.00054 / +0.00039 at τ 0.80, more balanced than the rule but
  4× smaller in the France-like direction. The LB-proven rule gc70 stays.
- **Group-structure columns alone are transferable but empty.** On top of gc70:
  - `p_group`: +0.0003 / 0.0000;
  - monotone variant: +0.0004 / 0.0000;
  - with group similarity (`p_group_gt`): +0.00046 / +0.00038 at τ 0.96.
  - The group-shape signal is the one P37's rule already uses.
- **Removals (addrm / full modes).** They win in India>US (removed chain pairs are 39–52% true) and lose or are flat in
  US>India. On France the model barely removes (1,530 pairs).
- **Competitor-p columns were not built.** In LOCO an argmax pair's second-best S1 is off-frame (fold 0 is ~10% of
  train S1), so the column would be observed on France and mostly missing in training.
- **Re-assignment was not attempted.** The stolen SX's other S1 cannot be scored in LOCO, and P35 measured
  re-assignment at +0.000016.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `frame.py loco <src> <tgt> | france | selftest` → `data/work/p39/{loco_<src>_<tgt>, france, s1_*}.parquet`,
  13–41 MB each, 6–11 s, ≤ 1.8 GB.
  - One row per (S1, SX-argmax pair): p (chain), lt = logit(p) − logit(t), q (within-country quantile), rank within
    S1 per source, accepted / live counts per source, best lt in the other source and in the same source, hn relation,
    name kind, name ratios, place / state overlap, web flag, same-address flag, similarity / hn match to the group's
    accepted members.
  - `selftest` passes.
- `fit.py <arm…>`: both LOCO directions at the training τ, modes full / add / addrm, with ledger rows. 1–2.5 min per
  arm, ≤ 0.8 GB.
  - Arms run: `p_only p_group p_group_gt p_group_nm p_group_mono full full_nohn`.
  - Defined but not run: `p_text full_noq full_nolt`.
  - `decide(d, s, tau, mode)` also has a `gc` mode (curve.py).
- `curve.py <arm…>`: dF vs τ, both directions. Caches the target scores in `data/work/p39/s_<arm>_<dir>.npy`.
- `ontop.py <arm…> [--hn90]`: the same, on top of the rescue90 + gc70 (+ hn90) analogue, with P37's density map.
  `gc(d, pred, t, hn_extra)` reimplements P37's `gc_add` oneall / allhn.
- `diag.py oracle | adds | lens`: headroom, error anatomy, label-free lens.
- `overlap.py <arm> <train> [tau]`: France adds vs hn90's extras.
- `build.py check [r90|gc70|hn90]`: the frame rebuild of each base's France.
  - r90: 0 rows differ.
  - gc70: 2 of 259,452 rows differ.
  - hn90: 353 rows differ (P37's hn guard differs slightly).
- `build.py make <arm> <add|addrm> <tau> <out|-> [pool|India|US] [gc70|hn90|r90]`:
  - the model's deltas are applied to the base file's own France lists;
  - India / US lines are copied byte for byte, and the candidates are a hard link;
  - about 20 s and 0.8 GB.
- Logs are aggregates only, in `probes/P39-fr-group-model/logs/`. `data/work/p39` holds ~110 MB.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- LightGBM (MIT), rapidfuzz (MIT). No external data. Nothing is keyed on a country name: the France model uses the same
  columns, computed from France's own records and p.
- **Gate reading.** P37 was staged on raw dF, and the master accepted India→US as France-like. So I used raw dF ≥ +0.0005
  in both directions. The brief's literal "+0.0005 LB-eq" (dF ≥ +0.0033) fails even more clearly.
- **The LOCO chain analogue is partial.** It has no P26 hn matcher, P29 tier or P28 same-address boost. On it,
  rescue90 and gc70 score worse in US>India than on P37's M2-only base (−0.0037 and −0.0010).
- **sub39_ind80 rests on the direction argument, not on the gate.**
  - The deployed model is India-labelled, and its only transfer test is India>US.
  - The mirrored configuration (US-labelled, India target) is catastrophic.
  - France matches the US target on the slices that caused that failure (lens). But it holds 2× more below-t hn-equal
    pairs than either target. On France the model adds at 2× the LOCO rate and removes at 1/6 of it.
- The group columns are computed at the chain t (France 0.9868; LOCO 0.915 / 0.92), not after rescue / gc.
- The frame's gc70 / hn90 rebuilds differ from P37's files on 2 / 353 France rows. The build applies deltas to the real
  files, so this only affects which deltas exist.

## Open questions   (for the master; each with your best guess)
- **Is a group-aware second stage worth more France effort?**
  - Best guess: no. What transfers (group shape) is already in gc70.
  - What carries signal (hn / name semantics) flips between countries. Only France labels, or a France-like proxy
    with labels, could calibrate it.
- **Upload sub39_ind80?**
  - Only after hn90, and only if hn90 gains. In that case rebuild it on the hn90 base (`… India hn90`): +6.6k / −3.7k
    France pairs, LOCO India>US +0.0038.
  - Best-guess LB: about +0.0003 if France behaves like India>US. The downside is unmeasured.
- **Why does gc70 lose in US>India on this base but win on P37's?**
  - Best guess: the bge blend + veto base already accepts more of the true one-sided completions.
  - The LB (+0.00034) agreed with the India>US reading, as P37 noted.
