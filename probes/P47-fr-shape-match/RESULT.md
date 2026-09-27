# P47-fr-shape-match — RESULT (final, 27 Sep 03:25 IST)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **Shape matching fails LOCO. Do not adopt it.**
  - France's gap to the seen countries' predicted shape is real: 3.256 vs 3.386 SX/S1, i.e. 33.6k pairs.
  - The LOCO targets carry the same gap, and filling it from the free argmax pool adds pairs at **45–67% precision**.
  - Stopping at the shape target costs **−0.0098 / −0.0010** France-F0.5 (chain analogue, US→India / India→US) and −0.022 / −0.0002 (m2). The truth-shape target is worse.
  - Every mode (mean, per-source CDF, one-sided-first) loses US→India. The F0.5 optimum sits at 0–30% of the gap, not at 100%.
- **Why:** the missing true pairs are not at the top of the pool. In LOCO the pool holds 0.14–0.21 true pairs per S1 among 0.65–1.16 pool pairs per S1. Its top 5% by p is only 67–75% true. France's pool top looks no better: 69% "hn missing" (the address-less twin floor), 14% hx − h1 = +1/+2 (sibling decoys).
- **One salvageable slice** (not shape matching): house-number-change pairs. It is LOCO-positive on all 4 readings, but France holds 3–4× fewer such pairs than LOCO. Staged `sub47_cls80` (+1,888 France pairs), validator PASS: expected **≈ +0.00003 LB**, stack-only. It is not worth an upload slot on its own.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
1. **Adopt no shape-matching stage**, in any mode and at any stop.
   - Do not chase France's 0.13 SX/S1 deficit with generic free pairs.
   - France's true mean is uncertain. P40's label-free estimate is 5.53 SX/S1 × 0.60 owned = **3.32**, not 3.39.
   - In every LOCO frame the deficit is recall that the pool does not rank.
2. **Optional, stack-only: the hn-change class stage** at France p ≥ 0.80.
   - Rule: add every free, SX-argmax, non-SIB3 France pair whose house number differs from the S1's by −1/−2, or by any amount other than a SIB3 offset or +1/+2. Best p first, caps 5/6, existing members count, all group states.
   - Folders:
     - `output/sub47_cls80` = #46_on43 + 1,888 pairs to 1,806 S1 (43 empties filled);
     - `output/sub47_cls80_onrecipe` = the same pairs on `sub46on43_recipe`, the base P43's zip rebuilds.
     - Both: India/US byte-identical, validator `--check-ids` PASS.
   - It is disjoint from #41 / #40x / #46, whose pairs are hn-equal or address-less.
   - run.py (France only, after P46's swap tier, on the post-rules p):
     ```python
     import p47 as P47                               # probes/P47-fr-shape-match/p47.py: class_add + hn_rel (numpy, pandas)
     hr = P47.hn_rel(h1, hx)                         # P37's fixed '(n)' house numbers: rf S1 hn, P29 w_fix SX hn
     pool = free & ~acc & ~owned[qx]                 # free = P42 base()'s mask: SX argmax, non-SIB3, not capped out
     acc |= P47.class_add(q1, is2, p, acc, pool, n_s1, hr, 0.80)      # keep=(4, 5), caps (5, 6)
     ```
   - Upload it only if a slot would otherwise sit idle, or fold it into the final France recipe when porting.
3. **The +1/+2 remover** (`sub47_cls80rm`): drops France's 675 accepted hx − h1 ∈ {1, 2} pairs.
   - It is a one-direction bet: US target +0.0024, India target −0.0022 LOCO.
   - France's argmax +1/2 : −1/2 ratio is **26.7**, against US 11.6 and India 3.7, which points US-like. Scaled, it is worth only ≈ ±0.00004 LB.
   - Built and validated, then deleted for disk. Rebuild command in Interfaces.

## Evidence         (table: ledger name | world | key metrics | verdict)
**Protocol.**
- Frames: P42's LOCO frames (target fold 0, SX-argmax pairs, source-only model, veto τ 0.1).
- Base: the **#43_r90 recipe analogue**, i.e. P42 `decide` with t 0.99, rescue 0.90, gc 0.70, hn extras 0.80, caps 5/6, at density-mapped thresholds.
- Shape target: the SOURCE country's test-prediction shape from `sub46_on43`, which is label-free for the target. Train truth's shape is reported alongside.
- The shape stage walks the free pool (argmax, non-SIB3, not accepted, not capped out) in descending p.
- dF: paired per-S1 France-F0.5 vs the base over all fold-0 target S1. SE is of the paired difference.

**France vs target** (`shape.py`, `targets.json`):

| world | SX/S1 | both | S3-only | S2-only | empty | S2 = 0 / 1 / 2 | S3 = 0 / 1 / 2 |
|---|---|---|---|---|---|---|---|
| France `sub46_on43` | 3.256 | 0.771 | 0.090 | 0.079 | 0.0599 | .150 / .376 / .284 | .139 / .345 / .295 |
| India + US test predictions | 3.386 | 0.789 | 0.080 | 0.073 | 0.0578 | .138 / .361 / .294 | .130 / .329 / .300 |
| train truth (India = US to 0.005) | 3.461 | 0.805 | 0.075 | 0.065 | 0.0558 | .130 / .358 / .296 | .121 / .325 / .303 |

France's CDF sits above the target's in every bucket of both sources: its excess is all in small groups ((1,1) +1.0 pp, (0,1) +0.57 pp, (1,0) +0.50 pp). **No large-group excess, so there is no shape remover.**

| ledger name | world | key metrics | verdict |
|---|---|---|---|
| loco-shape-mean-f{0.25,0.5,1.0} | LOCO | chain US→India / India→US: f0.25 **+0.00038 / +0.00021** (prec 0.78 / 0.78); f0.5 −0.00101 / +0.00007; **f1.0 −0.00977 / −0.00103** (prec 0.46 / 0.67). m2: f0.25 −0.0038 / −0.00003; f1.0 −0.0219 / −0.0002 (prec 0.43 / 0.68). Truth target f1.0: chain −0.0227 / −0.0066 | drop |
| loco-shape-cdf-f* | LOCO | per-source CDF matching, same volumes: f0.25 −0.00069 / +0.00003; f1.0 −0.0116 / −0.0015 (chain); m2 −0.0224 / −0.0005 | drop |
| loco-shape-onesided-f* | LOCO | one-sided groups get their best missing-source pair until the one-sided share matches: f1.0 chain −0.0028 (prec 0.35) / −0.00004 (0.69); m2 −0.0020 / +0.00013 | drop |
| (empty mode) | LOCO + France | LOCO bases are already below the target's empty share (0.050 / 0.054 vs 0.058), so 0 adds. France (0.0599) would fill 542 S1 at p 0.75–0.90, which is #37_gc70_r70's idea (staged by P37, a coin flip) | — |
| (cond mode) | LOCO | CDF matching within the other source's bucket: over-fills (to p = 0), −0.003 to −0.07 | drop |
| loco-sweep-mean-{all,no+12,other_only} | LOCO | optimum frac of the gap: chain 0.2 (+0.00044) / 0.25–0.3 (+0.00021); **m2 0 in both directions**. No +1/2: India→US rises to +0.0013 at f0.75, US→India unchanged. hn-change pairs only: positive in all 4 frames up to f0.4–0.5 | drop / inconclusive |
| loco-class-hn45-*-t{0.9,0.8,0.7,0.6} | LOCO, France bar density-mapped (P42 rule) | at **0.80, all groups**: chain **+0.00092 ± 0.00009 / +0.00057 ± 0.00007** (prec 0.83 / 0.80); m2 **+0.00212 ± 0.00010 / +0.00024 ± 0.00006** (0.92 / 0.76); both halves positive. Every bar from 0.6 to 0.97 and every group restriction is positive in all 4 readings. LOCO adds 0.025–0.03 pairs per S1 | keep (stack-only) |
| (P40 loco_allops, cross-check) | P27 M2 frames, gc70 base | A_HN_OTHER@0.7: prec **0.913 / 0.665**, LB-scaled +0.00007 / **−0.00001**. So the US-target reading is near break-even (~0.73) on P40's analogue | caveat |
| loco-remove-accepted-hn+12 | LOCO chain | accepted +1/2 precision 0.955 (India target) / 0.487 (US target); removal −0.0022 / **+0.0024**. Argmax +1/2 : −1/2 = 3.7 (India) / 11.6 (US) / **26.7 (France)**. True +1/2 ≈ true −1/2 in both targets (1,867 vs 1,633; 2,126 vs 2,168): the generator's ±1/2 noise op is symmetric, and the +1/2 excess is decoys | inconclusive |
| build-sub47_cls80 / _onrecipe / _cls80rm | test | +1,888 pairs (1,809 hn-other, 79 hn −1/2), 1,806 S1, p 0.80–0.99 (median 0.965); 3.256 → 3.264 SX/S1; empty 5.99 → 5.97%; 0 over cap. rm: −675 pairs from 619 S1. `check.py`: 0 non-France lines differ, 0 SX twice, every added pair is in `candidate_pairs.tsv` | staged |
| validator-check-ids-sub47 | test | official validator `--check-ids` **PASS** on all three (SageMaker p47-val-cls80 / -cls80rm / -cls80r, 45 s each; md5 57f33a02 / 7779560d / eb80af56; candidates b3ccba0d = #46_on43's) | keep |

**Where LOCO's missing true pairs are** (chain; per S1):

| target | truth | base TP / FP | missing | in the free pool | pool size | not in frame |
|---|---|---|---|---|---|---|
| India (US model) | 3.475 | 3.185 / 0.086 | 0.290 | 0.211 | 1.156 | 0.075 |
| US (India model) | 3.460 | 3.249 / 0.065 | 0.211 | 0.142 | 0.652 | 0.068 |

**Pool top precision by hn class** (top 0.03 pairs per S1; chain US→India / India→US / m2 US→India / m2 India→US):

| class | precision |
|---|---|
| hn missing | 0.68 / 0.91 / 0.34 / 0.80 |
| hn other | 0.93 / 0.83 / 0.96 / 0.78 |
| hn −1/2 | 0.94 / 0.98 / — / 0.95 |
| hn +1/2 | 0.76 / **0.12** / 0.69 / **0.09** |

- France's pool top (p ≥ 0.917, 0.03 per S1) is 69% hn missing / 14% +1/2 / 17% other.
- Filling France's gap takes the pool down to **p 0.46**. That fill would add 11.4k hn-missing, 12.4k hn-equal (below P42's 0.80 bar), 6.6k +1/2 and 3.0k hn-other pairs.
- Splitting hn missing by P39's `city` flag reverses between directions (same city: 0.26 vs 0.97; other city: 0.87 vs 0.58). This is P40's address-less twin floor again.

**Expected LB of `sub47_cls80`:** LOCO dF per (added pair per S1) × France's 0.0073 added pairs per S1 × 0.1497:
- chain: +0.000032 / +0.000025;
- m2: +0.000079 / +0.000012;
- P40's reading: ≈ −0.00001.

## What failed      (approaches tried and dropped, with the number that killed them)
- **Shape matching, every variant.**
  - Mean-mode stop at 50 / 75 / 100% of the gap:
    - chain: −0.0010 / −0.0046 / −0.0098 (US→India) and +0.00007 / −0.00034 / −0.00103 (India→US);
    - m2: −0.0092 / −0.0154 / −0.0219.
  - Per-source CDF, one-sided-first and conditional-CDF variants are worse (table).
  - Truth-shape targets are worse still, as they sit further away.
- **Stopping at the shape target is not near the F0.5 optimum.** The optimum is at 0–30% of the gap. The m2 analogue wants no fill at all.
- **Qwen-ordered fill: not built.** P38's France scores had not landed by 03:25, and it cannot rescue the idea anyway:
  - P38's gate puts Qwen over bge at +0.0014 / +0.0011 France-F for the whole France procedure;
  - the target fill loses 0.0098 US→India.
  - `build … POST` takes a Qwen post file if wanted (see Interfaces).
- **A generic-pool remover.** France has no shape that India/US lack. Its S2 = 5 and S3 ≥ 5 shares are below the target's.
- **The hn-missing subclasses** (city, kind, is_web): no cell holds in both directions.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `shape.py <dirs>`: the per-country shape of a submission, plus the seen-country pool (train S1 countries). 5 s, 0.9 GB.
- `shape.py targets`: writes `targets.json` (pred_* from `sub46_on43`, truth_* from train truth, with 6 × 7 joint tables).
- `p47.py` (no arguments) runs the self-check of `shape_add`, `class_add` and `hn_rel`.
- `p47.py pool [base]`: the France free-pool profile by p band. 4 s, 0.9 GB.
- `p47.py loco [out]`, `sweep`, `classloco`: the LOCO runs. 20–30 s each, 0.2 GB. They write `loco.json` / `loco2.json` / `sweep.json` / `classloco.json`.
- `p47.py build BASE OUT SPEC [POST]`: 7 s, 0.9 GB.
  - SPEC is `cls=0.8[,groups=two|one+two][,rm12=1]` or `shape=mean|cdf|onesided|empty,frac=F[,tgt=truth_seen][,excl=3]`.
  - POST is a P38-style France post-rules parquet (s1, sx, p in the P31 chain's row order), for example the Qwen re-blend. It replaces p10 and is row-order asserted. Checked: POST = `fr31v0.1_t9868` reproduces the default exactly.
  - India/US lines byte-identical; `candidate_pairs.tsv` / `code.zip` are hard links.
- `check.py BASE NEW`: the independent streamed check (0 non-France diffs, SX unique, caps, adds ⊆ candidates). 5 s, 0.9 GB.
- `shape_add(s1, is2, p, pred, pool, n, tgt, mode, frac, p_min, cap, order)` and `class_add(s1, is2, p, pred, pool, n, hr, bar, cap, keep=(4, 5), groups)`: each returns an added-pair mask.
- `fr_state(base, post)` returns France arrays, the base mask, the free pool and hn_rel. `loco_frame(analogue, dr)` returns a P42 frame plus the #43_r90 analogue base.
- Rebuild commands:
  - the remover: `.venv/bin/python probes/P47-fr-shape-match/p47.py build sub46_on43 sub47_cls80rm cls=0.8,rm12=1`
  - the half-gap diagnostic (+16,789 pairs, reaching P40's 3.32 estimate; LOCO −0.0011 / +0.0011 chain, −0.0094 m2): `… build sub46_on43 sub47_m50 shape=mean,frac=0.5,excl=3`

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Transfer of the class precision.** France's generator is light on house-number noise:
  - accepted hn-other pairs are 0.031 per S1, against 0.23–0.29 in the LOCO targets;
  - pool hn-other at the bar is 0.22× the accepted count, against LOCO's 0.11×.
  - France's decoy share in this class may therefore be higher than LOCO's. P40's analogue already puts the US-target reading at precision 0.665, below the ~0.73 break-even.
- **LOCO frames are the train world** (4.7 SX/S1); test has 5.8. Test-only distractors, possibly orphans of dropped S1, cannot be emulated. This cuts against every France add, mine included.
- **The shape target is itself uncertain.** France has 5.53 SX/S1 against 5.79 for the seen countries, and 6–8× their sibling-decoy rate (P15: 22 per 1k accepted vs 2.6 / 3.9). Its true mean may be ~3.3, not 3.39–3.46.
- **Rules and licences:** label-free and model-free. The class stage uses only house numbers, p, and the unlabelled test file (ruling 2). Nothing is keyed on a country name: seen countries = train S1 labels.
- **Worktree:** worked in the main checkout's `probes/P47-fr-shape-match/`. Disk was 2.1 → 0.94 GB free (memguard paused one build on DISK LOW), so I made no worktree.
  - `output/sub47_cls80` + `sub47_cls80_onrecipe` take ~196 MB; `sub47_cls80rm` was deleted.
  - Nothing is committed, and nothing under `src/` was touched.
- **Cost:** 3 SageMaker validator jobs × 45 s spot on ml.m5.xlarge ≈ $0.01. No grid job was needed: the LOCO runs take 20 s locally.

## Open questions   (for the master; each with your best guess)
- **Is France under-confident at all, as the brief assumes?**
  - Best guess: only in specific classes, which P40 / P41 / P46 and this class stage have now harvested. Not generically.
  - The one LB test would be the half-gap diagnostic `sub47_m50` (rebuild command above). LOCO, scaled to its volume (0.065 pairs per S1), predicts −0.0008 (m2 US→India) / −0.0002 (chain US→India) to +0.0003 (chain India→US) LB. Spend a slot on it only if the question still decides something.
- **Upload `sub47_cls80`?** Best guess: no, not as a standalone slot (≈ +0.00003). Fold `class_add` into the final France recipe if P43 re-freezes run.py anyway.
- **The +1/+2 remover?** Best guess: skip. France's 26.7 ratio says its +1/2 pairs are US-like decoys, but 675 pairs are worth ≈ 0.00004 LB.
- **Qwen (P38):** the class stage can be rebuilt on P38's France post file with `build <p38 base> sub47_cls80_q cls=0.8 <post.parquet>`. The volume and expected gain should be about the same.
