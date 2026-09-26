# P27-france-pseudo: learn France from France itself — FILTERED pseudo-label self-training (now allowed)
**Context (26 Sep 15:05).** Best LB = #9ab 0.984072 (India/US ≈ 0.990, France ≈ 0.952). LB #1 = 0.990556 → France
≈ 0.985. The organisers ruled pseudo-labelling on test is allowed (docs/PROBLEM.md "Organiser rulings", asked for France).
**Known failure to avoid:** P6's plain self-training (source labels + target pseudo-labels at p ≥ .98 / ≤ .02) LOST
3.5 pt in LOCO (US+psIndia→India 0.8701 vs 0.9054) — confident mistakes (1.6%) taught the model to trust its errors.
P15 found the dominant confident mistake: generator siblings (hn offset ∈ {3,4,5,7,9,11,13,21}); France accepts 8× more.
All training/eval on AWS (reuse P22/P25/P16's SageMaker train + eval path: /home/harsh1/github-repos/amc-26-wt/
p22-ensemble/…, p25-lgbm-bag/…; train X in S3 amc26-p16/); laptop RAM/disk are tight. AWS `--profile harsh` only.
Read first: probes/P6-france/RESULT.md (loco-dense-selftrain row + why it failed), P12 (synthetic failed: rates must
match the target), P13 (LOCO protocol, label-free France threshold, bge France recipe), P14, P15, P17, P19, SUBMISSIONS.md.
Own worktree; code under probes/P27-france-pseudo/.

**Phase A — analyse (≤ 1 h):** on the LOCO proxy (US-trained model scoring India, and reverse), characterise the
confident pseudo-labels: precision of positives/negatives by p bin, by hn class (sibling offsets, digit drop, far),
by chain size, by agreement between m2 and the bge CE, by "argmax + margin". Find a filter that pushes pseudo-positive
precision to ≥ 99.8% and negatives ≥ 99.9% while keeping volume.
**Phase B — LOCO go/no-go:** arms: (0) P6's plain self-training (reproduce the −3.5 pt), (1) filtered pseudo-labels
(the Phase A filter: agreement m2∧bge, no sibling offsets, margin to 2nd candidate, chain-size guard), (2) (1) with
pseudo-labels down-weighted (e.g. 0.3) and/or only pseudo-NEGATIVE look-alikes (hard negatives are what the target
lacks), (3) a small France-only recalibration GBDT on pseudo-labels on top of m2 p (few robust features: p, hn class,
margins). Measure on the unseen country's full-corpus fold 0 with the label-free threshold procedure.
Gate: ≥ +1 pt F0.5 on the unseen country in BOTH directions and no in-country loss.
**Phase C — France:** if an arm passes, apply it to France test (IDF-fixed candidates, #6 pruner, bge France blend as in
#9ab), pick France t at D4-equivalent empty rate AND report P6's rule; build the France file +
`probes/P14-france-shift/combine.py output/sub8c <file> <t> <out>` + validator.
Deliver RESULT.md (tables, $ spent, commands). If nothing passes, say so with the numbers. Memory 2 GB locally.
Never end your turn to wait. Deadline 26 Sep 23:00.
