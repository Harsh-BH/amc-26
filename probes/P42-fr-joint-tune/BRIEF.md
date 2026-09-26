# P42-fr-joint-tune: re-tune ALL France levers jointly on top of the current recipe, on AWS CPU
Why: France levers were tuned one at a time on older bases: t 0.9868 (nearest-empty-rate rule from #9ab), veto τ 0.1,
cap 5/6, rescue p ≥ 0.90, gc p ≥ 0.70, hn90 extras p ≥ 0.90 (staged). The LB has tracked the India→US LOCO direction
in both measured uploads (P37: rescue90 LB +0.0021 France-F0.5 vs LOCO +0.0009 / −0.0017; gc70 LB +0.0023 vs +0.0020 /
+0.0008), and France's hn behaviour is US-like (92% hn-equal accepted pairs; US 78%, India 50%). Nobody has tuned the
levers jointly. Base = output/sub37_gc70 (LB 0.986304).
Read probes/P37-fr-complete (p37.py LOCO machinery, density-matched thresholds, gc/hn90 rules, RESULT.md),
probes/P39-fr-group-model/RESULT.md in ../amc-26-wt/p39-fr-group-model (chain analogue, build.py), probes/P31-final-france,
src/france.py. Own worktree; code in probes/P42-fr-joint-tune/.
Task:
1. Reproduce gc70 in the LOCO analogue (P37/P39 already do: reuse). Define the joint grid: France t (via density
   matching), veto τ, rescue bar, gc bar, hn90 bar (incl. off), cap. ~500–2000 cells.
2. Run the grid as SageMaker CPU training jobs in parallel (sm_launch.py --solution <script> --data-s3 … --instance
   ml.m5.4xlarge / ml.m6i.4xlarge (4 each, $0.97/h, 64 GB) / ml.r6i.4xlarge (4, ≈$1.25/h, 128 GB); bucket
   sagemaker-114872709824-ap-south-1, prefix amc26-p42/, profile harsh). Smoke-run first; hyperparameters from a config
   file, never argv. Budget $25.
3. Report the frontier: India→US gain vs US→India gain for every cell. Pick (a) the best cell with both directions ≥ 0
   vs gc70, and (b) the best India→US cell with US→India ≥ −0.0005. Also report where the current recipe sits.
4. Build output/sub42_both and output/sub42_ius on sub37_gc70 (India/US byte-identical), validator PASS, rank them in the
   SUBMISSIONS.md queue against hn90 / sub39_ind80, with expected LB (France-F0.5 × 0.1498).
RESULT.md with tables. Local memory 1.5 GB (the heavy work runs on AWS). DISK ~3 GB free: small files only. Never end
your turn to wait. Deadline: first staged file by 27 Sep 06:00 IST, final by 08:00.
