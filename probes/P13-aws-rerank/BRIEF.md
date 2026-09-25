# P13-aws-rerank: fine-tune a big multilingual reranker on ONE SageMaker A10G job — get it right the first time
The user approved AWS for this and asked that it be done properly so no second attempt is needed. Budget: <= $15 of
credits, ONE training job (a second job only if the first fails for an infrastructure reason you could not rehearse).

Read first: probes/P4b-rerank/RESULT.md (current CE: e5-small, band recipe, blend, LOCO -0.0016), probes/P11-ce-big/
(RESULT.md draft + ce.py: band builder, token store, eval against the #5 yardstick — reuse it, do not rewrite it),
probes/MERGE-2.md, sm_launch.py + sm_train.py + test_sm.py (the launcher: packs top-level *.py + sm_requirements.txt,
--solution, --data-s3, --instance, --max-run, --no-spot, --status/--logs/--fetch, checkpoint dir). Own worktree; code
only under probes/P13-aws-rerank/ (a solution file may live there and be passed with --solution).

AWS: ALWAYS `--profile harsh` / `AWS_PROFILE=harsh` (default is a different account). Account 114872709824, ap-south-1,
bucket sagemaker-114872709824-ap-south-1 (private; use prefix amc26-p13/), role AMCSageMakerExecutionRole.
Quota: exactly 1 x ml.g5.xlarge (A10G 24 GB, 4 vCPU, 16 GB RAM) ON-DEMAND for training — spot is NOT available for g5,
so --no-spot. Do not file quota requests, create IAM users, endpoints, notebooks or anything else. Look up the exact
on-demand price (aws pricing API / awsdoc skill at ~/.claude/skills/aws-docs/awsdoc.py) and set MaxRuntime so the
worst case stays under $15. Read APIs before calling them (awsdoc.py <svc> <Op>).

Models (licences verified on HF by the master): BAAI/bge-reranker-v2-m3 (Apache-2.0, 568M, XLM-R-large cross-encoder)
first; Qwen/Qwen3-Reranker-0.6B (Apache-2.0, 596M, yes/no-logit reranker) second, only if time/budget allow.

Goal — two questions, in one job:
1. India/US: does a fine-tuned big reranker beat #5's e5-small CE on the SAME yardstick (full-corpus holdout, P5 report
   half, blend refit on tune half; #5 = 0.9885)? Train on the P4b/P11 train band (never folds 0/1).
2. France (the real prize): does it transfer to an unseen country? LOCO: fine-tune on US only -> score India band,
   and India only -> US band; compare with e5-small's LOCO (P4b: India>US_loco 0.9806 vs base 0.9822, US>India 0.9801
   vs 0.9802). Also zero-shot (no fine-tune) band AUC per country. A France CE is justified only if LOCO blend gain is
   > 0 in BOTH directions.
Outputs from the job: per-model/per-setting scores for the holdout band and the test band of all three countries
(India, US, France) as parquet (s1, sx, r), written INCREMENTALLY to the checkpoint dir / S3 so a timeout keeps
everything finished so far. Evaluation (blend, F0.5) happens back on the laptop with P11's eval code.

"No second attempt" discipline:
- Build the band text data locally (CPU), upload once to S3; log sizes.
- Rehearse the EXACT solution script end to end on the laptop first: tiny slice (~2k pairs per phase), a tiny model
  (e5-small) on CPU or the local GPU (flock /tmp/amc-gpu.lock; P11 uses it), same phases, same output files; check they
  load and have the right row counts. Rehearse the S3 read/write paths too (a small `aws s3 cp`). Pin package versions
  in the requirements you ship; make sure the DLC's torch/CUDA works with them (check the image's torch version).
- Size the job: measure/estimate pairs/s for a 568M model on A10G (bf16, max_len ~96, length buckets) and cap the
  training set per phase so the whole job fits MaxRuntime with 25% slack. Order phases by value: zero-shot AUCs ->
  LOCO (both directions, bge) -> pooled bge + test scoring -> Qwen3 (optional).
- Launch once, poll with --status/--logs in the foreground (bounded waits, never end your turn to wait), --fetch at the end.
- Then evaluate on the laptop and write RESULT.md: table, cost actually spent, recommendation (adopt for India/US? France?).
  If adopt: data/work/p13/prob_test_{c}_blend.parquet for the master (s1, sx, p) like P4b/P11 produce.
Memory cap 4 GB incl. the CLI locally, swap allowed; disk ~9 GB free. Never end your turn to wait on a background job.
Record every run in the ledger (ledger.py). Report the $ spent.
