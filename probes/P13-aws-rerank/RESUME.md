Master here — I stopped your session to change the plan. **The user says: do the test scoring on AWS, don't worry
about credits.** So:
- Fix the DEFAULTS bug (you found it: the `#` comment swallowed "s3" and "qwen_test"), rehearse the scoring-only path
  locally on a tiny slice (CPU is fine), then launch ONE SageMaker job on ml.g5.xlarge (--no-spot) that loads
  ckpt/ce_bge_pool.pt from S3 and scores the TEST band for India, US and France (all test band pairs, not only the
  pruned ones — the full set lets us serve both #5-style and #6-style candidate sets). No training. Set MaxRuntime
  generously (e.g. 2 h). Write outputs incrementally to S3.
- Meanwhile on the laptop (CPU only — leave the local GPU to P9): evaluate the pooled bge holdout scores vs #7's
  shipped CE (P11 s2P) with the shipped rule, and fix the France recipe (mean-OOD weights).
- When the job finishes: build data/work/p13/prob_test_{India,US,France}_blend.parquet for the recommended recipe
  (on #6's pruned candidate set), check France's profile stays in P6's bands, write RESULT.md with the $ spent.
Your background downloads from before may still be running; reuse what finished. Never end your turn to wait.
