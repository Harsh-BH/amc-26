# P38-qwen4b-fr: a bigger multilingual cross-encoder for France — Qwen3-Reranker-4B (Apache-2.0, ~4B ≤ 8B) on AWS
Why: the bge-reranker-v2-m3 CE (568M) gave France +1.2 pt, the biggest single France gain, because France is
out-of-domain text and the CE transfers better than GBDT features. A stronger multilingual reranker is the one untried
model lever. Qwen3-0.6B was planned in P13 but never ran (crash); no bigger model was tried for France.
Read probes/P13-aws-rerank (recipe, sm launch code, pair sets, LOCO protocol, the dict-comment crash), src/bge_train.py,
probes/P26* / P31-final-france (France band 0.02–0.98 blend with out-of-domain weights), sm_launch.py. Verify the licence
on the HF card (must be Apache-2.0/MIT) and param count before anything.
Plan (AWS, --profile harsh / AWS_PROFILE=harsh, ap-south-1; GPU quota 1 per instance type, so run the two LOCO jobs on
different types in parallel, e.g. ml.g6e.xlarge and ml.g5.2xlarge / g6; pull weights on AWS, never locally):
1. LoRA fine-tune Qwen3-Reranker-4B (bf16, grad checkpointing) on the P13 pair recipe, two LOCO directions
   (train US → score India holdout; train India → score US holdout), capped pair counts so each job ≤ 4 h.
2. Gate = same as P13/P26: on the LOCO holdout, blend into the France-style chain in place of bge and compare with bge
   at equal settings; ≥ +1 pt F0.5 in both directions (or ≥ +0.0005 LB-eq both ways) to proceed.
3. If it passes: pooled India+US training job, score the France band pairs (data/work/p31 France pair set), build
   output/sub38 = sub34v_rescue90 with France re-blended (India/US byte-identical), validator PASS, run.py notes.
Publish progress in RESULT.md early (LOCO numbers as soon as they land). Budget: don't worry about credits, but no
quota requests. Test every SageMaker script with --local or a tiny smoke job first — a crash after 3 h is the real cost.
Memory 1.5 GB locally. DISK NEARLY FULL (1.6 GB): keep everything on S3. Never end your turn to wait (poll jobs).
Deadline: LOCO verdict by 27 Sep 09:00 IST, sub38 by 16:00 IST (final upload cut-off 22:00).
