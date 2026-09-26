# P32-bge-dapt: domain-adaptive pre-training of the bge cross-encoder on UNLABELLED French text (AWS) + a portable kit
Idea (user-approved): continue bge-reranker-v2-m3's masked-language-model (MLM) pre-training on the unlabelled France
records (names + addresses of test S1/S2/S3 France — allowed by the organiser ruling on unlabelled test data), THEN
fine-tune it as our cross-encoder on the US/India labelled band exactly as P13 did, then score France. Literature:
Gururangan 2020 (DAPT/TAPT), DADER. Read first: probes/P13-aws-rerank/ (RESULT + solution_p13.py + p13.py: training
recipe, band data in S3 amc26-p13/data/v1/, LOCO protocol with out-of-domain blend weights, France blend
[0.9245, 0.2595, 0.0515]), P19/P26 (France files, bge top-up), P14 combine.py. Own worktree; code under probes/P32-bge-dapt/.
AWS `--profile harsh`; 1× ml.g5.xlarge (A10G) on-demand quota (P30 may hold it until ~18:05 — queue after it or check
quotas for ml.g5.2xlarge/g4dn); credits fine; MaxRuntime per job ≤ 5 h. Never end your turn to wait.

**Honest LOCO gate (do this first; it is the go/no-go):** DAPT on the TARGET country's unlabelled text, fine-tune on the
SOURCE country's labels, score the target: (a) MLM on India text → fine-tune on US band → score India holdout band;
(b) MLM on US text → fine-tune on India band → score US holdout band. Compare with P13's LOCO bge (no DAPT) using P13's
evaluator (out-of-domain weights, label-free threshold). Gate: ≥ +0.0005 F0.5 on the unseen country in BOTH directions.
To save time run (a),(b) and the France chain (MLM on France text → fine-tune on US+India band → score France band on
#9ab's / P29's France candidates) in parallel if quota allows; only USE France if the gate passes.
MLM details: start from BAAI/bge-reranker-v2-m3's encoder (XLM-R large); ~1–2 epochs over the unique France strings
(~1.7M rows, "name | address"), max_len 64, 15% masking, lr ~5e-5, bf16; then P13's exact fine-tune recipe (2 epochs,
778k pairs). Keep the tokenizer unchanged.
**Portable kit for an external compute service (user asked):** build `probes/P32-bge-dapt/kit/` containing: README.md
(what it does, GPU needs ≥ 24 GB, expected hours, exact commands), requirements.txt (pinned), run.sh (one command:
dapt → finetune → score), the Python script(s) (no AWS dependency; local paths), and a data folder with ONLY what is
needed as parquet: France unlabelled texts, the US/India fine-tune band (text pairs + labels), and the holdout/test band
pairs to score. Output contract: r files (s1, sx, r) per country, same format as data/work/p13/r/. Zip it
(kit + data) and upload to gdrive:amc26-offload/dapt_kit/ (rclone copy; NEVER `rclone link` — keep it private).
Report the zip size and the exact Drive path. Mark in README that the data is Amazon ML Challenge data — share only
within team Yeti.
Deliver RESULT.md: LOCO table, France r files (if gate passes) + France file on the #8c base via combine.py at #9ab's
empty rate + validator, $ spent, kit location. Deadline: LOCO verdict by ~23:30; France file by 27 Sep 09:00.
