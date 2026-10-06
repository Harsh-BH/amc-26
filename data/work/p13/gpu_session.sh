#!/usr/bin/env bash
# P13 local GPU session (one flock): CUDA rehearsal of the job (small cached models, forced-OOM micro-batch probe),
# then #5's e5-small CEs (ce5 text check + P4b LOCO CEs) on m2_big's holdout band.
cd /home/harsh1/github-repos/amc-26-wt/p13-aws-rerank
flock /tmp/amc-gpu.lock bash -c '
rm -rf /tmp/p13cuda && mkdir -p /tmp/p13cuda
HF_HUB_OFFLINE=1 HF_HOME=/home/harsh1/.cache/huggingface .venv/bin/python -c "
import sys; sys.path.insert(0, \"probes/P13-aws-rerank\")
import solution_p13 as S
hp = {\"bge\": \"intfloat/multilingual-e5-small\", \"qwen\": \"trl-internal-testing/tiny-Qwen3ForCausalLM\",
      \"micro\": 4096, \"qmicro\": 4096, \"cap_s\": {\"bge_locoUS\": 60, \"bge_locoIndia\": 60, \"bge_pool\": 60, \"qwen_pool\": 60},
      \"deadline_s\": 3600, \"qwen_test\": \"always\", \"min_train_s\": 5, \"s3\": \"\", \"checkpoint_dir\": \"/tmp/p13cuda/ckpt\"}
S.main(\"data/work/p13/sm_tiny\", None, \"/tmp/p13cuda/model\", \"/tmp/p13cuda/out\", hp)
" > data/work/p13/cuda_rehearsal.log 2>&1
echo "REHEARSAL_EXIT $?" >> data/work/p13/cuda_rehearsal.log
HF_HUB_OFFLINE=1 .venv/bin/python probes/P13-aws-rerank/p13.py e5 > data/work/p13/e5.log 2>&1
echo "E5_EXIT $?" >> data/work/p13/e5.log
'
