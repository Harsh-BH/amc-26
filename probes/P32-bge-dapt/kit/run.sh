#!/usr/bin/env bash
# P32 DAPT kit - Amazon ML Challenge 2026 data, for team Yeti (do not share outside the team).
#   bash run.sh [all|loco|france] [--gpus N]
# all (default) = the LOCO gate (both directions + no-DAPT controls) and the France chain. Each chain is its own process
# pinned to one GPU (CUDA_VISIBLE_DEVICES=i); with G GPUs the chains are dealt round-robin (longest first) and each GPU
# runs its share in order. GPUs: CUDA_VISIBLE_DEVICES if set, else --gpus N, else `nvidia-smi -L`, else 1.
# P32_EXTRA='"key": value, ...' overrides any job param (e.g. "dapt_epochs": 1) for every chain.
set -euo pipefail
cd "$(dirname "$0")"
WHAT=all; NG=""
while [ $# -gt 0 ]; do
  case "$1" in
    all|loco|france) WHAT=$1 ;;
    --gpus) NG=$2; shift ;;
    *) echo "usage: bash run.sh [all|loco|france] [--gpus N]"; exit 2 ;;
  esac
  shift
done

declare -A CH                                  # chain tag -> its params (P13's fine-tune recipe, see README)
CH[daptFrance_pool]='{"dapt": "France", "ft": ["India", "US"], "tag": "daptFrance_pool", "tests": ["France"], "save_ft": true, "epochs": {"daptFrance_pool": 2}}'
CH[daptIndia_locoUS]='{"dapt": "India", "ft": ["US"], "tag": "daptIndia_locoUS", "save_dapt": false}'
CH[daptUS_locoIndia]='{"dapt": "US", "ft": ["India"], "tag": "daptUS_locoIndia", "save_dapt": false}'
CH[ctl_locoUS]='{"dapt": "", "ft": ["US"], "tag": "ctl_locoUS"}'
CH[ctl_locoIndia]='{"dapt": "", "ft": ["India"], "tag": "ctl_locoIndia"}'
case "$WHAT" in                                # longest first, so round-robin balances the GPUs
  all) ORDER=(daptFrance_pool daptIndia_locoUS daptUS_locoIndia ctl_locoUS ctl_locoIndia) ;;
  loco) ORDER=(daptIndia_locoUS daptUS_locoIndia ctl_locoUS ctl_locoIndia) ;;
  france) ORDER=(daptFrance_pool) ;;
esac

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then IFS=, read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
  [ -z "$NG" ] && NG=$( (nvidia-smi -L 2>/dev/null || true) | grep -c '^GPU' || true)
  [ "${NG:-0}" -ge 1 ] 2>/dev/null || NG=1
  GPUS=($(seq 0 $((NG - 1))))
fi
mkdir -p out/r logs
(nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null || echo "no nvidia-smi") | tee out/gpu.txt
echo "chains: ${ORDER[*]} | GPUs: ${GPUS[*]}"

export HF_HOME="$PWD/out/hf" TOKENIZERS_PARALLELISM=true
[ -n "${P32_SKIP_DOWNLOAD:-}" ] || python -c "from huggingface_hub import snapshot_download as s; print('model at', s('BAAI/bge-reranker-v2-m3'))"   # once, not per process

run_gpu() {                                    # run_gpu GPU TAG... : this GPU's chains, one after another
  local g=$1; shift
  for tag in "$@"; do
    echo "[gpu $g] start $tag  $(date +%H:%M:%S)"
    if CUDA_VISIBLE_DEVICES=$g python solution_p32.py data "out/$tag" \
        "{\"chains\": [${CH[$tag]}], \"deadline_s\": 108000, \"dapt_reserve_s\": 0${P32_EXTRA:+, $P32_EXTRA}}" \
        > "logs/$tag.log" 2>&1; then
      echo "[gpu $g] done  $tag  $(date +%H:%M:%S)"
    else
      echo "[gpu $g] FAILED $tag (see logs/$tag.log)"; return 1
    fi
  done
}

pids=()
for i in "${!GPUS[@]}"; do
  mine=()
  for k in "${!ORDER[@]}"; do [ $((k % ${#GPUS[@]})) -eq "$i" ] && mine+=("${ORDER[$k]}"); done
  [ ${#mine[@]} -gt 0 ] && { run_gpu "${GPUS[$i]}" "${mine[@]}" & pids+=($!); }
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done

cp out/*/r/*.parquet out/r/ 2>/dev/null || true
for tag in "${ORDER[@]}"; do [ -f "out/$tag/log.jsonl" ] && cp "out/$tag/log.jsonl" "logs/$tag.jsonl"; done
python - "${ORDER[@]}" <<'PY'
import pathlib, sys
import pandas as pd
bad = 0
for tag in sys.argv[1:]:
    want = [f"r_{tag}_holdout.parquet"] + ([f"r_{tag}_test_France.parquet"] if tag == "daptFrance_pool" else [])
    for name in want:
        f = pathlib.Path("out/r") / name
        if not f.exists():
            print(f"MISSING {name}"); bad = 1; continue
        d = pd.read_parquet(f)
        ok = list(d.columns) == ["s1", "sx", "r"] and d.r.between(0, 1).all() and not d.r.isna().any()
        print(f"{name}: {len(d):,} rows {'ok' if ok else 'BAD'}"); bad |= not ok
sys.exit(bad)
PY
[ $fail -eq 0 ] || { echo "a chain failed - see logs/"; exit 1; }
echo "done. Send back: out/r/ (all .parquet) and logs/*.jsonl. The .pt checkpoints in out/<chain>/ only if asked."
