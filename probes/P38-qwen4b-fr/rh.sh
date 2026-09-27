#!/usr/bin/env bash
# Local rehearsal of the exact SageMaker bundle (sm_train.py + solution.py + hyperparameters.json, JSON-encoded the way
# sm_launch.build_request sends them) on CPU with a tiny random Qwen3 and P13's sm_tiny channel.
#   probes/P38-qwen4b-fr/rh.sh <name> '<hp json>'
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
D=$ROOT/data/work/p38/rh/$1
rm -rf "$D"; mkdir -p "$D/code" "$D/config"
cp "$ROOT/sm_train.py" "$D/code/"; cp "$ROOT/probes/P38-qwen4b-fr/solution_p38.py" "$D/code/solution.py"
"$ROOT/.venv/bin/python" -c "import json,sys; print(json.dumps({k: json.dumps(v) for k, v in json.loads(sys.argv[1]).items()}))" "$2" > "$D/config/hyperparameters.json"
cd "$D/code"
CUDA_VISIBLE_DEVICES="" AWS_PROFILE=harsh AWS_DEFAULT_REGION=ap-south-1 SM_CHANNEL_TRAIN=$ROOT/data/work/p13/sm_tiny \
SM_MODEL_DIR=$D/model SM_OUTPUT_DATA_DIR=$D/output SM_INPUT_CONFIG_DIR=$D/config \
  "$ROOT/.venv/bin/python" sm_train.py 2>&1 | tee "$D/run.log"
