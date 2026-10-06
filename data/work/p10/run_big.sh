#!/usr/bin/env bash
set -e
cd /home/harsh1/github-repos/amc-26-wt/p10-candidates
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, "probes/P10-candidates")
from prune import train, PARAMS, OUT
train(rounds=600, params=PARAMS | {"num_leaves": 63}, out=OUT / "pruner_big.txt")
PY
.venv/bin/python probes/P10-candidates/prep.py holdq pruner_big.txt _big
.venv/bin/python probes/P10-candidates/sweep.py pruner_big.txt _big "prune-tau0.0"
echo BIG_DONE
