"""Submit one P60 SageMaker job: P38's solution_p38.py with P13's bundle (p13.launch), quota lookup skipped
(ListServiceQuotas is rate-limited when several probes launch at once; quota is known: 1 per GPU type).
    launch60.py <name> <instance> <channel s3 uri> '<hp json>' [max_run_s=14400]
SPOT=1 submits on spot (g6e spot quota is separate from on-demand: 1 per size). An interruption restarts the job
from its `ckpt` (the partial LoRA is not resumed): acceptable for 1-2 h jobs.
"""
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "probes/P13-aws-rerank")]
import p13
import sm_launch

name, inst, data, hp = sys.argv[1:5]
mx = sys.argv[5] if len(sys.argv) > 5 else "14400"
assert int(mx) <= 14400, "brief: MaxRuntime <= 4 h"
sm_launch.quota_advice = lambda *a, **k: (1, None)
p13.launch(["--profile", "harsh", "--instance", inst, *([] if os.environ.get("SPOT") else ["--no-spot"]), "--prefix", "amc26-p60", "--name", name,
            "--data-s3", data, "--solution", str(ROOT / "probes/P38-qwen4b-fr/solution_p38.py"), "--max-run", mx,
            "--hp", hp])
(ROOT / "data/work/p60").mkdir(parents=True, exist_ok=True)
with open(ROOT / "data/work/p60/jobs.txt", "a") as f:
    f.write(f"{name} {inst}\n")
