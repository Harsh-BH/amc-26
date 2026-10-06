"""Launch a P67 GPU job with only sm_train.py + the solution + its unchanged base job file + P13's pinned requirements.
    launch67.py bge  <name> <instance> <runs: India,US | France> [channel]
    launch67.py qwen <name> <instance> <k: India | US | France> [channel]
"""
import io
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "probes/P13-aws-rerank")]
import p13  # noqa: E402
import sm_launch  # noqa: E402

B = "s3://sagemaker-114872709824-ap-south-1"
HERE = ROOT / "probes/P67-fr-synth"
CH = f"{B}/amc26-p67/{sys.argv[5] if len(sys.argv) > 5 else 'ch'}/"   # channel: ch = v1, ch2 = v2, ch_v1h
BGE = {"India": (f"{B}/amc26-p30/results/p30-veto-0926-1651/ckpt/ce_bge_locoUS.pt", "fb_US_India"),
       "US": (f"{B}/amc26-p30/results/p30-veto-0926-1651/ckpt/ce_bge_locoIndia.pt", "fb_India_US"),
       "France": (f"{B}/amc26-p13/results/p13-rerank-0925-2257/ckpt/ce_bge_pool.pt", "fb_France")}
QWEN = {"India": (f"{B}/amc26-p38/results/p38-locoUS-0926-2346/ckpt/lora_qwen_locoUS.pt", "fb_US_India"),
        "US": (f"{B}/amc26-p38/results/p38-locoIndia-0926-2346/ckpt/lora_qwen_locoIndia.pt", "fb_India_US"),
        "France": (f"{B}/amc26-p38/results/p38-pool-0927-0145/ckpt/lora_qwen_pool.pt", "fb_France")}


def pack(sol, base_file, base_name):
    files = {"sm_train.py": ROOT / "sm_train.py", "solution.py": sol, base_name: base_file, "requirements.txt": p13.REQS}

    def _p(solution=None):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for arc in sorted(files):
                tar.add(files[arc], arcname=arc)
        return buf.getvalue(), sorted(files)
    return _p


arm, name, inst, what = sys.argv[1:5]
res = f"{B}/amc26-p67/results/{name}"
if arm == "bge":
    hp = {"runs": [{"k": k, "ckpt": BGE[k][0], "score": [BGE[k][1]], "pre": k != "France"} for k in what.split(",")],
          "s3": res, "lr": 1e-5, "train_cap_s": 2400, "deadline_s": 6600}
    sm_launch.pack_source = pack(HERE / "solution_p67.py", ROOT / "probes/P13-aws-rerank/solution_p13.py", "p13sol.py")
else:
    k = what
    ck, fr = QWEN[k]
    common = {"ckpt": ck, "score": [fr], "mid": [], "val": [], "score_speed": 150, "deadline_s": 5400}
    stages = ([] if k == "France" else [dict(common, tag=f"qpre_{k}", train=[], s3=f"{res}/qpre")]) + \
             [dict(common, tag=f"qsyn_{k}", train=[f"run{k}"], cap_s=720, lr=5e-5, s3=f"{res}/qsyn")]
    hp = {"stages": stages, "weights_s3": f"{B}/amc26-p38/weights/Qwen3-Reranker-4B"}
    sm_launch.pack_source = pack(HERE / "solution_p67q.py", ROOT / "probes/P38-qwen4b-fr/solution_p38.py", "p38sol.py")
sm_launch.main(["--profile", "harsh", "--instance", inst, "--no-spot", "--prefix", "amc26-p67", "--name", name,
                "--data-s3", CH, "--solution", str(HERE / "solution_p67.py"), "--max-run", "7200", "--hp", json.dumps(hp)])
