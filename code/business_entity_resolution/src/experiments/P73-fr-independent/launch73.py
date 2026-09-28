"""Launch a P73 GPU job (P67's packing: sm_train.py + solution + unchanged P13 job file + P13's pinned requirements).
    launch73.py <name> <instance> <run,run,...>     run = <k>:<m>, k in India|US|France, m in MODELS
    .venv/bin/python probes/P73-fr-independent/launch73.py p73-bge-loco-0927-1530 ml.g5.2xlarge India:bge,US:bge
"""
import io
import json
import os
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "probes/P13-aws-rerank")]
import p13  # noqa: E402
import sm_launch  # noqa: E402

B = "s3://" + os.environ.get("BUCKET", "your-bucket")
HERE = ROOT / "probes/P73-fr-independent"
# public checkpoints only (licences checked on the HF cards 27 Sep): name, lr, epochs
MODELS = {"bge": ("BAAI/bge-reranker-v2-m3", 1e-5, 2),                          # Apache-2.0, 568M, multilingual
          "cam": ("antoinelouis/crossencoder-camembert-base-mmarcoFR", 2e-5, 2),  # MIT, 111M, French
          "ele": ("cross-encoder/ms-marco-electra-base", 2e-5, 2),
          "ele4": ("cross-encoder/ms-marco-electra-base", 2e-5, 4),   # 4 epochs (user, 16:45)
          "mxb": ("mixedbread-ai/mxbai-rerank-base-v1", 2e-5, 4),      # Apache-2.0, 184M (DeBERTa-v3)
          "mxl": ("mixedbread-ai/mxbai-rerank-large-v1", 1e-5, 2),     # Apache-2.0, 435M
          "mml": ("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", 3e-5, 4),  # Apache-2.0, 118M, multilingual
          "bgb": ("BAAI/bge-reranker-base", 2e-5, 4),                  # MIT, 278M
          "bgl": ("BAAI/bge-reranker-large", 1e-5, 2),                 # MIT, 560M
          "ele6": ("cross-encoder/ms-marco-electra-base", 2e-5, 6),    # 6 epochs
          "ele4s0": ("cross-encoder/ms-marco-electra-base", 2e-5, 4), "ele4s1": ("cross-encoder/ms-marco-electra-base", 2e-5, 4),
          "ele4s2": ("cross-encoder/ms-marco-electra-base", 2e-5, 4)}
SEED = {"ele4s1": 1, "ele4s2": 2}   # seed arm (ele4s0 = seed 0, + fc scoring)
FC = {"ele4s0", "ele4s1", "ele4s2"}  # these runs also score fc_<frame> (confident rows, veto arm)
                # Apache-2.0, 109M, English
FRAME = {"India": "fb_US_India", "US": "fb_India_US", "France": "fb_France"}


def pack(sol, base_file, base_name):
    files = {"sm_train.py": ROOT / "sm_train.py", "solution.py": sol, base_name: base_file, "requirements.txt": p13.REQS}

    def _p(solution=None):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for arc in sorted(files):
                tar.add(files[arc], arcname=arc)
        return buf.getvalue(), sorted(files)
    return _p


name, inst, what = sys.argv[1:4]
runs = []
for r in what.split(","):
    k, m = r.split(":")
    mod, lr, ep = MODELS[m]
    h = "h" if k in ("Indiah", "USh") else "n" if k in ("Indian", "USn", "Francen") else "r" if k in ("Indiar", "USr", "Francer") else "v" if k in ("Indiav", "USv", "Francev") else ""   # h: strict v2h; n: v3 noise list
    fb = FRAME[k.removesuffix(h) if h else k]
    runs.append({"k": k, "m": m + h, "model": mod, "lr": lr, "epochs": ep, "seed": SEED.get(m, 0),
                 "score": [fb] + ([fb.replace("fb_", "fc_")] if m in FC else []),
                 "save": k.startswith("France"),
                 "cap_s": 2400})
hp = {"runs": runs, "s3": f"{B}/amc26-p73/results/{name}", "deadline_s": 6600}
sm_launch.pack_source = pack(HERE / "solution_p73.py", ROOT / "probes/P13-aws-rerank/solution_p13.py", "p13sol.py")
sm_launch.main(["--profile", os.environ.get("AWS_PROFILE", "default"), "--instance", inst, "--no-spot", "--prefix", "amc26-p73", "--name", name,
                "--data-s3", f"{B}/amc26-p73/ch/", "--solution", str(HERE / "solution_p73.py"), "--max-run", "7200",
                "--hp", json.dumps(hp)])
