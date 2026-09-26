"""Launch solution_p48.py with P13's bundle (solution_p13.py as p13sol.py), = P45's launch_c.py.
    launch48.py <name> <instance>"""
import io, json, sys, tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "probes/P13-aws-rerank")]
import p13
import sm_launch

B = "s3://sagemaker-114872709824-ap-south-1"


def pack(solution=None):
    files = {"sm_train.py": ROOT / "sm_train.py", "solution.py": ROOT / "probes/P48-fr-ce-all/solution_p48.py",
             "p13sol.py": ROOT / "probes/P13-aws-rerank/solution_p13.py", "requirements.txt": p13.REQS}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for arc in sorted(files):
            tar.add(files[arc], arcname=arc)
    return buf.getvalue(), sorted(files)


name, inst = sys.argv[1:3]
hp = {"runs": {"France": f"{B}/amc26-p13/results/p13-rerank-0925-2257/ckpt/ce_bge_pool.pt",
               "India": f"{B}/amc26-p30/results/p30-veto-0926-1651/ckpt/ce_bge_locoUS.pt",
               "US": f"{B}/amc26-p30/results/p30-veto-0926-1651/ckpt/ce_bge_locoIndia.pt"},
      "s3": f"{B}/amc26-p48/results/{name}", "deadline_s": 4800}
sm_launch.pack_source = pack
sm_launch.main(["--profile", "harsh", "--instance", inst, "--no-spot", "--prefix", "amc26-p48", "--name", name,
                "--data", str(ROOT / "data/work/p48/sm"), "--solution", str(ROOT / "probes/P48-fr-ce-all/solution_p48.py"),
                "--max-run", "5400", "--hp", json.dumps(hp)])
