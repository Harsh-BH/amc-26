"""Launch solution_p45c.py (full-coverage CE on test) with P13's bundle + solution_p13.py as p13sol.py.
    launch_c.py <name> <instance> '<hp json>'"""
import io, sys, tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "probes/P13-aws-rerank")]
import p13
import sm_launch


def pack(solution=None):
    files = {"sm_train.py": ROOT / "sm_train.py", "solution.py": ROOT / "probes/P45-ens-stages/solution_p45c.py",
             "p13sol.py": ROOT / "probes/P13-aws-rerank/solution_p13.py", "requirements.txt": p13.REQS}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for arc in sorted(files):
            tar.add(files[arc], arcname=arc)
    return buf.getvalue(), sorted(files)


name, inst, hp = sys.argv[1:4]
sm_launch.pack_source = pack
sm_launch.main(["--profile", "harsh", "--instance", inst, "--no-spot", "--prefix", "amc26-p45", "--name", name,
                "--data-s3", "s3://sagemaker-114872709824-ap-south-1/amc26-p45/data/cetest/",
                "--solution", str(ROOT / "probes/P45-ens-stages/solution_p45c.py"), "--max-run", "7200", "--hp", hp])
