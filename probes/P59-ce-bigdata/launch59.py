"""Submit one P59 SageMaker job: sm_launch with P13's pinned bundle, quota lookup skipped (ListServiceQuotas is
rate-limited when several probes launch at once; quota is known: 1 per GPU type).
    launch59.py <name> <instance> <channel s3 uri> '<hp json>' [solution=probes/P13-aws-rerank/solution_p13.py]
A solution other than solution_p13.py ships with solution_p13.py as p13sol.py (P45's full-coverage wrapper)."""
import io, sys, tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "probes/P13-aws-rerank")]
import p13
import sm_launch

P13SOL = ROOT / "probes/P13-aws-rerank/solution_p13.py"
name, inst, data, hp = sys.argv[1:5]
sol = Path(sys.argv[5]) if len(sys.argv) > 5 else P13SOL


def pack(solution=None):
    files = {"sm_train.py": ROOT / "sm_train.py", "solution.py": sol, "requirements.txt": p13.REQS}
    if sol.resolve() != P13SOL:
        files["p13sol.py"] = P13SOL
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for arc in sorted(files):
            tar.add(files[arc], arcname=arc)
    return buf.getvalue(), sorted(files)


sm_launch.pack_source = pack
sm_launch.quota_advice = lambda *a, **k: (1, None)
sm_launch.main(["--profile", "harsh", "--instance", inst, "--no-spot", "--prefix", "amc26-p59", "--name", name,
                "--data-s3", data, "--solution", str(sol), "--max-run", "10800", "--hp", hp])
with open(ROOT / "data/work/p59/jobs.txt", "a") as f:
    f.write(f"{name} {inst}\n")
