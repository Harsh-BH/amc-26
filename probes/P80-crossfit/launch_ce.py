"""Submit one half of the P80 CE cross-fit: sm_launch with P13's pinned bundle (sm_train.py, solution_p80ce.py as
solution.py, solution_p13.py as p13sol.py, P13's requirements_sm.txt), on-demand, profile harsh only.
    launch_ce.py <A|B> <instance>        -> job p80ce-<A|B>-MMDD-HHMM, name appended to probes/P80-crossfit/JOBS"""
import io, json, sys, tarfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUCKET, PREFIX = "sagemaker-114872709824-ap-south-1", "amc26/p80/ce"
MAX_RUN = 10800                                  # expected ~85 min; worst case 3 h x ($3.37 + $4.51) = $23.6 for both


def hp(split: str, name: str) -> dict:
    """P13 bge_pool defaults = the shipped CE recipe; only the planner's prior speeds (L40S, P59) and the deadline differ."""
    return {"split": split, "phases": ["bge_pool"], "s3": f"s3://{BUCKET}/{PREFIX}/results/{name}",
            "deadline_s": MAX_RUN - 1200, "speed": {"bge_train": 600, "bge_score": 2500}}


def pack(solution=None):
    files = {"sm_train.py": ROOT / "sm_train.py", "solution.py": HERE / "solution_p80ce.py",
             "p13sol.py": ROOT / "probes/P13-aws-rerank/solution_p13.py",
             "requirements.txt": ROOT / "probes/P13-aws-rerank/requirements_sm.txt"}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for arc in sorted(files):
            tar.add(files[arc], arcname=arc)
    return buf.getvalue(), sorted(files)


if __name__ == "__main__":
    split, inst = sys.argv[1:3]
    name = f"p80ce-{split}-{time.strftime('%m%d-%H%M')}"
    sys.path.insert(0, str(ROOT))
    import sm_launch
    sm_launch.pack_source = pack
    sm_launch.main(["--profile", "harsh", "--region", "ap-south-1", "--instance", inst, "--no-spot", "--prefix", PREFIX,
                    "--name", name, "--data-s3", f"s3://{BUCKET}/{PREFIX}/data/", "--solution", str(HERE / "solution_p80ce.py"),
                    "--max-run", str(MAX_RUN), "--hp", json.dumps(hp(split, name))] + sys.argv[3:])
    if "--dry-run" not in sys.argv:
        with open(HERE / "JOBS", "a") as f:
            f.write(f"{name} {inst}\n")
