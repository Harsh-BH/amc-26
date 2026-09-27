"""Submit P80 CE jobs: sm_launch with P13's pinned bundle (sm_train.py, solution_p80ce.py as solution.py, solution_p13.py
as p13sol.py, P13's requirements_sm.txt), on-demand, profile harsh only. Job names are appended to probes/P80-crossfit/JOBS.
    launch_ce.py <A|B> <instance>                    # cross-fit half -> p80ce-<A|B>-MMDD-HHMM
    launch_ce.py score <India|US> <k> <n> <instance> # slice k of n of stage C's missing rows, both checkpoints
                                                     # -> p80cs-<c><k>-<instance>-MMDD-HHMM"""
import io, json, sys, tarfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUCKET, PREFIX = "sagemaker-114872709824-ap-south-1", "amc26/p80/ce"
MAX_RUN = 10800                                  # expected ~85 min; worst case 3 h x ($3.37 + $3.13) = $19.5 for A (2xlarge) + B (xlarge)


def hp(split: str, name: str) -> dict:
    """P13 bge_pool defaults = the shipped CE recipe; only the planner's prior speeds (L40S, P59) and the deadline differ."""
    return {"split": split, "phases": ["bge_pool"], "s3": f"s3://{BUCKET}/{PREFIX}/results/{name}",
            "deadline_s": MAX_RUN - 1200, "speed": {"bge_train": 600, "bge_score": 2500}}


CKPTS = {s: f"s3://{BUCKET}/{PREFIX}/results/{j}/ckpt/ce_bge_pool.pt"
         for s, j in (("A", "p80ce-A-0927-2141"), ("B", "p80ce-B-0927-2149"))}


def score_hp(c: str, k: int, n: int, name: str) -> dict:
    """P13 bge_ckpt per checkpoint; planner prior 1200 pairs/s (A10G / L4 worst case), one shared deadline."""
    return {"mode": "score", "country": c, "part": [int(k), int(n)], "ckpts": CKPTS, "phases": ["bge_ckpt"],
            "s3": f"s3://{BUCKET}/{PREFIX}/results/{name}", "deadline_s": MAX_RUN - 1200, "speed": {"bge_score": 1200}}


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
    sys.path.insert(0, str(ROOT))
    import sm_launch
    sm_launch.pack_source = pack
    if sys.argv[1] == "score":
        c, k, n, inst = sys.argv[2:6]
        name = f"p80cs-{c}{k}-{inst.split('.', 1)[1].replace('.', '')}-{time.strftime('%m%d-%H%M')}"
        data, h, rest = f"s3://{BUCKET}/{PREFIX}/score_data/", score_hp(c, k, n, name), sys.argv[6:]
        sm_launch.quota_advice = lambda *a, **kw: (1, None)     # 1 per GPU type (known); ListServiceQuotas throttles
    else:
        split, inst = sys.argv[1:3]
        name = f"p80ce-{split}-{time.strftime('%m%d-%H%M')}"
        data, h, rest = f"s3://{BUCKET}/{PREFIX}/data/", hp(split, name), sys.argv[3:]
    sm_launch.main(["--profile", "harsh", "--region", "ap-south-1", "--instance", inst, "--no-spot", "--prefix", PREFIX,
                    "--name", name, "--data-s3", data, "--solution", str(HERE / "solution_p80ce.py"),
                    "--max-run", str(MAX_RUN), "--hp", json.dumps(h)] + rest)
    if "--dry-run" not in sys.argv:
        with open(HERE / "JOBS", "a") as f:
            f.write(f"{name} {inst}\n")
