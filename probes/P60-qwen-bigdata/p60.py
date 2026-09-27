"""P60 laptop side (jobs: launch60.py; job list data/work/p60/jobs.txt).

    p60.py fetch <job>     # results/<job>/r/*.parquet (no smoke) + log.jsonl -> data/work/p60/
    p60.py gate            # P38's gate procedure: P60 LOCO Qwen (warm-started on folds 1-3) vs P38's LOCO Qwen, same
                           # holdout pairs. Gate: qwen60 - qwen38 >= +0.0005 France-F0.5 in both directions (mean mode).

r files live in data/work/p60/r; P38's LOCO Qwen and P13's LOCO bge files are symlinked in, so p38.gate reads one dir.
"""
import json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "probes/P38-qwen4b-fr")]
OUT = ROOT / "data/work/p60"
R = OUT / "r"
BUCKET, PREFIX = "sagemaker-114872709824-ap-south-1", "amc26-p60"
BAR = 0.0005


def fetch(job: str):
    R.mkdir(parents=True, exist_ok=True)
    for f in (ROOT / "data/work/p38/r").glob("r_*_loco*_holdout*.parquet"):
        if not (R / f.name).exists():
            (R / f.name).symlink_to(f.resolve())
    aws = ["--profile", "harsh", "--region", "ap-south-1", "--only-show-errors"]
    subprocess.run(["aws", "s3", "sync", f"s3://{BUCKET}/{PREFIX}/results/{job}/r/", str(R), "--exclude", "*_smoke_*", *aws], check=True)
    subprocess.run(["aws", "s3", "cp", f"s3://{BUCKET}/{PREFIX}/results/{job}/log.jsonl", str(OUT / f"log_{job}.jsonl"), *aws], check=True)
    print(sorted(p.name for p in R.glob("*.parquet")))


def gate(q_us="qwen60_locoUS", q_in="qwen60_locoIndia", o_us="qwen_locoUS_holdout", o_in="qwen_locoIndia_holdout"):
    """p38.gate with P38's LOCO Qwen in its 'bge' slot: its 'qwen-bge' rows are then new - old Qwen."""
    import p38
    import p13
    from ledger import log
    p38.R = p13.R = R
    p38.OUT = OUT
    os.environ["P38_NOLOG"] = "1"                       # p38.gate's own row is P38's (bar 0.0033, bge wording)
    t0 = time.time()
    res = p38.gate(q_us, q_in, o_us, o_in)
    dirs = (("US", "India"), ("India", "US"))
    g = {f"{a}>{b}": res[f"qwen-bge_mean_{a}>{b}"] for a, b in dirs}
    ok = all(v["gain"] >= BAR for v in g.values())
    m = {}
    for k, v in res.items():
        if k.startswith(("qwen-bge", "qwen_mean", "bge_mean", "qwen_xfit", "bge_xfit")):
            for q, x in v.items():
                if q != "w":
                    m[f"{k.replace('qwen-bge', 'new-old').replace('bge_', 'old_').replace('qwen_', 'new_')}_{q}"] = \
                        round(x, 6) if isinstance(x, float) else x
    log("exp", probe="P60-qwen-bigdata", subproblem="rerank-france", name="gate-qwen60_loco-vs-qwen38_loco",
        world="holdout-dense-half1", metrics=m,
        config={"new": [q_us, q_in], "old": [o_us, o_in], "warm_start": "P38 LOCO LoRA + 1 pass over folds 1-3 (P59 channel)",
                "lr": 5e-5, "procedure": "P38 gate (P13 France procedure, OOD mean weights, P6 empty-rate t)",
                "bar": f">= +{BAR} France-F0.5 both directions, mean mode",
                "w": {k: v["w"] for k, v in res.items() if "w" in v}},
        hypothesis="Qwen3-Reranker-4B LoRA that has also seen folds 1-3 (1.42x pairs) is a better out-of-domain France CE",
        verdict="keep" if ok else "drop", runtime_s=round(time.time() - t0))
    print("P60 GATE", "PASS" if ok else "FAIL", {k: {q: round(x, 6) for q, x in v.items() if isinstance(x, float)} for k, v in g.items()})
    return res


if __name__ == "__main__":
    cmd, pos = sys.argv[1], sys.argv[2:]
    {"fetch": lambda: fetch(*pos), "gate": lambda: gate(*pos)}[cmd]()
