"""P51 laptop side (the job is P13's solution_p13.py on the P51 channel; launch with launch51.py).
    p51.py fetch <job> [tag=ser]    # results/<job>/r/*.parquet -> data/work/p51/r/, renamed r_bge_* -> r_<tag>_*
                                    # (r_bge_locoUS_holdout -> r_ser_locoUS_holdout)
    p51.py gate <tag_US> <tag_India> [vs_US vs_India]   # P38's gate (P13's France procedure on LOCO), serialized
                                    # bge vs raw bge (default) or vs Qwen; per-S1 paired; own ledger row
r dir data/work/p51/r also symlinks P13's raw-bge LOCO files and P38's Qwen LOCO files.
"""
import json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "probes/P38-qwen4b-fr")]
OUT = ROOT / "data/work/p51"
R = OUT / "r"
B = "s3://sagemaker-114872709824-ap-south-1/amc26-p51/results"
AWS = ["--profile", "harsh", "--region", "ap-south-1", "--only-show-errors"]


def fetch(job: str, tag: str = "ser"):
    R.mkdir(parents=True, exist_ok=True)
    for f in [*(ROOT / "data/work/p13/r").glob("r_bge_loco*_holdout.parquet"),
              *(ROOT / "data/work/p38/r").glob("r_qwen_loco*_holdout.parquet")]:
        if not (R / f.name).exists():
            (R / f.name).symlink_to(f.resolve())
    tmp = OUT / f"dl_{job}"
    subprocess.run(["aws", "s3", "sync", f"{B}/{job}/r/", str(tmp), *AWS], check=True)
    subprocess.run(["aws", "s3", "cp", f"{B}/{job}/log.jsonl", str(OUT / f"log_{job}.jsonl"), *AWS], check=True)
    for f in tmp.glob("r_bge_*.parquet"):
        f.replace(R / f.name.replace("r_bge_", f"r_{tag}_", 1))
    tmp.rmdir()
    print(sorted(p.name for p in R.glob(f"r_{tag}_*")))


def gate(t_us: str, t_in: str, v_us: str = "bge_locoUS_holdout", v_in: str = "bge_locoIndia_holdout"):
    os.environ["P38_NOLOG"] = "1"
    import p38
    import p13
    from ledger import log
    p38.R = p13.R = R
    p38.OUT = OUT
    t0 = time.time()
    res = p38.gate(t_us, t_in, v_us, v_in)        # its 'qwen' arm = serialized bge, its 'bge' arm = the control
    d = {f"{a}>{b}": res[f"qwen-bge_mean_{a}>{b}"] for a, b in (("US", "India"), ("India", "US"))}
    ok = all(v["gain"] >= 0.0010 for v in d.values())
    nm = f"gate51-{t_us}+{t_in}-vs-{v_us}+{v_in}"
    if os.environ.get("P51_NOLOG"):                     # code checks stay out of the ledger
        log = lambda *a, **k: None
    log("exp", probe="P51-ce-serialize", subproblem="rerank-france", name=nm, world="holdout-dense-half1",
        metrics={f"{k}_{q}": (round(v, 5) if isinstance(v, float) else v) for k, m in res.items() for q, v in m.items()
                 if q != "w"},
        config={"band": [0.02, 0.98], "gbdt": "m2_big", "rule": "shipped (argmax, p >= t)",
                "threshold": "P6 empty-rate match", "arm": [t_us, t_in], "control": [v_us, v_in], "maxlen": 160,
                "gate": "arm-control mean-OOD-w France-F0.5 >= +0.0010 both directions",
                "w": {k: m["w"] for k, m in res.items() if "w" in m}},
        hypothesis="Ditto-style serialized fields (parsed name core / legal / hn / street / city / state / flags + raw "
                   "name) make the bge CE transfer better to an unseen country than P3's raw 'name | addr' text",
        verdict="keep" if ok else "drop", runtime_s=round(time.time() - t0), peak_gb=round(p38.peak(), 2))
    print("GATE51", "PASS" if ok else "FAIL", json.dumps(d))
    return res


if __name__ == "__main__":
    globals()[sys.argv[1]](*sys.argv[2:])
