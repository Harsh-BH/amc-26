"""P25 evaluation job (ONE CPU run) = P22's solution_p22ev.py with ev25 (P25's configs) in place of ev.evaluate.
Shipped as solution.py with ev25.py, P22's ev.py and the src modules ev calls (rerank, stack, model, train_matcher).

Channel 'train' = P22's eval channel (s3 amc26-p22/data/ev/: frame, s1 npz, bge + e5 CE caches, P17 stacker).
params: models = [variant, ...]; results = {variant: s3 prefix of its training job} (pred/holdout_<c>.parquet);
runs = [[models, name, only|null], ...]; s3 = where repro.json and eval_<name>.json go.
"""
import json, shutil, sys, time, types
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path("/tmp/work")


def stub(name: str, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m


def main(train_dir, test_dir, model_dir, out_dir, params):
    import boto3
    t0 = time.time()
    shutil.copytree(train_dir, WORK, dirs_exist_ok=True)
    stub("worlds", WORK=WORK)
    stub("block", TXT={}, peak_gb=lambda: 0.0)
    stub("features")
    sys.path.insert(0, str(HERE))
    s3 = boto3.client("s3")
    out = WORK / "p22"                               # ev.load reads the frame from ev.OUT = p22/
    for m, uri in params["results"].items():
        b, pre = uri[5:].rstrip("/").split("/", 1)
        for c in ("India", "US"):
            dst = out / m / "pred" / f"holdout_{c}.parquet"
            dst.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(b, f"{pre}/pred/holdout_{c}.parquet", str(dst))
    import ev
    import ev25
    ev.WORK, ev.OUT, ev.STACK = WORK, out, WORK / "p17/stack_sup-bge-l21.0_US.txt"
    ev.PRED = {m: out / m / "pred/holdout_{c}.parquet" for m in params["results"]}
    b, pre = params["s3"][5:].rstrip("/").split("/", 1)
    model_dir = Path(model_dir)
    res = ev.repro()
    (model_dir / "repro.json").write_text(json.dumps(res, default=float))
    s3.upload_file(str(model_dir / "repro.json"), b, f"{pre}/repro.json")
    print(f"[p25ev] repro at {time.time() - t0:.0f}s: all {res['all']:.5f} h1 {res['h1']:.5f}", flush=True)
    for models, name, only in params["runs"]:
        ev25.evaluate(tuple(models), name, only)
        f = out / f"eval_{name}.json"
        shutil.copy(f, model_dir / f.name)
        s3.upload_file(str(f), b, f"{pre}/{f.name}")
        print(f"[p25ev] {name} done at {time.time() - t0:.0f}s", flush=True)
