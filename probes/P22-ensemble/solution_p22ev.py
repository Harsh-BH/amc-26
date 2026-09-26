"""P22 evaluation job (ONE CPU run): ev.py's #8c-yardstick evaluation, run on SageMaker because the laptop's memguard
keeps pausing it. Shipped as solution.py with ev.py and the pipeline modules it calls (rerank, stack, model,
train_matcher: the real files from src/); their unused heavy imports (block, features, worlds) are stubbed.

Channel 'train' = a data/work subset: p9/r_holdout_<c>_<ce>.parquet (bge + e5 CE caches), p9/m2_holdout_<c>_<p16>
(the dry-run stand-in), p17/stack_sup-bge-l21.0_US.txt, p22/frame_<c>.parquet + s1_<c>.npz [+ p22/r_extra_*].
params: runs = [[models, name], ...]; results = {model: s3 prefix of its training job} (pred/holdout_<c>.parquet);
s3 = where eval_<name>.json goes. Also writes repro.json (#8c control = the pipe row).
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
    for m, uri in params.get("results", {}).items():
        b, pre = uri[5:].rstrip("/").split("/", 1)
        for c in ("India", "US"):
            dst = WORK / "p22" / m / "pred" / f"holdout_{c}.parquet"
            dst.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(b, f"{pre}/pred/holdout_{c}.parquet", str(dst))
    import ev
    ev.WORK, ev.OUT, ev.STACK = WORK, WORK / "p22", WORK / "p17/stack_sup-bge-l21.0_US.txt"
    ev.PRED = {m: WORK / "p22" / m / "pred/holdout_{c}.parquet" for m in params.get("results", {})} | \
              {"p16": WORK / "p9/m2_holdout_{c}_0726cb9a_6a23c2ec_0.03.parquet"}
    b, pre = params["s3"][5:].rstrip("/").split("/", 1)
    model_dir = Path(model_dir)
    res = ev.repro()
    (model_dir / "repro.json").write_text(json.dumps(res, default=float))
    s3.upload_file(str(model_dir / "repro.json"), b, f"{pre}/repro.json")
    for models, name in params["runs"]:
        ev.evaluate(tuple(models), name)
        f = ev.OUT / f"eval_{name}.json"
        shutil.copy(f, model_dir / f.name)
        s3.upload_file(str(f), b, f"{pre}/{f.name}")
        print(f"[p22ev] {name} done at {time.time() - t0:.0f}s", flush=True)
