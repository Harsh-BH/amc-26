"""P27-france-pseudo, laptop side. The SageMaker job is solution_p27.py (self-contained, shipped as solution.py).

    p27.py prep                  # aux channel -> data/work/p27/aux/: fold-0 truth counts, house numbers, P6 LOCO models
    p27.py tiny                  # 1% S1-hash slice of every train X file -> data/work/p27/tiny/ (rehearsal)
    p27.py rehearse              # sm_train.py -> solution.py on the slice, SageMaker hp encoding, S3 writes
    p27.py launch SRC TGT [dry]  # one CPU job per LOCO direction, train = amc26-p16/data/v1/, aux = amc26-p27/data/aux/
    p27.py status [job]          # DescribeTrainingJob + the job's status.json
    p27.py fetch job             # results -> data/work/p27/jobs/<job>/
"""
import io, json, os, resource, shutil, subprocess, sys, tarfile, tempfile, time
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
os.environ.setdefault("AWS_PROFILE", "harsh")

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = ROOT / "code/business_entity_resolution/src"
sys.path[:0] = [str(HERE), str(SRC), str(ROOT)]
WORK = ROOT / "data/work"
OUT = WORK / "p27"
AUX = OUT / "aux"
TINY = OUT / "tiny"
BUCKET, PREFIX, REGION = "sagemaker-114872709824-ap-south-1", "amc26-p27", "ap-south-1"
TRAIN_S3 = f"s3://{BUCKET}/amc26-p16/data/v1/"          # P16's copy of data/work/p4/X/train_*
AUX_S3 = f"s3://{BUCKET}/{PREFIX}/data/aux/"
INSTANCE, MAX_RUN = "ml.m7i.4xlarge", 3 * 3600         # 16 vCPU / 64 GB, $1.018/h on-demand
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _boto(svc):
    import boto3
    return boto3.Session(profile_name="harsh", region_name=REGION).client(svc)


def prep():
    """Per train country: tn_<c>.parquet (fold-0 S1 row q1 -> number of true matches, 0 = singleton; = P4
    train.truth_counts(0)), hn_<c>_{s1,sx}.npy (house number as int64 per q1 / qx row, -1 = none); P6's models."""
    import features as F
    AUX.mkdir(parents=True, exist_ok=True)
    gt = pd.read_parquet(WORK / "gt_pairs.parquet", columns=["s1"]).s1.value_counts()
    fo = pd.read_parquet(WORK / "s1_folds.parquet").set_index("entity_id").fold
    num = lambda v: pd.to_numeric(pd.Series(v).fillna("").replace("", "-1"), errors="coerce").fillna(-1).to_numpy(np.int64)
    for c in ("India", "US"):
        s1, sx = F.ids("train", c)
        ids1 = pd.Index(s1.to_numpy(zero_copy_only=False))
        q = np.flatnonzero(fo.reindex(ids1).to_numpy() == 0)
        tn = pd.DataFrame({"q1": q.astype(np.int64), "n_true": gt.reindex(ids1[q]).fillna(0).to_numpy(np.int64)})
        tn.to_parquet(AUX / f"tn_{c}.parquet")
        for side, ids in (("s1", s1), ("sx", sx)):
            rf = pq.read_table(WORK / f"rf_train_{side}_{c}.parquet", columns=["id", "hn"])
            pos = pc.index_in(ids, value_set=rf.column("id")).to_numpy(zero_copy_only=False)
            assert not np.isnan(pos.astype(float)).any(), f"{c} {side}: ids missing from rf"
            hn = num(rf.column("hn").to_numpy(zero_copy_only=False))[pos.astype(np.int64)]
            np.save(AUX / f"hn_{c}_{side}.npy", hn)
            print(f"{c} {side}: {len(hn):,} rows, {np.mean(hn >= 0):.3f} with hn", flush=True)
            del rf, hn
        print(f"{c}: fold-0 S1 {len(tn):,}, singletons {np.mean(tn.n_true == 0):.4f}, peak {peak():.2f} GB", flush=True)
    for m in ("US", "India"):
        shutil.copy(OUT / "p6" / f"{m}.txt", AUX / f"model_{m}.txt")


def upload_aux():
    s3 = _boto("s3")
    for f in sorted(AUX.iterdir()):
        s3.upload_file(str(f), BUCKET, f"{PREFIX}/data/aux/{f.name}")
        print("up", f.name, flush=True)


def tiny(frac: float = 0.03):
    """Code-path rehearsal data only (the laptop has no train X; WARP pulls 0.2 MB/s): a 3% S1 slice of one TEST X
    file per country dressed as train (fold = q1 % 10, is_true = rule_acc > 0) + matching fake aux (random hn)."""
    d, a = TINY / "train/p4/X", TINY / "aux"
    d.mkdir(parents=True, exist_ok=True)
    a.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for c in ("India", "US"):
        t = pq.read_table(WORK / f"p4/X/test_{c}_sx_000.parquet")
        q1 = t.column("q1").to_numpy().astype(np.uint64)
        t = t.filter((q1 * np.uint64(2654435761) + np.uint64(7)) % np.uint64(1000) < frac * 1000)
        q1 = t.column("q1").to_numpy().astype(np.int64)
        y = (t.column("rule_acc").to_numpy() > 0)
        t = t.append_column("is_true", pa.array(y)).append_column("fold", pa.array((q1 % 10).astype(np.int8)))
        pq.write_table(t, d / f"train_{c}_sx_000.parquet")
        f0 = q1 % 10 == 0
        tn = pd.Series(y[f0]).groupby(q1[f0]).sum()
        pd.DataFrame({"q1": tn.index.to_numpy(np.int64), "n_true": tn.to_numpy(np.int64)}).to_parquet(a / f"tn_{c}.parquet")
        np.save(a / f"hn_{c}_s1.npy", rng.integers(-1, 30, q1.max() + 1))
        np.save(a / f"hn_{c}_sx.npy", rng.integers(-1, 30, t.column("qx").to_numpy().max() + 1))
        shutil.copy(AUX / f"model_{c}.txt", a / f"model_{c}.txt")
        print(c, t.num_rows, flush=True)


def _hp_json(h: dict) -> dict:
    return {k: json.dumps(v) for k, v in {"sagemaker_program": "sm_train.py", "sagemaker_submit_directory": "x", **h}.items()}


def rehearse(src="US", tgt="India"):
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="p27-", dir=OUT))
    shutil.copy(ROOT / "sm_train.py", tmp / "sm_train.py")
    shutil.copy(HERE / "solution_p27.py", tmp / "solution.py")
    (tmp / "cfg").mkdir()
    s3 = f"s3://{BUCKET}/{PREFIX}/rehearse/"
    (tmp / "cfg/hyperparameters.json").write_text(json.dumps(_hp_json({"src": src, "tgt": tgt, "s3": s3,
                                                                        "rounds": 30, "arms": ARMS2})))
    env = dict(os.environ, SM_CHANNEL_TRAIN=str(TINY / "train"), SM_CHANNEL_TEST=str(TINY / "aux"),
               SM_MODEL_DIR=str(tmp / "model"), SM_OUTPUT_DATA_DIR=str(tmp / "o"), SM_INPUT_CONFIG_DIR=str(tmp / "cfg"),
               OMP_NUM_THREADS="4")
    subprocess.run([sys.executable, str(tmp / "sm_train.py")], cwd=tmp, env=env, check=True)
    res = json.loads((tmp / "model/result.json").read_text())
    for k, v in res["dirs"].items():
        print(k, "pool", v["pool"])
        for arm, r in v["arms"].items():
            print(" ", arm, {a: b for a, b in r.items() if not a.startswith("curve")})
    s3c = _boto("s3")
    for o in s3c.list_objects_v2(Bucket=BUCKET, Prefix=f"{PREFIX}/rehearse/").get("Contents", []):
        s3c.delete_object(Bucket=BUCKET, Key=o["Key"])
    shutil.rmtree(tmp)
    print(f"rehearse ok, {time.time() - t0:.0f}s")


ARMS2 = ["tgtonly_nohn", "tgtonly_plain", "tgtonly_nohn_plain"]
REQS = HERE / "requirements_sm.txt"


def _pack(solution=None):
    files = {"sm_train.py": ROOT / "sm_train.py", "solution.py": HERE / "solution_p27.py", "requirements.txt": REQS}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for arc in sorted(files):
            tar.add(files[arc], arcname=arc)
    return buf.getvalue(), sorted(files)


def launch(src: str, tgt: str, dry: bool = False, instance: str = INSTANCE, **extra):
    import sm_launch
    sm_launch.pack_source = _pack
    name = f"p27-{src[:2].lower()}{tgt[:2].lower()}-{time.strftime('%m%d-%H%M%S')}"
    h = {"src": src, "tgt": tgt, "s3": f"s3://{BUCKET}/{PREFIX}/results/{name}/", "deadline_s": MAX_RUN - 900} | extra
    argv = ["--profile", "harsh", "--region", REGION, "--prefix", PREFIX, "--name", name, "--data-s3", TRAIN_S3,
            "--test-s3", AUX_S3, "--solution", str(HERE / "solution_p27.py"), "--instance", instance, "--no-spot",
            "--volume", "30", "--max-run", str(MAX_RUN), "--hp", json.dumps(h)] + (["--dry-run"] if dry else [])
    sm_launch.main(argv)
    if not dry:
        with open(OUT / "jobs.txt", "a") as f:
            f.write(name + "\n")
    return name


def status(job):
    d = _boto("sagemaker").describe_training_job(TrainingJobName=job)
    try:
        st = json.loads(_boto("s3").get_object(Bucket=BUCKET, Key=f"{PREFIX}/results/{job}/status.json")["Body"].read())
    except Exception:
        st = None
    print(json.dumps({"job": job, "status": d["TrainingJobStatus"], "secondary": d["SecondaryStatus"],
                      "failure": d.get("FailureReason"), "billable_s": d.get("BillableTimeInSeconds"),
                      "s3_status": st}, default=str, indent=1))
    return d, st


def fetch(job):
    s3 = _boto("s3")
    dst = OUT / "jobs" / job
    dst.mkdir(parents=True, exist_ok=True)
    for o in s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{PREFIX}/results/{job}/").get("Contents", []):
        f = dst / o["Key"].split(f"{job}/", 1)[1]
        f.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(BUCKET, o["Key"], str(f))
        print("->", f, flush=True)


if __name__ == "__main__" and sys.argv[1] not in ("prep_fr", "upload_fr", "launch_fr"):
    cmd, *a = sys.argv[1:]
    if cmd == "prep":
        prep()
    elif cmd == "upload":
        upload_aux()
    elif cmd == "tiny":
        tiny()
    elif cmd == "rehearse":
        rehearse(*a)
    elif cmd == "launch":
        kw = json.loads(a[3]) if len(a) > 3 else {}
        launch(a[0], a[1], dry=len(a) > 2 and a[2] == "dry", **kw)
    elif cmd == "status":
        status(a[0])
    elif cmd == "fetch":
        fetch(a[0])


def prep_fr():
    """France aux -> data/work/p27/fr/: #9ab's France rows as (q1, qx, p, p_gbdt) on F.ids('test', 'France') indices,
    test house numbers, m2_big (alignment check), and the IDF-fixed France X (P14 idf144 rebuilt with --keep-x)."""
    import features as F
    d = OUT / "fr"
    (d / "X").mkdir(parents=True, exist_ok=True)
    s1, sx = F.ids("test", "France")
    t = pq.read_table(WORK / "p19/prob_test_France_idf144_bge.parquet")
    q1 = pc.index_in(t.column("s1"), value_set=s1).to_numpy(zero_copy_only=False)
    qx = pc.index_in(t.column("sx"), value_set=sx).to_numpy(zero_copy_only=False)
    assert not (np.isnan(q1.astype(float)).any() or np.isnan(qx.astype(float)).any())
    pd.DataFrame({"q1": q1.astype(np.int64), "qx": qx.astype(np.int64), "p": t.column("p").to_numpy(),
                  "p_gbdt": t.column("p_gbdt").to_numpy()}).to_parquet(d / "keys.parquet")
    num = lambda v: pd.to_numeric(pd.Series(v).fillna("").replace("", "-1"), errors="coerce").fillna(-1).to_numpy(np.int64)
    for side, ids in (("s1", s1), ("sx", sx)):
        rf = pq.read_table(WORK / f"rf_test_{side}_France.parquet", columns=["id", "hn"])
        pos = pc.index_in(ids, value_set=rf.column("id")).to_numpy(zero_copy_only=False).astype(np.int64)
        np.save(d / f"hn_France_{side}.npy", num(rf.column("hn").to_numpy(zero_copy_only=False))[pos])
    json.dump({"n_s1": len(s1)}, open(d / "meta.json", "w"))
    for f in sorted((WORK / "p14/w_idfx/p4/X").glob("test_France_sx_*.parquet")):
        dst = d / "X" / f.name
        if not dst.exists():
            dst.symlink_to(f.resolve())
    if not (d / "m2_big.txt").exists():
        (d / "m2_big.txt").symlink_to((WORK / "p4/m2_big.txt").resolve())
    print(f"France keys {t.num_rows:,}, S1 {len(s1):,}, SX {len(sx):,}, peak {peak():.2f} GB")


def upload_fr():
    s3 = _boto("s3")
    d = OUT / "fr"
    for f in sorted(d.rglob("*")):
        if f.is_file():
            k = f"{PREFIX}/data/fr/{f.relative_to(d)}"
            s3.upload_file(str(f.resolve()), BUCKET, k)
            print("up", k, flush=True)


def launch_fr(dry: bool = False, **extra):
    import sm_launch
    files = {"sm_train.py": ROOT / "sm_train.py", "solution.py": HERE / "solution_p27fr.py", "requirements.txt": REQS}

    def pack(solution=None):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for arc in sorted(files):
                tar.add(files[arc], arcname=arc)
        return buf.getvalue(), sorted(files)
    sm_launch.pack_source = pack
    name = f"p27-fr-{time.strftime('%m%d-%H%M%S')}"
    h = {"s3": f"s3://{BUCKET}/{PREFIX}/results/{name}/", "deadline_s": 5400} | extra
    argv = ["--profile", "harsh", "--region", REGION, "--prefix", PREFIX, "--name", name, "--data-s3", TRAIN_S3,
            "--test-s3", f"s3://{BUCKET}/{PREFIX}/data/fr/", "--solution", str(HERE / "solution_p27fr.py"),
            "--instance", INSTANCE, "--no-spot", "--volume", "30", "--max-run", "7200", "--hp", json.dumps(h)] \
        + (["--dry-run"] if dry else [])
    sm_launch.main(argv)
    if not dry:
        with open(OUT / "jobs.txt", "a") as f:
            f.write(name + "\n")
    return name


if __name__ == "__main__" and sys.argv[1] in ("prep_fr", "upload_fr", "launch_fr"):
    {"prep_fr": prep_fr, "upload_fr": upload_fr, "launch_fr": launch_fr}[sys.argv[1]]()
