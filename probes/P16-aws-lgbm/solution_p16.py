"""P16 SageMaker job (ONE CPU run): m2_big's recipe (src/train_matcher.main) on train folds 2-9 instead of 2-5.
Shipped alone as solution.py next to sm_train.py; sm_train calls main(train_dir, test_dir, model_dir, out_dir, params).

Same as train_matcher.main: the LightGBM params come in as hp['params'] (= model.PARAMS, read by the launcher, only
num_threads is set to the instance's vCPUs); train rows = every pair of the S1 in hp['folds'], held as float16 in an
lgb.Sequence; validation = the 25% S1-hash sample of fold 1 (float32), early stopping 100 rounds, at most 8000.
load() is train_matcher.load on pyarrow (the DLC has no pandas 3 / src deps); p16.py check asserts they are equal.
Channel 'train' = data/work/ layout: p4/X/train_<country>_sx_NNN.parquet (sorted = train_matcher's file order).

Deadline-aware: past hp['deadline_s'] the run stops like an early stop (the best iteration so far is kept). The
booster (-> model_dir/m2_p16.txt), checkpoints every hp['ckpt_every'] rounds and status.json go to hp['s3'] too.
"""
import json, os, resource, time
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

T0 = time.time()
ID = ["q1", "qx", "is_true", "fold"]
DROP = {"s1", "sx", "q1", "qx", "y", "is_true", "fold", "country"}      # model.DROP
DEFAULTS = {"folds": [2, 3, 4, 5, 6, 7, 8, 9], "val_frac": 0.25, "frac": 1.0, "rounds": 8000, "es": 100,
            "deadline_s": 13200, "ckpt_every": 1000, "s3": "", "params": None}
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def log(msg: str):
    print(f"[p16 {time.time() - T0:7.0f}s peak {peak():5.1f} GB] {msg}", flush=True)


class S3:
    """Best-effort uploads under s3://bucket/prefix/; '' disables. A failed upload is logged, never fatal (the
    booster also lands in model_dir -> model.tar.gz)."""

    def __init__(self, uri: str):
        self.on = bool(uri)
        if self.on:
            import boto3
            self.bucket, self.prefix = uri[5:].rstrip("/").split("/", 1)
            self.c = boto3.client("s3")

    def put(self, path: Path, key: str):
        if self.on:
            try:
                self.c.upload_file(str(path), self.bucket, f"{self.prefix}/{key}")
            except Exception as e:                      # network / throttling: the model_dir copy still exists
                log(f"s3 {key} failed: {e!r}")

    def put_json(self, obj: dict, key: str):
        if self.on:
            try:
                self.c.put_object(Bucket=self.bucket, Key=f"{self.prefix}/{key}", Body=json.dumps(obj).encode())
            except Exception as e:
                log(f"s3 {key} failed: {e!r}")


def files(train_dir: Path) -> list:
    return sorted(Path(train_dir).glob("p4/X/train_*_sx_*.parquet"))


def load(fs: list, cols: list, folds, frac: float = 1.0, seed: int = 0, dtype=np.float32):
    """train_matcher.load: (X [n, len(cols)], y) of the pairs of S1 in `folds`, files in order; frac < 1 keeps an
    S1-hash sample. Preallocated, so RAM is one copy of X plus one file."""
    def keep(t):
        q1, f = t.column("q1").to_numpy().astype(np.uint64), t.column("fold").to_numpy()
        return np.isin(f, folds) & ((q1 * np.uint64(2654435761) + np.uint64(seed)) % np.uint64(1000) < frac * 1000)

    n = sum(int(keep(pq.read_table(p, columns=["q1", "fold"])).sum()) for p in fs)
    X, y, a = np.empty((n, len(cols)), dtype), np.empty(n, bool), 0
    for p in fs:
        t = pq.read_table(p, columns=ID + cols)
        t = t.filter(keep(t))
        for j, c in enumerate(cols):
            X[a:a + len(t), j] = t.column(c).to_numpy().astype(np.float32).astype(dtype)   # nulls -> NaN
        y[a:a + len(t)] = t.column("is_true").to_numpy()
        a += len(t)
        del t
    assert a == n
    return X, y


def main(train_dir, test_dir, model_dir, out_dir, params):
    import lightgbm as lgb
    hp = DEFAULTS | {k: v for k, v in params.items() if k in DEFAULTS}
    assert hp["params"], "pass params = model.PARAMS"
    assert not {0, 1} & set(hp["folds"]), "never train on folds 0 (holdout) / 1 (early stopping)"
    P = dict(hp["params"], num_threads=os.cpu_count())
    s3, model_dir = S3(hp["s3"]), Path(model_dir)
    log(f"lightgbm {lgb.__version__}, {os.cpu_count()} vCPU, hp {hp | {'params': P}}")

    class Seq(lgb.Sequence):
        def __init__(self, a):
            self.a, self.batch_size = a, 65_536

        def __getitem__(self, i):
            return self.a[i].astype(np.float64)       # LightGBM samples rows as double

        def __len__(self):
            return len(self.a)

    fs = files(train_dir)
    assert fs, f"no p4/X/train_* under {train_dir}"
    cols = [c for c in pq.read_schema(fs[0]).names if c not in DROP]
    Xtr, ytr = load(fs, cols, hp["folds"], frac=hp["frac"], dtype=np.float16)
    Xva, yva = load(fs, cols, (1,), frac=hp["val_frac"])
    info = {"train_pairs": len(ytr), "train_true": float(ytr.mean()), "val_pairs": len(yva), "feats": len(cols),
            "files": len(fs), "load_s": round(time.time() - T0)}
    log(f"loaded {info}")
    s3.put_json(info | {"state": "loaded"}, "status.json")
    d = lgb.Dataset([Seq(Xtr)], label=ytr.astype(np.float32), feature_name=cols, free_raw_data=True)
    dv = lgb.Dataset(Xva, yva.astype(np.float32), reference=d)
    d.construct()
    del Xtr
    log("dataset constructed")

    best = {"i": -1, "v": np.inf, "res": None}
    ck = model_dir / "ckpt.txt"

    def guard(env):
        """Track the best val logloss; checkpoint + status every ckpt_every rounds; stop at the deadline."""
        _, name, v, _ = env.evaluation_result_list[0]
        if v < best["v"]:
            best.update(i=env.iteration, v=v, res=env.evaluation_result_list)
        it = env.iteration + 1
        if it % 250 == 0:
            log(f"iter {it} val_loss={v:.6f} best {best['v']:.6f} @ {best['i'] + 1}")
        if it % hp["ckpt_every"] == 0:
            env.model.save_model(str(ck), num_iteration=best["i"] + 1)
            s3.put(ck, "ckpt/m2_p16_ckpt.txt")
            s3.put_json(info | {"state": "training", "iter": it, "val_loss": v, "best_iter": best["i"] + 1,
                                "best_val_loss": best["v"], "elapsed_s": round(time.time() - T0)}, "status.json")
        if time.time() - T0 > hp["deadline_s"]:
            log(f"deadline {hp['deadline_s']}s reached at iter {it}: keeping best iter {best['i'] + 1}")
            info["deadline_hit"] = it
            raise lgb.callback.EarlyStopException(best["i"], best["res"])

    guard.order = 40                                    # after early_stopping (30)
    m = lgb.train(P, d, hp["rounds"], valid_sets=[dv],
                  callbacks=[lgb.early_stopping(hp["es"], verbose=False), lgb.log_evaluation(250), guard])
    out = model_dir / "m2_p16.txt"
    m.save_model(str(out))
    ck.unlink(missing_ok=True)
    res = info | {"state": "done", "trees": m.num_trees(), "best_iter": m.best_iteration,
                  "best_val_loss": float(m.best_score["valid_0"]["binary_logloss"]),
                  "elapsed_s": round(time.time() - T0), "peak_gb": round(peak(), 2)}
    (model_dir / "result.json").write_text(json.dumps(res))
    s3.put(out, "m2_p16.txt")
    s3.put_json(res, "status.json")
    log(f"done {res}; val_loss={res['best_val_loss']:.6f}")
