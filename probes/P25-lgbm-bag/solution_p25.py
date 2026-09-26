"""P25 SageMaker job (ONE CPU run per LightGBM variant): m2_big's recipe (train_matcher.main) with the variant's
LightGBM params and folds, then its p on #6's pruned holdout and test pairs. Shipped alone as solution.py.

Training = P16's solution_p16.main (float16 lgb.Sequence of every pair of the S1 in hp['folds'], early stopping
hp['es'] rounds on the 25% S1-hash sample of fold 1, deadline-aware). DART has no early stopping in LightGBM: it runs
hp['rounds'] (or to the deadline) and keeps every tree. Scoring = P22's score_split (every key found exactly once).
Channel 'train' = p4/X/train_<c>_sx_NNN.parquet (s3 amc26-p16/data/v1/).
Channel 'test'  = keys/{holdout,test}_<c>.parquet + p4/X/test_<c>_sx_NNN.parquet (s3 amc26-p22/data/test/).
-> model_dir + hp['s3']: <name>.txt (booster), pred/{holdout,test}_<c>.parquet (q1, qx, p), status.json, result.json.
"""
import json, os, resource, time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

T0 = time.time()
ID = ["q1", "qx", "is_true", "fold"]
DROP = {"s1", "sx", "q1", "qx", "y", "is_true", "fold", "country"}      # model.DROP
DEFAULTS = {"name": "var", "folds": [2, 3, 4, 5], "val_frac": 0.25, "frac": 1.0, "rounds": 8000, "es": 100,
            "deadline_s": 8400, "ckpt_every": 1000, "s3": "", "params": None}
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def log(msg: str):
    print(f"[p25 {time.time() - T0:7.0f}s peak {peak():5.1f} GB] {msg}", flush=True)


class S3:
    """Best-effort uploads under s3://bucket/prefix/; '' disables; a failed upload is logged, never fatal."""

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
            except Exception as e:
                log(f"s3 {key} failed: {e!r}")

    def put_json(self, obj: dict, key: str):
        if self.on:
            try:
                self.c.put_object(Bucket=self.bucket, Key=f"{self.prefix}/{key}", Body=json.dumps(obj).encode())
            except Exception as e:
                log(f"s3 {key} failed: {e!r}")


def files(d: Path, split: str = "train") -> list:
    return sorted(Path(d).glob(f"p4/X/{split}_*_sx_*.parquet"))


def load(fs: list, cols: list, folds, frac: float = 1.0, seed: int = 0, dtype=np.float32):
    """train_matcher.load on pyarrow (P16): (X [n, len(cols)], y) of the pairs of S1 in `folds`, files in order."""
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


def score_split(predict, xs: list, keys_dir: Path, split: str, cols: list, out: Path) -> dict:
    """P22's: p of every pair in keys/<split>_<c>.parquet from that country's X files -> out/<split>_<c>.parquet."""
    res = {}
    for kf in sorted(Path(keys_dir).glob(f"{split}_*.parquet")):
        c = kf.stem.split("_", 1)[1]
        kt = pq.read_table(kf, columns=["q1", "qx"])
        want = pa.array((kt.column("q1").to_numpy().astype(np.int64) << 32) | kt.column("qx").to_numpy().astype(np.int64))
        parts = []
        for p in (f for f in xs if f.name.split("_")[1] == c):
            t = pq.read_table(p, columns=["q1", "qx"] + cols)
            k = (t.column("q1").to_numpy().astype(np.int64) << 32) | t.column("qx").to_numpy().astype(np.int64)
            t = t.filter(pc.is_in(pa.array(k), value_set=want))
            if not t.num_rows:
                continue
            X = np.column_stack([t.column(col).to_numpy().astype(np.float32) for col in cols])
            parts.append(pa.table({"q1": t.column("q1").cast(pa.int32()), "qx": t.column("qx").cast(pa.int32()),
                                   "p": pa.array(predict(X).astype(np.float32))}))
            del t, X
        tab = pa.concat_tables(parts)
        assert tab.num_rows == len(want), f"{split} {c}: {tab.num_rows} scored of {len(want)} keys"
        out.mkdir(parents=True, exist_ok=True)
        pq.write_table(tab, out / f"{split}_{c}.parquet")
        res[f"{split}_{c}"] = tab.num_rows
        log(f"scored {split} {c}: {tab.num_rows:,} pairs")
    return res


def main(train_dir, test_dir, model_dir, out_dir, params):
    import lightgbm as lgb
    hp = DEFAULTS | {k: v for k, v in params.items() if k in DEFAULTS}
    assert hp["params"], "pass params = the variant's LightGBM params"
    assert not {0, 1} & set(hp["folds"]), "never train on folds 0 (holdout) / 1 (early stopping)"
    P = dict(hp["params"], num_threads=os.cpu_count())
    dart = P.get("boosting") == "dart"
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
    info = {"name": hp["name"], "train_pairs": len(ytr), "train_true": float(ytr.mean()), "val_pairs": len(yva),
            "feats": len(cols), "files": len(fs), "load_s": round(time.time() - T0)}
    log(f"loaded {info}")
    s3.put_json(info | {"state": "loaded"}, "status.json")
    d = lgb.Dataset([Seq(Xtr)], label=ytr.astype(np.float32), feature_name=cols, free_raw_data=True)
    dv = lgb.Dataset(Xva, yva.astype(np.float32), reference=d)
    d.construct()
    del Xtr, ytr
    log("dataset constructed")

    best = {"i": -1, "v": np.inf, "res": None}

    def guard(env):
        """Best val logloss; status every 250 rounds; stop at the deadline (DART keeps every tree so far)."""
        _, _, v, _ = env.evaluation_result_list[0]
        if v < best["v"]:
            best.update(i=env.iteration, v=v, res=env.evaluation_result_list)
        it = env.iteration + 1
        if it % 250 == 0:
            log(f"iter {it} val_loss={v:.6f} best {best['v']:.6f} @ {best['i'] + 1}")
            s3.put_json(info | {"state": "training", "iter": it, "val_loss": v, "best_iter": best["i"] + 1,
                                "best_val_loss": best["v"], "elapsed_s": round(time.time() - T0)}, "status.json")
        if time.time() - T0 > hp["deadline_s"]:
            keep = env.iteration if dart else best["i"]
            log(f"deadline {hp['deadline_s']}s reached at iter {it}: keeping iter {keep + 1}")
            info["deadline_hit"] = it
            raise lgb.callback.EarlyStopException(keep, env.evaluation_result_list if dart else best["res"])

    guard.order = 40                                    # after early_stopping (30)
    cbs = [lgb.log_evaluation(250), guard] + ([] if dart else [lgb.early_stopping(hp["es"], verbose=False)])
    m = lgb.train(P, d, hp["rounds"], valid_sets=[dv], callbacks=cbs)
    del d
    out = model_dir / f"{hp['name']}.txt"
    m.save_model(str(out))                              # best_iteration (early stop / deadline), else every tree
    pv = m.predict(Xva, num_threads=os.cpu_count())
    vl = float(-np.mean(np.where(yva, np.log(np.clip(pv, 1e-12, 1)), np.log(np.clip(1 - pv, 1e-12, 1)))))
    info |= {"trees": m.num_trees(), "best_iter": m.best_iteration, "best_val_loss": best["v"],
             "val_logloss_saved": vl, "train_s": round(time.time() - T0)}
    del Xva, yva, pv
    log(f"trained {info}")
    s3.put(out, out.name)
    s3.put_json(info | {"state": "trained"}, "status.json")
    predict = lambda X: m.predict(X, num_threads=os.cpu_count())
    pred = model_dir / "pred"
    info["scored"] = score_split(predict, fs, Path(test_dir) / "keys", "holdout", cols, pred)
    info["scored"] |= score_split(predict, files(test_dir, "test"), Path(test_dir) / "keys", "test", cols, pred)
    for f in sorted(pred.glob("*.parquet")):
        s3.put(f, f"pred/{f.name}")
    res = info | {"state": "done", "elapsed_s": round(time.time() - T0), "peak_gb": round(peak(), 2)}
    (model_dir / "result.json").write_text(json.dumps(res))
    s3.put_json(res, "status.json")
    log(f"done {res}; val_loss={vl:.6f}")
