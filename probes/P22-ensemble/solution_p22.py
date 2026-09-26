"""P22 SageMaker job (one CPU run per model family): a second GBDT family on m2_big's data -- XGBoost (hist) or
CatBoost -- then its p on the pruned holdout and test pairs. Shipped alone as solution.py next to sm_train.py.

Data = m2_big's: every pair of the S1 in hp['folds'] (2-5), the 65 X features (model.DROP excluded), early stopping
(100 rounds, at most hp['rounds']) on the 25% S1-hash sample of fold 1. load() is P16's (= train_matcher.load).
Channel 'train' = data/work layout p4/X/train_<c>_sx_NNN.parquet (the S3 copy under amc26-p16/data/v1/).
Channel 'test'  = keys/{holdout,test}_<c>.parquet (q1, qx of #6's pruned pairs: the m2_big p9 caches) and
                  p4/X/test_<c>_sx_NNN.parquet. After training, every kept pair is scored:
                  -> pred/{holdout,test}_<c>.parquet (q1, qx, p) in model_dir and hp['s3'].
Deadline-aware like P16: past hp['deadline_s'] training stops and the best iteration so far is kept.
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
XGB = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", max_depth=8, eta=0.05,
           subsample=0.8, colsample_bytree=0.8, min_child_weight=1, reg_lambda=1.0, max_bin=256)
CAT = dict(loss_function="Logloss", depth=8, learning_rate=0.08, l2_leaf_reg=3, border_count=254,
           boosting_type="Plain", bootstrap_type="Bernoulli", subsample=0.8, rsm=0.8, verbose=250)
DEFAULTS = {"model": "xgb", "folds": [2, 3, 4, 5], "val_frac": 0.25, "frac": 1.0, "rounds": 8000, "es": 100,
            "deadline_s": 9000, "ckpt_every": 500, "s3": "", "params": {}}
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def log(msg: str):
    print(f"[p22 {time.time() - T0:7.0f}s peak {peak():5.1f} GB] {msg}", flush=True)


class S3:
    """Best-effort uploads under s3://bucket/prefix/; '' disables. A failed upload is logged, never fatal (every
    file also lands in model_dir -> model.tar.gz)."""

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
    """P16's load (= train_matcher.load): (X [n, len(cols)], y) of the pairs of S1 in `folds`, files in order."""
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


def country_of(p: Path) -> str:
    return p.name.split("_")[1]                      # <split>_<country>_sx_NNN.parquet


def score_split(predict, xs: list, keys_dir: Path, split: str, cols: list, out: Path) -> dict:
    """p of every pair in keys/<split>_<c>.parquet, read from the X files of that country -> out/<split>_<c>.parquet
    (q1, qx, p; X-file order). Asserts every key was found exactly once."""
    res = {}
    for kf in sorted(Path(keys_dir).glob(f"{split}_*.parquet")):
        c = kf.stem.split("_", 1)[1]
        kt = pq.read_table(kf, columns=["q1", "qx"])
        want = pa.array((kt.column("q1").to_numpy().astype(np.int64) << 32) | kt.column("qx").to_numpy().astype(np.int64))
        parts = []
        for p in (f for f in xs if country_of(f) == c):
            t = pq.read_table(p, columns=["q1", "qx"] + cols)
            k = (t.column("q1").to_numpy().astype(np.int64) << 32) | t.column("qx").to_numpy().astype(np.int64)
            t = t.filter(pc.is_in(pa.array(k), value_set=want))
            if not t.num_rows:
                continue
            X = np.column_stack([t.column(col).to_numpy().astype(np.float32) for col in cols])
            parts.append(pa.table({"q1": t.column("q1").cast(pa.int32()), "qx": t.column("qx").cast(pa.int32()),
                                   "p": pa.array(predict(X).astype(np.float32))}))
            del t, X
        tab = pa.concat_tables(parts) if parts else pa.table({"q1": pa.array([], pa.int32()),
                                                               "qx": pa.array([], pa.int32()), "p": pa.array([], pa.float32())})
        assert tab.num_rows == len(want), f"{split} {c}: {tab.num_rows} scored of {len(want)} keys"
        out.mkdir(parents=True, exist_ok=True)
        pq.write_table(tab, out / f"{split}_{c}.parquet")
        res[f"{split}_{c}"] = tab.num_rows
        log(f"scored {split} {c}: {tab.num_rows:,} pairs")
    return res


def train_xgb(Xtr, ytr, Xva, yva, cols, hp, s3, model_dir, info):
    import xgboost as xgb
    P = XGB | dict(hp["params"]) | {"nthread": os.cpu_count()}
    d = xgb.QuantileDMatrix(Xtr, ytr.astype(np.float32), feature_names=cols, max_bin=P["max_bin"])
    dv = xgb.QuantileDMatrix(Xva, yva.astype(np.float32), feature_names=cols, ref=d)
    log(f"xgboost {xgb.__version__} DMatrix built; params {P}")
    ck = model_dir / "ckpt.ubj"

    class Guard(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            v = evals_log["val"]["logloss"][-1]
            it = epoch + 1
            if it % hp["ckpt_every"] == 0:
                model.save_model(str(ck))
                s3.put(ck, "ckpt/model_ckpt.ubj")
                s3.put_json(info | {"state": "training", "iter": it, "val_loss": v,
                                    "best_val_loss": float(min(evals_log["val"]["logloss"])),
                                    "elapsed_s": round(time.time() - T0)}, "status.json")
            if time.time() - T0 > hp["deadline_s"]:
                log(f"deadline {hp['deadline_s']}s reached at iter {it}")
                info["deadline_hit"] = it
                return True
            return False

    es = xgb.callback.EarlyStopping(rounds=hp["es"], metric_name="logloss", data_name="val", save_best=True)
    m = xgb.train(P, d, hp["rounds"], evals=[(dv, "val")], verbose_eval=250, callbacks=[es, Guard()])
    del d, dv
    out = model_dir / "xgb.ubj"
    m.save_model(str(out))                        # save_best=True: already cut at the best iteration
    res = {"trees": m.num_boosted_rounds(), "best_iter": m.num_boosted_rounds(), "best_val_loss": float(m.best_score)}
    predict = lambda X: m.inplace_predict(X)
    return out, res, predict


def train_cat(Xtr, ytr, Xva, yva, cols, hp, s3, model_dir, info):
    import catboost as cb
    P = CAT | dict(hp["params"]) | {"thread_count": os.cpu_count(), "iterations": hp["rounds"],
                                    "od_type": "Iter", "od_wait": hp["es"], "use_best_model": True,
                                    "train_dir": str(model_dir / "catboost_info"), "allow_writing_files": False}
    d = cb.Pool(Xtr, ytr.astype(np.int8), feature_names=cols)
    dv = cb.Pool(Xva, yva.astype(np.int8), feature_names=cols)
    log(f"catboost {cb.__version__} pools built; params {P}")

    class Guard:
        def after_iteration(self, i):
            it = i.iteration + 1
            v = i.metrics["validation"]["Logloss"][-1]
            if it % hp["ckpt_every"] == 0:
                s3.put_json(info | {"state": "training", "iter": it, "val_loss": v,
                                    "best_val_loss": float(min(i.metrics["validation"]["Logloss"])),
                                    "elapsed_s": round(time.time() - T0)}, "status.json")
            if time.time() - T0 > hp["deadline_s"]:
                log(f"deadline {hp['deadline_s']}s reached at iter {it}")
                info["deadline_hit"] = it
                return False
            return True

    m = cb.CatBoostClassifier(**P)
    m.fit(d, eval_set=dv, callbacks=[Guard()])
    del d, dv
    out = model_dir / "cat.cbm"
    m.save_model(str(out))
    res = {"trees": m.tree_count_, "best_iter": int(m.get_best_iteration()) + 1,
           "best_val_loss": float(m.get_best_score()["validation"]["Logloss"])}
    predict = lambda X: m.predict(X, prediction_type="Probability", thread_count=os.cpu_count())[:, 1]
    return out, res, predict


def main(train_dir, test_dir, model_dir, out_dir, params):
    hp = DEFAULTS | {k: v for k, v in params.items() if k in DEFAULTS}
    assert hp["model"] in ("xgb", "cat"), hp["model"]
    assert not {0, 1} & set(hp["folds"]), "never train on folds 0 (holdout) / 1 (early stopping)"
    s3, model_dir = S3(hp["s3"]), Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    log(f"{os.cpu_count()} vCPU, hp {hp}")
    fs = files(train_dir)
    assert fs, f"no p4/X/train_* under {train_dir}"
    cols = [c for c in pq.read_schema(fs[0]).names if c not in DROP]
    Xtr, ytr = load(fs, cols, hp["folds"], frac=hp["frac"])
    Xva, yva = load(fs, cols, (1,), frac=hp["val_frac"])
    info = {"model": hp["model"], "train_pairs": len(ytr), "train_true": float(ytr.mean()), "val_pairs": len(yva),
            "feats": len(cols), "files": len(fs), "load_s": round(time.time() - T0)}
    log(f"loaded {info}")
    s3.put_json(info | {"state": "loaded"}, "status.json")
    fit = train_xgb if hp["model"] == "xgb" else train_cat
    out, res, predict = fit(Xtr, ytr, Xva, yva, cols, hp, s3, model_dir, info)
    del Xtr, ytr
    info |= res | {"train_s": round(time.time() - T0)}
    log(f"trained {info}")
    s3.put(out, out.name)
    s3.put_json(info | {"state": "trained"}, "status.json")
    pv = predict(Xva)                                # sanity: the val logloss from our own predict path
    info["val_logloss_check"] = float(-np.mean(np.where(yva, np.log(np.clip(pv, 1e-12, 1)), np.log(np.clip(1 - pv, 1e-12, 1)))))
    del Xva, yva
    pred = model_dir / "pred"
    info["scored"] = score_split(predict, fs, Path(test_dir) / "keys", "holdout", cols, pred)
    info["scored"] |= score_split(predict, files(test_dir, "test"), Path(test_dir) / "keys", "test", cols, pred)
    for f in sorted(pred.glob("*.parquet")):
        s3.put(f, f"pred/{f.name}")
    res = info | {"state": "done", "elapsed_s": round(time.time() - T0), "peak_gb": round(peak(), 2)}
    (model_dir / "result.json").write_text(json.dumps(res))
    s3.put_json(res, "status.json")
    log(f"done {res}; val_loss={res['best_val_loss']:.6f}")
