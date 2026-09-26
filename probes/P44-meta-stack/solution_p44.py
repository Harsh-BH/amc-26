"""P44-meta-stack SageMaker job: a 2nd, house-number-focused LightGBM on m2_big's folds (train 2-5, early stop on
fold 1's 25% S1 sample, never fold 0 = holdout), then its p on #6's pruned India/US holdout and test pairs.

Shipped as solution.py next to p25.py (P25's job: S3 helper, file globbing) and config.json (the variants; only the
variant NAME comes through hyperparameters.json). Extra channel 'hn' = hn/{split}_{c}_{s1|sx}.parquet (hn_tables.py):
the derived house-number corruption features (HN) are computed per X file from hn1[q1], hnx[qx].
-> model_dir + s3: <name>.txt, pred/{holdout,test}_<c>.parquet (q1, qx, p), status.json, result.json.
    .venv/bin/python probes/P44-meta-stack/solution_p44.py        # self-check of hn_feats
"""
import json, os, time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

T0 = time.time()
C = ("India", "US")
DROP = {"s1", "sx", "q1", "qx", "y", "is_true", "fold", "country"}      # model.DROP
SIB = np.array([1, 2, 3, 4, 5, 7, 9, 11, 13, 21])                     # stack.SIB
SIB3 = np.array([3, 4, 5, 7, 9, 11, 13, 21])                          # stack.SIB3
HN = ["hn_d", "hn_absd", "hn_cls", "hn_sib3", "hn_lev", "hn_lendiff", "hn_len1"]


def log(msg: str):
    import resource
    print(f"[p44 {time.time() - T0:7.0f}s peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:5.1f} GB] "
          f"{msg}", flush=True)


def hn_feats(h1: pa.Array, hx: pa.Array) -> dict:
    """Signed delta, |delta|, stack.hn_cls (0 none/equal, 1 sibling +d, 2 digit drop/add, 3 small other, 4 other),
    SIB3 flag, digit Levenshtein, length delta, S1 hn length. NaN where a side has no house number."""
    from rapidfuzz.distance import Levenshtein
    from rapidfuzz.process import cpdist
    h1, hx = pc.fill_null(h1, ""), pc.fill_null(hx, "")
    l1, lx = (pc.utf8_length(h).to_numpy(zero_copy_only=False).astype(np.int64) for h in (h1, hx))
    both = (l1 > 0) & (lx > 0)
    num = lambda h, l: np.where(l > 0, pc.cast(pc.if_else(pc.equal(h, ""), "0", h), pa.int64()).to_numpy(), 0)
    a, b = num(h1, l1), num(hx, lx)
    d = b - a
    eq = both & (d == 0) & (l1 == lx)
    lo, sh = np.where(l1 > lx, a, b), np.where(l1 > lx, b, a)                 # longer / shorter number
    L, S = np.maximum(l1, lx), np.minimum(l1, lx)
    fix = both & (l1 != lx)
    ps = fix & ((lo % np.power(10, np.clip(S, 0, 18), dtype=np.int64) == sh) |
                (lo // np.power(10, np.clip(L - S, 0, 18), dtype=np.int64) == sh))
    cls = np.where(~both | eq, 0, np.where(np.isin(d, SIB), 1, np.where(ps, 2, np.where(np.abs(d) <= 21, 3, 4))))
    nan = lambda v: np.where(both, v, np.nan).astype(np.float32)
    lev = cpdist(h1.to_pylist(), hx.to_pylist(), scorer=Levenshtein.distance, workers=-1)
    return {"hn_d": nan(np.clip(d, -1000, 1000)), "hn_absd": nan(np.minimum(np.abs(d), 1000)),
            "hn_cls": cls.astype(np.float32), "hn_sib3": (both & np.isin(d, SIB3)).astype(np.float32),
            "hn_lev": nan(lev), "hn_lendiff": nan(lx - l1), "hn_len1": l1.astype(np.float32)}


class HnTab:
    """hn1 / hnx lookups of one split (country -> arrow string arrays in q1 / qx order)."""

    def __init__(self, d: Path, split: str):
        self.t = {c: {s: pq.read_table(Path(d) / f"{split}_{c}_{s}.parquet").column("hn").combine_chunks()
                      for s in ("s1", "sx")} for c in C if (Path(d) / f"{split}_{c}_s1.parquet").exists()}

    def feats(self, c: str, q1: np.ndarray, qx: np.ndarray) -> dict:
        return hn_feats(self.t[c]["s1"].take(pa.array(q1)), self.t[c]["sx"].take(pa.array(qx)))


def matrix(t: pa.Table, cols: list, hn: "HnTab | None", c: str, dtype=np.float32) -> np.ndarray:
    base = [col for col in cols if col not in HN]
    X = np.empty((t.num_rows, len(cols)), dtype)
    extra = hn.feats(c, t.column("q1").to_numpy(), t.column("qx").to_numpy()) if hn and len(base) < len(cols) else {}
    for j, col in enumerate(cols):
        v = extra[col] if col in HN else t.column(col).to_numpy().astype(np.float32)      # nulls -> NaN
        X[:, j] = v.astype(dtype)
    return X


def load(fs: list, cols: list, folds, hn, frac: float = 1.0, dtype=np.float32):
    """P25/P16 load + the derived hn columns: (X, y) of the pairs of S1 in `folds`."""
    base = [col for col in cols if col not in HN]

    def keep(t):
        q1, f = t.column("q1").to_numpy().astype(np.uint64), t.column("fold").to_numpy()
        return np.isin(f, folds) & ((q1 * np.uint64(2654435761)) % np.uint64(1000) < frac * 1000)

    n = sum(int(keep(pq.read_table(p, columns=["q1", "fold"])).sum()) for p in fs)
    X, y, a = np.empty((n, len(cols)), dtype), np.empty(n, bool), 0
    for p in fs:
        t = pq.read_table(p, columns=["q1", "qx", "is_true", "fold"] + base)
        t = t.filter(keep(t))
        X[a:a + len(t)] = matrix(t, cols, hn, p.name.split("_")[1], dtype)
        y[a:a + len(t)] = t.column("is_true").to_numpy()
        a += len(t)
        del t
    assert a == n
    return X, y


def score_split(predict, xs: list, keys_dir: Path, split: str, cols: list, hn, out: Path) -> dict:
    """P22's scorer (every key of keys/<split>_<c>.parquet found exactly once), India / US only."""
    base, res = [col for col in cols if col not in HN], {}
    for c in C:
        kf = Path(keys_dir) / f"{split}_{c}.parquet"
        kt = pq.read_table(kf, columns=["q1", "qx"])
        want = pa.array((kt.column("q1").to_numpy().astype(np.int64) << 32) | kt.column("qx").to_numpy().astype(np.int64))
        parts = []
        for p in (f for f in xs if f.name.split("_")[1] == c):
            t = pq.read_table(p, columns=["q1", "qx"] + base)
            k = (t.column("q1").to_numpy().astype(np.int64) << 32) | t.column("qx").to_numpy().astype(np.int64)
            t = t.filter(pc.is_in(pa.array(k), value_set=want))
            if not t.num_rows:
                continue
            X = matrix(t, cols, hn, c)
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
    from p25 import S3, files
    cfg = json.loads((Path(__file__).parent / "config.json").read_text())
    name = params["name"]
    v = cfg[name]
    assert not {0, 1} & set(v["folds"]), "never train on folds 0 (holdout) / 1 (early stopping)"
    P = dict(cfg["_base"] | v["params"], num_threads=os.cpu_count())
    s3, model_dir = S3(params.get("s3", "")), Path(model_dir)
    hn_dir = Path(os.environ.get("SM_CHANNEL_HN", "/opt/ml/input/data/hn"))
    fs = files(train_dir)
    assert fs, f"no p4/X/train_* under {train_dir}"
    fs = [f for f in fs if f.name.split("_")[1] in C]
    allc = [c for c in pq.read_schema(fs[0]).names if c not in DROP]
    cols = (allc if v["cols"] == "all" else cfg[v["cols"]]) + (HN if v.get("hn") else [])
    log(f"lightgbm {lgb.__version__}, {os.cpu_count()} vCPU, variant {name} {v}, {len(cols)} cols {cols}")
    hn_tr = HnTab(hn_dir, "train")

    class Seq(lgb.Sequence):
        def __init__(self, a):
            self.a, self.batch_size = a, 65_536

        def __getitem__(self, i):
            return self.a[i].astype(np.float64)

        def __len__(self):
            return len(self.a)

    Xtr, ytr = load(fs, cols, v["folds"], hn_tr, frac=v.get("frac", 1.0), dtype=np.float16 if len(cols) > 40 else np.float32)
    Xva, yva = load(fs, cols, (1,), hn_tr, frac=0.25)
    info = {"name": name, "train_pairs": len(ytr), "train_true": float(ytr.mean()), "val_pairs": len(yva),
            "feats": len(cols), "cols": cols, "load_s": round(time.time() - T0)}
    log(f"loaded {info}")
    s3.put_json(info | {"state": "loaded"}, "status.json")
    d = lgb.Dataset([Seq(Xtr)], label=ytr.astype(np.float32), feature_name=cols, free_raw_data=True)
    dv = lgb.Dataset(Xva, yva.astype(np.float32), reference=d)
    d.construct()
    del Xtr, ytr

    def status(env):
        it = env.iteration + 1
        if it % 250 == 0:
            s3.put_json(info | {"state": "training", "iter": it, "val_loss": env.evaluation_result_list[0][2],
                                "elapsed_s": round(time.time() - T0)}, "status.json")

    m = lgb.train(P, d, v["rounds"], valid_sets=[dv],
                  callbacks=[lgb.log_evaluation(100), lgb.early_stopping(v["es"], verbose=False), status])
    del d
    out = model_dir / f"{name}.txt"
    m.save_model(str(out))
    pv = m.predict(Xva, num_threads=os.cpu_count())
    vl = float(-np.mean(np.where(yva, np.log(np.clip(pv, 1e-12, 1)), np.log(np.clip(1 - pv, 1e-12, 1)))))
    imp = dict(zip(cols, m.feature_importance("gain").round(1).tolist()))
    info |= {"trees": m.num_trees(), "best_iter": m.best_iteration, "val_logloss": vl, "gain": imp,
             "train_s": round(time.time() - T0)}
    del Xva, yva, pv
    log(f"trained {info}")
    s3.put(out, out.name)
    s3.put_json(info | {"state": "trained"}, "status.json")
    predict = lambda X: m.predict(X, num_threads=os.cpu_count())
    pred = model_dir / "pred"
    info["scored"] = score_split(predict, fs, Path(test_dir) / "keys", "holdout", cols, hn_tr, pred)
    del hn_tr
    info["scored"] |= score_split(predict, files(test_dir, "test"), Path(test_dir) / "keys", "test", cols,
                                  HnTab(hn_dir, "test"), pred)
    for f in sorted(pred.glob("*.parquet")):
        s3.put(f, f"pred/{f.name}")
    res = info | {"state": "done", "elapsed_s": round(time.time() - T0)}
    (model_dir / "result.json").write_text(json.dumps(res))
    s3.put_json(res, "status.json")
    log(f"done; val_loss={vl:.6f}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code/business_entity_resolution/src"))
    import stack
    h1 = ["10", "10", "10", "10", "", "07", "1795", "12", "5"]
    hx = ["13", "1", "10", "300", "4", "7", "179", "33", "5"]
    f = hn_feats(pa.array(h1), pa.array(hx))
    want = stack.hn_cls(np.array(h1, object), np.array(hx, object))
    assert f["hn_cls"].tolist() == want.tolist(), (f["hn_cls"], want)
    assert f["hn_sib3"].tolist() == [1, 0, 0, 0, 0, 0, 0, 1, 0]
    assert f["hn_d"][0] == 3 and np.isnan(f["hn_d"][4]) and f["hn_lev"][6] == 1 and f["hn_lendiff"][6] == -1
    # the vectorized class equals stack.hn_cls on real pairs too
    hn = HnTab(Path(__file__).resolve().parents[2] / "data/work/p44ms/hn", "test")
    rng = np.random.default_rng(0)
    q1, qx = rng.integers(0, len(hn.t["US"]["s1"]), 200_000), rng.integers(0, len(hn.t["US"]["sx"]), 200_000)
    f = hn.feats("US", q1, qx)
    want = stack.hn_cls(hn.t["US"]["s1"].take(pa.array(q1)).to_numpy(zero_copy_only=False).astype(object),
                        hn.t["US"]["sx"].take(pa.array(qx)).to_numpy(zero_copy_only=False).astype(object))
    assert (f["hn_cls"] == want).all(), np.flatnonzero(f["hn_cls"] != want)[:5]
    print("solution_p44 self-check ok", np.bincount(want, minlength=5))
