"""P59 train channel: P13's CE recipe on folds 1-9 (fold 0 = the #8c holdout, never touched).

    build59.py band <c>     # folds 1-2 band: P4b's quick LightGBM, fitted on a 30% sample of fold 3 (P4b fit fold 2,
                            # which cannot label fold 2) -> data/work/p59/trainband12_<c>.parquet
    build59.py chan         # + P4b trainband (folds 3-9): val = 20k fold-1 band pairs, train = the rest with P13's
                            # sampling (band + 0.5x as many out-of-band) -> data/work/p59/sm_data/{train,val}_<c>.parquet
"""
import gc, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "probes/P4b-rerank")]
import rerank as rr

W = ROOT / "data/work"
OUT = W / "p59"
DATA = OUT / "sm_data"
ID = rr.ID


def band(c: str, frac: float = 0.3, seed: int = 0):
    import lightgbm as lgb
    t0, rng = time.time(), np.random.default_rng(seed)
    files = sorted((W / "p4/X").glob(f"train_{c}_sx_*.parquet"))
    cols = [x for x in pq.read_schema(files[0]).names if x not in ID]
    parts = []
    for p in files:
        d = pq.read_table(p, columns=cols + ["is_true"], filters=[("fold", "==", 3)]).to_pandas()
        parts.append(d[rng.random(len(d)) < frac])
        del d
    d = pd.concat(parts, ignore_index=True)
    del parts
    X, y = d[cols].to_numpy(np.float32), d.is_true.to_numpy(np.float32)
    del d
    gc.collect()
    m = lgb.train(dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100,
                       feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=6),
                  lgb.Dataset(X, y, feature_name=cols, free_raw_data=True), 300)
    print(f"{c}: fit on {len(y):,} fold-3 pairs {time.time() - t0:.0f}s, peak {rr.peak():.2f} GB", flush=True)
    del X, y
    gc.collect()
    out = []
    for p in files:
        d = pq.read_table(p, filters=[("fold", "<=", 2), ("fold", ">=", 1)]).to_pandas()
        pr = m.predict(d[cols].to_numpy(np.float32), num_threads=6).astype(np.float32)
        inb = (pr > rr.LO) & (pr < rr.HI)
        keep = inb | (rng.random(len(d)) < 0.05)
        out.append(d.loc[keep, ID].assign(p=pr[keep], inband=inb[keep]))
        del d, pr
        gc.collect()
    b = pd.concat(out, ignore_index=True)
    b.to_parquet(OUT / f"trainband12_{c}.parquet", index=False)
    ib = b[b.inband]
    print(f"{c}: {len(ib):,} band pairs (true {ib.is_true.mean():.3f}), {len(b) - len(ib):,} out-of-band, "
          f"{time.time() - t0:.0f}s, peak {rr.peak():.2f} GB", flush=True)
    print(b.groupby("fold").agg(n=("p", "size"), inb=("inband", "sum"), pos=("is_true", "sum")), flush=True)


def chan():
    DATA.mkdir(parents=True, exist_ok=True)
    for c in ("India", "US"):
        b = pd.concat([pd.read_parquet(OUT / f"trainband12_{c}.parquet"), pd.read_parquet(W / "p4b" / f"trainband_{c}.parquet")],
                      ignore_index=True)
        assert (b.fold >= 1).all() and b.fold.between(1, 9).all()
        v = b[(b.fold == 1) & b.inband].sample(20_000, random_state=0)
        b = b.drop(v.index)
        ib, ob = b[b.inband], b[~b.inband]
        b = pd.concat([ib, ob.sample(min(len(ob), int(0.5 * len(ib))), random_state=0)])
        print(f"{c}: train {len(b):,} (band {len(ib):,}, true {b.is_true.mean():.3f}, folds "
              f"{sorted(b.fold.unique().tolist())}), val {len(v):,}", flush=True)
        for nm, d in (("train", b), ("val", v)):
            ta, tb = rr.texts("train", c, d.q1.to_numpy(), d.qx.to_numpy())
            f = DATA / f"{nm}_{c}.parquet"
            pd.DataFrame({"ta": ta, "tb": tb, "y": d.is_true.to_numpy(np.int8), "p": d.p.to_numpy(np.float32)}).to_parquet(
                f, index=False, compression="zstd")
            print(f"{f.name}: {len(d):,} rows, {f.stat().st_size / 1e6:.1f} MB, peak {rr.peak():.2f} GB", flush=True)
            del ta, tb
        del b, v, ib, ob
        gc.collect()


if __name__ == "__main__":
    {"band": band, "chan": chan}[sys.argv[1]](*sys.argv[2:])
