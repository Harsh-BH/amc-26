"""P35: fit the address-less group-decision GBDT on #8c holdout half h, report the other half (paired per-S1 F0.5 vs #8c).

    .venv/bin/python probes/P35-twin-propagate/fit.py feats            # cache data/work/p35/gx_<c>.parquet (~2 min)
    .venv/bin/python probes/P35-twin-propagate/fit.py eval [arm ...]   # arms: see ARMS
"""
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from ev import W, load, f_of  # noqa: E402
from gdec import FEATS, decide, features  # noqa: E402

T = 0.71
LBW = {"India": 0.468, "US": 0.383}
PARAMS = dict(objective="binary", num_leaves=31, learning_rate=0.05, min_data_in_leaf=200, lambda_l2=1.0, verbose=-1,
              num_threads=4, seed=0)
TWIN = ["tw_p", "tw_sim", "n_tw", "tw_x", "tw_same"]
ARMS = {"full": FEATS, "notwin": [f for f in FEATS if f not in TWIN],
        "twin_only": ["lp", "rank", "margin", "n_cand"] + TWIN, "ctx_only": ["lp", "rank", "margin", "n_cand"],
        # no competitor-p columns (margin / rank / x_best lean on e5-blended competitor rows on holdout only)
        "nomargin": [f for f in FEATS if f not in ("margin", "rank", "x_best")],
        "nomargin_notwin": [f for f in FEATS if f not in ("margin", "rank", "x_best") + tuple(TWIN)]}


def core1(c, ids):
    v = pa.array(sorted(ids), pa.large_string()); f = pq.ParquetFile(W / f"rf_train_s1_{c}.parquet"); out = []
    for i in range(f.num_row_groups):
        t = f.read_row_group(i, columns=["id", "core"])
        out.append(t.filter(pc.is_in(t["id"].cast(pa.large_string()), value_set=v)).to_pandas()); del t
    return pd.concat(out).set_index("id").core


def feats():
    for c in ("India", "US"):
        t0 = time.time()
        d = pd.read_parquet(W / f"p35/ho_{c}.parquet", columns=["s1", "sx", "p", "noaddr", "core", "src"])
        x = features(d, core1(c, set(pd.unique(d.s1))), T)
        x.drop(columns=["s1", "sx", "p", "src"]).to_parquet(W / f"p35/gx_{c}.parquet")
        print(c, len(x), f"{time.time() - t0:.0f}s", flush=True)


def load_all():
    S = []
    for c in ("India", "US"):
        s = load(c, ["s1", "sx", "holdout", "y", "p", "pred", "noaddr"])
        s["x"] = pd.read_parquet(W / f"p35/gx_{c}.parquet")
        g = s["g"][s["x"].index.to_numpy()]
        s["xh"] = np.where(g >= 0, s["h"][np.maximum(g, 0)].astype(np.int64), -1)            # half of the row's S1, -1 = not holdout
        S.append(s)
    return S


def fit(S, cols, fit_half, rounds=200):
    X = pd.concat([s["x"].loc[s["xh"] == fit_half, cols] for s in S])
    y = np.concatenate([s["d"].y.to_numpy()[s["x"].index.to_numpy()][s["xh"] == fit_half] for s in S])
    b = lgb.train(PARAMS, lgb.Dataset(X, y, init_score=X.lp.to_numpy()), rounds)
    return b, [1 / (1 + np.exp(-(b.predict(s["x"][cols], raw_score=True) + s["x"].lp.to_numpy()))) for s in S]


def evaluate(S, ps, fit_half, t):
    rep, out = 1 - fit_half, {}
    for s, p in zip(S, ps):
        pred = decide(s["d"], s["x"], p, t)
        dl = f_of(s, pred) - s["f0"]
        x = dl[s["h"] == rep]; xf = dl[s["h"] == fit_half]
        base = s["d"].pred.to_numpy()
        out[s["c"]] = dict(d=float(x.mean()), se=float(x.std(ddof=1) / np.sqrt(len(x))), d_fit=float(xf.mean()),
                           added=int((pred & ~base & (s["g"] >= 0)).sum()), dropped=int((base & ~pred & (s["g"] >= 0)).sum()),
                           dl=x)
    pool = np.concatenate([out[c]["dl"] for c in LBW])
    out["pooled"] = dict(d=float(pool.mean()), se=float(pool.std(ddof=1) / np.sqrt(len(pool))))
    out["lb"] = sum(LBW[c] * out[c]["d"] for c in LBW)
    out["lb_se"] = float(np.sqrt(sum((LBW[c] * out[c]["se"]) ** 2 for c in LBW)))
    return out


def show(name, fh, t, r):
    print(f"{name:10s} fit h{fh} t {t:.2f}: pooled {r['pooled']['d']:+.6f} ± {r['pooled']['se']:.6f} | LB-eq {r['lb']:+.6f} "
          f"± {r['lb_se']:.6f} | " + " ".join(f"{c} {r[c]['d']:+.6f} ± {r[c]['se']:.6f} (fit-half {r[c]['d_fit']:+.6f}, "
                                              f"+{r[c]['added']}/-{r[c]['dropped']})" for c in LBW), flush=True)


def main(arms):
    S = load_all()
    res = {}
    for name in arms:
        for fh in (0, 1):
            b, ps = fit(S, ARMS[name], fh)
            # t tuned on the fit half (pooled), then reported on the other half
            best = max(np.round(np.arange(0.55, 0.91, 0.04), 2),
                       key=lambda t: np.mean(np.concatenate([
                           (f_of(s, decide(s["d"], s["x"], p, t)) - s["f0"])[s["h"] == fh] for s, p in zip(S, ps)])))
            for t in sorted({T, float(best)}):
                r = evaluate(S, ps, fh, t)
                show(name + ("*" if t == best else ""), fh, t, r)
                res[(name, fh, t)] = r
            if name == "full" and fh == 0:
                imp = pd.Series(b.feature_importance("gain"), index=ARMS[name]).sort_values(ascending=False)
                print("   gain importance:", (imp / imp.sum()).round(3).to_dict(), flush=True)
    return res


if __name__ == "__main__":
    if sys.argv[1] == "feats":
        feats()
    else:
        main(sys.argv[2:] or ["full"])
