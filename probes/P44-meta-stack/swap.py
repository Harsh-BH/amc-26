"""P44: m2hn in place of m2_big under #8c's CE blend (every holdout-world row, argmax recomputed from each p).

#8c = run.rerank with CE_MISSING=fallback: band (0.02 < p < 0.98) pairs with a bge r get the bge blend (w), the rest of
the band gets #7's e5 blend (W7), then P17's stacker for US. Self-check: that rebuild from m2_big reproduces P23's p.
Variants swap p_m2 -> p_m2hn with w refit on the fit half's band rows (P13's refit_w), US with / without P17's stack.
Band pairs of m2hn that have no CE score keep p_m2hn (rerank.blend's NaN rule).
    probes/P44-meta-stack/py.sh probes/P44-meta-stack/swap.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ev as E  # noqa: E402
import a10  # noqa: E402

W8 = [0.5901490903244407, 0.6319357246516834, 0.29732167912424273]      # run.RECIPES sub8c w (P22 ev.W8)
W7 = [0.617, 0.666, 0.261]                                              # run.FALLBACK w (#7's e5 CE)
BGE, E5 = "3bd0b930", "bb0c29f0"
STACK = E.W / "p17/stack_sup-bge-l21.0_US.txt"
LO, HI = 0.02, 0.98
GRID = np.round(np.arange(0.55, 0.86, 0.01), 2)


def lookup(f: Path, k: np.ndarray) -> np.ndarray:
    t = pd.read_parquet(f)
    s = pd.Series(t.r.to_numpy(np.float32), index=t.k.to_numpy(np.int64))
    return s[~s.index.duplicated()].reindex(k).to_numpy(np.float32)


def frame(c: str) -> dict:
    import features as F
    from solution_p44 import HnTab
    d = pd.read_parquet(E.W / f"p23/pairs_{c}.parquet", columns=["s1", "sx", "holdout", "y", "p_m2", "r", "p"])
    id1, idx = F.ids("train", c)
    q1 = pc.index_in(pa.array(d.s1.to_numpy(object), pa.string()), value_set=id1).to_numpy(zero_copy_only=False)
    qx = pc.index_in(pa.array(d.sx.to_numpy(object), pa.string()), value_set=idx).to_numpy(zero_copy_only=False)
    del id1, idx
    q1, qx = q1.astype(np.int64), qx.astype(np.int64)
    k = E.keys(q1, qx)
    t = pq.read_table(E.OUT / f"pred_m2hn/holdout_{c}.parquet")
    s = pd.Series(t.column("p").to_numpy(), index=E.keys(t.column("q1").to_numpy(), t.column("qx").to_numpy()))
    hn = HnTab(E.W / "p44ms/hn", "train")
    S = pd.read_parquet(E.OUT / f"S1_{c}.parquet")
    ctx = pd.read_parquet(E.W / f"p23/s1ctx_{c}.parquet", columns=["n_true"])
    g = pd.Series(np.arange(len(ctx)), index=ctx.index).reindex(d.s1).to_numpy()
    g = np.where(d.holdout.to_numpy() & ~np.isnan(g), np.nan_to_num(g), -1).astype(np.int64)
    return {"q1": q1, "qx": qx, "y": d.y.to_numpy(), "g": g, "p_m2": d.p_m2.to_numpy(np.float64),
            "r_bge": lookup(E.W / f"p9/r_holdout_{c}_{BGE}.parquet", k), "r_e5": lookup(E.W / f"p9/r_holdout_{c}_{E5}.parquet", k),
            "p8": d.p.to_numpy(np.float64), "p_m2hn": s.reindex(k).to_numpy(np.float64),
            "hn1": hn.t[c]["s1"].take(pa.array(q1)).to_numpy(zero_copy_only=False).astype(object),
            "hnx": hn.t[c]["sx"].take(pa.array(qx)).to_numpy(zero_copy_only=False).astype(object), "S": S}


def blended(d, p, w):
    import rerank as RR
    b = (p > LO) & (p < HI)
    out = RR.blend(p, np.where(b, d["r_bge"], np.nan).astype(np.float32), w, LO, HI)
    f = b & np.isnan(d["r_bge"])
    out[f] = RR.blend(p[f], d["r_e5"][f], W7, LO, HI)
    return out.astype(np.float64)


def stacked(d, p, booster=None):
    import lightgbm as lgb
    import stack as S
    return S.stacked(booster or lgb.Booster(model_file=str(STACK)), S.X_of(d["q1"], p, d["hn1"], d["hnx"])).astype(np.float64)


def fit_w(ds, key, fit):
    import rerank as RR
    xs = []
    for c, d in ds.items():
        p = d[key]
        m = (p > LO) & (p < HI) & ~np.isnan(d["r_bge"]) & (d["g"] >= 0)
        m[m] &= d["S"].half.to_numpy()[d["g"][m]] == fit
        xs.append((p[m], d["r_bge"][m], d["y"][m]))
    return list(RR.fit_blend(*(np.concatenate(z) for z in zip(*xs))))


def f_s1(d, p, t):
    import model as M
    am = M.argmax_sx(d["qx"], p)
    acc = am & (p >= t) & (d["g"] >= 0)
    g, n = d["g"], len(d["S"])
    return a10.fb(np.bincount(g[acc & d["y"]], minlength=n), np.bincount(g[acc], minlength=n), d["S"].n_true.to_numpy())


def main():
    ds = {c: frame(c) for c in E.C}
    for c, d in ds.items():
        p = blended(d, d["p_m2"], W8)
        if c == "US":
            p = stacked(d, p)
        dd = np.abs(p - d["p8"])
        f = f_s1(d, p, E.T)
        print(c, f"#8c rebuild: max |dp| {dd.max():.2e}, F0.5 all {f.mean():.6f} (P23 f8 {d['S'].f8.mean():.6f}); "
                 f"m2hn band pairs without a CE score: "
                 f"{int(((d['p_m2hn'] > LO) & (d['p_m2hn'] < HI) & np.isnan(d['r_bge']) & np.isnan(d['r_e5'])).sum()):,}",
              flush=True)
        assert np.allclose(f, d["S"].f8.to_numpy()), "#8c does not reproduce"
    out = []
    for fit in (0, 1):
        rep = 1 - fit
        w = fit_w(ds, "p_m2hn", fit)
        w8 = fit_w(ds, "p_m2", fit)
        print(f"fit h{fit}: w m2hn {np.round(w, 3).tolist()}, w m2 refit {np.round(w8, 3).tolist()}", flush=True)
        for c, d in ds.items():
            S = d["S"]
            hm = S.half.to_numpy()
            V = {"m2_refitw": blended(d, d["p_m2"], w8), "m2hn_blend": blended(d, d["p_m2hn"], w)}
            if c == "US":
                V["m2_refitw"] = stacked(d, V["m2_refitw"])
                V["m2hn_blend+p17"] = stacked(d, V["m2hn_blend"])
            for k, p in V.items():
                for tt in ("t071", "tuned"):
                    t = E.T if tt == "t071" else float(GRID[int(np.argmax([f_s1(d, p, x)[hm == fit].mean() for x in GRID]))])
                    f = f_s1(d, p, t)
                    d8, d34 = (f - S.f8.to_numpy())[hm == rep], (f - S.f34.to_numpy())[hm == rep]
                    row = {"fit": fit, "c": c, "v": k, "tt": tt, "t": t, "f_rep": f[hm == rep].mean(),
                           "d8": d8.mean(), "se8": d8.std(ddof=1) / np.sqrt(len(d8)),
                           "d34": d34.mean(), "se34": d34.std(ddof=1) / np.sqrt(len(d34))}
                    out.append(row)
                    print(f"  {c:5s} h{fit}->h{rep} {k:15s} {tt} t {t:.2f}: F0.5 {row['f_rep']:.6f} | vs #8c "
                          f"{row['d8']:+.6f} ± {row['se8']:.6f} | vs #34 {row['d34']:+.6f} ± {row['se34']:.6f}", flush=True)
    (E.OUT / "swap.json").write_text(json.dumps(out, default=float, indent=0))


if __name__ == "__main__":
    main()
