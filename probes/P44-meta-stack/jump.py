"""P44: does the report's "m2 alone -> meta" jump (0.9903 -> 0.9924) exist on OUR holdout?

Every row of P23's holdout frame (holdout S1 + competing train S1), each score decides on its own: am = argmax over the
SX's rows, acc = am & s >= t & holdout S1, t per country tuned on the fit half. Scores: m2_big alone, the P44 model
alone, the report's logistic on [p_m2, p_hn, |diff|, min, max, has_hn, exact_hn] fit on the fit half's holdout rows
(every candidate row, not only argmax pairs), and #8c's p as the reference. Absolute F0.5 on the report half + delta
vs m2 alone and vs #34.
    probes/P44-meta-stack/py.sh probes/P44-meta-stack/jump.py hn21
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

GRID = np.round(np.arange(0.30, 0.961, 0.01), 2)


def frame(c: str, m: str) -> tuple:
    import features as F
    import model as M
    from solution_p44 import HnTab
    d = pd.read_parquet(E.W / f"p23/pairs_{c}.parquet", columns=["s1", "sx", "holdout", "y", "p_m2", "p"])
    id1, idx = F.ids("train", c)
    q1 = pc.index_in(pa.array(d.s1.to_numpy(object), pa.string()), value_set=id1).to_numpy(zero_copy_only=False)
    qx = pc.index_in(pa.array(d.sx.to_numpy(object), pa.string()), value_set=idx).to_numpy(zero_copy_only=False)
    del id1, idx
    q1, qx = q1.astype(np.int64), qx.astype(np.int64)
    t = pq.read_table(E.OUT / f"pred_{m}/holdout_{c}.parquet")
    s = pd.Series(t.column("p").to_numpy(), index=E.keys(t.column("q1").to_numpy(), t.column("qx").to_numpy()))
    d["p_hn"] = s.reindex(E.keys(q1, qx)).to_numpy(np.float64)
    assert d.p_hn.notna().all()
    f = HnTab(E.W / "p44ms/hn", "train").feats(c, q1, qx)
    d["has_hn"] = np.isfinite(f["hn_lev"]).astype(float)
    d["exact_hn"] = ((f["hn_cls"] == 0) & (d.has_hn > 0)).astype(float)
    d["qx"] = pd.factorize(qx)[0]
    S = pd.read_parquet(E.OUT / f"S1_{c}.parquet")
    ctx = pd.read_parquet(E.W / f"p23/s1ctx_{c}.parquet", columns=["n_true"])
    gi = pd.Series(np.arange(len(ctx)), index=ctx.index)
    d["g"] = gi.reindex(d.s1).to_numpy()
    d.loc[~d.holdout, "g"] = -1
    d["g"] = d.g.fillna(-1).astype(np.int64)
    return d, S, M


def f_s1(d, S, M, s, t):
    am = M.argmax_sx(d.qx.to_numpy(), s)
    acc = am & (s >= t) & (d.g.to_numpy() >= 0)
    g, y, n = d.g.to_numpy(), d.y.to_numpy(), len(S)
    return a10.fb(np.bincount(g[acc & y], minlength=n), np.bincount(g[acc], minlength=n), S.n_true.to_numpy())


def main(m: str):
    from sklearn.linear_model import LogisticRegression
    D = {c: frame(c, m) for c in E.C}
    X_of = lambda d: np.column_stack([d.p_m2, d.p_hn, (d.p_m2 - d.p_hn).abs(), np.minimum(d.p_m2, d.p_hn),
                                      np.maximum(d.p_m2, d.p_hn), d.has_hn, d.exact_hn])
    out = []
    for fit in (0, 1):
        rep = 1 - fit
        tr = {c: (D[c][0].g.to_numpy() >= 0) & (D[c][1].half.to_numpy()[np.maximum(D[c][0].g.to_numpy(), 0)] == fit)
              for c in E.C}
        lr = LogisticRegression(max_iter=2000).fit(np.vstack([X_of(D[c][0])[tr[c]] for c in E.C]),
                                                   np.concatenate([D[c][0].y.to_numpy()[tr[c]] for c in E.C]))
        w = dict(zip(["p_m2", "p_hn", "diff", "min", "max", "has_hn", "exact_hn"], lr.coef_[0].round(3).tolist()))
        print(f"fit h{fit}: logistic w {w}, b {lr.intercept_[0]:.3f}", flush=True)
        for c in E.C:
            d, S, M = D[c]
            hm = S.half.to_numpy()
            scores = {"m2_alone": d.p_m2.to_numpy(), f"{m}_alone": d.p_hn.to_numpy(),
                      "report_meta": lr.predict_proba(X_of(d))[:, 1], "p8c(ref)": d.p.to_numpy()}
            base = None
            for k, s in scores.items():
                ft = [f_s1(d, S, M, s, t)[hm == fit].mean() for t in GRID]
                t = float(GRID[int(np.argmax(ft))])
                f = f_s1(d, S, M, s, t)
                if k == "m2_alone":
                    base = f
                x = f[hm == rep]
                dm, d34 = (f - base)[hm == rep], (f - S.f34.to_numpy())[hm == rep]
                row = {"fit": fit, "c": c, "score": k, "t": t, "f05_rep": x.mean(), "d_vs_m2": dm.mean(),
                       "se_vs_m2": dm.std(ddof=1) / np.sqrt(len(dm)), "d_vs_34": d34.mean(),
                       "se_vs_34": d34.std(ddof=1) / np.sqrt(len(d34)), "w": w}
                out.append(row)
                print(f"  {c:5s} h{fit}->h{rep} {k:12s} t {t:.2f}: F0.5 {x.mean():.6f} | vs m2 alone "
                      f"{row['d_vs_m2']:+.6f} ± {row['se_vs_m2']:.6f} | vs #34 {row['d_vs_34']:+.6f} ± "
                      f"{row['se_vs_34']:.6f}", flush=True)
    (E.OUT / f"jump_{m}.json").write_text(json.dumps(out, default=float, indent=0))


if __name__ == "__main__":
    main(sys.argv[1])
