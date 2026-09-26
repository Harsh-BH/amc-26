"""P49 test 1: does the owner of an ambiguous address-less SX follow the group-size likelihood ratio?
Among ambiguous SX whose owner is inside the twin set: w_j = r(k_j) (or h(k_j)), P(owner = j) = w_j / Σ w.
k = the twin's OTHER members from the SX's source, three ways: 'oracle' (all true, incl. address-less), 'addr' (true
addressed members; what a ~99%-precise decision sees), 'dec' (label-free #8c accepted addressed argmax SX; only sets
whose twins are all holdout S1, where the frame covers 99%). Aggregates only.
    probes/P49-twin-prior/py.sh probes/P49-twin-prior/test1.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from law import W, ratios  # noqa: E402

L = json.load(open(W / "p49/law.json"))


def wfun(c, src, k, kind):
    h, r = ratios(L[f"{c}|{src}"])
    v = r if kind == "r" else h
    return v[np.clip(k, 0, len(v) - 1)]


def se(p, n):
    return np.sqrt(p * (1 - p) / max(n, 1))


rows = []
for c in ("India", "US"):
    tw = pd.read_parquet(W / f"p49/tw_{c}.parquet")
    tw["own"] = tw.groupby("sx").y_full.transform("any")
    tw["allho"] = tw.groupby("sx").holdout.transform("all")
    tw["anyho"] = tw.groupby("sx").holdout.transform("any")
    q = tw[tw.own & tw.anyho].copy()
    q["k_oracle"] = q.kt_src + q.kta_src
    q["k_addr"] = q.kt_src
    q["k_dec"] = q.kd_src
    n_sx = q.sx.nunique()
    am_ok = q[q.am].y_full.sum() / n_sx
    print(f"\n{c}: ambiguous SX, owner in set, >=1 holdout twin: {n_sx:,}; #8c argmax is the owner {am_ok:.3f} "
          f"(uniform 1/nset {q.groupby('sx').size().rdiv(1).mean():.3f})", flush=True)
    for kname in ("k_oracle", "k_addr", "k_dec"):
        qq = q[q.allho] if kname == "k_dec" else q
        for kind in ("r", "h"):
            w = np.array([wfun(c, s, k, kind) for s, k in zip(qq.src, qq[kname])]) + 1e-9
            post = w / pd.Series(w).groupby(qq.sx.to_numpy()).transform("sum").to_numpy()
            qq = qq.assign(post=post)
            top = qq.loc[qq.groupby("sx").post.idxmax()]
            ties = qq.groupby("sx").post.transform(lambda x: (x == x.max()).sum()).loc[top.index] > 1
            dec = top[~ties.to_numpy()]
            ll = np.log(qq.post[qq.y_full].clip(1e-9)).mean() - np.log(1 / qq[qq.y_full].nset).mean()
            rate, n = dec.y_full.mean(), len(dec)
            base = (1 / dec.nset).mean()
            exp_ = dec.post.mean()
            rows.append(dict(c=c, k=kname, kind=kind, n_sx=len(top), n_decided=n, top_true=rate, se=se(rate, n),
                             uniform=base, predicted=exp_, dll=ll))
            print(f"  {kname:8s} {kind}: decided {n:6,} of {len(top):6,} | top-prior twin is owner {rate:.3f} ± {se(rate, n):.3f} "
                  f"(uniform {base:.3f}, model says {exp_:.3f}) | Δ mean log-lik vs uniform {ll:+.4f}", flush=True)
    # nset == 2: bucket by (k_A, k_B), k = addressed truth; A = the smaller-k twin
    two = q[q.nset == 2].copy()
    for kname in ("k_oracle", "k_addr"):
        g = two.sort_values(["sx", kname])
        a, b = g.groupby("sx").head(1).set_index("sx"), g.groupby("sx").tail(1).set_index("sx")
        t = pd.DataFrame({"ka": a[kname].clip(upper=5), "kb": b[kname].clip(upper=5), "ya": a.y_full, "src": a.src})
        t = t[t.ka != t.kb]
        pr = [wfun(c, s, ka, "r") / (wfun(c, s, ka, "r") + wfun(c, s, kb, "r")) for s, ka, kb in zip(t.src, t.ka, t.kb)]
        t["pred"] = pr
        tab = t.groupby(["ka", "kb"]).agg(n=("ya", "size"), true_small=("ya", "mean"), pred=("pred", "mean"))
        tab = tab[tab.n >= 30]
        tab["se"] = np.sqrt(tab.true_small * (1 - tab.true_small) / tab.n)
        print(f"  nset=2, {kname}: owner = the smaller-k twin {t.ya.mean():.3f} ± {se(t.ya.mean(), len(t)):.3f} over {len(t):,} "
              f"(model {t.pred.mean():.3f})\n" + tab.round(3).to_string(), flush=True)
pd.DataFrame(rows).to_csv(W / "p49/test1.csv", index=False)
