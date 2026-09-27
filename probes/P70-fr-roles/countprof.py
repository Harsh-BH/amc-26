"""Count-profile pricing of rule classes (lead 19:35; idea from a public repo's 'count profile' lesson). For a class of France
pairs (adds or vetoes), regress class pairs per S1 on the S1's accepted-member count k in the base (S1 with k >= 1):
rate(k) = a + b*k (least squares on per-k means, weighted by S1 count). True copies scale with cluster size, decoys are
flat, so the true share ~= b*sum(k) / (a*N + b*sum(k)). Calibrated on LB-backed classes (P71a dev ~.9, hnst ~.3)."""
import csv, json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P42-fr-joint-tune")]
import p42
W = ROOT / "data/work"
_L = {}


def lines(base):
    if base not in _L:
        _L[base] = p42.read_fr(ROOT / "output" / base)
    return _L[base]


def diff(base, new):
    A, B = lines(base), lines(new)
    rows = []
    for k in A.index[A.to_numpy() != B.reindex(A.index).to_numpy()]:
        a, b = set(filter(None, A[k].split(","))), set(filter(None, B[k].split(",")))
        rows += [(k, x, "add") for x in b - a] + [(k, x, "rm") for x in a - b]
    return pd.DataFrame(rows, columns=["s1", "sx", "act"])


def price(pairs_s1, base, veto=False):
    L = lines(base)
    k = L.map(lambda v: len(v.split(",")) if v else 0)
    c = pd.Series(pairs_s1).value_counts().reindex(k.index).fillna(0)
    if veto:                     # the vetoed pair is one of the S1's members: copy count without it
        k = k - c
    m = k >= 1
    df = pd.DataFrame({"k": k[m].clip(upper=8), "c": c[m]})
    g = df.groupby("k").agg(N=("c", "size"), mean=("c", "mean"))
    w = g.N.to_numpy(float)
    X = np.c_[np.ones(len(g)), g.index.to_numpy(float)]
    a, b = np.linalg.lstsq(X * np.sqrt(w)[:, None], g["mean"].to_numpy() * np.sqrt(w), rcond=None)[0]
    N, sk, T = int(m.sum()), float(df.k.sum()), float(df.c.sum())
    share = b * sk / max(1e-9, a * N + b * sk)
    return {"pairs_on_k>=1": int(T), "a": round(float(a), 6), "b": round(float(b), 6), "true_share_est": round(float(share), 3),
            "rate_by_k": {int(kk): round(float(v), 5) for kk, v in g["mean"].items()}}


if __name__ == "__main__":
    out = {}
    dev = diff("sub66_final", "sub71a"); dev = dev[dev.act == "add"]
    out["CAL dev swaps (P71a, LB ~.9) vs sub66_final"] = price(dev.s1, "sub66_final")
    hn = pd.read_parquet(W / "p70/changes_sub77_hnst.parquet")
    out["CAL hnst (LB ~.3) vs sub77"] = price(hn.s1, "sub77")
    gr = diff("sub77", "sub78g"); gr = gr[gr.act == "add"]
    out["groupe swaps (sub78g) vs sub77"] = price(gr.s1, "sub77")
    x = pd.read_parquet(W / "p70/changes_sub77_xlf_split.parquet")
    for sp, g in x.groupby("sp"):
        out[f"xlf {sp} vs sub77"] = price(g.s1, "sub77")
    q = diff("sub77", "sub79q")
    out["sub79q vetoes vs sub77"] = price(q.s1[q.act == "rm"], "sub77", veto=True)
    out["sub79q adds vs sub77"] = price(q.s1[q.act == "add"], "sub77")
    for k, v in out.items():
        print(f"{k:45s} pairs {v['pairs_on_k>=1']:>5}  a {v['a']:.6f}  b {v['b']:.6f}  est true share {v['true_share_est']:.3f}  rate(k) {v['rate_by_k']}", flush=True)
    (Path(__file__).parent / "countprof.json").write_text(json.dumps(out, indent=1))
