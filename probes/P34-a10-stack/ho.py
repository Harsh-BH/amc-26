"""P34 holdout: a10 twin_p2mask on #8c's holdout argmax pairs (P23 dump), features rebuilt from the raw sources by
a10.features, parity-checked against P23's scratch files, then fit on one half of the S1 (India + US pooled, as a13) and
reported on the other, per country, with the paired per-S1 SE on the #8c yardstick (s1ctx f05).

    probes/P34-a10-stack/py.sh probes/P34-a10-stack/ho.py build     # -> data/work/p34/ho_{c}.parquet (+ parity)
    probes/P34-a10-stack/py.sh probes/P34-a10-stack/ho.py eval      # fit h0 -> h1, swap; ledger rows
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[2])]
import a10 as A

OUT = A.W / "p34"
SP = OUT / "sp"                      # P23's /tmp scratch files (amc_, tw_, tw2_), copied here
LB_W = {"India": 0.47, "US": 0.38}   # test S1 shares: a per-S1 holdout delta times this = its LB contribution
half = lambda ids: (pd.util.hash_pandas_object(pd.Series(list(ids)), index=False).to_numpy() >> 7) % 2   # = run.half


def build(c: str):
    t0 = time.time()
    pp = pd.read_parquet(A.W / f"p23/pairs_{c}.parquet", columns=["s1", "sx", "holdout", "y", "p_m2", "r", "p", "am"])
    q1, s1u = pd.factorize(pp.s1)
    qx, sxu = pd.factorize(pp.sx)
    s1t, sxt = A.texts("train", c, s1u, sxu)
    a = A.features(q1, qx, pp.p.to_numpy(), pp.p_m2.to_numpy(), pp.r.to_numpy(), pp.am.to_numpy(),
                   pp.holdout.to_numpy(), s1t, sxt)
    a["y"] = pp.y.to_numpy()[a.row]
    a["s1"], a["sx"] = np.asarray(s1u)[a.q1], np.asarray(sxu)[a.qx]
    # parity with P23's files: a10's twin / cat columns and a13's mask
    ref = pd.read_parquet(SP / f"amc_{c}.parquet", columns=["s1", "sx", "cat"]).merge(
        pd.read_parquet(SP / f"tw_{c}.parquet"), on=["s1", "sx"]).merge(pd.read_parquet(SP / f"tw2_{c}.parquet"), on=["s1", "sx"])
    m = a.merge(ref, on=["s1", "sx"], suffixes=("", "_ref"))
    assert len(m) == len(a) == len(ref), (len(m), len(a), len(ref))
    ru = pp[~pp.am].sort_values("p", ascending=False).drop_duplicates("sx")
    e5 = ((ru.p_m2 > 0.02) & (ru.p_m2 < 0.98) & ru.r.isna()).to_numpy()
    mk = m.sx.map(pd.Series(e5, index=ru.sx.to_numpy())).fillna(False).to_numpy(bool)
    par = {k: int((m[k].to_numpy() != m[f"{k}_ref"].to_numpy()).sum()) for k in ("n_twins", "n_atwins", "n_ctwins", "n_cand_sx")}
    par["p2"] = int((~np.isclose(m.p2, m.p2_ref)).sum())
    par["cat"] = int((np.asarray(A.CATS, object)[m["cat"]] != m.cat_ref.to_numpy(object)).sum())
    par["mask"] = int((np.isnan(m.p2m.to_numpy()) != mk).sum())
    print(f"{c}: {len(a):,} argmax pairs, parity mismatches vs P23 {par}, {time.time() - t0:.0f}s", flush=True)
    a.drop(columns=["q1", "qx", "row"]).to_parquet(OUT / f"ho_{c}.parquet", index=False)
    return par


def load(c: str) -> dict:
    a = pd.read_parquet(OUT / f"ho_{c}.parquet")
    ctx = pd.read_parquet(A.W / f"p23/s1ctx_{c}.parquet", columns=["n_true", "f05"])
    ids = ctx.index.to_numpy()
    a["g"] = pd.Series(np.arange(len(ids)), index=ids).reindex(a.s1).to_numpy()
    assert not a.g.isna().any()
    s = dict(c=c, a=a, n=len(ids), n_true=ctx.n_true.to_numpy(), f0=ctx.f05.to_numpy(), h=half(ids))
    assert np.allclose(f_of(s, a.p.to_numpy() >= A.T), s["f0"]), "the #8c yardstick does not reproduce"
    return s


def f_of(s: dict, keep: np.ndarray) -> np.ndarray:
    g, y = s["a"].g.to_numpy(), s["a"].y.to_numpy()
    return A.fb(np.bincount(g[keep & y], minlength=s["n"]), np.bincount(g[keep], minlength=s["n"]), s["n_true"])


def report(S: list, ps: list, fit_half: int, t: float) -> dict:
    """Deltas on the report half vs #8c: per country (mean ± SE), pooled, LB-equivalent; pair flips."""
    rep, out = 1 - fit_half, {}
    for s, p in zip(S, ps):
        d = (f_of(s, p >= t) - s["f0"])[s["h"] == rep]
        hr = s["h"][s["a"].g] == rep
        base, new = s["a"].p.to_numpy() >= A.T, p >= t
        out[s["c"]] = dict(d=d.mean(), se=d.std(ddof=1) / np.sqrt(len(d)), n=len(d), added=int((new & ~base & hr).sum()),
                           dropped=int((base & ~new & hr).sum()), acc_base=int((base & hr).sum()),
                           s1_changed=int((d != 0).sum()), d_sum=d.sum(), var_sum=d.var(ddof=1) * len(d))
    n = sum(o["n"] for o in out.values())
    pooled = sum(o["d_sum"] for o in out.values()) / n
    out["pooled"] = dict(d=pooled, se=np.sqrt(sum(o["var_sum"] for o in out.values())) / n)   # ponytail: per-country var, not pooled around one mean
    out["lb"] = dict(d=sum(LB_W[c] * out[c]["d"] for c in LB_W), se=np.sqrt(sum((LB_W[c] * out[c]["se"]) ** 2 for c in LB_W)))
    return out


def fmt(o: dict) -> str:
    return (f"LB-eq {o['lb']['d']:+.6f} ± {o['lb']['se']:.6f} ({o['lb']['d'] / o['lb']['se']:.1f} SE) | pooled "
            f"{o['pooled']['d']:+.6f} ± {o['pooled']['se']:.6f} | " + " | ".join(
                f"{c} {o[c]['d']:+.6f} ± {o[c]['se']:.6f} (+{o[c]['added']}/-{o[c]['dropped']} of {o[c]['acc_base']:,}, "
                f"{o[c]['s1_changed']} S1)" for c in LB_W))


def evaluate():
    import lightgbm as lgb
    sys.path.insert(0, str(A.ROOT))
    from ledger import log
    S = [load(c) for c in ("India", "US")]
    res = {}
    for fh in (0, 1):
        t0 = time.time()
        tr = [s["h"][s["a"].g] == fh for s in S]
        X = pd.concat([s["a"][A.COLS][m] for s, m in zip(S, tr)])
        y = np.concatenate([s["a"].y.to_numpy()[m] for s, m in zip(S, tr)])
        b = A.fit(X, y)
        b.save_model(str(OUT / f"a10_h{fh}.txt"))
        for guard in (True, False):
            ps = [A.stacked(b, s["a"], guard) for s in S]
            tag = "" if guard else "_raw"
            for s, p in zip(S, ps):
                np.save(OUT / f"ho_p{tag}_{s['c']}_h{fh}.npy", p)
            for t in (A.T, 0.69, 0.73):
                o = report(S, ps, fh, t)
                res[(fh, guard, t)] = o
                print(f"guard={guard!s:5s} fit h{fh} -> h{1 - fh} t={t:.2f}: {fmt(o)}", flush=True)
            o = res[(fh, guard, A.T)]
            log("exp", probe="P34-a10-stack", subproblem="decide",
                name=f"a10-twin_p2mask{'-sibguard' if guard else ''}-fit_h{fh}-report_h{1 - fh}",
                world="holdout", metrics={"lb_eq_delta": round(o["lb"]["d"], 6), "lb_eq_se": round(o["lb"]["se"], 6),
                                          "pooled_delta": round(o["pooled"]["d"], 6), "pooled_se": round(o["pooled"]["se"], 6),
                                          **{f"{c}_delta": round(o[c]["d"], 6) for c in LB_W},
                                          **{f"{c}_se": round(o[c]["se"], 6) for c in LB_W},
                                          **{f"{c}_added": o[c]["added"] for c in LB_W},
                                          **{f"{c}_dropped": o[c]["dropped"] for c in LB_W}},
                config={"cols": A.COLS, "rounds": A.ROUNDS, "params": A.PRM, "t": A.T, "fit": f"India+US holdout half {fh}",
                        "guard": "stack.hn_cls == 1 keeps #8c's p" if guard else None, "base": "#8c (P23 dump, s1ctx f05)"},
                hypothesis="P23 a13 twin_p2mask: the residual stacker on #8c argmax pairs adds >= +0.00015 LB-eq at >= 3 SE",
                verdict="keep" if fh == 0 and guard and o["lb"]["d"] >= 0.00015 and o["lb"]["d"] >= 3 * o["lb"]["se"]
                and min(o[c]["d"] for c in LB_W) >= -0.00005 else "inconclusive",
                runtime_s=round(time.time() - t0), peak_gb=None)
        if fh == 0:
            imp = pd.Series(b.feature_importance("gain"), index=A.COLS).sort_values(ascending=False)
            print("   gain importance:", (imp / imp.sum()).round(3).to_dict(), flush=True)
    json.dump({f"h{fh}_guard{g}_t{t}": o for (fh, g, t), o in res.items()}, open(OUT / "ho_eval.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        for c in ("India", "US"):
            build(c)
    else:
        evaluate()
