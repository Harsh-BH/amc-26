"""P49: re-decide the ambiguous twin SX on top of sub44r's holdout decision (out of fold on both halves), holdout half split.

Base: sub44r = #8c argmax & p'' >= 0.71; p'' out of fold (half-1 S1 from P44's half-0 fit, half-0 S1 from swapfit.py).
For every ambiguous SX (tw_{c}, >= 1 holdout twin) a small LightGBM scores each twin row (label-free columns except the
competitor-count proxy, see K below); the SX goes to its best-scored twin iff that score >= tau, else nowhere.
Fit rows: ambiguous SX with no holdout twin in the report half; tau: tuned on the other half's S1. Report: per-S1 F0.5 delta
vs sub44r over the report half's S1, paired SE, LB-eq 0.468 / 0.383.
K: the twin's other accepted same-source members = label-free #8c count (kd) for holdout S1; for competing train S1 the
frame holds 16% of their rows, so truth-addressed count kt stands in (a ~99%-precise decision's count).
    probes/P49-twin-prior/py.sh probes/P49-twin-prior/dec.py [arm ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein as Lv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from law import W, ratios  # noqa: E402
from test4 import script  # noqa: E402  (module guard below keeps test4's __main__ work off)

T = 0.71
LB_W = {"India": 0.468, "US": 0.383}
L = json.load(open(W / "p49/law.json"))
lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
PRM = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=100, lambda_l2=1.0,
           feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, seed=0, deterministic=True, verbose=-1)
BASE = ["lp", "dlp", "prank", "nset", "kfull", "kout", "is_am", "lpa_am", "exact", "is_s3", "p_max_other"]
CNT = ["lr_k", "dlr_k", "k", "k_oth"]
RAW = ["rawdiff", "d_rel", "d_min_uniq", "dl_rel"]
SCR = ["scr_rel"]
ARMS = {"base": BASE, "cnt": BASE + CNT, "raw": BASE + RAW, "all": BASE + CNT + RAW + SCR}


def fb(tp, npred, ntrue):
    tp, npred, ntrue = (np.asarray(x, float) for x in (tp, npred, ntrue))
    with np.errstate(divide="ignore", invalid="ignore"):
        P, R = tp / npred, tp / ntrue
        f = 1.25 * P * R / (0.25 * P + R)
    f = np.where(tp == 0, 0.0, f)
    return np.where(ntrue == 0, np.where(npred == 0, 1.0, 0.0), f)


def load(c: str) -> dict:
    ids = pd.read_parquet(W / f"p23/s1ctx_{c}.parquet", columns=["n_true"]).index
    S = pd.read_parquet(W / f"p44ms/S1_{c}.parquet")
    A = pd.read_parquet(W / f"p44ms/A_{c}.parquet", columns=["s1", "sx", "g", "half", "y"])
    p0, p1 = np.load(W / f"p44ms/ho_p44r_{c}.npy"), np.load(W / f"p49/ho_p44r_h1fit_{c}.npy")
    A["pa"] = np.where(A.half.to_numpy() == 1, p0, p1)         # out of fold on both halves
    A["acc"] = A.pa >= T
    n = len(S)
    g, y, acc = A.g.to_numpy(), A.y.to_numpy(), A.acc.to_numpy()
    tp, npred = np.bincount(g[acc & y], minlength=n), np.bincount(g[acc], minlength=n)
    gi = pd.Series(np.arange(n), index=ids)
    tw = pd.read_parquet(W / f"p49/tw_{c}.parquet")
    tw["g"] = gi.reindex(tw.s1).fillna(-1).to_numpy(np.int64)
    tw = tw.merge(A[["s1", "sx", "pa", "acc"]], on=["s1", "sx"], how="left")
    tw["acc"] = tw.acc.fillna(False).astype(bool)
    return dict(c=c, S=S, tp=tp, npred=npred, n_true=S.n_true.to_numpy(), half=S.half.to_numpy().astype(np.int64), tw=tw)


def feats(c: str, tw: pd.DataFrame) -> pd.DataFrame:
    sx = tw.sx.to_numpy()
    grp = tw.groupby("sx")
    tw["lp"] = lg(tw.p.to_numpy())
    mx_other = grp.lp.transform(lambda v: np.array([np.max(np.delete(v.to_numpy(), i)) for i in range(len(v))]))
    tw["dlp"] = tw.lp - mx_other
    tw["p_max_other"] = 1 / (1 + np.exp(-mx_other))
    tw["prank"] = grp.p.rank(ascending=False, method="min")
    tw["kout"] = tw.kfull - tw.nset
    tw["is_am"] = tw.am.astype(np.float32)
    lpa = tw.pa.where(tw.am)
    tw["lpa_am"] = lg(lpa.groupby(sx).transform("max").to_numpy())     # p'' of the SX's argmax row (NaN: competitor)
    tw["is_s3"] = (tw.src == "S3").astype(np.float32)
    tw["exact"] = tw.exact.astype(np.float32)
    tw["k"] = np.where(tw.holdout, tw.kd_src, tw.kt_src)
    tw["k_oth"] = np.where(tw.holdout, tw.kd_oth, tw.kt_oth)
    tab = {s: ratios(L[f"{c}|{s}"])[1] for s in ("S2", "S3")}
    tw["lr_k"] = np.log([tab[s][min(k, len(tab[s]) - 1)] + 1e-6 for s, k in zip(tw.src, tw.k)])
    tw["dlr_k"] = tw.lr_k - tw.groupby("sx").lr_k.transform("mean")
    tw["d"] = [Lv.distance(a, b) for a, b in zip(tw.sx_name, tw.s1_name)]
    tw["dl"] = [Lv.distance(a.lower(), b.lower()) for a, b in zip(tw.sx_name, tw.s1_name)]
    tw["rawdiff"] = (tw.groupby("sx").s1_name.transform("nunique") > 1).astype(np.float32)
    tw["d_rel"] = tw.d - tw.groupby("sx").d.transform("min")
    tw["dl_rel"] = tw.dl - tw.groupby("sx").dl.transform("min")
    nmin = tw.assign(m=tw.d_rel == 0).groupby("sx").m.transform("sum")
    tw["d_min_uniq"] = ((tw.d_rel == 0) & (nmin == 1)).astype(np.float32)
    tw["scr_rel"] = np.nan
    if c == "India":                      # the table is India's; any other country has no Indic names (all NaN)
        P = pd.read_parquet(W / "p49/script_state.parquet")
        st = pd.read_parquet(W / "p49/s1_India.parquet").set_index("s1").state
        sc = [script(x) for x in tw.sx_name]
        stt = st.reindex(tw.s1).to_numpy()
        v = np.array([np.log(P.loc[s, k]) if (k is not None and s in P.index and k in P.columns) else np.nan
                      for s, k in zip(stt, sc)])
        tw["scr_rel"] = v - pd.Series(v).groupby(sx).transform("max").to_numpy()
    return tw


def apply(d: dict, tw: pd.DataFrame, s: np.ndarray, tau: float) -> np.ndarray:
    """Per-S1 F0.5 after re-deciding every SX of tw: best-scored twin iff score >= tau."""
    tp, npred = d["tp"].copy(), d["npred"].copy()
    old = tw[tw.acc & (tw.g >= 0)]                           # sub44r's accepted pairs of these SX
    np.subtract.at(npred, old.g.to_numpy(), 1)
    np.subtract.at(tp, old.g.to_numpy(), old.y_full.to_numpy().astype(int))
    t = tw.assign(s=s)
    best = t.loc[t.groupby("sx").s.idxmax()]
    new = best[(best.s >= tau) & (best.g >= 0)]
    np.add.at(npred, new.g.to_numpy(), 1)
    np.add.at(tp, new.g.to_numpy(), new.y_full.to_numpy().astype(int))
    return fb(tp, npred, d["n_true"])


def run(arms):
    import lightgbm as lgb
    D = {c: load(c) for c in LB_W}
    for c, d in D.items():
        d["tw"] = feats(c, d["tw"])
        tw = d["tw"]
        # which halves each SX's holdout twins sit in
        hh = np.where(tw.g >= 0, d["half"][tw.g.clip(0)], -1)
        tw["h0"] = pd.Series(hh == 0).groupby(tw.sx.to_numpy()).transform("any").to_numpy()
        tw["h1"] = pd.Series(hh == 1).groupby(tw.sx.to_numpy()).transform("any").to_numpy()
        f0 = fb(d["tp"], d["npred"], d["n_true"])
        d["f0"] = f0
        ho = tw[tw.holdout]
        kd_ok = (ho.kd_src == ho.kt_src).mean()
        print(f"{c}: sub44r OOF F0.5 h0 {f0[d['half'] == 0].mean():.6f} h1 {f0[d['half'] == 1].mean():.6f}; "
              f"holdout twins kd == kt {kd_ok:.3f}, |kd-kt| <= 1 {(abs(ho.kd_src - ho.kt_src) <= 1).mean():.3f}", flush=True)
    out = {}
    for arm in arms:
        cols = ARMS[arm]
        res = {}
        for rep in (1, 0):
            fit = 1 - rep
            Xs, ys = [], []
            for c, d in D.items():
                tw = d["tw"]
                m = ~tw[f"h{rep}"].to_numpy()                    # SX that touch no report-half S1
                Xs.append(tw.loc[m, cols]), ys.append(tw.y_full.to_numpy()[m])
            b = lgb.train(PRM, lgb.Dataset(pd.concat(Xs), np.concatenate(ys)), 150)
            for c, d in D.items():
                tw = d["tw"]
                s = b.predict(tw[cols])
                ev = tw[f"h{fit}"].to_numpy() | tw[f"h{rep}"].to_numpy()
                # tau on the fit half's S1 (every SX touching a holdout S1; the score of fit-half SX is in-sample)
                grid = np.round(np.arange(0.5, 0.96, 0.025), 3)
                sub = tw[ev].reset_index(drop=True)
                ss = s[ev]
                gains = [(apply(d, sub, ss, t) - d["f0"])[d["half"] == fit].mean() for t in grid]
                tau = float(grid[int(np.argmax(gains))])
                dl = (apply(d, sub, ss, tau) - d["f0"])[d["half"] == rep]
                res[(c, rep)] = (dl.mean(), dl.std(ddof=1) / np.sqrt(len(dl)), tau, int((dl > 0).sum()), int((dl < 0).sum()))
        for rep in (1, 0):
            lb = sum(LB_W[c] * res[(c, rep)][0] for c in LB_W)
            se = np.sqrt(sum((LB_W[c] * res[(c, rep)][1]) ** 2 for c in LB_W))
            print(f"[{arm}] fit h{1 - rep} -> report h{rep}: " + " | ".join(
                f"{c} {res[(c, rep)][0]:+.6f} ± {res[(c, rep)][1]:.6f} (tau {res[(c, rep)][2]}, +{res[(c, rep)][3]}/-{res[(c, rep)][4]})"
                for c in LB_W) + f" | LB-eq {lb:+.6f} ± {se:.6f} ({lb / se:.1f} SE)", flush=True)
            out[f"{arm}_h{rep}"] = dict(lb=lb, se=se, **{c: res[(c, rep)][:3] for c in LB_W})
    return D, out


if __name__ == "__main__":
    arms = sys.argv[1:] or list(ARMS)
    D, out = run(arms)
    json.dump(out, open(W / "p49/dec.json", "w"), indent=1, default=float)
