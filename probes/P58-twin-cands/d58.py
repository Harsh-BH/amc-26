"""P58: re-decide every address-less twin SX of the expanded frame (ex_{c}, every same-key S1 a candidate) on top of
sub44r's holdout decision (out of fold on both halves; the holdout stand-in for sub54's India/US decision), half split.

Score: LightGBM on twin rows. Model for report half r is fit on SX that touch no half-r S1, so every score a half sees is
out of sample (tau is tuned on the fit half with that half's own out-of-sample scores). Decision per SX: its best-scored
twin iff score >= tau (tau_e when the twin holds no other accepted pair), else nowhere.
Counts (label-free on test): holdout S1 from sub44r's accepted pairs; competing train S1 from truth-addressed members
(P49's proxy; the base accepts addressed pairs at ~0.99 precision).
    probes/P58-twin-cands/py.sh probes/P58-twin-cands/d58.py [arm ...]    -> data/work/p58/d58.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz.distance import Levenshtein as Lv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "P49-twin-prior"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dec import fb, lg, PRM, LB_W, T, L  # noqa: E402
from law import ratios  # noqa: E402
from frame58 import W, noaddr_sx  # noqa: E402

RAW = ["d", "dl", "d_rel", "dl_rel", "case_rel", "d_min_uniq", "n_min", "rawdiff", "exact_raw", "len_rel"]
SET = ["kfull", "nset_old", "in_set", "is_s3"]
CNT = ["k_src", "k_oth", "lr_k", "dlr_k", "emp", "k_rel"]
P8 = ["lp", "dlp", "is_am", "lp_am"]
ARMS = {"raw": SET + RAW + CNT, "rawp": SET + RAW + CNT + P8, "set": SET + CNT, "p": SET + CNT + P8}


def base(c: str) -> dict:
    ids = pd.read_parquet(W / f"p23/s1ctx_{c}.parquet", columns=["n_true"]).index
    S = pd.read_parquet(W / f"p44ms/S1_{c}.parquet")
    A = pd.read_parquet(W / f"p44ms/A_{c}.parquet", columns=["s1", "sx", "g", "half", "y", "noaddr"])
    p0, p1 = np.load(W / f"p44ms/ho_p44r_{c}.npy"), np.load(W / f"p49/ho_p44r_h1fit_{c}.npy")
    A["acc"] = np.where(A.half.to_numpy() == 1, p0, p1) >= T
    n = len(S)
    g, y, acc = A.g.to_numpy(), A.y.to_numpy(), A.acc.to_numpy()
    return dict(c=c, A=A, tp=np.bincount(g[acc & y], minlength=n), npred=np.bincount(g[acc], minlength=n),
                n_true=S.n_true.to_numpy(), half=S.half.to_numpy().astype(np.int64), gi=pd.Series(np.arange(n), index=ids))


def counts(c: str, d: dict, s1s: set, na: set) -> pd.DataFrame:
    """Per S1 (index) x source: addressed accepted members (holdout: sub44r; others: truth), all accepted (n_acc)."""
    A = d["A"]
    a = A[A.acc].assign(src=lambda t: t.sx.str[:2], ad=lambda t: ~t.sx.isin(na))
    ho = a.groupby(["s1", "src"]).ad.sum().unstack(fill_value=0)
    ho["n_acc"] = a.groupby("s1").size()
    gt = pq.read_table(W / "gt_pairs.parquet", filters=[("s1", "in", list(s1s - set(d["gi"].index)))]).to_pandas()
    gt = gt.assign(src=gt.sx.str[:2], ad=~gt.sx.isin(na))
    ot = gt.groupby(["s1", "src"]).ad.sum().unstack(fill_value=0)
    ot["n_acc"] = gt.groupby("s1").ad.sum()                      # the base's decision ~ its addressed truth
    out = pd.concat([ho, ot]).fillna(0)
    return out.reindex(sorted(s1s), fill_value=0).astype(np.int64)


def feats(c: str, ex: pd.DataFrame, cnt: pd.DataFrame) -> pd.DataFrame:
    sx = ex.sx.to_numpy()
    ex["lp"] = np.where(ex.in_set, lg(ex.p.fillna(0.5).to_numpy()), np.nan)
    mx = ex.groupby(sx).lp.transform("max")
    ex["lp_am"] = ex.lp.where(ex.am).groupby(sx).transform("max")
    second = ex.assign(r=ex.groupby(sx).lp.rank(ascending=False, method="first")).query("r == 2").set_index("sx").lp
    ex["dlp"] = np.where(ex.lp >= mx, ex.lp - second.reindex(ex.sx).to_numpy(), ex.lp - mx)
    ex["is_am"] = ex.am.astype(np.float32)
    ex["in_set"] = ex.in_set.astype(np.float32)
    ex["is_s3"] = (ex.src == "S3").astype(np.float32)
    ex["d"] = [Lv.distance(a, b) for a, b in zip(ex.sx_name, ex.s1_name)]
    ex["dl"] = [Lv.distance(a.lower(), b.lower()) for a, b in zip(ex.sx_name, ex.s1_name)]
    ex["case"] = ex.d - ex.dl
    ex["len_rel"] = (ex.s1_name.str.len() - ex.sx_name.str.len()).abs()
    for a in ("d", "dl", "case"):
        ex[f"{a}_rel"] = ex[a] - ex.groupby(sx)[a].transform("min")
    ex["n_min"] = (ex.d_rel == 0).groupby(sx).transform("sum")
    ex["d_min_uniq"] = ((ex.d_rel == 0) & (ex.n_min == 1)).astype(np.float32)
    ex["rawdiff"] = (ex.groupby(sx).s1_name.transform("nunique") > 1).astype(np.float32)
    ex["exact_raw"] = (ex.sx_name == ex.s1_name).astype(np.float32)
    oth = {"S2": "S3", "S3": "S2"}
    ci = cnt.reindex(ex.s1)
    ex["k_src"] = np.where(ex.src == "S2", ci.S2.to_numpy(), ci.S3.to_numpy())
    ex["k_oth"] = np.where(ex.src == "S2", ci.S3.to_numpy(), ci.S2.to_numpy())
    ex["n_acc"] = ci.n_acc.to_numpy() - ex.cur.to_numpy().astype(int)     # other accepted pairs of the twin
    ex["emp"] = (ex.n_acc == 0).astype(np.float32)
    tab = {s: ratios(L[f"{c}|{s}"])[1] for s in ("S2", "S3")}
    ex["lr_k"] = np.log([tab[s][min(k, len(tab[s]) - 1)] + 1e-6 for s, k in zip(ex.src, ex.k_src)])
    ex["dlr_k"] = ex.lr_k - ex.groupby(sx).lr_k.transform("mean")
    ex["k_rel"] = ex.k_src - ex.groupby(sx).k_src.transform("min")
    return ex


def load(c: str) -> tuple:
    d = base(c)
    ex = pd.read_parquet(W / f"p58/ex_{c}.parquet")
    na = set(noaddr_sx(c).entity_id)
    A = d["A"]
    acc = set(zip(A.s1[A.acc], A.sx[A.acc]))
    ex["cur"] = [(a, b) in acc for a, b in zip(ex.s1, ex.sx)]
    ex["g"] = d["gi"].reindex(ex.s1).fillna(-1).to_numpy(np.int64)
    ex = feats(c, ex, counts(c, d, set(ex.s1), na))
    hh = np.where(ex.g >= 0, d["half"][ex.g.clip(0)], -1)
    for h in (0, 1):
        ex[f"h{h}"] = pd.Series(hh == h).groupby(ex.sx.to_numpy()).transform("any").to_numpy()
    return d, ex


def apply(d: dict, ex: pd.DataFrame, s: np.ndarray, tau: float, tau_e: float) -> np.ndarray:
    """Per-S1 F0.5 after re-deciding every SX of ex: best-scored twin iff score >= tau (tau_e for an empty twin)."""
    tp, npred = d["tp"].copy(), d["npred"].copy()
    old = ex[ex.cur & (ex.g >= 0)]
    np.subtract.at(npred, old.g.to_numpy(), 1)
    np.subtract.at(tp, old.g.to_numpy(), old.y.to_numpy().astype(int))
    t = ex.assign(s=s)
    best = t.loc[t.groupby("sx").s.idxmax()]
    bar = np.where(best.emp.to_numpy() > 0, tau_e, tau)
    new = best[(best.s.to_numpy() >= bar) & (best.g.to_numpy() >= 0)]
    np.add.at(npred, new.g.to_numpy(), 1)
    np.add.at(tp, new.g.to_numpy(), new.y.to_numpy().astype(int))
    return fb(tp, npred, d["n_true"])


GRID = np.round(np.arange(0.3, 0.96, 0.05), 3)


def run(arms, D):
    import lightgbm as lgb
    out = {}
    for arm in arms:
        cols = ARMS[arm]
        res = {}
        S = {}
        for rep in (0, 1):                                   # score for half-rep SX: model fit on SX touching no rep S1
            Xs, ys = [], []
            for c, (d, ex) in D.items():
                m = ~ex[f"h{rep}"].to_numpy()
                Xs.append(ex.loc[m, cols]), ys.append(ex.y.to_numpy()[m])
            b = lgb.train(PRM, lgb.Dataset(pd.concat(Xs), np.concatenate(ys)), 150)
            for c, (d, ex) in D.items():
                S[(c, rep)] = b.predict(ex[cols])
        for rep in (1, 0):
            fit = 1 - rep
            for c, (d, ex) in D.items():
                f0 = fb(d["tp"], d["npred"], d["n_true"])
                sf, sr = ex[f"h{fit}"].to_numpy(), ex[f"h{rep}"].to_numpy()
                subf, subr = ex[sf].reset_index(drop=True), ex[sr].reset_index(drop=True)
                best, bt = -1e9, None
                for t in GRID:
                    for te in GRID:
                        gv = (apply(d, subf, S[(c, fit)][sf], t, te) - f0)[d["half"] == fit].mean()
                        if gv > best:
                            best, bt = gv, (float(t), float(te))
                dl = (apply(d, subr, S[(c, rep)][sr], *bt) - f0)[d["half"] == rep]
                res[(c, rep)] = (dl.mean(), dl.std(ddof=1) / np.sqrt(len(dl)), bt, int((dl > 0).sum()), int((dl < 0).sum()))
        for rep in (1, 0):
            lb = sum(LB_W[c] * res[(c, rep)][0] for c in LB_W)
            se = np.sqrt(sum((LB_W[c] * res[(c, rep)][1]) ** 2 for c in LB_W))
            print(f"[{arm}] fit h{1 - rep} -> report h{rep}: " + " | ".join(
                f"{c} {res[(c, rep)][0]:+.6f} ± {res[(c, rep)][1]:.6f} (tau {res[(c, rep)][2]}, +{res[(c, rep)][3]}/-{res[(c, rep)][4]})"
                for c in LB_W) + f" | LB-eq {lb:+.6f} ± {se:.6f} ({lb / se:.1f} SE)", flush=True)
            out[f"{arm}_h{rep}"] = dict(lb=lb, se=se, **{c: list(res[(c, rep)]) for c in LB_W})
    return out


def oracle(D):
    for c, (d, ex) in D.items():
        f0 = fb(d["tp"], d["npred"], d["n_true"])
        s = ex.y.to_numpy().astype(float)
        dl = apply(d, ex, s, 0.5, 0.5) - f0
        own = ex[ex.y].drop_duplicates("sx")
        print(f"{c}: frame SX {ex.sx.nunique():,}, touching holdout {ex[ex.g >= 0].sx.nunique():,}; base acc rows {int(ex.cur.sum()):,} "
              f"(prec {ex.y[ex.cur].mean():.3f}); oracle re-decision {dl.mean():+.6f} (h0 {dl[d['half'] == 0].mean():+.6f}, "
              f"h1 {dl[d['half'] == 1].mean():+.6f}); owner-in-group SX {len(own):,}, owner is a new pair {(own.in_set == 0).mean():.3f}", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    D = {c: load(c) for c in LB_W}
    oracle(D)
    out = run(sys.argv[1:] or list(ARMS), D)
    json.dump(out, open(W / "p58/d58.json", "w"), indent=1, default=float)
    print(f"{time.time() - t0:.0f}s")


def ef_gain(q, npo, lam, ps):
    """Expected per-S1 F0.5 gain of giving one more SX to a twin that holds npo other accepted pairs (assumed true) and
    lam further missed members, if the twin owns it with probability q. Empty twin: singleton with probability ps."""
    npo = npo.astype(float)
    g_tp = fb(npo + 1, npo + 1, npo + 1 + lam) - fb(npo, npo, npo + 1 + lam)
    l_fp = fb(npo, npo, npo + lam) - fb(npo, npo + 1, npo + lam)
    l_fp = np.where(npo == 0, ps, l_fp)
    return q * g_tp - (1 - q) * l_fp


def apply_ef(d, ex, s, lam, ps):
    """Re-decide every SX: the twin with the largest expected gain, iff that gain > 0."""
    tp, npred = d["tp"].copy(), d["npred"].copy()
    old = ex[ex.cur & (ex.g >= 0)]
    np.subtract.at(npred, old.g.to_numpy(), 1)
    np.subtract.at(tp, old.g.to_numpy(), old.y.to_numpy().astype(int))
    t = ex.assign(e=ef_gain(s, ex.n_acc.to_numpy(), lam, ps))
    best = t.loc[t.groupby("sx").e.idxmax()]
    new = best[(best.e.to_numpy() > 0) & (best.g.to_numpy() >= 0)]
    np.add.at(npred, new.g.to_numpy(), 1)
    np.add.at(tp, new.g.to_numpy(), new.y.to_numpy().astype(int))
    return fb(tp, npred, d["n_true"])
