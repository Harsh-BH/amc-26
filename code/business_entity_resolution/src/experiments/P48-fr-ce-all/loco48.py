"""P48 LOCO: does the CE outside the band (0.02 < g < 0.98) help France's recipe? P42's chain analogue (P6 source-only
M2 g, P30 source-only bge r, OOD blend, veto tau 0.1) at the current France recipe (t 0.99 / rescue 0.9 / gc 0.7 /
hn 0.8, caps 5/6), vs three arms:
  veto   drop accepted pairs whose CE r < v (post-decision; 'hi' = only g >= 0.98 pairs, 'all' = every accepted pair)
  resc   add free (argmax, non-sibling, under cap) pairs with g < 0.02 whose CE > u   (scope: any / empty / onesided S1)
  blend  the OOD band blend applied to g >= 0.98 too (instead of the tau veto); France thresholds re-mapped by density
CE for the low side: 'dom' = P45's in-domain r_all (optimistic: the CE saw the target country's other folds; P45's
pruned pairs only), 'src' = source-only CE scores (P48 GPU job, if fetched: data/work/p48/r_lo_<dr>.parquet).

    .venv/bin/python probes/P48-fr-ce-all/loco48.py [arm ...]      # -> probes/P48-fr-ce-all/loco.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "probes/P42-fr-joint-tune"), str(ROOT)]
import solution_p42 as S  # noqa: E402

W = ROOT / "data/work"
G = json.loads((W / "p42/sm/grid.json").read_text())
CUR = {"tau": 0.1, "t": 0.99, "cap": (5, 6), "r": 0.9, "gc": 0.7, "hn": 0.8}      # sub46_on43's P42 part
DIRS = {"US_India": ("US", "India"), "India_US": ("India", "US")}


def frame(dr):
    src, tgt = DIRS[dr]
    z = dict(np.load(W / f"p42/loco_{dr}.npz"))
    ids = pd.read_parquet(W / f"p48/loco_{dr}_ids.parquet")
    z["r_dom"] = ids.r_all.to_numpy(np.float64)
    f = W / f"p48/r_lo_{dr}.parquet"                     # source-only CE on the g < 0.02 rows (GPU job)
    z["r_src"] = z["r"].copy()
    if f.exists():
        lo = pd.read_parquet(f)
        k = ids[["s1", "sx"]].reset_index().merge(lo, on=["s1", "sx"])
        z["r_src"][k["index"].to_numpy()] = np.where(np.isnan(z["r_src"][k["index"].to_numpy()]), k.r.to_numpy(), z["r_src"][k["index"].to_numpy()])
    return z


def recipe(z, dr, p, tau=CUR["tau"], dens=None, an="chain"):
    """Current recipe's stage outputs on LOCO p (thresholds density-mapped from France)."""
    s1, is2, n = z["s1"].astype(np.int64), z["is2"], len(z["ntrue"])
    am = np.ones(len(s1), bool)
    Gx = G if dens is None else dict(G, fr_dens={str(tau): dens})
    tm = S.fr_to_loco(Gx, an, dr, tau, p, am, n)
    mp = lambda x: None if x is None else tm[str(x)]
    st = S.decide(s1, is2, p, am, z["hn_rel"] == 2, z["hn_rel"] == 1, n, mp(CUR["t"]), CUR["cap"], mp(CUR["r"]),
                  mp(CUR["gc"]), mp(CUR["hn"]), steps=True)
    return st, tm


def F(z, pred):
    s1, n = z["s1"].astype(np.int64), len(z["ntrue"])
    return S.fbeta(np.bincount(s1[pred & z["y"]], minlength=n), np.bincount(s1[pred], minlength=n), z["ntrue"].astype(np.int64))


def cmp(z, ref, new, fref):
    f = F(z, new)
    d, hv = f - fref, z["hv"]
    add, rm = new & ~ref, ref & ~new
    se = lambda x: float(x.std() / np.sqrt(len(x)))
    return {"dF": round(float(d.mean()), 6), "se": round(se(d), 6), "dF_h0": round(float(d[hv == 0].mean()), 6),
            "dF_h1": round(float(d[hv == 1].mean()), 6), "se_h1": round(se(d[hv == 1]), 6),
            "add": int(add.sum()), "add_prec": round(float(z["y"][add].mean()), 3) if add.any() else None,
            "rm": int(rm.sum()), "rm_false": round(float(1 - z["y"][rm].mean()), 3) if rm.any() else None}


def free_mask(z, st, p, tm):
    """P42 base()'s free set: argmax, non-sibling, not capped out."""
    s1, is2, n = z["s1"].astype(np.int64), z["is2"], len(z["ntrue"])
    _, free = S.base(s1, is2, p, np.ones(len(s1), bool), z["hn_rel"] == 2, tm[str(CUR["t"])], CUR["cap"])
    return free


def rescue(z, pred, free, cand, scope):
    """Add cand pairs (best CE first) to pred up to the caps; scope any / empty / onesided (missing source only)."""
    s1, is2, n = z["s1"].astype(np.int64), z["is2"], len(z["ntrue"])
    n2, n3 = S.counts(s1, is2, pred, n)
    m2, m3 = n2[s1], n3[s1]
    ok = {"any": np.ones(len(s1), bool), "empty": m2 + m3 == 0,
          "onesided": np.where(is2, (m2 == 0) & (m3 > 0), (m3 == 0) & (m2 > 0))}[scope]
    c = np.flatnonzero(free & ~pred & ok & cand)
    return c, m2, m3


def add_capped(z, pred, c, score, m2, m3, one=False):
    s1, is2 = z["s1"].astype(np.int64), z["is2"]
    if one:
        o, rk = S.rank_in(s1, score, c)
        take = o[rk == 0]
    else:
        o, rk = S.rank_in(s1 * 2 + is2, score, c)
        take = o[rk < np.where(is2[o], CUR["cap"][0] - m2[o], CUR["cap"][1] - m3[o])]
    out = pred.copy()
    out[take] = True
    return out


def fr_dens_blend():
    """France argmax densities for the current recipe's bars with the OOD blend on g >= 0.98 (rule-exempt pairs, where
    the tier / bxneg rules moved p off g, keep their p)."""
    z = np.load(W / "p42/france.npz")
    g, r, p0 = z["g"], z["r"], z["p0"]
    hi = (g >= S.HI) & ~np.isnan(r) & (p0.astype(np.float32) == g.astype(np.float32))
    p = p0.copy()
    p[hi] = 1 / (1 + np.exp(-(S.W_FR[0] * S.logit(g[hi]) + S.W_FR[1] * S.logit(r[hi]) + S.W_FR[2])))
    am = S.argmax_sx(z["sx"].astype(np.int64), p)
    n = int(z["n_s1"])
    return {str(x): S.sdens(p, am, S.T_FR, x, n) for x in (CUR["t"], CUR["r"], CUR["gc"], CUR["hn"])}, int(hi.sum())


def main(arms):
    t0 = time.time()
    out = {}
    lg = S.logit
    for dr in DIRS:
        z = frame(dr)
        g, y = z["g"], z["y"]
        p = S.loco_p(g, z["r"], CUR["tau"], "chain")
        st, tm = recipe(z, dr, p)
        ref = st["hn"]
        fref = F(z, ref)
        free = free_mask(z, st, p, tm)
        R = out[dr] = {"t_map": {k: round(v, 4) for k, v in tm.items() if float(k) in (0.99, 0.9, 0.7, 0.8)},
                       "ref_F": round(float(fref.mean()), 5), "ref_pairs": int(ref.sum())}
        lo = g < S.LO
        R["low"] = {"rows": int(lo.sum()), "true": int((lo & y).sum()), "free": int((lo & free & ~ref).sum()),
                    "free_true": int((lo & free & ~ref & y).sum()),
                    "dom_cov": round(float((~np.isnan(z["r_dom"][lo])).mean()), 3),
                    "src_cov": round(float((~np.isnan(z["r_src"][lo])).mean()), 3)}
        print(dr, R, flush=True)
        if "veto" in arms:
            for which, rr in (("src", z["r"]), ("dom", z["r_dom"])):
                for scope in ("hi", "all"):
                    for v in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7):
                        m = ref & (np.nan_to_num(rr, nan=1.0) < v) & ((g >= S.HI) if scope == "hi" else True)
                        R[f"veto:{which}:{scope}:{v}"] = cmp(z, ref, ref & ~m, fref)
        if "resc" in arms:
            for which in ("dom", "src"):
                rr = z[f"r_{which}"]
                if np.isnan(rr[lo]).all():
                    continue
                for scope in ("any", "empty", "onesided"):
                    for u in (0.5, 0.7, 0.9, 0.95, 0.98, 0.99, 0.995):
                        cand = lo & (np.nan_to_num(rr, nan=0.0) > u)
                        c, m2, m3 = rescue(z, ref, free, cand, scope)
                        R[f"resc:{which}:{scope}:{u}"] = cmp(z, ref, add_capped(z, ref, c, rr, m2, m3, one=scope == "empty"), fref)
        if "vdec" in arms:                                  # where the post-decision veto (source-only r) bites
            rr = np.nan_to_num(z["r"], nan=1.0)
            stage = np.select([st["base"], st["rescue"], st["gc"], st["hn"]], ["base", "rescue", "gc", "hn"], "none")
            reg = np.select([g < S.LO, g >= S.HI], ["lo", "hi"], "band")
            for v in (0.05, 0.1, 0.2, 0.3):
                m = ref & (rr < v)
                R[f"vdec:{v}"] = {f"{a}/{b}": [int((m & (stage == a) & (reg == b)).sum()), round(float(1 - y[m & (stage == a) & (reg == b)].mean()), 3)]
                                  for a in ("base", "rescue", "gc", "hn") for b in ("band", "hi") if (m & (stage == a) & (reg == b)).any()}
                for sc, keep in (("band", reg == "band"), ("base", stage == "base"), ("added", stage != "base"),
                                 ("band_added", (reg == "band") & (stage != "base"))):
                    R[f"veto:src:{sc}:{v}"] = cmp(z, ref, ref & ~(m & keep), fref)
                # pre-decision: band pairs with r < v get p = 0 (the tau veto extended to the band), thresholds unchanged
                pv = np.where((reg == "band") & (rr < v), 0.0, p)
                s1_, is2_, n_ = z["s1"].astype(np.int64), z["is2"], len(z["ntrue"])
                mp = lambda x: tm[str(x)]
                pre = S.decide(s1_, is2_, pv, np.ones(len(g), bool), z["hn_rel"] == 2, z["hn_rel"] == 1, n_, mp(CUR["t"]), CUR["cap"],
                               mp(CUR["r"]), mp(CUR["gc"]), mp(CUR["hn"]))
                R[f"veto:pre_band:{v}"] = cmp(z, ref, pre, fref)
        if "vfine" in arms:                                 # France-applicable scope: band pairs added by rescue / gc / hn
            rr = np.nan_to_num(z["r"], nan=1.0)
            added = (g > S.LO) & (g < S.HI) & ~st["base"]
            for v in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
                R[f"vadd:{v}"] = cmp(z, ref, ref & ~(added & (rr < v)), fref)
        if "vstage" in arms:                                # per-stage veto, chain and m2 (P37 M2-only) analogues
            rr = np.nan_to_num(z["r"], nan=1.0)
            band = (g > S.LO) & (g < S.HI)
            for an in ("chain", "m2"):
                pa_ = p if an == "chain" else S.loco_p(g, z["r"], CUR["tau"], "m2")
                sa, _ = recipe(z, dr, pa_, an=an) if an == "m2" else (st, tm)
                ra, fa = sa["hn"], F(z, sa["hn"])
                for nm, m in (("rescue", sa["rescue"] & ~sa["base"]), ("gc", sa["gc"] & ~sa["rescue"]), ("hn", sa["hn"] & ~sa["gc"]),
                              ("gc+rescue", sa["gc"] & ~sa["base"]), ("all3", sa["hn"] & ~sa["base"])):
                    for v in (0.1, 0.2, 0.3):
                        R[f"vst:{an}:{nm}:{v}"] = cmp(z, ra, ra & ~(m & band & (rr < v)), fa)
        if "blend" in arms:
            dens, nhi = fr_dens_blend()
            pb = S.loco_p(g, z["r"], 0.0, "chain")
            h = g >= S.HI
            pb[h] = 1 / (1 + np.exp(-(S.W_FR[0] * lg(g[h]) + S.W_FR[1] * lg(z["r"][h]) + S.W_FR[2])))
            stb, tmb = recipe(z, dr, pb, tau=0.0, dens=dens)
            R["blend:hi"] = cmp(z, ref, stb["hn"], fref) | {"fr_dens": {k: round(v, 4) for k, v in dens.items()}, "fr_hi_blended": nhi}
        for k, v in R.items():
            if ":" in k:
                print(dr, k, v, flush=True)
    (HERE / f"loco{'_' + '_'.join(arms) if arms else ''}.json").write_text(json.dumps(out, indent=1))
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:] or ["veto", "resc", "blend"])
