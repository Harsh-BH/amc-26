"""P55: P44's learned combiner, transferred to France's decision stack and tested leave-one-country-out.

Frames: P42's LOCO chain analogues (P48 loco48.frame: target fold-0 SX-argmax pairs; g = source-only M2, r = source-only
bge CE, filled on g < 0.02 by P48's GPU job). Reference = the current France recipe analogue (t .99 / rescue .9 / gc .7 /
hn .8, caps 5/6, tau 0.1, density-mapped bars) + P48's veto (rescue / gc band pairs with r < 0.2).
Combiner = LightGBM on country-invariant inputs only: within-country densities of p and g (argmax pairs per S1 above
the pair), CE r, SX exclusivity (competitor mass / counts / logit gap over every candidate pair), S1 group counts, rank
and gaps, accepted counts, stage, region, hn relation. Direction A->B: fit on A's labels (frame loco_<B>_<A>, i.e. A's
pairs under B-trained models: the same out-of-domain situation France is in), bars picked on A (5-fold cross-fit),
applied to B (frame loco_<A>_<B>).
  arm B (veto)  drop accepted pairs with combiner c < v
  arm A (both)  the veto, then add free argmax pairs (non-sibling, not capped out) with c > u, best first, up to caps

    .venv/bin/python probes/P55-fr-combiner/p55.py loco [plain|resid]   # -> probes/P55-fr-combiner/loco_<mode>.json
"""
import json
import resource
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "probes/P48-fr-ce-all"), str(ROOT / "probes/P42-fr-joint-tune"), str(ROOT)]
import loco48 as L  # noqa: E402
import solution_p42 as S  # noqa: E402

W = ROOT / "data/work"
CI = {"India": 0, "US": 1}
CUR = L.CUR
V_GRID = (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4)
U_GRID = (0.8, 0.9, 0.95, 0.97, 0.98, 0.99)
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, lambda_l2=1.0,
              bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8, deterministic=True, verbose=-1, num_threads=4)
ROUNDS = 200
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def sx_feats(sx, g):
    """Per candidate pair: its SX's candidate count, count with g >= 0.5 and g mass excluding the pair, and the best
    competitor's g (every other S1 of the SX)."""
    o = np.lexsort((-g, sx))
    so = sx[o]
    first = np.r_[True, so[1:] != so[:-1]]
    start = np.flatnonzero(first)
    grp = np.cumsum(first) - 1
    n = np.diff(np.r_[start, len(o)])
    top1 = g[o][start]
    top2 = np.where(n > 1, g[o][np.minimum(start + 1, len(o) - 1)], 0.0)
    rank = np.arange(len(o)) - start[grp]
    out = np.empty((len(g), 4))
    gs = g[o]
    out[o, 0] = n[grp]
    out[o, 1] = np.bincount(grp, weights=(gs >= 0.5).astype(float))[grp] - (gs >= 0.5)
    out[o, 2] = np.bincount(grp, weights=gs)[grp] - gs
    out[o, 3] = np.where(rank == 0, top2[grp], top1[grp])
    return out


def dens(v, n):
    """Pairs of the frame with value >= v, per S1 (the recipe's density map, as a feature)."""
    q = np.sort(v)
    return (len(q) - np.searchsorted(q, v, side="left")) / n


def feats(s1, is2, g, r, p, hn_rel, ref, stage, sxf, n):
    """Country-invariant feature frame on the argmax pairs."""
    s1 = s1.astype(np.int64)
    key = s1 * 2 + is2
    rows = np.arange(len(s1))
    o, rk = S.rank_in(key, p, rows)
    po = p[o]
    same = np.r_[False, key[o][1:] == key[o][:-1]]
    gap_prev = np.full(len(s1), np.nan)
    gap_next = np.full(len(s1), np.nan)
    gap_prev[o[same]] = po[np.flatnonzero(same) - 1] - po[same]
    nxt = np.r_[same[1:], False]
    gap_next[o[nxt]] = po[nxt] - po[np.flatnonzero(nxt) + 1]
    cnt = np.bincount(key, minlength=2 * n)
    acc = np.bincount(key[ref], minlength=2 * n)
    mx = np.full(2 * n, -1.0)
    np.maximum.at(mx, key, p)
    oth = s1 * 2 + (~is2)
    lg = lambda x: S.logit(np.asarray(x, float))
    return pd.DataFrame({
        "dens_p": dens(p, n), "dens_g": dens(g, n), "lr": np.where(np.isnan(r), np.nan, lg(np.nan_to_num(r, nan=0.5))),
        "is2": is2.astype(np.int8), "hn_rel": hn_rel.astype(np.int8), "stage": stage.astype(np.int8),
        "region": np.select([g < S.LO, g >= S.HI], [0, 2], 1).astype(np.int8),
        "s1_rank": rk[np.argsort(o)], "gap_prev": gap_prev, "gap_next": gap_next,
        "n_same": cnt[key], "n_oth": cnt[oth], "acc_same": acc[key] - ref, "acc_oth": acc[oth],
        "max_oth": mx[oth], "s1_psum": np.bincount(s1, weights=p, minlength=n)[s1],
        "sx_n": sxf[:, 0], "sx_n05": sxf[:, 1], "sx_psum": sxf[:, 2], "sx_gap": lg(g) - lg(np.maximum(sxf[:, 3], 1e-6)),
    }).astype(np.float32)


def stage_code(st):
    return np.select([st["base"], st["rescue"], st["gc"], st["hn"]], [1, 2, 3, 4], 0)


def p48_veto(st, g, r, stages=("rescue", "gc")):
    prev = {"rescue": "base", "gc": "rescue", "hn": "gc"}
    added = np.any([st[s] & ~st[prev[s]] for s in stages], 0)
    return added & (g > S.LO) & (g < S.HI) & (np.nan_to_num(r, nan=1.0) < 0.2)


def loco_frame(dr):
    """LOCO target frame dr with features, the reference decision and the free set."""
    src, tgt = L.DIRS[dr]
    z = L.frame(dr)
    t = pq.read_table(W / f"p27/p6/S_holdout_{src}.parquet", columns=["sx", "country", "p", "am_all"],
                      filters=[("country", "=", CI[tgt])])
    sx, ga, am = t.column("sx").to_numpy(), t.column("p").to_numpy().astype(np.float64), t.column("am_all").to_numpy(zero_copy_only=False)
    del t
    assert am.sum() == len(z["g"]) and (ga[am] == z["g"]).all()
    sxf = sx_feats(sx.astype(np.int64), ga)[am]
    del sx, ga, am
    g, r = z["g"], z["r_src"]
    p = S.loco_p(g, z["r"], CUR["tau"], "chain")
    st, tm = L.recipe(z, dr, p)
    ref = st["hn"] & ~p48_veto(st, g, z["r"])
    free = L.free_mask(z, st, p, tm)
    n = len(z["ntrue"])
    X = feats(z["s1"], z["is2"], g, r, p, z["hn_rel"], ref, stage_code(st), sxf, n)
    return {"z": z, "X": X, "ref": ref, "free": free, "p": p, "n": n}


def fit(X, y, init, seed=0):
    ds = lgb.Dataset(X, y.astype(float), init_score=init, free_raw_data=True)
    return lgb.train(dict(PARAMS, seed=seed), ds, ROUNDS)


def score(b, X, init):
    raw = b.predict(X, raw_score=True) + (0 if init is None else init)
    return 1 / (1 + np.exp(-raw))


def decide(fr, c, v, u):
    """Arm B (u None): ref minus pairs with c < v. Arm A: also add free pairs with c > u up to the caps."""
    z, ref = fr["z"], fr["ref"]
    new = ref & ~(c < v) if v is not None else ref.copy()
    if u is not None:
        s1, is2 = z["s1"].astype(np.int64), z["is2"]
        n2, n3 = S.counts(s1, is2, new, fr["n"])
        cand = np.flatnonzero(fr["free"] & ~new & (c > u))
        o, rk = S.rank_in(s1 * 2 + is2, c, cand)
        take = o[rk < np.where(is2[o], CUR["cap"][0] - n2[s1[o]], CUR["cap"][1] - n3[s1[o]])]
        new[take] = True
    return new


def grid(fr, c):
    fref = L.F(fr["z"], fr["ref"])
    out = {}
    for v in V_GRID:
        out[f"B:{v}"] = L.cmp(fr["z"], fr["ref"], decide(fr, c, v, None), fref)
    for u in U_GRID:
        out[f"add:{u}"] = L.cmp(fr["z"], fr["ref"], decide(fr, c, None, u), fref)
        for v in V_GRID:
            out[f"A:{v}:{u}"] = L.cmp(fr["z"], fr["ref"], decide(fr, c, v, u), fref)
    return out


def best(g, arm):
    k = max((k for k in g if k.startswith(arm + ":")), key=lambda k: g[k]["dF"])
    return k, g[k]


def cross_fit(fr, mode, k=5):
    s1 = fr["z"]["s1"].astype(np.int64)
    fold = (s1 * 2654435761 % 2**32) % k
    c = np.zeros(len(s1))
    y = fr["z"]["y"]
    init = S.logit(fr["p"]) if mode == "resid" else None
    for f in range(k):
        tr = fold != f
        b = fit(fr["X"][tr], y[tr], None if init is None else init[tr])
        c[~tr] = score(b, fr["X"][~tr], None if init is None else init[~tr])
    return c


def loco(mode="plain"):
    t0 = time.time()
    frames = {dr: loco_frame(dr) for dr in L.DIRS}
    print(f"frames {time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)
    out = {"mode": mode}
    # direction name = source labels -> target:  India->US trains on loco_US_India (India pairs), tests on loco_India_US
    for name, trn, tst in (("India->US", "US_India", "India_US"), ("US->India", "India_US", "US_India")):
        a, b = frames[trn], frames[tst]
        ca = cross_fit(a, mode)
        ga = grid(a, ca)
        init_a = S.logit(a["p"]) if mode == "resid" else None
        bst = fit(a["X"], a["z"]["y"], init_a)
        cb = score(bst, b["X"], S.logit(b["p"]) if mode == "resid" else None)
        gb = grid(b, cb)
        R = out[name] = {"src_ref_F": round(float(L.F(a["z"], a["ref"]).mean()), 5), "tgt_ref_F": round(float(L.F(b["z"], b["ref"]).mean()), 5)}
        for arm in ("B", "A", "add"):
            k, v = best(ga, arm)
            R[f"pick_{arm}"] = {"cell": k, "src": v, "tgt": gb[k]}
            kt, vt = best(gb, arm)
            R[f"oracle_{arm}"] = {"cell": kt, "tgt": vt}
        # rate transfer for the veto: veto the same share of accepted pairs as the source's best cell
        kB = R["pick_B"]["cell"]
        vB = float(kB.split(":")[1])
        share = float((a["ref"] & (ca < vB)).sum() / a["ref"].sum())
        vt = float(np.quantile(cb[b["ref"]], share)) if share > 0 else 0.0
        R["rate_B"] = {"share": round(share, 5), "v_tgt": round(vt, 4), "tgt": L.cmp(b["z"], b["ref"], decide(b, cb, vt, None), L.F(b["z"], b["ref"]))}
        R["tgt_grid"] = gb
        R["src_grid"] = ga
        imp = pd.Series(bst.feature_importance("gain"), index=a["X"].columns).sort_values(ascending=False)
        R["imp"] = {k: round(float(v / imp.sum()), 3) for k, v in imp.items()}
        R["auc_tgt_ref"] = auc(b["z"]["y"][b["ref"]], cb[b["ref"]])
        print(name, json.dumps({k: v for k, v in R.items() if k not in ("tgt_grid", "src_grid")}), flush=True)
    out["runtime_s"] = round(time.time() - t0)
    out["peak_gb"] = round(peak(), 2)
    (HERE / f"loco_{mode}.json").write_text(json.dumps(out, indent=1))
    print(f"{time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)


def auc(y, s):
    o = np.argsort(s)
    rk = np.empty(len(s))
    rk[o] = np.arange(1, len(s) + 1)
    npos = y.sum()
    return round(float((rk[y].sum() - npos * (npos + 1) / 2) / max(npos * (len(y) - npos), 1)), 4)


def _selfcheck():
    sx = np.array([0, 0, 1, 0]); g = np.array([.9, .5, .7, .2])
    f = sx_feats(sx, g)
    assert f[:, 0].tolist() == [3, 3, 1, 3] and f[:, 1].tolist() == [1, 1, 0, 2]
    assert np.allclose(f[:, 2], [.7, 1.1, 0, 1.4]) and np.allclose(f[:, 3], [.5, .9, 0, .9])
    assert dens(np.array([.1, .5, .5, .9]), 2).tolist() == [2, 1.5, 1.5, .5]
    print("p55 self-check ok")


if __name__ == "__main__":
    _selfcheck()


# ------------------------------------------------------------------ France side + breakdowns
def france_frame():
    """France argmax pairs (france.npz, P42's decide at the current recipe) with the same features; ref = recipe minus
    P48's veto; vetoable = recipe pairs whose p is the model's own (no tier / bxneg rule moved it)."""
    import p42
    from build48 import veto_mask
    z = p42.fr_frame()
    p = p42.fr_p(z, CUR["tau"])
    sx, s1, is2 = z["sx"].astype(np.int64), z["s1"].astype(np.int64), z["is2"]
    am = S.argmax_sx(sx, p)
    st, p_ = p42.fr_decide(z, CUR)
    assert (p_ == p).all()
    g, r = z["g"], z["r"].copy()
    i1, ix = p42.fr_ids()
    lo = pd.read_parquet(W / "p48/r_lo_France.parquet")
    k = pd.DataFrame({"s1": i1[z["s1"]], "sx": ix[z["sx"]]}).reset_index().merge(lo, on=["s1", "sx"])
    ki = k["index"].to_numpy()
    r[ki] = np.where(np.isnan(r[ki]), k.r.to_numpy(), r[ki])
    hn = np.load(W / "p52/france_hn_rel.npy")
    sxf = sx_feats(sx, g)
    ref = st["hn"] & ~veto_mask(z, st, 0.2, ("rescue", "gc"))
    n = int(z["n_s1"])
    _, free = S.base(s1, is2, p, am, z["sib"], CUR["t"], tuple(CUR["cap"]))
    band = (g > S.LO) & (g < S.HI)
    bl = 1 / (1 + np.exp(-(S.W_FR[0] * S.logit(g) + S.W_FR[1] * S.logit(np.nan_to_num(z["r"], nan=0.5)) + S.W_FR[2])))
    own = np.where(band, np.abs(z["p0"] - bl) <= 1e-4, z["p0"].astype(np.float32) == g.astype(np.float32))
    X = feats(s1[am], is2[am], g[am], r[am], p[am], hn[am], ref[am], stage_code(st)[am], sxf[am], n)
    rows = np.flatnonzero(am)
    return {"z": z, "X": X, "rows": rows, "ref": ref[am], "free": free[am], "own": own[am], "p": p[am], "n": n,
            "r_cov": float(np.isfinite(r[am]).mean()), "stage": stage_code(st)[am], "g": g[am]}


def bag(X, y, init, seeds=(0, 1, 2)):
    return [fit(X, y, init, s) for s in seeds]


def bag_score(bs, X, init):
    lg = np.mean([S.logit(score(b, X, init)) for b in bs], 0)
    return 1 / (1 + np.exp(-lg))


def breakdown(fr, c, v):
    """Veto at v, split by stage x region: pairs, false share, and the dF of vetoing that cell alone."""
    z, ref = fr["z"], fr["ref"]
    fref = L.F(z, ref)
    st, rg = fr["X"]["stage"].to_numpy(), fr["X"]["region"].to_numpy()
    m = ref & (c < v)
    out = {}
    for a in (1, 2, 3, 4):
        for b in (0, 1, 2):
            mm = m & (st == a) & (rg == b)
            if mm.sum():
                out[f"st{a}/rg{b}"] = L.cmp(z, ref, ref & ~mm, fref)
    return out


def ana(mode="plain"):
    """Seed-bagged veto on both LOCO directions (stability + breakdown), then the France profile at the same bars."""
    t0 = time.time()
    frames = {dr: loco_frame(dr) for dr in L.DIRS}
    out = {}
    for name, trn, tst in (("India->US", "US_India", "India_US"), ("US->India", "India_US", "US_India")):
        a, b = frames[trn], frames[tst]
        ia, ib = (S.logit(a["p"]), S.logit(b["p"])) if mode == "resid" else (None, None)
        bs = bag(a["X"], a["z"]["y"], ia)
        cb = bag_score(bs, b["X"], ib)
        fref = L.F(b["z"], b["ref"])
        R = out[name] = {f"B:{v}": L.cmp(b["z"], b["ref"], decide(b, cb, v, None), fref) for v in V_GRID}
        R["veto_rate"] = {str(v): round(float((b["ref"] & (cb < v)).sum() / b["ref"].sum()), 5) for v in V_GRID}
        R["breakdown_0.4"] = breakdown(b, cb, 0.4)
        R["breakdown_0.2"] = breakdown(b, cb, 0.2)
        print(name, json.dumps(R), flush=True)
    # France: pooled fit on both frames (India + US labels)
    Xp = pd.concat([frames[d]["X"] for d in L.DIRS], ignore_index=True)
    yp = np.concatenate([frames[d]["z"]["y"] for d in L.DIRS])
    ip = np.concatenate([S.logit(frames[d]["p"]) for d in L.DIRS]) if mode == "resid" else None
    del frames
    bs = bag(Xp, yp, ip)
    del Xp, yp, ip
    f = france_frame()
    cf = bag_score(bs, f["X"], S.logit(f["p"]) if mode == "resid" else None)
    np.save(W / f"p55_fr_c_{mode}.npy", cf.astype(np.float32))
    for i, b in enumerate(bs):
        b.save_model(str(W / f"p55_{mode}_s{i}.txt"))
    st, rg = f["X"]["stage"].to_numpy(), f["X"]["region"].to_numpy()
    F_ = out["France"] = {"r_cov": round(f["r_cov"], 4), "ref": int(f["ref"].sum()), "own_ref": int((f["ref"] & f["own"]).sum())}
    for v in V_GRID:
        m = f["ref"] & (cf < v)
        F_[f"veto:{v}"] = {"all": int(m.sum()), "own": int((m & f["own"]).sum()), "rate_own": round(float((m & f["own"]).sum() / f["ref"].sum()), 5),
                           "by": {f"st{a}/rg{b}": int((m & f["own"] & (st == a) & (rg == b)).sum()) for a in (1, 2, 3, 4) for b in (0, 1, 2)
                                  if (m & f["own"] & (st == a) & (rg == b)).any()}}
    print("France", json.dumps(F_), flush=True)
    out["runtime_s"], out["peak_gb"] = round(time.time() - t0), round(peak(), 2)
    (HERE / f"ana_{mode}.json").write_text(json.dumps(out, indent=1))
    print(f"{time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)




DIRN = (("India->US", "US_India", "India_US"), ("US->India", "India_US", "US_India"))
V2 = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
U2 = (0.9, 0.95, 0.97, 0.98, 0.99)
SCOPES = {"all": lambda st, rg: st > 0, "added": lambda st, rg: st >= 2, "base": lambda st, rg: st == 1,
          "nobasehi": lambda st, rg: ~((st == 1) & (rg == 2)), "hn": lambda st, rg: st == 4,
          "band": lambda st, rg: rg == 1}      # = nobasehi on LOCO (its rescue / gc / hn pairs are all band pairs)


def fr_own(z):
    """France pairs whose p is the model's own: key tier (p := 1) and bxneg (p := max(p, 0.99)) moved the rest."""
    p0, g = z["p0"].astype(np.float32), z["g"]
    return ~(((p0 == np.float32(1.0)) & (g < 0.99995)) | ((p0 == np.float32(0.99)) & (g.astype(np.float32) != np.float32(0.99))))


def scores(mode="plain", seeds=(0, 1, 2)):
    """Bagged combiner scores on both LOCO targets and France (pooled fit), cached in data/work/p55/."""
    d = W / "p55"
    d.mkdir(exist_ok=True)
    frames = {dr: loco_frame(dr) for dr in L.DIRS}
    ini = lambda f: S.logit(f["p"]) if mode == "resid" else None
    for name, trn, tst in DIRN:
        bs = bag(frames[trn]["X"], frames[trn]["z"]["y"], ini(frames[trn]), seeds)
        np.save(d / f"cb_{mode}_{tst}.npy", bag_score(bs, frames[tst]["X"], ini(frames[tst])))
    Xp = pd.concat([frames[dr]["X"] for dr in L.DIRS], ignore_index=True)
    yp = np.concatenate([frames[dr]["z"]["y"] for dr in L.DIRS])
    ip = np.concatenate([S.logit(frames[dr]["p"]) for dr in L.DIRS]) if mode == "resid" else None
    del frames
    bs = bag(Xp, yp, ip, seeds)
    del Xp, yp, ip
    for i, b in enumerate(bs):
        b.save_model(str(d / f"fr_{mode}_s{i}.txt"))
    f = france_frame()
    np.save(d / f"cf_{mode}.npy", bag_score(bs, f["X"], ini(f)))
    print(f"scores {mode} done, peak {peak():.2f} GB", flush=True)


def ana2(mode="plain"):
    t0 = time.time()
    d = W / "p55"
    frames = {dr: loco_frame(dr) for dr in L.DIRS}
    out = {"mode": mode}
    cell = {}                                   # per-pair F-sum (dF x n_S1) by stage x region, for France reweighting
    for name, trn, tst in DIRN:
        b = frames[tst]
        cb = np.load(d / f"cb_{mode}_{tst}.npy")
        fref = L.F(b["z"], b["ref"])
        st, rg = b["X"]["stage"].to_numpy(), b["X"]["region"].to_numpy()
        R = out[name] = {}
        for sc, fn in SCOPES.items():
            for v in V2:
                m = b["ref"] & (cb < v) & fn(st, rg)
                R[f"B:{sc}:{v}"] = L.cmp(b["z"], b["ref"], b["ref"] & ~m, fref) | {"rate": round(float(m.sum() / b["ref"].sum()), 5)}
        for u in U2:
            new = decide(b, cb, None, u)
            R[f"add:{u}"] = L.cmp(b["z"], b["ref"], new, fref) | {"rate": round(float((new & ~b["ref"]).sum() / b["ref"].sum()), 5)}
            for v in (0.3, 0.4, 0.5):
                R[f"A:{v}:{u}"] = L.cmp(b["z"], b["ref"], decide(b, cb, v, u), fref)
        for v in V2:
            m = b["ref"] & (cb < v)
            for a in (1, 2, 3, 4):
                for r_ in (0, 1, 2):
                    mm = m & (st == a) & (rg == r_)
                    if mm.sum():
                        c = L.cmp(b["z"], b["ref"], b["ref"] & ~mm, fref)
                        cell[(name, v, a, r_)] = c["dF"] * b["n"] / mm.sum()
        print(name, f"{time.time() - t0:.0f}s", flush=True)
    del frames
    # France: counts at the same bars (own pairs only) and the LOCO-cell-reweighted dF estimate
    import p42
    z = p42.fr_frame()
    f = france_frame()
    cf = np.load(d / f"cf_{mode}.npy")
    own = fr_own(z)[f["rows"]]
    st, rg = f["X"]["stage"].to_numpy(), f["X"]["region"].to_numpy()
    F_ = out["France"] = {"ref": int(f["ref"].sum()), "own_ref": int((f["ref"] & own).sum())}
    for sc, fn in SCOPES.items():
        for v in V2:
            m = f["ref"] & own & (cf < v) & fn(st, rg)
            e = {nm: 0.0 for nm, _, _ in DIRN}
            for a in (1, 2, 3, 4):
                for r_ in (0, 1, 2):
                    k = int((m & (st == a) & (rg == r_)).sum())
                    for nm, _, _ in DIRN:
                        e[nm] += k * cell.get((nm, v, a, r_), 0.0)
            F_[f"B:{sc}:{v}"] = {"n": int(m.sum()), "rate": round(float(m.sum() / f["ref"].sum()), 5),
                                 "est": {nm: round(x / f["n"], 6) for nm, x in e.items()}}
    for u in U2:
        c = f["free"] & ~f["ref"] & own & (cf > u)
        F_[f"add:{u}"] = {"cand": int(c.sum()), "rate": round(float(c.sum() / f["ref"].sum()), 5),
                          "by_region": np.bincount(rg[c].astype(np.int64), minlength=3).tolist()}
    out["runtime_s"], out["peak_gb"] = round(time.time() - t0), round(peak(), 2)
    (HERE / f"ana2_{mode}.json").write_text(json.dumps(out, indent=1))
    print(f"{time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)




def build(base, out, spec):
    """output/<out> = output/<base> with France re-scored by the pooled combiner (cached cf_<mode>.npy): drop own recipe
    pairs with c < v (scope), then (u set) add free own argmax pairs with c > u, best first, up to caps 5 / 6 counted on
    the base's own France lines, skipping SX already listed for any France S1. India / US lines byte-identical;
    candidate_pairs.tsv and code.zip are hard links (every added pair is asserted to be a candidate)."""
    import os
    import p42
    t0 = time.time()
    cfg = dict(kv.split("=") for kv in spec.split(","))
    mode, sc, v = cfg.get("mode", "plain"), cfg.get("scope", "all"), float(cfg["v"])
    z = p42.fr_frame()
    f = france_frame()
    cf = np.load(W / "p55" / f"cf_{mode}.npy")
    own = fr_own(z)[f["rows"]]
    st, rg = f["X"]["stage"].to_numpy(), f["X"]["region"].to_numpy()
    i1, ix = p42.fr_ids()
    a1, ax = i1[z["s1"][f["rows"]]], ix[z["sx"][f["rows"]]]
    rm = f["ref"] & own & (cf < v) & SCOPES[sc](st, rg)
    lines = p42.read_fr(ROOT / "output" / base)
    cur = {k: set(x.split(",")) if x else set() for k, x in lines.items()}
    del lines
    minus = {}
    for a, b in zip(a1[rm], ax[rm]):
        minus.setdefault(a, set()).add(b)
    plus = {}
    if "u" in cfg:
        used = set().union(*cur.values())
        cnt = {}
        for k, s in cur.items():
            s = s - minus.get(k, set())
            cnt[k] = [sum(x.startswith("S2-") for x in s), sum(x.startswith("S3-") for x in s)]
        sup = ~((rg == 2) & (f["p"] > 0))       # LOCO's free g >= .98 pairs are all tau-vetoed (p 0): France's p > 0 ones are unseen
        c = np.flatnonzero(f["free"] & ~f["ref"] & own & sup & (cf > float(cfg["u"])))
        for i in c[np.argsort(-cf[c], kind="stable")]:
            a, b = a1[i], ax[i]
            j = 0 if b.startswith("S2-") else 1
            if b in used or cnt[a][j] >= CUR["cap"][j]:
                continue
            used.add(b)
            cnt[a][j] += 1
            plus.setdefault(a, set()).add(b)
        need = dict(plus)
        with open(ROOT / "output" / base / "candidate_pairs.tsv", encoding="utf-8") as fi:
            next(fi)
            for line in fi:
                k, vv = line.rstrip("\n").split("\t", 1)
                if k in need:
                    assert need.pop(k) <= set(vv.split(",")), k
        assert not need, len(need)
    src, od = ROOT / "output" / base, ROOT / "output" / out
    od.mkdir(parents=True, exist_ok=True)
    fr = set(i1)
    n_fr = n_chg = n_rm = n_add = n_empty = 0
    with open(src / "matching_results.tsv", encoding="utf-8", newline="") as fi, \
            open(od / "matching_results.tsv", "w", encoding="utf-8", newline="") as fo:
        fo.write(fi.readline())
        for line in fi:
            k = line.split("\t", 1)[0]
            if k in fr and (k in minus or k in plus):
                n_fr += 1
                old = cur[k]
                new = (old - minus.get(k, set())) | plus.get(k, set())
                n_rm += len(old - new)
                n_add += len(new - old)
                n_empty += bool(old) and not new
                n_chg += new != old
                fo.write(f"{k}\t{','.join(sorted(new))}\n")
                continue
            n_fr += k in fr
            fo.write(line)
    assert n_fr == len(i1), (n_fr, len(i1))
    for name in ("candidate_pairs.tsv", "code.zip"):
        (od / name).unlink(missing_ok=True)
        os.link(src / name, od / name)
    info = {"base": base, "spec": spec, "veto_pairs": int(rm.sum()), "removed_in_file": n_rm, "added_in_file": n_add,
            "s1_changed": n_chg, "s1_made_empty": n_empty,
            "veto_by_stage_region": {f"st{a}/rg{b}": int((rm & (st == a) & (rg == b)).sum()) for a in (1, 2, 3, 4) for b in (0, 1, 2) if (rm & (st == a) & (rg == b)).any()},
            "added_pairs": sum(len(x) for x in plus.values()),
            "runtime_s": round(time.time() - t0)}
    (od / "p55_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)


if __name__ == "__main__" and len(sys.argv) > 1:
    if sys.argv[1] == "build":
        build(*sys.argv[2:5])
    else:
        {"loco": loco, "ana": ana, "scores": scores, "ana2": ana2}[sys.argv[1]](*(sys.argv[2:3] or ["plain"]))
