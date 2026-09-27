"""P52: France's under-confidence as calibration. Every arm is a label-free PROCEDURE (source labels + target unlabelled)
that outputs the target's recipe bars; it is scored on LOCO against the current recipe analogue (P42 density-mapped France
bars t .99 / r .9 / gc .7 / hn .8, caps 5/6, tau 0.1, + P48's rescue/gc band veto r < 0.2).

  source S = the source country's OWN fold 0 (prep52.py: same source-only M2 g, in-domain CE r), labelled
  target T = P42's LOCO frame (source-only g and r on the other country's fold 0), labels used only to score

Arms (analogue p = chain | m2, both directions):
  naive   source-optimal bars (coordinate descent on S), applied to T raw
  em      BCTS (a*logit p + b, fit on S) -> Saerens EM prior pi_T on T -> logit shift -> source-optimal bars in q space
  emo     the same with the TRUE pi_T (the ceiling of any prior-shift correction)
  qmap    T's per-S1 top-1 p quantile-mapped onto S's top-1 distribution (monotone map of every T pair) -> source bars
  iw      source bars re-tuned on S with S1 weights p(T|x)/p(S|x) from an S1-level domain classifier -> applied raw

    .venv/bin/python probes/P52-fr-calib/p52.py loco [arms...]     # -> probes/P52-fr-calib/loco.json
    .venv/bin/python probes/P52-fr-calib/p52.py                    # self-check
"""
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "probes/P42-fr-joint-tune"), str(ROOT)]
import solution_p42 as S  # noqa: E402

W = ROOT / "data/work"
CUR = {"tau": 0.1, "t": 0.99, "cap": (5, 6), "r": 0.9, "gc": 0.7, "hn": 0.8}
VETO = 0.2                                              # P48: rescue/gc-added band pairs with r < 0.2 leave
DIRS = {"US_India": ("US", "India"), "India_US": ("India", "US")}
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
sig = lambda x: 1 / (1 + np.exp(-x))


def frames(dr):
    src, _ = DIRS[dr]
    T = dict(np.load(W / f"p42/loco_{dr}.npz"))
    Sf = dict(np.load(W / f"p52/src_{src}.npz"))
    for z in (T, Sf):
        z["s1"] = z["s1"].astype(np.int64)
        z["n"] = len(z["ntrue"])
        z["sib"], z["hn_eq"] = z["hn_rel"] == 2, z["hn_rel"] == 1
        z["am"] = np.ones(len(z["s1"]), bool)                     # frames hold SX-argmax pairs only
    return Sf, T


def run(z, p, bars, steps=False, veto=True):
    """The France recipe on frame z at bars (t, r, gc, hn in p's own space; None = stage off), then P48's veto."""
    st = S.decide(z["s1"], z["is2"], p, z["am"], z["sib"], z["hn_eq"], z["n"], bars["t"], CUR["cap"], bars["r"], bars["gc"],
                  bars["hn"], steps=True)
    g = z["g"]
    band = (g > S.LO) & (g < S.HI)
    pred = st["hn"] & ~((st["gc"] & ~st["base"]) & band & (np.nan_to_num(z["r"], nan=1.0) < VETO)) if veto else st["hn"]
    return (st, pred) if steps else pred


def fvec(z, pred):
    s1, n = z["s1"], z["n"]
    return S.fbeta(np.bincount(s1[pred & z["y"]], minlength=n), np.bincount(s1[pred], minlength=n), z["ntrue"].astype(np.int64))


def cmp(z, ref, new, fref):
    f = fvec(z, new)
    d, hv = f - fref, z["hv"]
    se = lambda x: float(x.std() / np.sqrt(len(x)))
    add, rm = new & ~ref, ref & ~new
    return {"dF": round(float(d.mean()), 6), "se": round(se(d), 6), "dF_h0": round(float(d[hv == 0].mean()), 6),
            "dF_h1": round(float(d[hv == 1].mean()), 6), "add": int(add.sum()),
            "add_prec": round(float(z["y"][add].mean()), 3) if add.any() else None, "rm": int(rm.sum()),
            "rm_false": round(float(1 - z["y"][rm].mean()), 3) if rm.any() else None, "F": round(float(f.mean()), 5)}


def ref_bars(dr, p, z, an):
    G = json.loads((W / "p42/sm/grid.json").read_text())
    tm = S.fr_to_loco(G, an, dr, CUR["tau"], p, z["am"], z["n"])
    return {k: tm[str(CUR[k])] for k in ("t", "r", "gc", "hn")}


# ------------------------------------------------------------------ source-optimal bars
T_GRID = [0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998]
S_GRID = [None, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]


def tune(z, p, w=None, rounds=2):
    """Coordinate descent on the recipe bars (t first with the stages off, then r / gc / hn, `rounds` passes) for the
    (weighted) mean per-S1 F0.5 on the labelled frame z. Stage bars at or above t are no-ops, so they are skipped."""
    score = lambda b: float(np.average(fvec(z, run(z, p, b)), weights=w))
    b = {"t": 0.9, "r": None, "gc": None, "hn": None}
    best = score(b)
    for _ in range(rounds):
        for k, grid in (("t", T_GRID), ("r", S_GRID), ("gc", S_GRID), ("hn", S_GRID)):
            for x in grid:
                if k != "t" and x is not None and x >= b["t"]:
                    continue
                c = dict(b, **{k: x})
                if k == "t" and any(c[j] is not None and c[j] >= x for j in ("r", "gc", "hn")):
                    continue
                s = score(c)
                if s > best + 1e-9:
                    best, b = s, c
    return b, best


# ------------------------------------------------------------------ calibration pieces
def bcts(p, y):
    """(a, b) of q = sigmoid(a * logit p + b), logloss fit on the open interval (veto zeros stay 0)."""
    from sklearn.linear_model import LogisticRegression
    m = (p > 1e-6) & (p < 1 - 1e-6)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(S.logit(p[m])[:, None], y[m])
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def calib(p, ab):
    return np.where(p <= 0, 0.0, sig(ab[0] * S.logit(p) + ab[1]))


def em_prior(q, pi_s, it=200):
    """Saerens et al. 2002: target prior by EM on source-calibrated posteriors q (q = 0 rows are hard rejects: excluded)."""
    q = q[q > 0]
    pi = pi_s
    for _ in range(it):
        a, b = pi / pi_s * q, (1 - pi) / (1 - pi_s) * (1 - q)
        new = float((a / (a + b)).mean())
        if abs(new - pi) < 1e-8:
            break
        pi = new
    return pi


def shift(q, pi_t, pi_s):
    d = S.logit(np.array([pi_t]))[0] - S.logit(np.array([pi_s]))[0]
    return np.where(q <= 0, 0.0, sig(S.logit(q) + d)), float(d)


def top1(z, p):
    return np.maximum.reduceat(p[np.argsort(z["s1"], kind="stable")], np.r_[0, np.flatnonzero(np.diff(np.sort(z["s1"])) != 0) + 1])


def qmap(pt, t_top, s_top):
    """Monotone map of every target p: u = F_T(p) over T's per-S1 top-1 scores, then F_S^-1(u) over S's."""
    ts, ss = np.sort(t_top), np.sort(s_top)
    u = (np.searchsorted(ts, pt, "left") + np.searchsorted(ts, pt, "right")) / 2 / len(ts)
    return np.where(pt <= 0, 0.0, np.quantile(ss, np.clip(u, 0, 1)))


def s1_feats(z, p):
    """Label-free S1-level features for the domain classifier: argmax pairs per source, top-3 p per source, hn mix."""
    s1, n, is2 = z["s1"], z["n"], z["is2"]
    cols = []
    for src in (True, False):
        m = is2 == src
        cols.append(np.bincount(s1[m], minlength=n))
        o, rk = S.rank_in(s1 * 1, p, np.flatnonzero(m))
        for k in range(3):
            v = np.full(n, -1.0)
            v[s1[o[rk == k]]] = p[o[rk == k]]
            cols.append(v)
    for h in range(6):
        cols.append(np.bincount(s1[z["hn_rel"] == h], minlength=n))
    cols.append(np.bincount(s1, weights=p, minlength=n) / np.maximum(np.bincount(s1, minlength=n), 1))
    return np.column_stack(cols).astype(np.float32)


def iw_weights(zs, ps, zt, pt, seed=0):
    """p(T|x)/p(S|x) for every source S1, 2-fold cross-fitted HistGradientBoosting; clipped at the 99.5th pct."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    Xs, Xt = s1_feats(zs, ps), s1_feats(zt, pt)
    X, d = np.vstack([Xs, Xt]), np.r_[np.zeros(len(Xs)), np.ones(len(Xt))]
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, 2, len(X))
    pr = np.zeros(len(X))
    for k in range(2):
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31).fit(X[fold != k], d[fold != k])
        pr[fold == k] = m.predict_proba(X[fold == k])[:, 1]
    auc = float(roc_auc_score(d, pr))
    ps_ = np.clip(pr[: len(Xs)], 1e-4, 1 - 1e-4)
    w = ps_ / (1 - ps_) * len(Xs) / len(Xt)
    w = np.minimum(w, np.quantile(w, 0.995))
    ess = float(w.sum() ** 2 / (w ** 2).sum() / len(w))
    return w, auc, ess


# ------------------------------------------------------------------ LOCO
def loco(arms):
    t0 = time.time()
    out = json.loads((HERE / "loco.json").read_text()) if (HERE / "loco.json").exists() else {}
    for an in ("chain", "m2"):
        for dr in DIRS:
            zs, zt = frames(dr)
            ps, pt = S.loco_p(zs["g"], zs["r"], CUR["tau"], an), S.loco_p(zt["g"], zt["r"], CUR["tau"], an)
            rb = ref_bars(dr, pt, zt, an)
            ref = run(zt, pt, rb)
            fref = fvec(zt, ref)
            R = out.setdefault(f"{an}:{dr}", {})
            R["ref"] = {"bars": {k: round(v, 5) for k, v in rb.items()}, "F": round(float(fref.mean()), 5), "pairs": int(ref.sum())}
            R["pi"] = {"S": round(float(zs["y"].mean()), 4), "T": round(float(zt["y"].mean()), 4)}
            if "naive" in arms or "em" in arms or "emo" in arms or "qmap" in arms:
                bs, fs = tune(zs, ps)
                R["src_opt_raw"] = {"bars": bs, "F_S": round(fs, 5)}
                if "naive" in arms:
                    R["naive"] = cmp(zt, ref, run(zt, pt, bs), fref) | {"bars": bs}
                if "qmap" in arms:
                    pq_ = qmap(pt, top1(zt, pt), top1(zs, ps))
                    R["qmap"] = cmp(zt, ref, run(zt, pq_, bs), fref) | {"bars": bs}
            if "em" in arms or "emo" in arms:
                ab = bcts(ps, zs["y"])
                qs, qt = calib(ps, ab), calib(pt, ab)
                pi_s = float(zs["y"].mean())
                bq, fq = tune(zs, qs)
                pi_hat = em_prior(qt, pi_s)
                R["em_info"] = {"bcts": [round(ab[0], 4), round(ab[1], 4)], "pi_S": round(pi_s, 4), "pi_T_true": round(float(zt["y"].mean()), 4),
                                "pi_T_hat": round(pi_hat, 4), "pi_T_meanq": round(float(qt[qt > 0].mean()), 4),
                                "pi_T_true_q>0": round(float(zt["y"][qt > 0].mean()), 4), "bars_q": bq, "F_S": round(fq, 5)}
                qe, d = shift(qt, pi_hat, pi_s)
                R["em"] = cmp(zt, ref, run(zt, qe, bq), fref) | {"delta": round(d, 4)}
                R["bcts_only"] = cmp(zt, ref, run(zt, qt, bq), fref)
                if "emo" in arms:
                    qo, d = shift(qt, float(zt["y"][qt > 0].mean()), pi_s)
                    R["emo"] = cmp(zt, ref, run(zt, qo, bq), fref) | {"delta": round(d, 4)}
            if "iw" in arms:
                w, auc, ess = iw_weights(zs, ps, zt, pt)
                bw, fw = tune(zs, ps, w)
                R["iw"] = cmp(zt, ref, run(zt, pt, bw), fref) | {"bars": bw, "auc": round(auc, 4), "ess": round(ess, 4)}
            print(an, dr, json.dumps(R), f"{time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)
            (HERE / "loco.json").write_text(json.dumps(out, indent=1))
    return out


def _selfcheck():
    q = np.r_[np.full(700, 0.9), np.full(300, 0.1)]          # calibrated at pi 0.66
    pi = em_prior(q, 0.66)
    assert 0.6 < pi < 0.8, pi
    qs, d = shift(np.array([0.5, 0.0]), 0.8, 0.5)
    assert abs(qs[0] - 0.8) < 1e-9 and qs[1] == 0 and d > 0
    m = qmap(np.array([0.1, 0.5, 0.9]), np.array([0.1, 0.5, 0.9]), np.array([0.2, 0.6, 0.95]))
    assert (np.diff(m) > 0).all()
    z = {"s1": np.array([0, 0, 1]), "n": 2}
    assert top1(z, np.array([0.3, 0.7, 0.2])).tolist() == [0.7, 0.2]
    print("p52 self-check ok")



# ------------------------------------------------------------------ transfer-rule tournament (LOCO frame A -> frame B)
def dmap(bars, pa_, za, ta, pb, zb, tb):
    """P42's rule between two unseen-country frames: same signed argmax density per S1 relative to each frame's
    label-free anchor (t_base)."""
    return {k: (None if v is None else S.inv_sdens(pb, zb["am"], tb, S.sdens(pa_, za["am"], ta, v, za["n"]), zb["n"])) for k, v in bars.items()}


def qbars(bars, pa_, za, pb, zb):
    """Quantile map of bars: A's per-S1 top-1 CDF value of each bar -> the same quantile of B's top-1 distribution."""
    ta_, tb_ = np.sort(top1(za, pa_)), np.sort(top1(zb, pb))
    return {k: (None if v is None else float(np.quantile(tb_, np.searchsorted(ta_, v, "left") / len(ta_)))) for k, v in bars.items()}


def xfer():
    """Every rule transfers frame A's oracle bars (tuned on A's labels) to frame B; B's labels only score. Frames =
    P42's two LOCO targets (India under the US model, US under the India model): both are unseen-country frames, the
    same relation France has to them. Rules: raw (same bars), density (P42's map, anchors t_base), qmap (top-1
    quantiles), em (BCTS on A, Saerens pi on B, logit shift), iw (A re-weighted to look like B)."""
    t0 = time.time()
    out = {}
    for an in ("chain", "m2"):
        Z, P = {}, {}
        for dr in DIRS:
            _, Z[dr] = frames(dr)
            P[dr] = S.loco_p(Z[dr]["g"], Z[dr]["r"], CUR["tau"], an)
        opt = {dr: tune(Z[dr], P[dr], rounds=2) for dr in DIRS}
        for a, b in (("US_India", "India_US"), ("India_US", "US_India")):
            za, zb, pa_, pb = Z[a], Z[b], P[a], P[b]
            ta, tb = S.T_BASE[an][a], S.T_BASE[an][b]
            rb = ref_bars(b, pb, zb, an)
            ref = run(zb, pb, rb)
            fref = fvec(zb, ref)
            ba, fa = opt[a]
            R = out[f"{an}:{a}->{b}"] = {"ref_in_sample": {"bars": rb, "F": round(float(fref.mean()), 5)},
                                         "A_opt": {"bars": ba, "F_A": round(fa, 5)}, "B_oracle": {"bars": opt[b][0], "F_B": round(opt[b][1], 5)}}
            R["B_oracle"]["dF_vs_ref"] = round(opt[b][1] - float(fref.mean()), 6)
            cands = {"raw": ba, "density": dmap(ba, pa_, za, ta, pb, zb, tb), "qmap": qbars(ba, pa_, za, pb, zb)}
            for k, bb in cands.items():
                R[k] = cmp(zb, ref, run(zb, pb, bb), fref) | {"bars": {j: (None if v is None else round(v, 5)) for j, v in bb.items()}}
            ab = bcts(pa_, za["y"])
            qa, qb = calib(pa_, ab), calib(pb, ab)
            pi_a = float(za["y"][qa > 0].mean())
            bq, _ = tune(za, qa, rounds=2)
            pi_hat = em_prior(qb, pi_a)
            qe, d = shift(qb, pi_hat, pi_a)
            R["em"] = cmp(zb, ref, run(zb, qe, bq), fref) | {"pi_A": round(pi_a, 4), "pi_B_hat": round(pi_hat, 4),
                                                             "pi_B_true": round(float(zb["y"][qb > 0].mean()), 4), "delta": round(d, 4)}
            w, auc, ess = iw_weights(za, pa_, zb, pb)
            bw, _ = tune(za, pa_, w, rounds=2)
            R["iw"] = cmp(zb, ref, run(zb, pb, bw), fref) | {"bars": bw, "auc": round(auc, 4), "ess": round(ess, 4)}
            # honest baselines differ from 'ref' (in-sample on B): the rule to beat is 'density'
            for k in ("raw", "qmap", "em", "iw"):
                R[k]["vs_density"] = round(R[k]["dF"] - R["density"]["dF"], 6)
            print(an, a, "->", b, json.dumps(R), f"{time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)
            (HERE / "xfer.json").write_text(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------ arm 2 for France: P42's bar choice, LOCO S1s weighted toward France
FR_LEV = {"t": [0.97, 0.98, 0.983, 0.985, 0.9868, 0.988, 0.99], "r": [None, 0.97, 0.95, 0.9, 0.8, 0.7, 0.5],
          "gc": [None, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4], "hn": [None, 0.97, 0.95, 0.9, 0.8, 0.7, 0.5]}


def fr_argmax_frame():
    """France chain pairs at tau 0.1 (P42 fr_p: post tier / bxneg / cap rules), SX-argmax rows only, as a p52 frame."""
    import p42
    z = p42.fr_frame()
    p = p42.fr_p(z, 0.1)
    am = S.argmax_sx(z["sx"].astype(np.int64), p)
    h = np.load(W / "p52/france_hn_rel.npy")
    zf = {"s1": z["s1"][am].astype(np.int64), "is2": z["is2"][am], "hn_rel": h[am], "n": int(z["n_s1"])}
    return zf, p[am]


def xfer_iwd():
    """Honest test of the France form of arm 2: frame A's bars tuned with S1 weights toward frame B, then P42's density
    map to B (the rule France's bars go through), vs the unweighted A-tuned bars through the same map."""
    out = json.loads((HERE / "xfer.json").read_text())
    for an in ("chain", "m2"):
        Z, P_ = {}, {}
        for dr in DIRS:
            _, Z[dr] = frames(dr)
            P_[dr] = S.loco_p(Z[dr]["g"], Z[dr]["r"], CUR["tau"], an)
        for a, b in (("US_India", "India_US"), ("India_US", "US_India")):
            za, zb, pa_, pb = Z[a], Z[b], P_[a], P_[b]
            ta, tb = S.T_BASE[an][a], S.T_BASE[an][b]
            rb = ref_bars(b, pb, zb, an)
            ref = run(zb, pb, rb)
            fref = fvec(zb, ref)
            R = out[f"{an}:{a}->{b}"]
            w, auc, ess = iw_weights(za, pa_, zb, pb)
            bw, _ = tune(za, pa_, w, rounds=2)
            R["iw_density"] = cmp(zb, ref, run(zb, pb, dmap(bw, pa_, za, ta, pb, zb, tb)), fref)
            R["iw_density"]["vs_density"] = round(R["iw_density"]["dF"] - R["density"]["dF"], 6)
            print(an, a, "->", b, "iw_density", json.dumps(R["iw_density"]), "density", R["density"]["dF"], flush=True)
    (HERE / "xfer.json").write_text(json.dumps(out, indent=1))


def fr_iw():
    """France bars = the P42 lever cell maximising the France-WEIGHTED LOCO F (mean over both frames) under P42's density
    map; weights p(France|x)/p(LOCO|x) per LOCO S1 (S1-level classifier, label-free). Reports the unweighted LOCO dF of
    the pick vs the current recipe (the ship gate)."""
    import itertools
    t0 = time.time()
    zf, pf = fr_argmax_frame()
    G = json.loads((W / "p42/sm/grid.json").read_text())
    out = {}
    D = {}
    for dr in DIRS:
        _, z = frames(dr)
        p = S.loco_p(z["g"], z["r"], CUR["tau"], "chain")
        w, auc, ess = iw_weights(z, p, zf, pf)
        tm = S.fr_to_loco(G, "chain", dr, CUR["tau"], p, z["am"], z["n"])
        ref = run(z, p, {k: tm[str(CUR[k])] for k in ("t", "r", "gc", "hn")})
        D[dr] = (z, p, w, tm, fvec(z, ref))
        out[f"w_{dr}"] = {"auc": round(auc, 4), "ess": round(ess, 4), "w_q": [round(float(x), 3) for x in np.quantile(w, [0.1, 0.5, 0.9, 0.99])]}
        print(dr, out[f"w_{dr}"], flush=True)
    rows = []
    mp = lambda tm, x: None if x is None else tm[str(x)]
    for t, r, gc, hn in itertools.product(FR_LEV["t"], FR_LEV["r"], FR_LEV["gc"], FR_LEV["hn"]):
        row = {"t": t, "r": r, "gc": gc, "hn": hn}
        for dr, (z, p, w, tm, fref) in D.items():
            f = fvec(z, run(z, p, {"t": tm[str(t)], "r": mp(tm, r), "gc": mp(tm, gc), "hn": mp(tm, hn)}))
            d, hv = f - fref, z["hv"]
            row[f"w_{dr}"], row[f"u_{dr}"] = float(np.average(d, weights=w)), float(d.mean())
            for h in (0, 1):
                m = hv == h
                row[f"w{h}_{dr}"], row[f"u{h}_{dr}"] = float(np.average(d[m], weights=w[m])), float(d[m].mean())
            row[f"wse_{dr}"] = float(np.sqrt((w ** 2 * (d - row[f"w_{dr}"]) ** 2).sum()) / w.sum())
        rows.append(row)
    import pandas as pd
    df = pd.DataFrame(rows)
    for x in ("w", "w0", "w1"):
        df[f"{x}_mean"] = (df[f"{x}_US_India"] + df[f"{x}_India_US"]) / 2
    df.to_parquet(W / "p52/fr_iw_grid.parquet")
    top = df.sort_values("w0_mean", ascending=False).head(10)                 # pick on the tune half, read h1
    out["top_w"] = top.round(6).to_dict("records")
    ok = df[(df.u_US_India >= 0) & (df.u_India_US >= 0.0003)]
    out["gate_cells"] = int(len(ok))
    out["best_gated_by_w"] = ok.sort_values("w0_mean", ascending=False).head(5).round(6).to_dict("records")
    out["runtime_s"] = round(time.time() - t0)
    (HERE / "fr_iw.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str)[:4000], flush=True)



def lbcheck(seeds=(0, 1, 2)):
    """Does France-weighted LOCO predict the two France levers the LB measured? rescue90 on #34_veto (t 0.9868, no
    stages): LB +0.00207 France-F; gc70 on #34v_r90: LB +0.00227 (P37 p37-loco-lb-calibration). No P48 veto then.
    Also the current recipe's hn stage (hn .8 vs .95 on t .99 / r .9 / gc .7, veto on). Weights: several variants."""
    G = json.loads((W / "p42/sm/grid.json").read_text())
    zf, pf = fr_argmax_frame()
    mp = lambda tm, x: None if x is None else tm[str(x)]
    cells = {"rescue90": ((0.9868, None, None, None, False), (0.9868, 0.9, None, None, False)),
             "gc70": ((0.9868, 0.9, None, None, False), (0.9868, 0.9, 0.7, None, False)),
             "hn95_vs_hn80": ((0.99, 0.9, 0.7, 0.8, True), (0.99, 0.9, 0.7, 0.95, True)),
             "hn_off_vs_hn80": ((0.99, 0.9, 0.7, 0.8, True), (0.99, 0.9, 0.7, None, True)),
             "r97_vs_r90": ((0.99, 0.9, 0.7, 0.8, True), (0.99, 0.97, 0.7, 0.8, True))}
    out = {}
    for dr in DIRS:
        _, z = frames(dr)
        p = S.loco_p(z["g"], z["r"], CUR["tau"], "chain")
        tm = S.fr_to_loco(G, "chain", dr, CUR["tau"], p, z["am"], z["n"])
        F = {}
        for nm, (a, b) in cells.items():
            fa, fb = (fvec(z, run(z, p, {"t": tm[str(c[0])], "r": mp(tm, c[1]), "gc": mp(tm, c[2]), "hn": mp(tm, c[3])}, veto=c[4])) for c in (a, b))
            F[nm] = fb - fa
        W_ = {}
        for sd in seeds:
            W_[f"s{sd}"], auc, ess = iw_weights(z, p, zf, pf, seed=sd)
        W_["sqrt"] = np.sqrt(W_["s0"])
        W_["clip90"] = np.minimum(W_["s0"], np.quantile(W_["s0"], 0.9))
        # no-p features: counts + hn mix only (the classifier cannot see the score scale)
        Xs, Xt = s1_feats(z, p), s1_feats(zf, pf)
        keep = [0, 4, 8, 9, 10, 11, 12, 13]                       # n per source, 6 hn classes
        from sklearn.ensemble import HistGradientBoostingClassifier
        Xx, d = np.vstack([Xs[:, keep], Xt[:, keep]]), np.r_[np.zeros(len(Xs)), np.ones(len(Xt))]
        fold = np.random.default_rng(0).integers(0, 2, len(Xx))
        pr = np.zeros(len(Xx))
        for k in range(2):
            pr[fold == k] = HistGradientBoostingClassifier(max_iter=200).fit(Xx[fold != k], d[fold != k]).predict_proba(Xx[fold == k])[:, 1]
        q = np.clip(pr[: len(Xs)], 1e-4, 1 - 1e-4)
        w = q / (1 - q)
        W_["nop"] = np.minimum(w, np.quantile(w, 0.995))
        R = out[dr] = {}
        for nm, dd in F.items():
            R[nm] = {"u": round(float(dd.mean()), 6), "u_se": round(float(dd.std() / np.sqrt(len(dd))), 6)}
            for k, w in W_.items():
                R[nm][k] = round(float(np.average(dd, weights=w)), 6)
                R[nm][k + "_se"] = round(float(np.sqrt((w ** 2 * (dd - np.average(dd, weights=w)) ** 2).sum()) / w.sum()), 6)
        R["ess"] = {k: round(float(w.sum() ** 2 / (w ** 2).sum() / len(w)), 4) for k, w in W_.items()}
        print(dr, json.dumps(R), flush=True)
    (HERE / "lbcheck.json").write_text(json.dumps(out, indent=1))



CANDS = {"current": (0.99, 0.9, 0.7, 0.8), "A_r80_gc50": (0.99, 0.8, 0.5, 0.8), "B_hn95": (0.99, 0.9, 0.7, 0.95),
         "C_r80_gc50_hn95": (0.99, 0.8, 0.5, 0.95), "D_hn90": (0.99, 0.9, 0.7, 0.9), "E_r97_hnoff": (0.99, 0.97, 0.7, None),
         "F_r80_gc50_hn90": (0.99, 0.8, 0.5, 0.9)}


def weight_variants(z, p, zf, pf):
    from sklearn.ensemble import HistGradientBoostingClassifier
    W_ = {}
    W_["full"], _, _ = iw_weights(z, p, zf, pf, seed=0)
    W_["sqrt"] = np.sqrt(W_["full"])
    W_["clip90"] = np.minimum(W_["full"], np.quantile(W_["full"], 0.9))
    Xs, Xt = s1_feats(z, p), s1_feats(zf, pf)
    keep = [0, 4, 8, 9, 10, 11, 12, 13]
    Xx, d = np.vstack([Xs[:, keep], Xt[:, keep]]), np.r_[np.zeros(len(Xs)), np.ones(len(Xt))]
    fold = np.random.default_rng(0).integers(0, 2, len(Xx))
    pr = np.zeros(len(Xx))
    for k in range(2):
        pr[fold == k] = HistGradientBoostingClassifier(max_iter=200).fit(Xx[fold != k], d[fold != k]).predict_proba(Xx[fold == k])[:, 1]
    q = np.clip(pr[: len(Xs)], 1e-4, 1 - 1e-4)
    W_["nop"] = np.minimum(q / (1 - q), np.quantile(q / (1 - q), 0.995))
    return W_


def cands():
    """The candidate France cells on both analogues x both frames: unweighted (all, h0, h1) and France-weighted (4 variants)."""
    G = json.loads((W / "p42/sm/grid.json").read_text())
    zf, pf = fr_argmax_frame()
    mp = lambda tm, x: None if x is None else tm[str(x)]
    out = {}
    for an in ("chain", "m2"):
        for dr in DIRS:
            _, z = frames(dr)
            p = S.loco_p(z["g"], z["r"], CUR["tau"], an)
            tm = S.fr_to_loco(G, an, dr, CUR["tau"], p, z["am"], z["n"])
            Wv = weight_variants(z, p, zf, pf)
            f0 = fvec(z, run(z, p, {k: mp(tm, CUR[k]) for k in ("t", "r", "gc", "hn")}))
            R = out[f"{an}:{dr}"] = {}
            for nm, c in CANDS.items():
                if nm == "current":
                    continue
                d = fvec(z, run(z, p, {"t": tm[str(c[0])], "r": mp(tm, c[1]), "gc": mp(tm, c[2]), "hn": mp(tm, c[3])})) - f0
                hv = z["hv"]
                r = {"u": round(float(d.mean()), 6), "u_se": round(float(d.std() / np.sqrt(len(d))), 6),
                     "u_h0": round(float(d[hv == 0].mean()), 6), "u_h1": round(float(d[hv == 1].mean()), 6)}
                for k, w in Wv.items():
                    r[k] = round(float(np.average(d, weights=w)), 6)
                R[nm] = r
            print(an, dr, json.dumps(R), flush=True)
    (HERE / "cands.json").write_text(json.dumps(out, indent=1))



if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("xfer", "xfer_iwd", "fr_iw", "lbcheck", "cands"):
        globals()[sys.argv[1]]()
    elif len(sys.argv) > 1 and sys.argv[1] == "loco":
        loco(sys.argv[2:] or ["naive", "em", "emo", "qmap", "iw"])
    else:
        _selfcheck()
