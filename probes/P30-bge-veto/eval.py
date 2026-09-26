"""P30 evaluation, all CPU (< 1.5 GB). Needs the job's r files in data/work/p30/r (p30.py fetch <job>).

    eval.py loco      # LOCO: precision of confident pairs by bge r bin + veto rules at the label-free t, both directions
    eval.py ind       # in-domain: the same rules on #8c's holdout decisions with the pooled bge (the #8c yardstick)
    eval.py ctl       # restored pooled model vs P13's r on the control pairs (numerics)
    eval.py france <rule> [ref_t]   # France: veto #9ab's file, t re-matched to #9ab's empty rate, write the prob file
    eval.py selftest

LOCO proxy = the France procedure on P6's worlds: P6's source-only LightGBM scores the unseen country's fold 0 (every
train S1 competing; argmax = P6's am_all, fixed); the band 0.02<p<0.98 is blended with a bge trained on the source
only (P13's recipe, retrained in this job) using P13's out-of-domain France weights; t = P6's label-free rule
(target empty-S1 rate matched to the source's at its own label-tuned t*). A veto removes an accepted pair whose
GBDT p >= 0.98 when the CE disagrees; t is then re-matched to the base's empty rate (France keeps #9ab's).
"""
import json, os, resource, sys, time
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "probes/P4b-rerank"), str(ROOT / "code/business_entity_resolution/src")]
import rerank as rr
from ledger import log
from p30 import CI, OFF1, OFF2, OTHER, WORK, loco_frame

OUT = WORK / "p30"
R = OUT / "r"
W_FR = [0.9245, 0.2595, 0.0515]                  # P13's label-free out-of-domain France weights
LO, HI = 0.02, 0.98
TS = np.round(np.arange(0.5, 0.9996, 0.0025), 4)
TAUS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
RBINS = (0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 0.95, 1.0001)
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def fbeta(tp, npred, ntrue):
    """Vectorised metric.f_beta (beta 0.5) from per-S1 counts."""
    tp, npred, ntrue = (np.asarray(x, np.float64) for x in (tp, npred, ntrue))
    with np.errstate(divide="ignore", invalid="ignore"):
        P, Rc = tp / npred, tp / ntrue
        f = 1.25 * P * Rc / (0.25 * P + Rc)
    f = np.where(tp == 0, 0.0, f)
    return np.where(ntrue == 0, (npred == 0).astype(np.float64), f)


def per_s1(s1, keep, y, n_s1, ntrue):
    """Per-S1 F0.5 over every S1 0..n_s1-1 (pairs carry dense S1 codes)."""
    npred = np.bincount(s1[keep], minlength=n_s1)
    tp = np.bincount(s1[keep & y], minlength=n_s1)
    return fbeta(tp, npred, ntrue)


def empty_curve(s1, am, p, n_s1, ts=TS):
    top = np.full(n_s1, -1.0)
    np.maximum.at(top, s1[am], p[am])
    return pd.Series([(top < t).mean() for t in ts], index=ts)


def match(curve, ref):
    return float((curve - ref).abs().idxmin())


def veto_mask(g, r, rule):
    """Pairs a rule vetoes. rule 'tau:<x>': GBDT p >= 0.98 and r < x. 'full': handled by blend (never vetoes)."""
    if rule.startswith("tau:"):
        return (g >= HI) & (r < float(rule[4:]))
    return np.zeros(len(g), bool)


def apply_rule(g, r, rule):
    """Final p for a rule: base = band blend; 'full' = the blend over every p > 0.02; 'tau:x' = base, vetoed -> 0."""
    rb = np.where(np.isnan(r), 0.5, r)
    if rule == "full":
        return rr.blend(g, rb, W_FR, LO, 1.0001)
    p = rr.blend(g, rb, W_FR, LO, HI)
    if rule == "nob":
        return g.copy()
    return np.where(veto_mask(g, r, rule), 0.0, p)


def truth_counts(c):
    """Fold-0 S1 of country c -> (entity ids, number of true matches)."""
    f = pd.read_parquet(WORK / "s1_folds.parquet")
    f = f[(f.country == c) & (f.fold == 0)]
    gt = pd.read_parquet(WORK / "train_ground_truth.parquet")
    gt = gt[gt.source1_entity_id.isin(set(f.entity_id))]
    n = gt.matched_entity_ids.fillna("").map(lambda s: len(s.split(",")) if s else 0)
    cnt = pd.Series(n.to_numpy(), index=gt.source1_entity_id.to_numpy())
    return f.entity_id.to_numpy(), cnt.reindex(f.entity_id.to_numpy()).fillna(0).to_numpy(np.int64)


def rbin_table(y, r, m):
    out = []
    for lo, hi in zip(RBINS[:-1], RBINS[1:]):
        k = m & (r >= lo) & (r < hi)
        out.append({"r": f"[{lo},{min(hi, 1)})", "n": int(k.sum()), "prec": round(float(y[k].mean()), 4) if k.any() else None})
    return out


def loco(srcs: str = "US,India"):
    t0 = time.time()
    import features as F
    res = {}
    for src in srcs.split(","):
        tgt, ci = OTHER[src], CI[OTHER[src]]
        # source: in-domain GBDT p on the source's fold 0 -> t* (labels) and its empty rate (the reference)
        S = pd.read_parquet(WORK / f"p30/p6/S_holdout_{src}.parquet", columns=["s1", "sx", "country", "p", "is_true", "am_all"])
        S = S[S.country == CI[src]]
        ids_s, nt_s = truth_counts(src)
        q1 = S.s1.to_numpy() - CI[src] * OFF1
        e1 = np.asarray(F.ids("train", src)[0].take(q1).to_pylist(), object)
        code_s = pd.Index(ids_s).get_indexer(e1)
        assert (code_s >= 0).all()
        ys, ps, ams = S.is_true.to_numpy(), S.p.to_numpy(np.float64), S.am_all.to_numpy()
        fs = {t: per_s1(code_s, ams & (ps >= t), ys, len(ids_s), nt_s).mean() for t in TS}
        t_star = max(fs, key=fs.get)
        ref = float(empty_curve(code_s, ams, ps, len(ids_s)).loc[t_star])
        del S, e1, code_s, ys, ps, ams
        # target
        T = loco_frame(src)
        ids, nt = truth_counts(tgt)
        e1 = np.asarray(F.ids("train", tgt)[0].take(T.s1.to_numpy() - ci * OFF1).to_pylist(), object)
        s1 = pd.Index(ids).get_indexer(e1)
        assert (s1 >= 0).all()
        del e1
        rf = pd.read_parquet(R / f"r_loco{src}_tgt_{tgt}.parquet")
        T = T.merge(rf, on=["s1", "sx"], how="left")
        g, y, am, r = T.p.to_numpy(np.float64), T.y.to_numpy(bool), T.am.to_numpy(bool), T.r.to_numpy(np.float64)
        assert not np.isnan(r[g > LO]).any(), "every p > 0.02 pair has a LOCO r"
        conf = am & (g >= HI)
        out = {"t_star_src": t_star, "ref_empty_src": round(ref, 4), "n_s1": len(ids), "n_conf": int(conf.sum()),
               "prec_conf": round(float(y[conf].mean()), 4), "rbin_conf": rbin_table(y, r, conf)}
        band = (g > LO) & (g < HI)
        from sklearn.metrics import roc_auc_score
        out["auc_r_band"] = round(float(roc_auc_score(y[band], r[band])), 4)
        out["auc_r_conf"] = round(float(roc_auc_score(y[conf], r[conf])), 4)
        rows = {}
        for rule in ("nob", "base", "full") + tuple(f"tau:{x}" for x in TAUS):
            p = apply_rule(g, r, rule)
            cur = empty_curve(s1, am, p, len(ids))
            rows[rule] = {"p": p, "cur": cur}
        tb = float(min(max(match(rows["base"]["cur"], ref), t_star), round(t_star + 0.2, 4)))   # P6's rule
        e_b = float(rows["base"]["cur"].loc[tb])
        f_base = per_s1(s1, am & (rows["base"]["p"] >= tb), y, len(ids), nt)
        tab = []
        for rule, v in rows.items():
            p, cur = v["p"], v["cur"]
            t_p6 = float(min(max(match(cur, ref), t_star), round(t_star + 0.2, 4)))
            t_m = match(cur, e_b)
            for how, t in (("fixed", tb), ("matched", t_m), ("p6", t_p6)):
                f = per_s1(s1, am & (p >= t), y, len(ids), nt)
                d = f - f_base
                vet = veto_mask(g, r, rule) & am
                tab.append({"rule": rule, "t_how": how, "t": t, "f05": round(float(f.mean()), 5),
                            "gain": round(float(d.mean()), 5), "se": round(float(d.std(ddof=1) / np.sqrt(len(d))), 5),
                            "empty": round(float(cur.loc[t]) if t in cur.index else float("nan"), 4),
                            "vetoed": int(vet.sum()), "vetoed_prec": round(float(y[vet].mean()), 4) if vet.any() else None,
                            "vetoed_share_conf": round(float(vet.sum() / conf.sum()), 4)})
        out |= {"t_base": tb, "empty_base": round(e_b, 4), "f05_base": round(float(f_base.mean()), 5), "table": tab}
        res[f"{src}>{tgt}"] = out
        print(f"\n=== {src}>{tgt}: t* {t_star} ref {ref:.4f} | base t {tb} empty {e_b:.4f} F {f_base.mean():.5f} | "
              f"conf {conf.sum():,} prec {y[conf].mean():.4f} | AUC r band {out['auc_r_band']} conf {out['auc_r_conf']}")
        for b in out["rbin_conf"]:
            print(f"  r {b['r']:<12} n {b['n']:>8,} prec {b['prec']}")
        for x in tab:
            print(f"  {x['rule']:<9} {x['t_how']:<8} t {x['t']:<7} F {x['f05']:.5f} gain {x['gain']:+.5f} ± {x['se']:.5f} "
                  f"empty {x['empty']} vetoed {x['vetoed']:,} @ {x['vetoed_prec']}")
        del T, g, y, am, r, rows
    json.dump(res, open(OUT / f"loco_{srcs.replace(',', '_')}.json", "w"), indent=1, default=float)
    if len(res) < 2:
        return
    # one ledger row per rule: gains in both directions at the matched t
    for rule in ("full",) + tuple(f"tau:{x}" for x in TAUS):
        m = {}
        for k, v in res.items():
            for x in v["table"]:
                if x["rule"] == rule:
                    m[f"{k}_{x['t_how']}"] = {kk: x[kk] for kk in ("t", "gain", "se", "vetoed", "vetoed_prec")}
        ok = all(m[f"{k}_matched"]["gain"] >= 0.0005 for k in res)
        log("exp", probe="P30-bge-veto", subproblem="france-bge-veto", name=f"loco-veto-{rule}",
            world="holdout-dense-full LOCO (P6 target fold 0)", metrics=m,
            config={"w_band": W_FR, "rule": rule, "t": "P6 label-free, re-matched to base empty rate"},
            hypothesis="the source-only bge vetoing GBDT-confident pairs lifts unseen-country F0.5 by >= 0.0005 both ways",
            verdict="keep" if ok else "drop", runtime_s=round(time.time() - t0), peak_gb=round(peak(), 2))
    print(f"{time.time() - t0:.0f}s peak {peak():.2f} GB")


def loco_r_ids(c):
    """The LOCO r of target c (bge trained on the other country) keyed by entity ids instead of P6's int ids."""
    import features as F
    T = pd.read_parquet(R / f"r_loco{OTHER[c]}_tgt_{c}.parquet")
    i1, ix = F.ids("train", c)
    return pd.DataFrame({"s1": i1.take(T.s1.to_numpy() - CI[c] * OFF1).to_pylist(),
                         "sx": ix.take(T.sx.to_numpy() - CI[c] * OFF2).to_pylist(), "r": T.r.to_numpy()})


def ind(which: str = "pool"):
    """#8c holdout decisions (P23 dump): remove predicted pairs with p_m2 >= 0.98 and r < tau, t fixed.
    which = pool: the pooled bge (in-domain, the #8c yardstick); loco: the bge trained on the OTHER country (an
    out-of-domain CE vetoing an in-domain GBDT: the pessimistic bound for France; pairs without a LOCO r keep)."""
    t0 = time.time()
    res = {}
    for c in ("India", "US"):
        q = pd.read_parquet(WORK / f"p23/pairs_{c}.parquet", columns=["s1", "sx", "holdout", "y", "p_m2", "p", "am", "pred"])
        q = q[q.holdout].reset_index(drop=True)
        ids, _ = truth_counts(c)
        miss = pd.read_parquet(WORK / f"p23/missed_{c}.parquet", columns=["s1"])
        s1 = pd.Index(ids).get_indexer(q.s1)
        assert (s1 >= 0).all()
        y = q.y.to_numpy(bool)
        nt = np.bincount(s1[y], minlength=len(ids)) + np.bincount(pd.Index(ids).get_indexer(miss.s1), minlength=len(ids))
        rf = pd.read_parquet(R / f"r_pool_{c}.parquet") if which == "pool" else loco_r_ids(c)
        q = q.merge(rf, on=["s1", "sx"], how="left")
        r, g, pred = q.r.to_numpy(np.float64), q.p_m2.to_numpy(np.float64), q.pred.to_numpy(bool)
        conf = q.am.to_numpy(bool) & (g >= HI)
        assert which == "loco" or not np.isnan(r[conf]).any()
        f0 = per_s1(s1, pred, y, len(ids), nt)
        out = {"n_s1": len(ids), "f05_8c": round(float(f0.mean()), 5), "n_conf": int(conf.sum()),
               "conf_with_r": int((conf & ~np.isnan(r)).sum()),
               "prec_conf": round(float(y[conf].mean()), 5), "rbin_conf": rbin_table(y, np.nan_to_num(r, nan=1), conf)}
        t = float(q.p.to_numpy()[pred].min())
        tab = []
        for rule in ("full",) + tuple(f"tau:{x}" for x in TAUS):
            if rule == "full":      # the France blend over the confident range on top of #8c's p
                pp = np.where(conf, rr.blend(g, np.nan_to_num(r, nan=0.5), W_FR, LO, 1.0001), q.p.to_numpy())
                keep = pred & (pp >= t)
                vet = pred & ~keep
            else:
                vet = pred & conf & (r < float(rule[4:]))
                keep = pred & ~vet
            d = per_s1(s1, keep, y, len(ids), nt) - f0
            tab.append({"rule": rule, "gain": round(float(d.mean()), 6), "se": round(float(d.std(ddof=1) / np.sqrt(len(d))), 6),
                        "vetoed": int(vet.sum()), "vetoed_prec": round(float(y[vet].mean()), 4) if vet.any() else None})
        out |= {"t_8c": round(t, 4), "table": tab}
        res[c] = out
        print(f"\n=== {c} in-domain (#8c, t {t:.3f}): F {f0.mean():.5f}, conf {conf.sum():,} prec {y[conf].mean():.5f}")
        for b in out["rbin_conf"]:
            print(f"  r {b['r']:<12} n {b['n']:>8,} prec {b['prec']}")
        for x in tab:
            print(f"  {x['rule']:<9} gain {x['gain']:+.6f} ± {x['se']:.6f} vetoed {x['vetoed']:,} @ {x['vetoed_prec']}")
    n = {c: res[c]["n_s1"] for c in res}
    assert sum(n.values()) == 220358, n
    json.dump(res, open(OUT / f"ind_{which}.json", "w"), indent=1, default=float)
    for i, rule in enumerate(("full",) + tuple(f"tau:{x}" for x in TAUS)):
        m = {c: res[c]["table"][i] for c in res}
        m["pooled_gain"] = round(sum(res[c]["table"][i]["gain"] * n[c] for c in res) / sum(n.values()), 6)
        log("exp", probe="P30-bge-veto", subproblem="france-bge-veto", name=f"ind{which}-veto-{rule}", world="holdout-dense (#8c)",
            metrics=m, config={"rule": rule, "r": "pooled bge (P13 ckpt)" if which == "pool" else "LOCO bge (other country)",
                               "t": "#8c's per country, fixed"},
            hypothesis="in-country effect of the same veto on #8c's India/US decisions", verdict="inconclusive",
            runtime_s=round(time.time() - t0), peak_gb=round(peak(), 2))
    print(f"{time.time() - t0:.0f}s peak {peak():.2f} GB")


def ctl():
    res = {}
    old = pd.read_parquet(WORK / "p13/r/r_bge_poolck_holdout.parquet")
    for nm, ref in (("holdout_India", old), ("holdout_US", old),
                    ("pool_France", pd.read_parquet(WORK / "p13/r/r_bge_pool_test_France.parquet"))):
        m = pd.read_parquet(R / f"r_{nm}.parquet").merge(ref, on=["s1", "sx"], suffixes=("", "_p13"))
        dz = np.abs(rr.logit(m.r.to_numpy()) - rr.logit(m.r_p13.to_numpy()))
        res[nm] = {"n": len(m), "corr": round(float(np.corrcoef(m.r, m.r_p13)[0, 1]), 6),
                   "med_dlogit": round(float(np.median(dz)), 4), "p99_dlogit": round(float(np.quantile(dz, 0.99)), 4)}
    print(json.dumps(res))
    ok = all(v["corr"] > 0.999 and v["p99_dlogit"] < 0.2 for v in res.values()) and res["pool_France"]["n"] >= 2000
    log("exp", probe="P30-bge-veto", subproblem="france-bge-veto", name="p30-sm-repro", world="test+holdout", metrics=res,
        config={}, hypothesis="the job's restored pooled bge reproduces P13's r on control pairs",
        verdict="keep" if ok else "drop", runtime_s=0, peak_gb=round(peak(), 2))
    assert ok, res


def france(rule: str, ref_t: float = 0.985):
    """#9ab's France file with the rule applied to its p_gbdt >= 0.98 pairs (argmax recomputed exactly as combine.py
    does); France profile at three t choices. Writes prob_test_France_veto_<rule> (the p file does not depend on t)."""
    import model as M
    t0 = time.time()
    d = pd.read_parquet(WORK / "p19/prob_test_France_idf144_bge.parquet")
    rf = pd.read_parquet(R / "r_pool_France.parquet").drop_duplicates(["s1", "sx"])
    d = d.merge(rf, on=["s1", "sx"], how="left")
    g, p0, r = d.p_gbdt.to_numpy(np.float64), d.p.to_numpy(np.float64), d.r.to_numpy(np.float64)
    hi = g >= HI
    assert not np.isnan(r[hi]).any(), "every p_gbdt >= 0.98 pair has an r"
    if rule == "full":
        p = np.where(hi, rr.blend(g, r, W_FR, LO, 1.0001), p0)
    else:
        p = np.where(veto_mask(g, r, rule), 0.0, p0)
    s1ids = pd.read_parquet(WORK / "test_source1.parquet", columns=["entity_id", "country"])
    s1ids = s1ids.entity_id[s1ids.country == "France"].to_numpy()
    code = pd.Index(s1ids).get_indexer(d.s1)
    assert (code >= 0).all()
    x = pd.factorize(d.sx)[0]
    am0, am1 = M.argmax_sx(x, p0), M.argmax_sx(x, p)
    ts = np.round(np.arange(0.95, 0.9996, 0.0005), 4)
    c0, c1 = empty_curve(code, am0, p0, len(s1ids), ts), empty_curve(code, am1, p, len(s1ids), ts)
    e_ref = float(c0.loc[ref_t])
    acc0 = am0 & (p0 >= ref_t)
    vet = acc0 & (p < ref_t)
    n1 = lambda a: np.bincount(code[a], minlength=len(s1ids))
    k0 = n1(acc0)
    res = {"rule": rule, "ref_t": ref_t, "empty_ref": round(e_ref, 5), "accepted_9ab": int(acc0.sum()),
           "vetoed_at_ref_t": int(vet.sum()), "vetoed_share": round(float(vet.sum() / acc0.sum()), 4),
           "france_r_conf": rbin_table(np.zeros(len(d)), np.nan_to_num(r, nan=1), acc0 & hi)}
    # t: fixed = #9ab's; matched = #9ab's empty rate; d4 = D4's 6.85% (the rule that chose #9ab's 0.985)
    for how, t in (("fixed", ref_t), ("matched", match(c1, e_ref)), ("d4", match(c1, 0.0685))):
        acc1 = am1 & (p >= t)
        k1 = n1(acc1)
        res[how] = {"t": t, "empty": round(float(c1.loc[t]), 5), "accepted": int(acc1.sum()),
                    "added_vs_9ab": int((acc1 & ~acc0).sum()), "removed_vs_9ab": int((acc0 & ~acc1).sum()),
                    "new_argmax_accepted": int((acc1 & ~am0).sum()),
                    "s1_changed": int(((k0 != k1) | (n1(acc0 & ~acc1) > 0)).sum()),
                    "matches_per_s1": round(float(acc1.sum() / len(s1ids)), 4)}
    for b in res["france_r_conf"]:
        b.pop("prec")
    # label-free sanity (P23's bar): house-number class mix of the vetoed pairs vs every #9ab-accepted pair
    from stack import hn_cls
    hn = lambda f, ids: pd.read_parquet(WORK / f, columns=["id", "hn"]).set_index("id").hn.fillna("").reindex(ids).fillna("").to_numpy(object)
    a = np.flatnonzero(acc0)
    cls = hn_cls(hn("rf_test_s1_France.parquet", d.s1.to_numpy()[a]), hn("rf_test_sx_France.parquet", d.sx.to_numpy()[a]))
    va = vet[a]
    names = ["none/eq", "+SIB", "digit", "small", "far"]
    res["hn_mix_accepted"] = {n: round(float((cls == k).mean()), 4) for k, n in enumerate(names)}
    res["hn_mix_vetoed"] = {n: round(float((cls[va] == k).mean()), 4) for k, n in enumerate(names)} if va.any() else {}
    f = OUT / f"prob_test_France_veto_{rule.replace(':', '')}.parquet"
    pd.DataFrame({"s1": d.s1, "sx": d.sx, "p": p.astype(np.float32), "p_gbdt": d.p_gbdt}).to_parquet(f, index=False)
    res["file"] = str(f)
    print(json.dumps(res, indent=1), f"{time.time() - t0:.0f}s peak {peak():.2f} GB")
    log("exp", probe="P30-bge-veto", subproblem="france-bge-veto", name=f"france-veto-{rule}", world="test-France",
        metrics=res, config={"base": "#9ab p19/prob_test_France_idf144_bge", "w": W_FR},
        hypothesis="label-free profile of the France veto; t keeps #9ab's empty rate", verdict="keep",
        runtime_s=round(time.time() - t0), peak_gb=round(peak(), 2))


def selftest():
    assert fbeta([1], [1], [1])[0] == 1 and fbeta([0], [0], [0])[0] == 1 and fbeta([0], [1], [0])[0] == 0
    assert fbeta([0], [0], [2])[0] == 0 and abs(fbeta([2], [3], [2])[0] - 0.714285) < 1e-5     # the PDF example
    s1, y = np.array([0, 0, 1]), np.array([True, False, True])
    assert np.allclose(per_s1(s1, np.array([True, True, False]), y, 3, np.array([1, 1, 0])), [0.625 * 0 + 1.25 * 0.5 / (0.25 * 0.5 + 1), 0, 1])
    g, r = np.array([0.99, 0.99, 0.5, 0.01]), np.array([0.01, 0.9, 0.1, np.nan])
    assert veto_mask(g, r, "tau:0.05").tolist() == [True, False, False, False]
    p = apply_rule(g, r, "tau:0.05")
    assert p[0] == 0 and p[1] == 0.99 and p[3] == 0.01 and 0 < p[2] < 0.5
    e = empty_curve(np.array([0, 1]), np.array([True, True]), np.array([0.9, 0.6]), 3, ts=[0.5, 0.7, 0.95])
    assert e.tolist() == [1 / 3, 2 / 3, 1.0], e
    print("eval self-check ok")


if __name__ == "__main__":
    cmd, pos = sys.argv[1], sys.argv[2:]
    {"loco": lambda: loco(*pos), "ind": lambda: ind(*pos), "ctl": ctl, "selftest": selftest,
     "france": lambda: france(pos[0], *(float(x) for x in pos[1:]))}[cmd]()
