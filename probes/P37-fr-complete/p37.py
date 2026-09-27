"""P37: complete one-sided France groups. In truth every S1 with matches usually has both an S2 and an S3; France's
predictions are one-sided far more often than India's (excess ~14k S1) and hold fewer members (3.10 vs 3.38 per S1).
Rule gc(t_gc): a free argmax, non-sibling (P15 SIB3) pair with p >= t_gc joins its S1's predicted group
  one    : S1 has members of one source only -> its best candidate of the missing source
  oneall : S1 has members of one source only -> every candidate of the missing source (caps 5 S2 / 6 S3)
  all    : any non-empty S1 -> every candidate (caps)
    .venv/bin/python probes/P37-fr-complete/p37.py fr                 # France test: shape + counts per variant
    .venv/bin/python probes/P37-fr-complete/p37.py loco               # LOCO proxy, both directions (P27's M2 scores)
    .venv/bin/python probes/P37-fr-complete/p37.py build MODE T OUT   # sub34v_rescue90 with France completed
"""
import csv
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P15-research"), str(ROOT)]
from sib_rule import sib_mask  # noqa: E402

W = ROOT / "data/work"
BASE = Path(os.environ.get("P37_BASE", ROOT / "output/sub34v_rescue90"))   # P37_BASE=output/sub34v_rescue70 for r70
FR = W / "p31/fr31v0.1_t9868.parquet"
T_FR, T_RESCUE = 0.9868, 0.90                      # France accept t, the empty-S1 rescue t of #34v_r90
CI = {"India": 0, "US": 1}
OFF1, OFF2 = 10_000_000, 100_000_000
TS = {("US", "India"): 0.97, ("India", "US"): 0.81}  # target t at France's rule (P6 empty-rate t_profile, P27)
RD = dict(sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def hn_cat(h1, hx):
    """0 hn equal, 1 both missing, 2 one missing, 3 different."""
    e1, ex = h1 == "", hx == ""
    return np.select([~e1 & (h1 == hx), e1 & ex, e1 | ex], [0, 1, 2], 3).astype(np.int8)


def best_per(key, p, c):
    """Mask: the highest-p row of c per key."""
    o = np.flatnonzero(c)
    o = o[np.lexsort((-p[o], key[o]))]
    m = np.zeros(len(p), bool)
    m[o[np.r_[True, key[o][1:] != key[o][:-1]]] if len(o) else o] = True
    return m


def one_sided(s1, is2, pred):
    """Per row: its S1 has predicted members of the OTHER source only (so this row's source is missing)."""
    ns = int(s1.max()) + 1
    n2 = np.bincount(s1[pred & is2], minlength=ns)[s1]
    n3 = np.bincount(s1[pred & ~is2], minlength=ns)[s1]
    return np.where(is2, (n2 == 0) & (n3 > 0), (n3 == 0) & (n2 > 0)), n2, n3


def gc_add(s1, sx, is2, p, am, pred, sib, t_gc, mode, guard=None):
    """Pairs the group-completion rule adds (never removes). s1 / sx: int codes. allhn: oneall + members of two-sided
    groups whose pair passes guard (hn equal)."""
    free = np.bincount(sx[pred], minlength=int(sx.max()) + 1)[sx] == 0
    cand = am & ~pred & free & ~sib & (p >= t_gc)
    miss, n2, n3 = one_sided(s1, is2, pred)
    if mode == "one":
        return best_per(s1, p, cand & miss)
    c = cand & {"oneall": miss, "all": n2 + n3 > 0, "allhn": miss | ((n2 + n3 > 0) & (guard if guard is not None else False))}[mode]
    o = np.flatnonzero(c)
    o = o[np.lexsort((-p[o], s1[o] * 2 + is2[o]))]
    k = s1[o] * 2 + is2[o]
    i = np.arange(len(o))
    rank = i - np.maximum.accumulate(np.where(np.r_[True, k[1:] != k[:-1]], i, 0))
    room = np.where(is2, 5 - n2, 6 - n3)
    m = np.zeros(len(p), bool)
    m[o[rank < room[o]]] = True
    return m


def rescue(s1, p, am, pred, sib, t_r):
    """#34v_r90's empty-S1 rescue: an S1 with no member gets its best argmax, non-sibling pair if p >= t_r."""
    has = np.bincount(s1[pred], minlength=int(s1.max()) + 1)[s1] > 0
    return best_per(s1, p, am & ~sib & ~has & ~pred & (p >= t_r))


def shape(n2, n3, n_s1):
    """Shares of S1 by group shape over n_s1 S1 (n2 / n3: per-S1 predicted counts, S1 with no row count as empty)."""
    both, s3o, s2o = ((n2 > 0) & (n3 > 0)).sum(), ((n2 == 0) & (n3 > 0)).sum(), ((n2 > 0) & (n3 == 0)).sum()
    return {"both": round(both / n_s1, 4), "s3_only": round(s3o / n_s1, 4), "s2_only": round(s2o / n_s1, 4),
            "empty": round(1 - (both + s3o + s2o) / n_s1, 4), "mps": round((n2.sum() + n3.sum()) / n_s1, 4)}


def per_s1_counts(s1, is2, pred, ns):
    return np.bincount(s1[pred & is2], minlength=ns), np.bincount(s1[pred & ~is2], minlength=ns)


def dens(p, am, t_base, t, n_s1):
    """Argmax pairs per S1 with t <= p < t_base: the France -> target quantile map matches this."""
    return ((am & (p >= t) & (p < t_base)).sum()) / n_s1


def inv_dens(p, am, t_base, g, n_s1):
    v = np.sort(p[am & (p < t_base)])[::-1]
    k = int(round(g * n_s1))
    return float(v[min(k, len(v)) - 1]) if k > 0 else t_base


# ---------------------------------------------------------------- France test
def fr_load():
    d = pd.read_parquet(FR, columns=["s1", "sx", "p"])
    s1_ids = pd.read_parquet(W / "test_source1.parquet", columns=["entity_id", "country"])
    fr_ids = s1_ids.entity_id[s1_ids.country == "France"].to_numpy()
    del s1_ids
    sub = pd.read_csv(BASE / "matching_results.tsv", **RD)
    sub = sub[sub.source1_entity_id.isin(set(fr_ids))]
    e = sub.assign(sx=sub.matched_entity_ids.str.split(",")).explode("sx")
    e = e[e.sx.notna() & (e.sx != "")]
    del sub
    s1c, s1u = pd.factorize(pd.concat([pd.Series(fr_ids), d.s1]))
    s1c = s1c[len(fr_ids):]
    sxc = pd.factorize(d.sx)[0]
    key = pd.MultiIndex.from_arrays([d.s1, d.sx])
    pred = key.isin(pd.MultiIndex.from_arrays([e.source1_entity_id, e.sx]))
    assert pred.sum() == len(e), ("every predicted France pair is in the file", pred.sum(), len(e))
    import model as M
    p = d.p.to_numpy(np.float64)
    am = M.argmax_sx(sxc, p)
    assert (pred <= am).all(), "predicted pairs are argmax pairs"
    h1 = pd.read_parquet(W / "rf_test_s1_France.parquet", columns=["id", "hn"]).set_index("id").hn
    hx = pd.read_parquet(W / "p29/w_fix/rf_test_sx_France.parquet", columns=["id", "hn"]).set_index("id").hn
    h1, hx = h1.reindex(d.s1).fillna("").to_numpy(), hx.reindex(d.sx).fillna("").to_numpy()
    sib, hc = sib_mask(h1, hx), hn_cat(h1, hx)
    is2 = d.sx.str.startswith("S2-").to_numpy()
    return d, dict(s1=s1c.astype(np.int64), sx=sxc.astype(np.int64), is2=is2, p=p, am=am, pred=pred, sib=sib, hc=hc,
                   n_s1=len(fr_ids), ns=len(s1u))


MODES, T_GC = ("one", "oneall", "all"), (0.95, 0.9, 0.8, 0.7)


def fr():
    t0 = time.time()
    d, a = fr_load()
    ns, n_s1 = a["ns"], a["n_s1"]
    n2, n3 = per_s1_counts(a["s1"], a["is2"], a["pred"], ns)
    res = {"base": shape(n2, n3, n_s1), "pairs": len(d), "pred": int(a["pred"].sum()), "sib_pairs": int(a["sib"].sum())}
    print("France base shape:", res["base"], flush=True)
    res["dens"] = {t: round(dens(a["p"], a["am"], T_FR, t, n_s1), 5) for t in T_GC + (T_RESCUE,)}
    for mode in MODES:
        for t in T_GC:
            add = gc_add(a["s1"], a["sx"], a["is2"], a["p"], a["am"], a["pred"], a["sib"], t, mode)
            m2, m3 = per_s1_counts(a["s1"], a["is2"], a["pred"] | add, ns)
            r = shape(m2, m3, n_s1) | {"added": int(add.sum()), "s1_changed": int(np.unique(a["s1"][add]).size)}
            res[f"{mode}_{t}"] = r
            print(f"  {mode:6s} t {t}: {r}", flush=True)
    res["runtime_s"], res["peak_gb"] = round(time.time() - t0), round(peak(), 2)
    print(json.dumps(res["dens"]), f"{res['runtime_s']}s peak {res['peak_gb']:.2f} GB", flush=True)
    (ROOT / "probes/P37-fr-complete/fr.json").write_text(json.dumps(res, indent=1))
    return res


# ---------------------------------------------------------------- LOCO proxy
def fbeta(tp, npred, ntrue):
    p, r = tp / np.maximum(npred, 1), tp / np.maximum(ntrue, 1)
    f = np.where(tp > 0, 1.25 * p * r / np.maximum(0.25 * p + r, 1e-12), 0.0)
    return np.where(ntrue == 0, (npred == 0).astype(float), f)


def loco_load(src, tgt):
    """P27's M2-trained-on-src scores of tgt's fold-0 pairs (argmax over all train S1) as France-like arrays."""
    import features as F
    from ho import half
    ci = CI[tgt]
    S = pd.read_parquet(W / f"p27/p6/S_holdout_{src}.parquet", columns=["s1", "sx", "country", "p", "is_true", "am_all"])
    S = S[S.country.to_numpy() == ci].reset_index(drop=True)
    id1, idx = F.ids("train", tgt)
    s1s = id1.take(pa.array(S.s1.to_numpy() - ci * OFF1)).to_numpy(zero_copy_only=False)
    sxs = idx.take(pa.array(S.sx.to_numpy() - ci * OFF2)).to_numpy(zero_copy_only=False)
    y, am, p = S.is_true.to_numpy(), S.am_all.to_numpy(), S.p.to_numpy(np.float64)
    del S, id1, idx
    fo = pd.read_parquet(W / "s1_folds.parquet")
    fo = fo[(fo.fold == 0) & (fo.country == tgt)].entity_id.to_numpy()
    pos = pd.Index(fo).get_indexer(s1s)
    assert (pos >= 0).all()
    gt = pd.read_parquet(W / "gt_pairs.parquet")
    gt = gt[gt.s1.isin(set(fo))]
    g2 = gt.sx.str.startswith("S2-").to_numpy()
    gpos = pd.Index(fo).get_indexer(gt.s1)
    t2, t3 = np.bincount(gpos[g2], minlength=len(fo)), np.bincount(gpos[~g2], minlength=len(fo))
    del gt
    h1 = pd.read_parquet(W / f"rf_train_s1_{tgt}.parquet", columns=["id", "hn"]).set_index("id").hn
    hx = pd.read_parquet(W / f"rf_train_sx_{tgt}.parquet", columns=["id", "hn"]).set_index("id").hn
    h1, hx = h1.reindex(s1s).fillna("").to_numpy(), hx.reindex(sxs).fillna("").to_numpy()
    sib, hc = sib_mask(h1, hx), hn_cat(h1, hx)
    del h1, hx
    is2 = pd.Series(sxs).str.startswith("S2-").to_numpy()
    return dict(s1=pos.astype(np.int64), sx=pd.factorize(sxs)[0].astype(np.int64), is2=is2, p=p, am=am, y=y, sib=sib, hc=hc,
                n_s1=len(fo), ns=len(fo), t2=t2, t3=t3, hv=half(fo), s1_ids=s1s, sx_ids=sxs)


def loco_one(src, tgt, fr_res):
    t0 = time.time()
    a = loco_load(src, tgt)
    s1, sx, is2, p, am, y, sib, n = a["s1"], a["sx"], a["is2"], a["p"], a["am"], a["y"], a["sib"], a["n_s1"]
    ntrue, hv = a["t2"] + a["t3"], a["hv"]
    tb = TS[(src, tgt)]
    tmap = {t: inv_dens(p, am, tb, g, n) for t, g in fr_res["dens"].items()}
    pred = am & (p >= tb)
    pred |= rescue(s1, p, am, pred, sib, tmap[str(T_RESCUE)])
    per = lambda pr: fbeta(np.bincount(s1[pr & y], minlength=n), np.bincount(s1[pr], minlength=n), ntrue)
    f0 = per(pred)
    n2, n3 = per_s1_counts(s1, is2, pred, n)
    res = {"t_base": tb, "tmap": {k: round(v, 4) for k, v in tmap.items()}, "base_f05": round(f0.mean(), 5),
           "base_shape": shape(n2, n3, n), "truth_shape": shape(a["t2"], a["t3"], n)}
    print(f"{src}->{tgt}: {len(p):,} pairs, t {tb}, base F0.5 {f0.mean():.5f}; tmap {res['tmap']}", flush=True)
    print("  shape base", res["base_shape"], "\n  shape true", res["truth_shape"], flush=True)
    grid = sorted(set([round(x, 3) for x in np.r_[np.arange(0.05, 0.95, 0.05), tb - 0.01]] + list(tmap.values())))
    grid = [t for t in grid if t < tb]
    for mode in MODES:
        for t in grid:
            add = gc_add(s1, sx, is2, p, am, pred, sib, t, mode)
            if not add.any():
                continue
            dl = per(pred | add) - f0
            x = dl[hv == 1]
            m2, m3 = per_s1_counts(s1, is2, pred | add, n)
            fr_t = [k for k, v in tmap.items() if v == t]
            r = {"t": t, "fr_t": fr_t[0] if fr_t else None, "d": round(dl.mean(), 6), "d_h1": round(x.mean(), 6),
                 "se_h1": round(x.std() / np.sqrt(len(x)), 6), "added": int(add.sum()), "prec": round(y[add].mean(), 4),
                 "up": int((dl > 0).sum()), "down": int((dl < 0).sum()), "shape": shape(m2, m3, n)}
            res[f"{mode}_{t}"] = r
            print(f"  {mode:6s} t {t:.4f}{' (FR ' + str(r['fr_t']) + ')' if r['fr_t'] else '':12s} dF {r['d']:+.5f} "
                  f"(h1 {r['d_h1']:+.5f} ± {r['se_h1']:.5f}) added {r['added']:,} prec {r['prec']:.3f} "
                  f"up/down {r['up']}/{r['down']} s3o {r['shape']['s3_only']} s2o {r['shape']['s2_only']}", flush=True)
    res["runtime_s"], res["peak_gb"] = round(time.time() - t0), round(peak(), 2)
    return res


def loco():
    fr_res = json.loads((ROOT / "probes/P37-fr-complete/fr.json").read_text())
    sys.path.insert(0, str(ROOT / "probes/P28-addr-anchor"))
    out = {}
    for src, tgt in (("US", "India"), ("India", "US")):
        out[f"{src}->{tgt}"] = loco_one(src, tgt, fr_res)
        import gc
        gc.collect()
    (ROOT / "probes/P37-fr-complete/loco.json").write_text(json.dumps(out, indent=1))
    return out


def hnscan():
    """Precision of the two-sided-group additions (all minus oneall) by hn category; dF of allhn (guard = hn equal)."""
    fr_res = json.loads((ROOT / "probes/P37-fr-complete/fr.json").read_text())
    sys.path.insert(0, str(ROOT / "probes/P28-addr-anchor"))
    out = {}
    for src, tgt in (("US", "India"), ("India", "US")):
        a = loco_load(src, tgt)
        s1, sx, is2, p, am, y, sib, hc, n = (a[k] for k in ("s1", "sx", "is2", "p", "am", "y", "sib", "hc", "n_s1"))
        ntrue, hv, tb = a["t2"] + a["t3"], a["hv"], TS[(src, tgt)]
        tmap = {t: inv_dens(p, am, tb, g, n) for t, g in fr_res["dens"].items()}
        pred = am & (p >= tb)
        pred |= rescue(s1, p, am, pred, sib, tmap[str(T_RESCUE)])
        per = lambda pr: fbeta(np.bincount(s1[pr & y], minlength=n), np.bincount(s1[pr], minlength=n), ntrue)
        f0 = per(pred)
        print(f"{src}->{tgt}: accepted pairs by hn cat (eq/both-miss/one-miss/diff):",
              np.bincount(hc[pred], minlength=4).round(-2).tolist(), "prec", [round(y[pred & (hc == k)].mean(), 3) for k in range(4)])
        for frt in ("0.9", "0.8", "0.7"):
            t = tmap[frt]
            ex = gc_add(s1, sx, is2, p, am, pred, sib, t, "all") & ~gc_add(s1, sx, is2, p, am, pred, sib, t, "oneall")
            cat = {k: (int((ex & (hc == k)).sum()), round(y[ex & (hc == k)].mean(), 3)) for k in range(4)}
            r = {"two_sided_add_by_hc": cat}
            for g, nm in (((hc == 0), "allhn"), ((hc == 0) | (hc == 1), "allhn01")):
                add = gc_add(s1, sx, is2, p, am, pred, sib, t, "allhn", g)
                dl = per(pred | add) - f0
                x = dl[hv == 1]
                r[nm] = {"d": round(dl.mean(), 6), "d_h1": round(x.mean(), 6), "se_h1": round(x.std() / np.sqrt(len(x)), 6),
                         "added": int(add.sum()), "prec": round(y[add].mean(), 4)}
            out[f"{src}->{tgt}_fr{frt}"] = r
            print(f"  FR {frt} (t {t:.4f}):", json.dumps(r), flush=True)
        del a
    (ROOT / "probes/P37-fr-complete/hnscan.json").write_text(json.dumps(out, indent=1))
    d, a = fr_load()
    for frt in (0.9, 0.8, 0.7):
        ex = gc_add(a["s1"], a["sx"], a["is2"], a["p"], a["am"], a["pred"], a["sib"], frt, "all") & \
            ~gc_add(a["s1"], a["sx"], a["is2"], a["p"], a["am"], a["pred"], a["sib"], frt, "oneall")
        oa = gc_add(a["s1"], a["sx"], a["is2"], a["p"], a["am"], a["pred"], a["sib"], frt, "oneall")
        print(f"France t {frt}: two-sided adds by hc", np.bincount(a["hc"][ex], minlength=4).tolist(),
              "oneall adds by hc", np.bincount(a["hc"][oa], minlength=4).tolist(),
              "accepted by hc", np.bincount(a["hc"][a["pred"]], minlength=4).tolist(), flush=True)
    return out


def loco2():
    """Lower gc bars and the r70 rescue, on LOCO: (rescue t, gc t) combos, France t -> target t by density;
    dF vs (0.9, 0.7) = the uploaded #37_gc70 and vs (0.9, none) = #34v_r90."""
    d, a = fr_load()
    FT = (0.9, 0.7, 0.6, 0.5, 0.4, 0.3)
    fd = {t: dens(a["p"], a["am"], T_FR, t, a["n_s1"]) for t in FT}
    fr_cnt = {}
    for rt in (0.9, 0.7):
        pr = a["pred"] | rescue(a["s1"], a["p"], a["am"], a["pred"], a["sib"], rt) if rt < T_RESCUE else a["pred"]
        for gt in (0.7, 0.6, 0.5, 0.4, 0.3):
            add = gc_add(a["s1"], a["sx"], a["is2"], a["p"], a["am"], pr, a["sib"], gt, "oneall")
            fr_cnt[f"r{rt}_gc{gt}"] = {"rescued": int((pr & ~a["pred"]).sum()), "gc_added": int(add.sum())}
    del d, a
    print("France densities", {k: round(v, 4) for k, v in fd.items()}, "\nFrance counts", json.dumps(fr_cnt), flush=True)
    sys.path.insert(0, str(ROOT / "probes/P28-addr-anchor"))
    out = {"fr_dens": fd, "fr_counts": fr_cnt}
    for src, tgt in (("US", "India"), ("India", "US")):
        a = loco_load(src, tgt)
        s1, sx, is2, p, am, y, sib, n = (a[k] for k in ("s1", "sx", "is2", "p", "am", "y", "sib", "n_s1"))
        ntrue, hv, tb = a["t2"] + a["t3"], a["hv"], TS[(src, tgt)]
        tm = {t: inv_dens(p, am, tb, g, n) for t, g in fd.items()}
        per = lambda pr: fbeta(np.bincount(s1[pr & y], minlength=n), np.bincount(s1[pr], minlength=n), ntrue)
        base = am & (p >= tb)
        f = {}
        preds = {}
        for rt in (0.9, 0.7):
            pr = base | rescue(s1, p, am, base, sib, tm[rt])
            preds[(rt, None)] = pr
            for gt in (0.7, 0.6, 0.5, 0.4, 0.3):
                preds[(rt, gt)] = pr | gc_add(s1, sx, is2, p, am, pr, sib, tm[gt], "oneall")
        for k, pr in preds.items():
            f[k] = per(pr)
        r = {"tmap": {str(k): round(v, 4) for k, v in tm.items()}}
        for k, pr in preds.items():
            row = {}
            for ref, nm in (((0.9, 0.7), "vs_gc70"), ((0.9, None), "vs_r90")):
                dl = f[k] - f[ref]
                x = dl[hv == 1]
                diff = pr ^ preds[ref]
                row[nm] = {"d": round(dl.mean(), 6), "d_h1": round(x.mean(), 6), "se_h1": round(x.std() / np.sqrt(len(x)), 6),
                           "pairs_diff": int(diff.sum()), "prec_new": round(y[pr & ~preds[ref]].mean(), 4) if (pr & ~preds[ref]).any() else None}
            r[f"r{k[0]}_gc{k[1]}"] = row
            print(f"{src}->{tgt} rescue {k[0]} gc {k[1]}: vs gc70 {row['vs_gc70']} | vs r90 d {row['vs_r90']['d']:+.6f}", flush=True)
        out[f"{src}->{tgt}"] = r
        del a, preds, f
    (ROOT / "probes/P37-fr-complete/loco2.json").write_text(json.dumps(out, indent=1, default=str))
    return out


# ---------------------------------------------------------------- staging
def build(mode, t, out):
    """BASE with France groups completed by gc(mode, t). India/US lines and candidate_pairs.tsv byte-identical
    (hard links: the added pairs are pairs of the France file, i.e. already candidates)."""
    t0, t, out = time.time(), float(t), Path(out)
    d, a = fr_load()
    add = gc_add(a["s1"], a["sx"], a["is2"], a["p"], a["am"], a["pred"], a["sib"], t, mode, a["hc"] == 0)
    new = d[a["pred"] | add].groupby("s1").sx.agg(lambda s: ",".join(sorted(s))).to_dict()
    chg = set(d.s1[add])
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(BASE / "matching_results.tsv", encoding="utf-8") as f, open(out / "matching_results.tsv", "w", encoding="utf-8") as g:
        g.write(next(f))
        for line in f:
            s = line.split("\t", 1)[0]
            if s in chg:
                g.write(f"{s}\t{new[s]}\n")
                n += 1
            else:
                g.write(line)
    assert n == len(chg)
    for name in ("candidate_pairs.tsv", "code.zip"):
        if not (out / name).exists():
            os.link(BASE / name, out / name)
    m2, m3 = per_s1_counts(a["s1"], a["is2"], a["pred"] | add, a["ns"])
    info = {"mode": mode, "t_gc": t, "added": int(add.sum()), "s1_changed": n, "shape": shape(m2, m3, a["n_s1"]),
            "added_p": np.quantile(a["p"][add], [0, .25, .5, .75, 1]).round(4).tolist() if add.any() else [],
            "runtime_s": round(time.time() - t0), "peak_gb": round(peak(), 2)}
    (out / "p37_info.json").write_text(json.dumps(info))
    print(json.dumps(info), flush=True)
    return info


if __name__ == "__main__":
    # self-check on a toy group: S1 0 has S3 only (row 0), S1 1 has S2 + S3; rows 2..5 free argmax candidates
    s1 = np.array([0, 1, 1, 0, 0, 1, 0]); sx = np.arange(7); is2 = np.array([0, 1, 0, 1, 1, 1, 0], bool)
    p = np.array([.99, .99, .99, .92, .95, .91, .93]); am = np.ones(7, bool); pred = np.r_[[True] * 3, [False] * 4]
    sib = np.zeros(7, bool)
    assert list(np.flatnonzero(gc_add(s1, sx, is2, p, am, pred, sib, .9, "one"))) == [4]
    assert list(np.flatnonzero(gc_add(s1, sx, is2, p, am, pred, sib, .9, "oneall"))) == [3, 4]
    assert list(np.flatnonzero(gc_add(s1, sx, is2, p, am, pred, sib, .9, "all"))) == [3, 4, 5, 6]
    assert list(np.flatnonzero(gc_add(s1, sx, is2, p, am, pred, sib, .94, "all"))) == [4]
    assert list(np.flatnonzero(gc_add(s1, sx, is2, p, am, pred, sib, .9, "allhn", np.r_[[False] * 6, True]))) == [3, 4, 6]
    assert list(np.flatnonzero(gc_add(s1, sx, is2, p, am, pred, sib | (p == .95), .9, "one"))) == [3]   # sibling skipped
    assert not rescue(s1, p, am, pred, sib, .9).any()                                                  # no empty S1
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fr"
    {"fr": fr, "loco": loco, "build": build, "hnscan": hnscan, "loco2": loco2}[cmd](*sys.argv[2:])
