"""P41: recover France's missing synthetic-brand ("gib", P36 gib.py) S2/S3.

France assigns 54.8% of its gib SX (India 89.5%, US 73.3%). A gib SX keeps (a perturbed copy of) its S1's address, so
the rule looks for the S1 whose address key (house number + street key, + city) equals the SX's.
    probes/P41-fr-brand/py.sh probes/P41-fr-brand/p41.py prof            # France test profile (label-free)
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
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P15-research"),
                str(ROOT / "probes/P36-iu-residual"), str(ROOT / "probes/P37-fr-complete"), str(ROOT)]
import gib as G  # noqa: E402
import france as FRM  # noqa: E402
from sib_rule import sib_mask  # noqa: E402

W = ROOT / "data/work"
OUT = W / "p41"
HERE = Path(__file__).resolve().parent
BASE = Path(os.environ.get("P41_BASE", ROOT / "output/sub37_gc70"))
os.environ["P37_BASE"] = str(BASE)                   # p37 reads its base at import
import p37 as P  # noqa: E402
FR = W / "p31/fr31v0.1_t9868.parquet"
RD = dict(sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
vc = lambda s: {str(k): int(v) for k, v in pd.Series(s).value_counts(sort=False).sort_index().items()}


def src_rows(split, c, srcs, cols):
    return pd.concat([pq.read_table(W / f"{split}_source{i}.parquet", columns=["entity_id", *cols],
                                    filters=[("country", "==", c)]).to_pandas() for i in srcs], ignore_index=True)


def keys(split, c, sx_rf=None):
    """Per record of a split / country: hn, city (rules.feats), street key (france.street_key), gib flag (SX only).
    Cached at data/work/p41/keys_{split}_{c}_{s1|sx}.parquet."""
    out = {}
    for side, srcs in (("s1", (1,)), ("sx", (2, 3))):
        path = OUT / f"keys_{split}_{c}_{side}.parquet"
        if not path.exists():
            t0 = time.time()
            r = src_rows(split, c, srcs, ["business_name", "business_address"])
            rf = sx_rf if (side == "sx" and sx_rf) else W / f"rf_{split}_{side}_{c}.parquet"
            f = pd.read_parquet(rf, columns=["id", "hn", "city"]).set_index("id").reindex(r.entity_id)
            k = pd.DataFrame({"id": r.entity_id.to_numpy(), "hn": f.hn.fillna("").to_numpy(),
                              "city": f.city.fillna("").to_numpy(),
                              "street": FRM.street_key(r.business_address).to_numpy()})
            if side == "sx":
                k["gib"] = G.is_gib(r.business_name.tolist(), G.learn("test" if split == "test" else "train"))
            del r, f
            k.to_parquet(path, index=False)
            print(f"keys {split} {c} {side}: {len(k):,} rows, {time.time() - t0:.0f}s", flush=True)
        out[side] = pd.read_parquet(path)
    return out["s1"], out["sx"]


def keys_train(c):
    """keys() for a train country, SX side restricted to gib SX (6M SX would not fit): cached like keys()."""
    p1, px = OUT / f"keys_train_{c}_s1.parquet", OUT / f"keys_train_{c}_sxgib.parquet"
    if not p1.exists():
        r = src_rows("train", c, (1,), ["business_address"])
        f = pd.read_parquet(W / f"rf_train_s1_{c}.parquet", columns=["id", "hn", "city"]).set_index("id").reindex(r.entity_id)
        pd.DataFrame({"id": r.entity_id.to_numpy(), "hn": f.hn.fillna("").to_numpy(), "city": f.city.fillna("").to_numpy(),
                      "street": FRM.street_key(r.business_address).to_numpy()}).to_parquet(p1, index=False)
        del r, f
    if not px.exists():
        M, parts = G.learn("train"), []
        for i in (2, 3):
            f = pq.ParquetFile(W / f"train_source{i}.parquet")
            for rg in range(f.num_row_groups):
                t = f.read_row_group(rg, columns=["entity_id", "business_name", "business_address", "country"])
                t = t.filter(pc.equal(t["country"], c)).to_pandas()
                parts.append(t[G.is_gib(t.business_name.tolist(), M)][["entity_id", "business_address"]])
        r = pd.concat(parts, ignore_index=True)
        f = pq.read_table(W / f"rf_train_sx_{c}.parquet", columns=["id", "hn", "city"],
                          filters=[("id", "in", r.entity_id.tolist())]).to_pandas().set_index("id").reindex(r.entity_id)
        pd.DataFrame({"id": r.entity_id.to_numpy(), "hn": f.hn.fillna("").to_numpy(), "city": f.city.fillna("").to_numpy(),
                      "street": FRM.street_key(r.business_address).to_numpy(), "gib": True}).to_parquet(px, index=False)
    return pd.read_parquet(p1), pd.read_parquet(px)


def rel(k1i, s1_ids, kx):
    """hn / street / city equality and sibling offset of SX rows kx vs the S1 ids s1_ids (k1i: keys indexed by id)."""
    a = k1i.reindex(s1_ids)
    h1, hx = a.hn.fillna("").to_numpy(), kx.hn.to_numpy()
    s1s, c1 = a.street.fillna("").to_numpy(), a.city.fillna("").to_numpy()
    return {"hn_eq": (h1 != "") & (h1 == hx), "sib": sib_mask(h1, hx), "st_eq": (s1s != "") & (s1s == kx.street.to_numpy()),
            "city_eq": (c1 != "") & (c1 == kx.city.to_numpy())}


def train_prof():
    """Train truth: what ties a gib SX to its S1 (India / US), and where the gib decoys (gib SX matching nothing) sit."""
    gt = pd.read_parquet(W / "gt_pairs.parquet")
    res = {}
    for c in ("India", "US"):
        k1, kx = keys_train(c)
        t = gt[gt.sx.isin(set(kx.id))].set_index("sx").s1
        kx["true_s1"] = t.reindex(kx.id).fillna("").to_numpy()
        km, kn = key_match(k1, kx)
        kx["key_s1"], kx["key_n"] = np.where(km >= 0, k1.id.to_numpy()[np.maximum(km, 0)], ""), kn
        km0, kn0 = key_match(k1, kx, city=False)
        kx["key0_s1"], kx["key0_n"] = np.where(km0 >= 0, k1.id.to_numpy()[np.maximum(km0, 0)], ""), kn0
        k1i = k1.set_index("id")
        tr, de = kx.true_s1 != "", kx.true_s1 == ""
        r = rel(k1i, kx.true_s1[tr], kx[tr])
        sh = lambda s: round(float(np.mean(s)), 4)
        # S1 sizes in truth and the number of gib SX in each true group
        ntrue = gt[gt.s1.isin(set(kx.true_s1[tr]))].groupby("s1").size()
        ngib = kx[tr].groupby("true_s1").size()
        u = kx.key_n == 1
        u0 = kx.key0_n == 1
        res[c] = {"gib": len(kx), "true_share": sh(tr),
                  "true": {k: sh(v) for k, v in r.items()} | {"hn_st_eq": sh(r["hn_eq"] & r["st_eq"]),
                                                              "hn_st_city_eq": sh(r["hn_eq"] & r["st_eq"] & r["city_eq"]),
                                                              "key_n": vc(np.minimum(kx.key_n[tr], 5)),
                                                              "key1_is_true": sh((kx.key_s1 == kx.true_s1)[tr & u])},
                  "decoy": {"key_n": vc(np.minimum(kx.key_n[de], 5)), "key0_n": vc(np.minimum(kx.key0_n[de], 5))},
                  "key_rule_prec": {"hn_st_city_unique": [int(u.sum()), sh((kx.key_s1 == kx.true_s1)[u])],
                                    "hn_st_unique": [int(u0.sum()), sh((kx.key0_s1 == kx.true_s1)[u0])]},
                  "true_group_size": vc(np.minimum(ntrue, 8)), "gib_per_true_group": vc(np.minimum(ngib, 4))}
        # decoys at a unique key: does their key S1 have a true gib member / is it a singleton in truth?
        dk = kx[de & u]
        tg = gt[gt.s1.isin(set(dk.key_s1))].groupby("s1").size()
        res[c]["decoy"]["key1_s1_true_size"] = vc(np.minimum(tg.reindex(dk.key_s1).fillna(0).astype(int), 6))
        res[c]["decoy"]["key1_s1_has_true_gib"] = sh(dk.key_s1.isin(set(kx.true_s1[tr])))
        kx.to_parquet(OUT / f"train_gib_{c}.parquet", index=False)
        print(c, json.dumps(res[c], default=str), flush=True)
        del k1, kx, k1i
    (HERE / "train_prof.json").write_text(json.dumps(res, indent=1, default=str))
    return res


# ---------------------------------------------------------------- the rule, France and LOCO
FT = (0.9, 0.7, 0.5, 0.3, 0.1, 0.0)                  # France p floors of the rule
GUARDS = ("key1", "key1nc", "eq", "key1_m10", "key1_bare", "eq_bare")


def bare(split, c, ids):
    """Per gib SX id: its name is one token (the brand alone, no word kept from the S1 name). Cached."""
    path = OUT / f"bare_{split}_{c}.parquet"
    if not path.exists():
        r = src_rows(split, c, (2, 3), ["business_name"])
        r = r[r.entity_id.isin(set(ids))]
        pd.DataFrame({"id": r.entity_id.to_numpy(), "bare": [len(G.toks(n)) == 1 for n in r.business_name]}).to_parquet(path, index=False)
    return pd.read_parquet(path).set_index("id").bare


def gib_keys(k1, kx):
    """Per gib SX id: its unique hn+street+city key S1 ('' if none / not unique), same without city, and the
    street / city of the SX (for the not-unique 'eq' guard)."""
    g = kx[kx.gib].reset_index(drop=True)
    km, kn = key_match(k1, g)
    km0, kn0 = key_match(k1, g, city=False)
    ids = k1.id.to_numpy()
    return pd.DataFrame({"key1": np.where(kn == 1, ids[np.maximum(km, 0)], ""),
                         "key1nc": np.where(kn0 == 1, ids[np.maximum(km0, 0)], ""),
                         "hn": g.hn.to_numpy(), "street": g.street.to_numpy(), "city": g.city.to_numpy()}, index=g.id.to_numpy())


def guards(s1_ids, sx_ids, p, gk, k1i):
    """Per pair: the guard masks (the SX is gib and the pair's S1 is its key S1 ...)."""
    x = gk.reindex(sx_ids)
    isg = x.hn.notna().to_numpy()
    out = {"gib": isg, "key1": isg & (x.key1.to_numpy() == s1_ids), "key1nc": isg & (x.key1nc.to_numpy() == s1_ids)}
    a = k1i.reindex(s1_ids)
    ne = lambda v: pd.Series(v).fillna("").to_numpy()
    eq = lambda c: (ne(a[c].to_numpy()) != "") & (ne(a[c].to_numpy()) == ne(x[c].to_numpy()))
    out["eq"] = isg & eq("hn") & eq("street") & eq("city")
    o = np.lexsort((-p, sx_ids))                      # second-best p of the SX
    q = sx_ids[o]
    first = np.r_[True, q[1:] != q[:-1]]
    sec = np.where(np.r_[~first[1:], False], p[o][np.minimum(np.arange(len(o)) + 1, len(o) - 1)], 0.0)
    p2 = np.zeros(len(p))
    p2[o[first]] = sec[first]
    gap = np.full(len(p), np.nan)
    gap[o[first]] = p[o[first]] - p2[o[first]]
    out["key1_m10"] = out["key1"] & (np.nan_to_num(gap, nan=-1) >= 0.10)
    b = x.bare.fillna(False).to_numpy(bool) if "bare" in x else np.zeros(len(p), bool)
    out["key1_bare"], out["eq_bare"] = out["key1"] & b, out["eq"] & b
    return out


def rule(P, s1, sx, is2, p, am, pred, sib, t, g, empty=False):
    """Pairs the rule adds: free argmax pairs passing guard g with p >= t into non-empty S1 groups (caps 5 S2 / 6 S3,
    counting members), or (empty=True) also the best such pair of an empty S1."""
    add = P.gc_add(s1, sx, is2, p, am, pred, sib | ~g, t, "all")
    if empty:
        add |= P.rescue(s1, p, am, pred, sib | ~g, t)
    return add


def fr_state():
    d, a = P.fr_load()
    k1, kx = keys("test", "France", W / "p29/w_fix/rf_test_sx_France.parquet")
    gk = gib_keys(k1, kx)
    gk["bare"] = bare("test", "France", gk.index).reindex(gk.index).fillna(False).to_numpy(bool)
    a["G"] = guards(d.s1.to_numpy(), d.sx.to_numpy(), a["p"], gk, k1.set_index("id")[["hn", "street", "city"]])
    return P, d, a


def fr_dens(a, P):
    return {t: P.dens(a["p"], a["am"], P.T_FR, t, a["n_s1"]) for t in FT}


def gib_ratio(sx, am, pred, gib):
    """Assigned share of gib and non-gib SX among the SX that have an argmax pair, and their ratio."""
    asg = np.bincount(sx[pred], minlength=int(sx.max()) + 1) > 0
    g, o = np.unique(sx[am & gib]), np.unique(sx[am & ~gib])
    return {"gib": round(float(asg[g].mean()), 4), "nongib": round(float(asg[o].mean()), 4),
            "ratio": round(float(asg[g].mean() / asg[o].mean()), 4)}


def loco(gm=""):
    """LOCO (P37 protocol): P27's M2 trained on src scores tgt's fold-0 pairs; base = t_profile + rescue(0.9) +
    gc70 (France t mapped by argmax density); the rule at France floors FT mapped two ways: density (P37) and the
    quantile of the rule's own pool. dF in France-F0.5 units over all fold-0 target S1."""
    t0 = time.time()
    P, d, a = fr_state()
    fd = fr_dens(a, P)
    frpool = {g: np.sort(a["p"][a["am"] & ~a["pred"] & a["G"][g]]) for g in GUARDS}
    fr_cnt = {}
    for g in GUARDS:
        pool = a["am"] & ~a["pred"] & a["G"][g]
        fr_cnt[g] = {"pool": int(pool.sum())}
        for t in FT:
            ad = rule(P, a["s1"], a["sx"], a["is2"], a["p"], a["am"], a["pred"], a["sib"], t, a["G"][g])
            ae = rule(P, a["s1"], a["sx"], a["is2"], a["p"], a["am"], a["pred"], a["sib"], t, a["G"][g], True)
            fr_cnt[g][str(t)] = [int(ad.sum()), int(ae.sum())]
    print("France densities", {k: round(v, 4) for k, v in fd.items()}, "\nFrance adds [non-empty, +empty]", json.dumps(fr_cnt), flush=True)
    frq = {g: {t: float((frpool[g] >= t).mean()) if len(frpool[g]) else 0 for t in FT} for g in GUARDS}
    ratio_fr = gib_ratio(a["sx"], a["am"], a["pred"], a["G"]["gib"])
    print("France gib / non-gib assigned (SX with an argmax pair):", ratio_fr, flush=True)
    del d, a
    import gc
    gc.collect()
    sys.path.insert(0, str(ROOT / "probes/P28-addr-anchor"))
    out = {"fr_dens": fd, "fr_counts": fr_cnt, "fr_pool_q": frq, "fr_gib_ratio": ratio_fr, "gm": bool(gm)}
    for src, tgt in (("US", "India"), ("India", "US")):
        k1, kx = keys_train(tgt)
        gk = gib_keys(k1, kx)
        gk["bare"] = bare("train", tgt, gk.index).reindex(gk.index).fillna(False).to_numpy(bool)
        k1i = k1.set_index("id")[["hn", "street", "city"]]
        del k1, kx
        A = P.loco_load(src, tgt)
        s1, sx, is2, p, am, y, sib, n = (A[k] for k in ("s1", "sx", "is2", "p", "am", "y", "sib", "n_s1"))
        ntrue, hv, tb = A["t2"] + A["t3"], A["hv"], P.TS[(src, tgt)]
        G_ = guards(A["s1_ids"], A["sx_ids"], p, gk, k1i)
        del k1i, gk
        tm = {t: P.inv_dens(p, am, tb, fd[t], n) for t in FT}
        tm07, tm09 = tm[0.7], tm[0.9]                  # the base's gc70 and rescue90, as P37 mapped them
        per = lambda pr: P.fbeta(np.bincount(s1[pr & y], minlength=n), np.bincount(s1[pr], minlength=n), ntrue)
        pred = am & (p >= tb)
        r0 = gib_ratio(sx, am, pred, G_["gib"])
        if gm:                                       # France-matched: raise t on gib pairs to France's gib ratio
            pg = np.sort(p[am & G_["gib"]])
            want = ratio_fr["ratio"] * r0["nongib"]
            t_g = float(np.quantile(pg, 1 - want))
            pred = am & (p >= np.where(G_["gib"], max(t_g, tb), tb))
            print(f"{src}->{tgt} gm: ratio {r0} -> t_g {t_g:.4f}", gib_ratio(sx, am, pred, G_["gib"]), flush=True)
        pred |= P.rescue(s1, p, am, pred, sib, tm09)
        pred |= P.gc_add(s1, sx, is2, p, am, pred, sib, tm07, "oneall")
        f0 = per(pred)
        r = {"t_base": tb, "gib_ratio_base": r0, "gib_ratio": gib_ratio(sx, am, pred, G_["gib"]), "n_s1": n,
             "tmap_dens": {str(k): round(v, 4) for k, v in tm.items()}, "base_f05": round(f0.mean(), 5)}
        gib_sx = G_["gib"]
        fsx = np.bincount(sx[pred], minlength=int(sx.max()) + 1) > 0
        gsx = np.unique(sx[gib_sx])
        r["gib_sx_visible"] = int(len(gsx))
        r["gib_sx_assigned"] = round(float(fsx[gsx].mean()), 4)
        r["gib_sx_true"] = round(float(np.bincount(sx[gib_sx & y], minlength=len(fsx))[gsx].astype(bool).mean()), 4)
        for g in GUARDS:
            pool = am & ~pred & G_[g] & ~(fsx[sx])
            tpool = np.sort(p[pool])
            r[f"{g}_pool"] = {"n": int(pool.sum()), "prec": round(float(y[pool].mean()), 4) if pool.any() else None,
                              "prec_by_p": {f"{lo}-{hi}": [int(((p >= lo) & (p < hi) & pool).sum()),
                                                           round(float(y[(p >= lo) & (p < hi) & pool].mean()), 3)
                                                           if ((p >= lo) & (p < hi) & pool).any() else None]
                                            for lo, hi in ((0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01))}}
            for mp in ("dens", "pool"):
                for ft in FT:
                    if mp == "dens":
                        t = tm[ft]
                    else:
                        qf = frq[g][ft]
                        t = float(np.quantile(tpool, 1 - qf)) if len(tpool) and qf > 0 else 2.0
                    for emp in (False, True):
                        ad = rule(P, s1, sx, is2, p, am, pred, sib, t, G_[g], emp)
                        if not ad.any():
                            continue
                        dl = per(pred | ad) - f0
                        x = dl[hv == 1]
                        ne = np.bincount(s1[pred], minlength=n)[s1] > 0
                        row = {"t": round(t, 4), "d": round(dl.mean(), 6), "d_h1": round(x.mean(), 6),
                               "se_h1": round(x.std() / np.sqrt(len(x)), 6), "added": int(ad.sum()),
                               "prec": round(float(y[ad].mean()), 4),
                               "prec_nonempty": round(float(y[ad & ne].mean()), 4) if (ad & ne).any() else None,
                               "n_empty": int((ad & ~ne).sum()),
                               "prec_empty": round(float(y[ad & ~ne].mean()), 4) if (ad & ~ne).any() else None}
                        key = f"{g}_{mp}_fr{ft}{'_E' if emp else ''}"
                        r[key] = row
                        print(f"{src}->{tgt} {key:28s} t {t:.4f} dF {row['d']:+.6f} (h1 {row['d_h1']:+.6f} ± {row['se_h1']:.6f}) "
                              f"added {row['added']:,} prec {row['prec']:.3f} (empty {row['n_empty']} @ {row['prec_empty']})", flush=True)
        print(f"{src}->{tgt} pools:", json.dumps({k: v for k, v in r.items() if k.endswith('_pool') or k.startswith('gib_')}), flush=True)
        out[f"{src}->{tgt}"] = r
        del A, G_, pred
        gc.collect()
    out["runtime_s"], out["peak_gb"] = round(time.time() - t0), round(peak(), 2)
    (HERE / f"loco{'_gm' if gm else ''}.json").write_text(json.dumps(out, indent=1, default=str))
    return out


# ---------------------------------------------------------------- staging
HN90 = ROOT / "output/sub37_gc70_hn90"


def build(t, out, guard="key1"):
    """BASE (P41_BASE) + the rule's France pairs (guard, France p >= t, non-empty groups, caps). India / US lines and
    candidate_pairs.tsv byte-identical (hard links: every added pair is a pair of the France file, so a candidate)."""
    t0, t, out = time.time(), float(t), Path(out)
    P, d, a = fr_state()
    add = rule(P, a["s1"], a["sx"], a["is2"], a["p"], a["am"], a["pred"], a["sib"], t, a["G"][guard])
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
    m2, m3 = P.per_s1_counts(a["s1"], a["is2"], a["pred"] | add, a["ns"])
    b2, b3 = P.per_s1_counts(a["s1"], a["is2"], a["pred"], a["ns"])
    gs = (b2 + b3)[a["s1"][add]]
    ad = d[add]
    info = {"base": BASE.name, "guard": guard, "t": t, "added": int(add.sum()), "s1_changed": n,
            "shape_before": P.shape(b2, b3, a["n_s1"]), "shape_after": P.shape(m2, m3, a["n_s1"]),
            "added_p": np.quantile(a["p"][add], [0, .25, .5, .75, 1]).round(4).tolist() if add.any() else [],
            "added_s2_share": round(float(a["is2"][add].mean()), 4), "added_group_size": vc(np.minimum(gs, 6)),
            "added_into_one_sided": int(((b2 > 0) ^ (b3 > 0))[a["s1"][add]].sum()),
            "gib_ratio_after": gib_ratio(a["sx"], a["am"], a["pred"] | add, a["G"]["gib"])}
    if HN90.exists() and BASE != HN90:
        h = fr_pred(pd.unique(ad.s1), HN90)
        info["in_hn90"] = int(pd.MultiIndex.from_arrays([ad.s1, ad.sx]).isin(pd.MultiIndex.from_arrays([h.s1, h.sx])).sum())
    info["runtime_s"], info["peak_gb"] = round(time.time() - t0), round(peak(), 2)
    (out / "p41_info.json").write_text(json.dumps(info, default=str))
    print(json.dumps(info, default=str), flush=True)
    return info


def key_match(k1, kx, city=True):
    """Per SX row of kx: the S1 (index into k1) whose hn + street (+ city) key equals the SX's, and how many S1 share
    that key (0 = none; the S1 is -1 then). Empty hn or street never matches."""
    cols = ["hn", "street"] + (["city"] if city else [])
    ok1 = (k1.hn != "") & (k1.street != "") & ((k1.city != "") if city else True)
    g = k1[ok1].reset_index().groupby(cols)["index"].agg(["first", "size"])
    okx = (kx.hn != "") & (kx.street != "") & ((kx.city != "") if city else True)
    j = kx[cols].merge(g, left_on=cols, right_index=True, how="left")
    s1 = np.where(okx.to_numpy(), j["first"].fillna(-1).to_numpy(), -1).astype(np.int64)
    n = np.where(okx.to_numpy(), j["size"].fillna(0).to_numpy(), 0).astype(np.int64)
    return np.where(n > 0, s1, -1), n


def fr_pred(ids1, base=None):
    """France predicted pairs of a submission folder (default BASE): DataFrame s1, sx."""
    sub = pd.read_csv((base or BASE) / "matching_results.tsv", **RD)
    sub = sub[sub.source1_entity_id.isin(set(ids1))]
    e = sub.assign(sx=sub.matched_entity_ids.str.split(",")).explode("sx")
    return e[e.sx.notna() & (e.sx != "")].rename(columns={"source1_entity_id": "s1"})[["s1", "sx"]].reset_index(drop=True)


def prof():
    t0 = time.time()
    k1, kx = keys("test", "France", W / "p29/w_fix/rf_test_sx_France.parquet")
    pr = fr_pred(k1.id)
    asg = pd.Series(pr.s1.to_numpy(), index=pr.sx.to_numpy())
    grp = pr.groupby("s1").sx.agg(n="size", n2=lambda s: s.str.startswith("S2-").sum())
    kx["assigned"] = kx.id.isin(asg.index)
    g = kx[kx.gib].reset_index(drop=True)
    res = {"base": BASE.name, "fr_sx": len(kx), "gib": len(g), "gib_assigned": round(g.assigned.mean(), 4),
           "nongib_assigned": round(kx.assigned[~kx.gib].mean(), 4)}
    km, kn = key_match(k1, g)
    km0, kn0 = key_match(k1, g, city=False)
    g["key_s1"] = np.where(km >= 0, k1.id.to_numpy()[np.maximum(km, 0)], "")
    g["key_n"], g["key_n_nocity"] = kn, kn0
    d = pd.read_parquet(FR, columns=["s1", "sx", "p"])
    d = d[d.sx.isin(set(g.id))]
    best = d.sort_values("p", ascending=False).drop_duplicates("sx").set_index("sx")
    g["in_file"] = g.id.isin(best.index)
    g["best_p"] = best.p.reindex(g.id).to_numpy()
    g["am_s1"] = best.s1.reindex(g.id).fillna("").to_numpy()
    g["key_in_file"] = pd.MultiIndex.from_arrays([g.key_s1, g.id]).isin(pd.MultiIndex.from_arrays([d.s1, d.sx]))
    full = pq.read_table(W / "p14/w_idfx/p3/cands_test_France.parquet", columns=["s1", "sx"],
                         filters=[("sx", "in", list(g.id[g.key_s1 != ""]))]).to_pandas()
    g["key_in_full"] = pd.MultiIndex.from_arrays([g.key_s1, g.id]).isin(pd.MultiIndex.from_arrays([full.s1, full.sx]))
    g["in_full"] = g.id.isin(set(full.sx))
    g["asg_s1"] = asg.reindex(g.id).fillna("").to_numpy()
    k1i = k1.set_index("id")
    for tag, col in (("am", "am_s1"), ("asg", "asg_s1"), ("key", "key_s1")):
        a = k1i.reindex(g[col])
        h1, hx = a.hn.fillna("").to_numpy(), g.hn.to_numpy()
        g[f"{tag}_hn_eq"] = (h1 != "") & (h1 == hx)
        g[f"{tag}_sib"] = sib_mask(h1, hx)
        g[f"{tag}_st_eq"] = (a.street.fillna("").to_numpy() != "") & (a.street.fillna("").to_numpy() == g.street.to_numpy())
        g[f"{tag}_city_eq"] = (a.city.fillna("").to_numpy() != "") & (a.city.fillna("").to_numpy() == g.city.to_numpy())
    kg = grp.reindex(g.key_s1)
    g["key_grp_n"] = kg.n.fillna(0).to_numpy()
    g["key_grp_n2"] = kg.n2.fillna(0).to_numpy()
    g["is2"] = g.id.str.startswith("S2-")
    g.to_parquet(OUT / "fr_gib.parquet", index=False)
    un, ag = g[~g.assigned], g[g.assigned]
    sh = lambda s: round(float(np.mean(s)), 4)
    res["unassigned"] = {
        "n": len(un), "in_file": sh(un.in_file), "in_full": sh(un.in_full),
        "best_p_q": un.best_p.quantile([.1, .25, .5, .75, .9]).round(4).tolist(),
        "best_p_bands": vc(pd.cut(un.best_p, [0, .01, .1, .3, .5, .7, .9, .9868, 1.01])),
        "hn_empty": sh(un.hn == ""), "street_empty": sh(un.street == ""),
        "am_hn_eq": sh(un.am_hn_eq[un.in_file]), "am_st_eq": sh(un.am_st_eq[un.in_file]),
        "am_city_eq": sh(un.am_city_eq[un.in_file]), "am_sib": sh(un.am_sib[un.in_file]),
        "key_n": vc(np.minimum(un.key_n, 5)),
        "key_n_nocity": vc(np.minimum(un.key_n_nocity, 5)),
        "key1_in_file": sh(un.key_in_file[un.key_n == 1]), "key1_in_full": sh(un.key_in_full[un.key_n == 1]),
        "key1_is_am": sh((un.key_s1 == un.am_s1)[un.key_n == 1]),
        "key1_grp_n": vc(np.minimum(un.key_grp_n[un.key_n == 1], 6)),
    }
    res["assigned"] = {
        "n": len(ag), "asg_hn_eq": sh(ag.asg_hn_eq), "asg_st_eq": sh(ag.asg_st_eq), "asg_city_eq": sh(ag.asg_city_eq),
        "asg_sib": sh(ag.asg_sib), "key_n": vc(np.minimum(ag.key_n, 5)),
        "key1_is_asg": sh((ag.key_s1 == ag.asg_s1)[ag.key_n == 1]),
        "asg_is_key_when_hn_st_eq": sh((ag.key_s1 == ag.asg_s1)[ag.asg_hn_eq & ag.asg_st_eq]),
    }
    res["runtime_s"], res["peak_gb"] = round(time.time() - t0), round(peak(), 2)
    print(json.dumps(res, indent=1, default=str), flush=True)
    (HERE / "prof.json").write_text(json.dumps(res, indent=1, default=str))
    return res


if __name__ == "__main__":
    k1 = pd.DataFrame({"hn": ["12", "12", "3", ""], "street": ["paix r", "paix r", "x", "paix r"],
                       "city": ["lille", "lyon", "a", "lille"]})
    kx = pd.DataFrame({"hn": ["12", "12", "3", ""], "street": ["paix r", "paix r", "x", "paix r"],
                       "city": ["lille", "", "b", "lille"]})
    s, n = key_match(k1, kx)
    assert s.tolist() == [0, -1, -1, -1] and n.tolist() == [1, 0, 0, 0], (s, n)
    s, n = key_match(k1, kx, city=False)
    assert s.tolist() == [0, 0, 2, -1] and n.tolist() == [2, 2, 1, 0], (s, n)
    # guards + rule on a toy France: SX 0 (gib) has unique key S1 0 (a non-empty group), SX 1 (gib) keys to S1 1
    # but its argmax is S1 0, SX 2 is not gib, SX 3 (gib) keys to the empty S1 2
    gk = pd.DataFrame({"key1": ["a", "b", "c"], "key1nc": ["a", "b", "c"], "hn": ["1", "2", "3"], "street": ["x"] * 3,
                       "city": ["l"] * 3}, index=["g0", "g1", "g3"])
    k1i = pd.DataFrame({"hn": ["1", "2", "3"], "street": ["x"] * 3, "city": ["l"] * 3}, index=["a", "b", "c"])
    s1i, sxi = np.array(["a", "a", "b", "a", "c", "a"]), np.array(["m", "g0", "g1", "g1", "g3", "n"])
    pp = np.array([.99, .8, .5, .9, .8, .6])
    G_ = guards(s1i, sxi, pp, gk, k1i)
    assert G_["key1"].tolist() == [False, True, True, False, True, False], G_["key1"]
    s1c, sxc = pd.factorize(s1i)[0], pd.factorize(sxi)[0]
    pred = np.array([1, 0, 0, 0, 0, 0], bool)
    import model as MD
    am = MD.argmax_sx(sxc, pp)
    z = np.zeros(6, bool)
    add = rule(P, s1c, sxc, z, pp, am, pred, z, .7, G_["key1"])
    assert add.tolist() == [False, True, False, False, False, False], add          # g1's argmax is S1 a; S1 c is empty
    assert rule(P, s1c, sxc, z, pp, am, pred, z, .7, G_["key1"], True).tolist() == [False, True] + [False] * 2 + [True, False]
    assert not rule(P, s1c, sxc, z, pp, am, pred, z, .85, G_["key1"]).any()        # below the floor
    print("p41 self-check ok")
    {"prof": prof, "train_prof": train_prof, "loco": loco, "build": build}[sys.argv[1]](*sys.argv[2:]) if len(sys.argv) > 1 else None
