"""P33: R05 (P23-research Brief 4) -- hn-equal single org-word swap tier, gated by a label-free sibling score s(w).

A pair is an R05 pair when (on top of P28's `bxsg` = sub1 & hn equal & city & street exact & not a learned sibling word):
  argmax  it is its SX's argmax pair (Brief 4; P28 also allowed SX-best < t)
  word    the swapped-in word w is not an R1 filler (fils / cie / associes / freres / frs: P29's rebuild handles them)
  s(w)    share of the country's sub1 pairs with swapped-in word w whose hn relation is a generator sibling offset (SIB3),
          over ALL of w's sub1 pairs (any hn relation); support n(w) >= 20; s(w) <= S_MAX (0.4). Learned label-free on
          the country's P3 blocked list (P28: the pruned list hides siblings), or on the scored list (Brief 4 literal)
  uniq    no other candidate S1 of the SX (in the scored list) contains the remaining core (SX core minus w)
  nocore  the SX's exact core is not another S1's core in the S1's city (S1 pool of the country; R21 negative rule)
  floor   p >= 0.02 (matches stay a subset of the m2_big p >= 0.02 candidate reading)
then p := max(p, 0.99) (P28's BOOST) before the usual argmax + t decision.

    .venv/bin/python probes/P33-orgword-tier/r05.py ho US India        # in-domain #8c holdout (P23 dump), t 0.71
    .venv/bin/python probes/P33-orgword-tier/r05.py loco India>US US>India   # P6 LOCO (P27 copies), t_profile + oracle
    .venv/bin/python probes/P33-orgword-tier/r05.py fr IN T OUT [variant]    # France file (post-rules) -> + R05 -> cap
    .venv/bin/python probes/P33-orgword-tier/r05.py selftest
"""
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P20-france-patterns"),
                str(ROOT / "probes/P28-addr-anchor"), str(ROOT / "probes/P29-fr-tiers")]
import model as M  # noqa: E402
from anchor import SIB3, _rf, boosted, decide, rule_masks  # noqa: E402
from patterns import name_rel  # noqa: E402

W = ROOT / "data/work"
PROBE = "P33-orgword-tier"
FILL = {"fils", "cie", "associes", "freres", "frs"}
S_MAX, SUPPORT, FLOOR = 0.4, 20, 0.02
A_MIN = 2.0        # R05a: w is at least twice as common in SX cores as in S1 cores (a corruptor-added word)
E_MIN = 0.8        # R05e: w's swap pairs keep the hn as often as true pairs (~0.9): a generator noise word, not a brand
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def s_table(pairs: pd.DataFrame, xs: str, c: str) -> pd.DataFrame:
    """word -> n (sub1 pairs with swapped-in word w), s (their SIB3 share over all hn relations). Label-free."""
    r1 = _rf(W / f"rf_{xs}_s1_{c}.parquet", pd.unique(pairs.s1), ["core", "hn"])
    rx = _rf(W / f"rf_{xs}_sx_{c}.parquet", pd.unique(pairs.sx), ["core", "hn", "is_web"])
    c1, cx = r1.core.reindex(pairs.s1).fillna("").to_numpy(), rx.core.reindex(pairs.sx).fillna("").to_numpy()
    wx = rx.is_web.reindex(pairs.sx).fillna(False).to_numpy()
    kw = [name_rel(a, b, x) for a, b, x in zip(c1, cx, wx)]
    sub = np.array([k == "sub1" for k, _ in kw])
    words = np.array([w for (k, w) in kw], object)[sub]
    del kw, c1, cx, wx
    h1 = r1.hn.reindex(pairs.s1.to_numpy()[sub]).fillna("").to_numpy()
    hx = rx.hn.reindex(pairs.sx.to_numpy()[sub]).fillna("").to_numpy()
    both = (h1 != "") & (hx != "")
    dl = np.full(len(h1), np.nan)
    dl[both] = pd.to_numeric(pd.Series(hx[both]), errors="coerce").to_numpy(float) - \
        pd.to_numeric(pd.Series(h1[both]), errors="coerce").to_numpy(float)
    g = pd.DataFrame({"w": words, "sib": np.isin(dl, SIB3), "eq": dl == 0, "hn": both}).groupby("w")
    g = g.agg(n=("sib", "size"), s=("sib", "mean"), n_hn=("hn", "sum"), n_eq=("eq", "sum"))
    g["e"] = g.n_eq / g.n_hn.clip(lower=1)                      # hn-equal share of w's hn-present swap pairs
    return g


def a_table(xs: str, c: str) -> pd.Series:
    """word -> a(w) = (share of the country's SX cores holding w) / (share of its S1 cores holding w), +1 smoothing.
    Label-free: the corruptor adds noise words to S2/S3 names only, so they sit far above 1 (P23 vocab_asym, R19)."""
    from collections import Counter
    import pyarrow.parquet as pq
    out = []
    for side in ("s1", "sx"):
        cnt, n = Counter(), 0
        for b in pq.ParquetFile(W / f"rf_{xs}_{side}_{c}.parquet").iter_batches(500_000, columns=["core"]):
            col = b.column(0).to_pylist()
            n += len(col)
            for x in col:
                cnt.update(set(x.split()))                      # document frequency
        out.append((pd.Series(cnt, dtype=float), n))
    (d1, n1), (dx, nx) = out
    idx = d1.index.union(dx.index)
    return ((dx.reindex(idx).fillna(0) + 1) / nx) / ((d1.reindex(idx).fillna(0) + 1) / n1)


def r05_masks(d: pd.DataFrame, split: str, c: str, t: float, am=None, best=None, learn=None, s_lists=None,
              a_tab=None) -> dict:
    """d: s1, sx, p (scored list); learn: blocked (s1, sx) list for P28's sibling guard; s_lists: {tag: s_table}.
    Returns {variant: mask} + 'info'. Variants: r05 (the brief), and ablations of one guard each."""
    xs = "train" if split == "holdout" else "test"
    m = rule_masks(d, split, c, t, am=am, best=best, learn=learn)
    p = d.p.to_numpy(np.float64)
    qx = pd.factorize(d.sx)[0]
    am = M.argmax_sx(qx, p) if am is None else am
    base_a = m["a"] & (m["kind"] == "sub1")                     # sub1 & hn eq & city (Brief 4's literal address)
    cand = np.flatnonzero(base_a & am)
    s1c, sxc = d.s1.to_numpy()[cand], d.sx.to_numpy()[cand]
    r1 = _rf(W / f"rf_{xs}_s1_{c}.parquet", pd.unique(s1c), ["core", "city"])
    rx = _rf(W / f"rf_{xs}_sx_{c}.parquet", pd.unique(sxc), ["core", "is_web"])
    c1, cx = r1.core.reindex(s1c).fillna("").to_numpy(), rx.core.reindex(sxc).fillna("").to_numpy()
    city1 = r1.city.reindex(s1c).fillna("").to_numpy()
    w = np.array([name_rel(a, b, False)[1] for a, b in zip(c1, cx)], object)
    rest = [frozenset(b.split()) - {x} for b, x in zip(cx, w)]
    # uniq: other candidate S1 of the same SX whose core contains the remaining core
    mine = pd.DataFrame({"sx": sxc, "s1": s1c, "i": np.arange(len(cand))})
    oth = mine[["sx", "i", "s1"]].merge(d[["sx", "s1"]].rename(columns={"s1": "o"}), on="sx")
    oth = oth[oth.o != oth.s1]
    oc = _rf(W / f"rf_{xs}_s1_{c}.parquet", pd.unique(oth.o), ["core"]).core.reindex(oth.o).fillna("").to_numpy()
    hit = np.array([rest[i] <= set(o.split()) for i, o in zip(oth.i.to_numpy(), oc)], bool)
    uniq = ~np.isin(np.arange(len(cand)), oth.i.to_numpy()[hit])
    del oth, oc, hit
    # nocore: (SX core, S1 city) is some other S1's (core, city) in the country's S1 pool
    pool = pd.read_parquet(W / f"rf_{xs}_s1_{c}.parquet", columns=["core", "city"])
    keys = set(zip(pool.core.to_numpy(), pool.city.to_numpy()))
    del pool
    nocore = np.array([(b, cc) not in keys for b, cc in zip(cx, city1)], bool)
    del keys
    fill = np.isin(w, list(FILL))
    floor = p[cand] >= FLOOR
    street = m["bxs"][cand]                                     # + street exact (P28 st_x)
    sib_ok = ~np.isin(np.array([f"sub1:{x}" for x in w], object), list(m["sibw"]))   # P28's sibling-word guard
    sg = {}
    for tag, st in (s_lists or {}).items():
        n = pd.Series(w).map(st.n).fillna(0).to_numpy()
        s = pd.Series(w).map(st.s).fillna(1.0).to_numpy()
        e = pd.Series(w).map(st.e).fillna(0.0).to_numpy()
        sg[tag] = (n >= SUPPORT, s, e)
    core_ok = ~fill & floor & uniq & nocore & sib_ok
    out, info = {}, {"sub1_hn_city_argmax": len(cand), "street": int(street.sum()), "filler": int(fill.sum()),
                     "uniq_fail": int((~uniq).sum()), "nocore_fail": int((~nocore).sum()), "sibguard_fail": int((~sib_ok).sum()), "sib_words": len(m["sibw"]),
                     "below_floor": int((~floor).sum())}
    tags = list(sg)
    main = tags[0] if tags else None

    def put(name, keep):
        mm = np.zeros(len(d), bool)
        mm[cand[keep]] = True
        out[name] = mm
        info[f"n_{name}"] = int(keep.sum())

    if main:
        sup, s, e = sg[main]
        put("r05", core_ok & street & sup & (s <= S_MAX))
        put("r05_s3", core_ok & street & sup & (s <= 0.3))
        put("r05_s2", core_ok & street & sup & (s <= 0.2))
        put("r05_nos", core_ok & street)
        put("r05_city", core_ok & sup & (s <= S_MAX))                 # Brief 4 literal: city, no street
        put("r05_nouniq", ~fill & floor & nocore & sib_ok & street & sup & (s <= S_MAX))
        put("r05_nonocore", ~fill & floor & uniq & sib_ok & street & sup & (s <= S_MAX))
        put("r05_p5", core_ok & street & sup & (s <= S_MAX) & (p[cand] >= 0.5))
        put("r05e", core_ok & street & sup & (s <= S_MAX) & (e >= E_MIN))     # + noise-word lens (pre-registered 0.8)
        put("r05e_nos", core_ok & street & sup & (e >= E_MIN))
        put("r05e7", core_ok & street & sup & (s <= S_MAX) & (e >= 0.7))
        put("r05e9", core_ok & street & sup & (s <= S_MAX) & (e >= 0.9))
        if a_tab is not None:
            a = pd.Series(w).map(a_tab).fillna(1.0).to_numpy()
            put("r05a", core_ok & street & sup & (s <= S_MAX) & (a >= A_MIN))    # pre-registered a >= 2
            put("r05a15", core_ok & street & sup & (s <= S_MAX) & (a >= 1.5))
            put("r05a4", core_ok & street & sup & (s <= S_MAX) & (a >= 4))
            put("r05a_nos", core_ok & street & sup & (a >= A_MIN))
            out["a"] = pd.Series(a, index=cand)
        for tg in tags[1:]:
            sup2, s2, _ = sg[tg]
            put(f"r05_{tg}", core_ok & street & sup2 & (s2 <= S_MAX))
        info["s_dist_cand"] = {f"le{q}": int((sup & (s <= q)).sum()) for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)}
    put("bxsg_am", m["bxsg"][cand])
    out["info"] = info
    out["w"] = pd.Series(w, index=cand)
    return out


VARS = ("r05", "r05_s3", "r05_s2", "r05_nos", "r05_city", "r05_nouniq", "r05_nonocore", "r05_p5", "r05_scored", "bxsg_am",
        "r05e", "r05e_nos", "r05e7", "r05e9", "r05a", "r05a15", "r05a4", "r05a_nos")


# ---------------------------------------------------------------- in-domain: #8c holdout dump (P28 ho.py protocol)
def ho(c: str) -> dict:
    from ho import T, half, per_s1
    t0 = time.time()
    s1 = pd.read_parquet(W / "worlds/holdout/s1.parquet", columns=["entity_id", "country"])
    ids = s1.entity_id[s1.country == c].tolist()
    tr = pd.read_parquet(W / "worlds/holdout/truth.parquet")
    truth = {s: set() for s in ids}
    for a, b in zip(tr.s1, tr.sx):
        if a in truth:
            truth[a].add(b)
    del s1, tr
    d = pd.read_parquet(W / f"p23/pairs_{c}.parquet", columns=["s1", "sx", "holdout", "y", "p", "pred"])
    p = d.p.to_numpy(np.float64)
    base = decide(d.sx, p, T)
    assert (base == d.pred.to_numpy()).all(), "the decision reproduces #8c's"
    blk = pd.read_parquet(W / f"p3/cands_holdout_full_{c}.parquet", columns=["s1", "sx"])
    st_b = s_table(blk, "train", c)
    st_s = s_table(d[["s1", "sx"]], "train", c)
    m = r05_masks(d[["s1", "sx", "p"]], "holdout", c, T, learn=blk, s_lists={"blocked": st_b, "scored": st_s},
                  a_tab=a_table("train", c))
    del blk
    f0 = per_s1(d, base, truth)
    hv = pd.Series(half(f0.index), index=f0.index)
    hmask, y = d.holdout.to_numpy(), d.y.to_numpy()
    res = {"base_h1": round(f0[hv == 1].mean(), 5), "base_all": round(f0.mean(), 5), **m["info"],
           "s_words_blocked": int((st_b.n >= SUPPORT).sum()), "s_words_le04_blocked": int(((st_b.n >= SUPPORT) & (st_b.s <= S_MAX)).sum())}
    print(f"{c}: {len(d):,} pairs, base half1 {res['base_h1']:.5f}; {m['info']}", flush=True)
    for v in VARS:
        mask = m[v]
        pred = decide(d.sx, boosted(p, mask), T)
        dl = per_s1(d, pred, truth) - f0
        r = {}
        for h in (0, 1):
            x = dl[hv == h]
            r[f"d_h{h}"], r[f"se_h{h}"] = x.mean(), x.std() / np.sqrt(len(x))
        rh, flip = mask & hmask, pred & ~base & hmask
        r |= {"d_all": dl.mean(), "rule_pairs": int(rh.sum()), "rule_prec": y[rh].mean() if rh.any() else np.nan,
              "flip": int(flip.sum()), "flip_prec": y[flip].mean() if flip.any() else np.nan,
              "lost": int((base & ~pred & hmask).sum())}
        print(f"  {v:13s} dF h1 {r['d_h1']:+.5f} ± {r['se_h1']:.5f} (h0 {r['d_h0']:+.5f}) | rule {r['rule_pairs']:,} "
              f"prec {r['rule_prec']:.3f} | flip {r['flip']:,} prec {r['flip_prec']:.3f} | lost {r['lost']}", flush=True)
        res |= {f"{v}_{k}": (round(float(x), 5) if isinstance(x, (float, np.floating)) else x) for k, x in r.items()}
    res["runtime_s"] = round(time.time() - t0)
    return res


# ---------------------------------------------------------------- LOCO: P6 dense holdout, M2 trained on the other country
def loco(src: str, tgt: str) -> dict:
    import features as F
    from ho import half
    from loco import CI, OFF1, OFF2, TS, fbeta
    t0 = time.time()
    ci = CI[tgt]
    S = pd.read_parquet(W / f"p27/p6/S_holdout_{src}.parquet", columns=["s1", "sx", "country", "p", "is_true", "am_all",
                                                                         "sx_best_p"])
    S = S[S.country.to_numpy() == ci].reset_index(drop=True)
    id1, idx = F.ids("train", tgt)
    d = pd.DataFrame({"s1": id1.take(pa.array(S.s1.to_numpy() - ci * OFF1)).to_numpy(zero_copy_only=False),
                      "sx": idx.take(pa.array(S.sx.to_numpy() - ci * OFF2)).to_numpy(zero_copy_only=False),
                      "p": S.p.to_numpy(np.float64)})
    y, am, best = S.is_true.to_numpy(), S.am_all.to_numpy(), S.sx_best_p.to_numpy(np.float64)
    del S
    fo = pd.read_parquet(W / "s1_folds.parquet")
    fo = fo[(fo.fold == 0) & fo.entity_id.isin(set(id1.to_numpy(zero_copy_only=False)))].entity_id.to_numpy()
    ntrue = pd.read_parquet(W / "gt_pairs.parquet", columns=["s1"]).s1.value_counts().reindex(fo).fillna(0).to_numpy()
    pos = pd.Index(fo).get_indexer(d.s1)
    assert (pos >= 0).all()
    hv = half(fo)
    st_b = s_table(d[["s1", "sx"]], "train", tgt)              # the LOCO frame IS the blocked (dense) list
    a_tab = a_table("train", tgt)
    res = {"s_words": int((st_b.n >= SUPPORT).sum()), "s_words_le04": int(((st_b.n >= SUPPORT) & (st_b.s <= S_MAX)).sum())}

    def per(pred):
        tp = np.bincount(pos[pred & y], minlength=len(fo))
        return fbeta(tp, np.bincount(pos[pred], minlength=len(fo)), ntrue)
    for t in TS[(src, tgt)]:
        m = r05_masks(d, "holdout", tgt, t, am=am, best=best, s_lists={"blocked": st_b, "scored": st_b}, a_tab=a_tab)
        base = am & (d.p.to_numpy() >= t)
        f0 = per(base)
        print(f"{src}->{tgt} t {t}: {len(d):,} pairs, base {f0.mean():.5f} (h1 {f0[hv == 1].mean():.5f}); {m['info']}", flush=True)
        res[f"t{t}_base"] = round(f0.mean(), 5)
        res[f"t{t}_info"] = m["info"]
        for v in VARS:
            r = m[v]                                           # argmax pairs only, one per SX
            pred = base | r
            dl = per(pred) - f0
            flip = pred & ~base
            k = {}
            for h in (0, 1):
                x = dl[hv == h]
                k[f"d_h{h}"], k[f"se_h{h}"] = x.mean(), x.std() / np.sqrt(len(x))
            k |= {"d_all": dl.mean(), "rule_pairs": int(r.sum()), "rule_prec": y[r].mean() if r.any() else np.nan,
                  "flip": int(flip.sum()), "flip_prec": y[flip].mean() if flip.any() else np.nan}
            print(f"  {v:13s} dF h1 {k['d_h1']:+.5f} ± {k['se_h1']:.5f} (h0 {k['d_h0']:+.5f}, all {k['d_all']:+.5f}) | "
                  f"rule {k['rule_pairs']:,} prec {k['rule_prec']:.3f} | flip {k['flip']:,} prec {k['flip_prec']:.3f}", flush=True)
            res |= {f"t{t}_{v}_{a}": (round(float(b), 5) if isinstance(b, (float, np.floating)) else b) for a, b in k.items()}
        if t == TS[(src, tgt)][0]:                              # flipped-pair precision by s bin (blocked s)
            fl = m["r05_nos"] & ~base
            ww = m["w"].reindex(np.flatnonzero(fl)).to_numpy()
            sv = pd.Series(ww).map(st_b.s).fillna(1).to_numpy()
            nv = pd.Series(ww).map(st_b.n).fillna(0).to_numpy()
            yy = y[fl]
            bins = {}
            for lo, hi in ((-1, .2), (.2, .3), (.3, .4), (.4, .6), (.6, 1.01)):
                k = (sv > lo) & (sv <= hi) & (nv >= SUPPORT)
                bins[f"s({max(lo, 0)},{hi}]"] = [int(k.sum()), round(float(yy[k].mean()), 3) if k.any() else None]
            k = nv < SUPPORT
            bins["n<20"] = [int(k.sum()), round(float(yy[k].mean()), 3) if k.any() else None]
            print("  flips of r05_nos by s(w) bin [n, precision]:", bins, flush=True)
            res[f"t{t}_nos_flip_by_s"] = bins
    res["runtime_s"] = round(time.time() - t0)
    return res


# ---------------------------------------------------------------- France: on top of #31's post-rules file, then the cap
def fr(src: str, t: str, out: str, variant: str = "r05") -> dict:
    import p29
    t0, t = time.time(), float(t)
    assert t < 0.99, "BOOST 0.99 must exceed t"
    d = pd.read_parquet(src, columns=["s1", "sx", "p"])
    p = d.p.to_numpy(np.float64).copy()
    n_fr = 259_452
    acc0 = decide(d.sx, p, t)
    learn = pd.read_parquet(W / "p3/cands_test_France.parquet", columns=["s1", "sx"])
    st_b = s_table(learn, "test", "France")
    st_s = s_table(d[["s1", "sx"]], "test", "France")
    m = r05_masks(d[["s1", "sx"]].assign(p=p), "test", "France", t, learn=learn,
                  s_lists={"blocked": st_b, "scored": st_s}, a_tab=a_table("test", "France"))
    del learn
    res = {"src": src, "t": t, "variant": variant, **m["info"],
           "base": {"empty": round(1 - d.s1[acc0].nunique() / n_fr, 5), "mps": round(int(acc0.sum()) / n_fr, 4)}}
    for v in VARS:
        a = decide(d.sx, boosted(p, m[v]), t)
        res[f"{v}_added"], res[f"{v}_s1"] = int((a & ~acc0).sum()), int(d.s1[a & ~acc0].nunique())
    q = boosted(p, m[variant])
    acc = decide(d.sx, q, t)
    drop = p29.cap_mask(d.s1.to_numpy(), d.sx.to_numpy(), q, acc)
    q[d.sx.isin(set(d.sx[drop])).to_numpy()] = 0.0
    acc = decide(d.sx, q, t)
    k = d.s1[acc].nunique()
    res |= {"added": int((acc & ~acc0).sum()), "dropped": int((acc0 & ~acc).sum()), "cap_pairs": int(drop.sum()),
            "s1_changed": int(d.s1[acc ^ acc0].nunique()), "empty": round(1 - k / n_fr, 5), "mps": round(int(acc.sum()) / n_fr, 4),
            "empties_filled": int(len(set(d.s1[acc]) - set(d.s1[acc0])))}
    res["in_p6_bands"] = bool(0.045 <= res["empty"] <= 0.075 and res["mps"] >= 2.9)
    fw = m["w"].reindex(np.flatnonzero(m[variant] & ~acc0)).value_counts().head(25).to_dict()
    res["top_words_added"] = fw
    d.assign(p=q.astype(np.float32)).to_parquet(out, index=False)
    print(json.dumps(res, default=str), f"{time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)
    return res


def selftest():
    assert name_rel("dupont ecole", "dupont comite", False) == ("sub1", "comite")
    assert name_rel("dupont ecole", "dupont ecol", False) == ("typo", "")
    rest = frozenset("dupont comite".split()) - {"comite"}
    assert rest <= set("dupont college".split()) and not rest <= set("martin college".split())
    print("r05 self-check ok")


if __name__ == "__main__":
    from ledger import log
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "selftest":
        selftest()
    elif cmd == "ho":
        for c in args or ["US", "India"]:
            r = ho(c)
            ok = r["r05_d_h1"] >= -0.0001
            log("exp", probe=PROBE, subproblem="france-orgword", name=f"r05-holdout-{c}", world="holdout-dense (#8c, P23 dump) t 0.71",
                metrics=r, config={"s_max": S_MAX, "support": SUPPORT, "floor": FLOOR, "boost": 0.99, "s_on": "P3 blocked list"},
                hypothesis="R05 flips no in-domain true pairs to false (gate: dF >= -0.0001)",
                verdict="keep" if ok else "drop", runtime_s=r["runtime_s"], peak_gb=round(peak(), 2))
            print(f"{c}: peak {peak():.2f} GB", flush=True)
    elif cmd == "loco":
        for a in args or ["India>US", "US>India"]:
            src, tgt = a.split(">")
            r = loco(src, tgt)
            from loco import TS
            tp = TS[(src, tgt)][0]
            log("exp", probe=PROBE, subproblem="france-orgword", name=f"r05-loco-{src}>{tgt}",
                world="holdout-dense-fold0 LOCO (P6 S_holdout, M2 no CE)", metrics=r,
                config={"s_max": S_MAX, "support": SUPPORT, "floor": FLOOR, "boost": 0.99, "t": TS[(src, tgt)]},
                hypothesis="an unseen-country matcher under-accepts hn/street-equal org-word swaps with a low-sibling word",
                verdict="keep" if r[f"t{tp}_r05_d_h1"] >= 0.0005 else ("drop" if r[f"t{tp}_r05_d_h1"] < 0 else "inconclusive"),
                runtime_s=r["runtime_s"], peak_gb=round(peak(), 2))
            print(f"{a}: peak {peak():.2f} GB", flush=True)
    elif cmd == "fr":
        t0 = time.time()
        r = fr(*args)
        log("exp", probe=PROBE, subproblem="france-orgword", name=f"r05-France-{Path(args[2]).stem}", world="test-France",
            metrics=r, config={"s_max": S_MAX, "support": SUPPORT, "floor": FLOOR, "boost": 0.99}, hypothesis="R05 on #31",
            verdict="inconclusive", runtime_s=round(time.time() - t0), peak_gb=round(peak(), 2))
