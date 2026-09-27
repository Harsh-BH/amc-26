"""Systematic pattern scan (label-free cells) for adds and vetoes, both countries.
cell = name_rel kind x hn relation x P64 legal-form class x address-less SX x synthetic-brand SX x S1 core unique.
  scan.py loco     -> data/work/p70/scan_loco.parquet: per LOCO target, cell n, precision of accepted (ref) pairs, of free pairs
  scan.py fr BASE  -> data/work/p70/scan_fr_<BASE>.parquet: France argmax pairs per cell, accepted / free in BASE
  scan.py rank BASE -> add candidates (free precision >= .75 Wilson LB in both targets) and veto candidates (accepted precision
                       Wilson UB <= .5 in both targets), with France volume"""
import csv, gc, json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P70-fr-roles"), str(ROOT / "probes/P68-iu-retrieval")]
import p70  # noqa: E402  (sets paths: patterns, lf, gib, normalize)
W = ROOT / "data/work"
SIB = {3, 4, 5, 7, 9, 11, 13, 21}
KINDS = {v: k for k, v in p70.KIND.items()}


def hrel(h1, hx):
    ok = pd.Series(h1).str.fullmatch(r"\d+").fillna(False).to_numpy() & pd.Series(hx).str.fullmatch(r"\d+").fillna(False).to_numpy()
    d = np.full(len(h1), np.nan)
    d[ok] = pd.to_numeric(pd.Series(hx[ok])).to_numpy(float) - pd.to_numeric(pd.Series(h1[ok])).to_numpy(float)
    return np.select([ok & (d == 0), ok & np.isin(np.abs(d), list(SIB)), ok], ["eq", "sib", "other"], "miss")


def cell(kind, hr, lfk, noaddr, gib, uniq):
    return pd.Series([f"{k}|{h}|lf{l}|na{int(a)}|gib{int(g)}|u{int(u)}" for k, h, l, a, g, u in zip(kind, hr, lfk, noaddr, gib, uniq)])


def loco():
    p55, L = p70.p55, p70.L
    rows = []
    for dr in L.DIRS:
        tgt = L.DIRS[dr][1]
        fr = p55.loco_frame(dr)
        ids = pd.read_parquet(W / f"p48/loco_{dr}_ids.parquet", columns=["s1", "sx"])
        R = pd.read_parquet(W / f"p70/roles_{dr}.parquet", columns=["kind", "lf_k", "gib_x", "noaddr_x", "twins1"])
        h1 = pd.read_parquet(W / f"rf_train_s1_{tgt}.parquet", columns=["id", "hn"]).set_index("id").hn.reindex(ids.s1).fillna("").to_numpy()
        hx = pd.read_parquet(W / f"rf_train_sx_{tgt}.parquet", columns=["id", "hn"]).set_index("id").hn.reindex(ids.sx).fillna("").to_numpy()
        kind = pd.Series(R.kind.to_numpy()).map(KINDS).fillna("none").to_numpy()
        c = cell(kind, hrel(h1, hx), R.lf_k.fillna(-1).astype(int).to_numpy(), R.noaddr_x.to_numpy() > 0, R.gib_x.to_numpy() > 0, R.twins1.to_numpy() == 1)
        y, ref, free = fr["z"]["y"].astype(bool), fr["ref"], fr["free"] & ~fr["ref"]
        df = pd.DataFrame({"cell": c, "y": y, "acc": ref, "free": free})
        g = df.groupby("cell").agg(n=("y", "size"), acc_n=("acc", "sum"), free_n=("free", "sum"))
        g["acc_k"] = df[df.acc].groupby("cell").y.sum().reindex(g.index).fillna(0)
        g["free_k"] = df[df.free].groupby("cell").y.sum().reindex(g.index).fillna(0)
        g["tgt"] = tgt
        rows.append(g.reset_index())
        print(dr, "cells", len(g), f"peak {p70.peak():.2f} GB", flush=True)
        del fr, ids, R, df; gc.collect()
    pd.concat(rows).to_parquet(W / "p70/scan_loco.parquet", index=False)


def fr(base):
    import fr_key, lf, gib as G
    import normalize as N
    from patterns import name_rel
    d = pd.read_parquet(W / "p31/fr31v0.1_t9868.parquet", columns=["s1", "sx", "p"]).sort_values("p", ascending=False).drop_duplicates("sx").reset_index(drop=True)
    r1 = pd.read_parquet(W / "rf_test_s1_France.parquet", columns=["id", "core", "hn"]).set_index("id")
    rx = pd.read_parquet(W / "p29/w_fix/rf_test_sx_France.parquet", columns=["id", "core", "hn", "is_web"]).set_index("id")
    c1, cx = r1.core.reindex(d.s1).fillna("").to_numpy(), rx.core.reindex(d.sx).fillna("").to_numpy()
    wx = rx.is_web.reindex(d.sx).fillna(False).to_numpy()
    kind = np.array([name_rel(a, b, w)[0] if a and b else "none" for a, b, w in zip(c1, cx, wx)], object)
    hr = hrel(r1.hn.reindex(d.s1).fillna("").to_numpy(), rx.hn.reindex(d.sx).fillna("").to_numpy())
    raw1 = p70._raw("test", set(d.s1), [1], ["business_name", "country"]).business_name
    rawx = p70._raw("test", set(d.sx), [2, 3], ["business_name", "business_address", "country"])
    lfk = np.asarray(lf.pair_class(pd.Series(fr_key.sigs(raw1.reindex(d.s1))), pd.Series(fr_key.sigs(rawx.business_name.reindex(d.sx)))))
    M = set(json.loads((W / "p36/morph_test.json").read_text()))
    un, inv = np.unique(rawx.business_name.reindex(d.sx).fillna("").astype(str).to_numpy(), return_inverse=True)
    gibx = G.is_gib(un, M)[inv]
    ax = rawx.business_address.reindex(d.sx).fillna("")
    na = (N.fold(ax).str.replace(r"<\s*null\s*>|\bnull\b|n/a", " ", regex=True).str.replace(r"[^a-z0-9]", "", regex=True).str.len() == 0).to_numpy()
    uniq = pd.Series(c1).map(r1.core.value_counts()).fillna(0).to_numpy() == 1
    d["cell"] = cell(kind, hr, lfk, na, gibx, uniq).to_numpy()
    m = pd.read_csv(ROOT / "output" / base / "matching_results.tsv", sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
    m = m[m.source1_entity_id.isin(r1.index)]
    acc = {(s, x) for s, v in zip(m.source1_entity_id, m.matched_entity_ids) if v for x in v.split(",")}
    used = {x for _, x in acc}
    d["acc"] = [(a, b) in acc for a, b in zip(d.s1, d.sx)]
    d["free"] = ~d.sx.isin(used)
    d[["s1", "sx", "p", "cell", "acc", "free"]].to_parquet(W / f"p70/scan_fr_{base}.parquet", index=False)
    print("France frame", len(d), "cells", d.cell.nunique(), f"peak {p70.peak():.2f} GB", flush=True)


def wil(k, n):
    if n == 0: return (np.nan, np.nan)
    p, dd = k / n, 1 + 3.8416 / n
    c, h = (p + 1.92 / n) / dd, 1.96 * math.sqrt(p * (1 - p) / n + 0.9604 / n / n) / dd
    return (c - h, c + h)


def rank(base):
    lo = pd.read_parquet(W / "p70/scan_loco.parquet")
    f = pd.read_parquet(W / f"p70/scan_fr_{base}.parquet").groupby("cell").agg(fr_n=("sx", "size"), fr_acc=("acc", "sum"), fr_free=("free", "sum"))
    t = {}
    for tgt, g in lo.groupby("tgt"):
        g = g.set_index("cell")
        t[tgt] = g
    I, U = t["India"], t["US"]
    cells = f.index
    out = []
    for c in cells:
        r = {"cell": c, **f.loc[c].to_dict()}
        for nm, g in (("IN", I), ("US", U)):
            if c in g.index:
                x = g.loc[c]
                r[f"{nm}_free_n"], r[f"{nm}_free_p"] = int(x.free_n), (x.free_k / x.free_n if x.free_n else np.nan)
                r[f"{nm}_free_lb"] = wil(x.free_k, x.free_n)[0]
                r[f"{nm}_acc_n"], r[f"{nm}_acc_p"] = int(x.acc_n), (x.acc_k / x.acc_n if x.acc_n else np.nan)
                r[f"{nm}_acc_ub"] = wil(x.acc_k, x.acc_n)[1]
                r[f"{nm}_freeshare"] = x.free_n / x.n
        out.append(r)
    o = pd.DataFrame(out)
    o["fr_freeshare"] = o.fr_free / o.fr_n
    add = o[(o.IN_free_lb >= .75) & (o.US_free_lb >= .75) & (o.fr_free >= 100)].sort_values("fr_free", ascending=False)
    veto = o[(o.IN_acc_ub <= .5) & (o.US_acc_ub <= .5) & (o.fr_acc >= 50)].sort_values("fr_acc", ascending=False)
    pd.set_option("display.width", 250)
    cols_a = ["cell", "fr_free", "fr_acc", "fr_freeshare", "IN_free_n", "IN_free_p", "IN_free_lb", "US_free_n", "US_free_p", "US_free_lb", "IN_freeshare", "US_freeshare"]
    cols_v = ["cell", "fr_acc", "fr_free", "IN_acc_n", "IN_acc_p", "IN_acc_ub", "US_acc_n", "US_acc_p", "US_acc_ub"]
    print("ADD candidates (free-pair Wilson LB >= .75 in BOTH targets, France free >= 100):\n", add[cols_a].round(3).to_string(index=False))
    print("\nVETO candidates (accepted-pair Wilson UB <= .5 in BOTH targets, France accepted >= 50):\n", veto[cols_v].round(3).to_string(index=False))
    o.to_parquet(W / f"p70/scan_rank_{base}.parquet", index=False)


if __name__ == "__main__":
    {"loco": lambda: loco(), "fr": lambda: fr(sys.argv[2]), "rank": lambda: rank(sys.argv[2])}[sys.argv[1]]()
