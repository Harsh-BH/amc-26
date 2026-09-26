"""N55: France same-address variants that swap/add one TRUE noise word (fils / cie / associes / groupe / developpement).

P46 accepted only 'developpement' (its R05 guards exclude R1 fillers); P28's same-address rule has a sibling-word guard
on the dropped org word; hn-extra needs p >= 0.8. The rest stay rejected at p 0.02-0.8.
    .venv/bin/python probes/N55-fr-noiseword-fill/n55.py analyse      # holdout analogue + France counts -> data/work/n55/
    .venv/bin/python probes/N55-fr-noiseword-fill/n55.py build <base> <out> [strict|all]
    .venv/bin/python probes/N55-fr-noiseword-fill/n55.py selftest
"""
import sys, os, csv, tempfile
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/N53-fr-drop1-sameaddr"), str(ROOT / "probes/P46-noise-word-tier")]
W = ROOT / "data/work"; OUT = W / "n55"
STRICT = {"fils", "cie", "associes"}            # France lens s 0.020 / 0.044 / 0.020, e 0.96 / 0.92 / 0.96
ALL = STRICT | {"groupe", "developpement"}       # groupe s 0.089 e 0.88; developpement s 0.015 (P46 already adds most)
CAP = {"S2": 5, "S3": 6}

def ident(d, split, c):
    """identical address (hn + street key + city) mask for pairs d (s1, sx) — reuse N53's helpers."""
    import p53_master as M                                 # noqa  (module runs its analysis on import: avoid)
def _helpers():
    src = (ROOT / "probes/N53-fr-drop1-sameaddr/p53_master.py").read_text().split("out = {}")[0]
    g = {"__file__": str(ROOT / "probes/N53-fr-drop1-sameaddr/p53_master.py")}
    os.chdir(ROOT); exec(src, g); return g

def gained(c1, cx):
    return [[t for t in b.split() if t not in set(a.split())] for a, b in zip(c1, cx)]

def cell_mask(d, split, c, words, H):
    """argmax-agnostic: kind in sub1/add1 (1 gained token, in words), hn equal, street key equal, city equal, not web."""
    r1 = H["rf"](W / f"rf_{split}_s1_{c}.parquet", pd.unique(d.s1), ["core", "hn", "places"])
    rx = H["rf"](W / f"rf_{split}_sx_{c}.parquet", pd.unique(d.sx), ["core", "hn", "places", "is_web"])
    c1, cx = r1.core.reindex(d.s1).fillna("").to_numpy(), rx.core.reindex(d.sx).fillna("").to_numpy()
    h1, hx = r1.hn.reindex(d.s1).fillna("").to_numpy(), rx.hn.reindex(d.sx).fillna("").to_numpy()
    wx = rx.is_web.reindex(d.sx).fillna(False).to_numpy().astype(bool)
    g = gained(c1, cx)
    shared = np.array([bool(set(a.split()) & set(b.split())) for a, b in zip(c1, cx)])
    pre = (h1 != "") & (h1 == hx) & ~wx & shared & np.array([len(x) == 1 and x[0] in words for x in g])
    idx = np.flatnonzero(pre)
    p1 = r1.places.reindex(d.s1.to_numpy()[idx]).fillna("").to_numpy(); px = rx.places.reindex(d.sx.to_numpy()[idx]).fillna("").to_numpy()
    idx = idx[np.array([a != "" and bool(set(a.split("|")) & set(b.split("|"))) for a, b in zip(p1, px)], bool)]
    s1i, sxi = d.s1.to_numpy()[idx], d.sx.to_numpy()[idx]
    a1, ax = H["addr"](split, pd.unique(s1i), True).reindex(s1i), H["addr"](split, pd.unique(sxi), False).reindex(sxi)
    key = H["french_street_key"] if c == "France" else H["street_key"]
    k1, kx = key(a1).to_numpy(), key(ax).to_numpy()
    idx = idx[(k1 == kx) & (k1 != "")]
    m = np.zeros(len(d), bool); m[idx] = True
    w = np.array([x[0] if len(x) == 1 else "" for x in g], object)
    return m, w

def analyse():
    import p46
    H = _helpers()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {}
    for c in ("US", "India"):
        d = pd.read_parquet(W / f"p23/pairs_{c}.parquet", columns=["s1", "sx", "holdout", "am", "y", "pred", "p"])
        d = d[d.holdout & d.am].reset_index(drop=True); n1 = pd.read_parquet(W / f"p23/pairs_{c}.parquet", columns=["s1", "holdout"]).query("holdout").s1.nunique()
        lt = p46.lens_table(d[["s1", "sx"]], "train", c)
        noise = set(lt[(lt.n >= 20) & (lt.s <= 0.05) & (lt.e >= 0.8)].index)          # the country's own noise words
        m, w = cell_mask(d, "train", c, noise, H)
        x = d[m]
        rows[c] = dict(noise_words=len(noise), n=len(x), per_s1=len(x) / n1, prec=x.y.mean(), neg_per_s1=(~x.y).sum() / n1,
                       rej=int((~x.pred).sum()), rej_prec=x[~x.pred].y.mean())
    res = pd.DataFrame(rows).T; print(res.round(5).to_string()); res.to_csv(OUT / "holdout.csv")
    fr = pd.read_parquet(W / "p31/pairs_France.parquet", columns=["s1", "sx", "p", "acc", "argmax", "hn_cls", "src"])
    m, w = cell_mask(fr, "test", "France", ALL, H)
    x = fr[m].assign(w=w[m]); x = x[x.argmax.astype(bool) & (x.hn_cls != 1)]
    x[["s1", "sx", "p", "w", "acc", "src"]].to_parquet(OUT / "cells.parquet", index=False)
    taken = set(fr.sx[fr.acc]); nonemp = set(fr.s1[fr.acc])
    r = x[~x.acc & ~x.sx.isin(taken)]
    r = r.assign(nonemp=r.s1.isin(nonemp))
    new = r[~((r.p >= 0.8) & r.nonemp) & (r.w != "developpement")]
    print("France cell", len(x), "accepted", int(x.acc.sum()), "| new after hn-extra & P46:", len(new), "fills empty:", int((~new.nonemp).sum()))
    print(new.w.value_counts().to_dict())
    for c in ("US", "India"):
        nb = rows[c]["neg_per_s1"] * 259452
        for v, ws in (("strict", STRICT), ("all", ALL)):
            k = new.w.isin(ws).sum(); print(f"  {c}-rate bound {v}: 1 - {nb:.0f}/{k} = {1 - nb / max(k, 1):.3f}")

def build(base, out, variant="strict", cell=None):
    base, out = Path(base), Path(out)
    c = cell if cell is not None else pd.read_parquet(OUT / "cells.parquet")
    c = c[c.w.isin(STRICT if variant == "strict" else ALL)].sort_values("p", ascending=False)
    rows, pred = [], {}
    with open(base / "matching_results.tsv", encoding="utf-8") as f:
        head = f.readline()
        for line in f:
            s1, _, ids = line.rstrip("\n").partition("\t"); pred[s1] = ids.split(",") if ids else []
    cand = {}
    with open(base / "candidate_pairs.tsv", encoding="utf-8") as f:
        f.readline()
        want = set(c.s1)
        for line in f:
            s1, _, ids = line.rstrip("\n").partition("\t")
            if s1 in want: cand[s1] = set(ids.split(",")) if ids else set()
    owned = {x for v in pred.values() for x in v}
    n = 0
    for s1, sx in zip(c.s1, c.sx):
        if sx in owned or s1 not in pred or sx not in cand.get(s1, ()): continue
        cur = pred[s1]
        if sum(x.startswith(sx[:2]) for x in cur) >= CAP[sx[:2]]: continue
        cur.append(sx); owned.add(sx); n += 1
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "matching_results.tsv", "w", encoding="utf-8") as g:
        g.write(head)
        for s1, v in pred.items(): g.write(f"{s1}\t{','.join(v)}\n")
    for name in ("candidate_pairs.tsv", "code.zip"):
        if (base / name).exists() and not (out / name).exists(): os.link(base / name, out / name)
    print(f"{variant}: added {n} France pairs -> {out}")
    return n

def selftest():
    with tempfile.TemporaryDirectory() as d:
        b = Path(d) / "b"; b.mkdir()
        (b / "matching_results.tsv").write_text("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\nS1-2\t\nS1-3\tS2-9\n")
        (b / "candidate_pairs.tsv").write_text("source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-1,S3-2\nS1-2\tS2-3\nS1-3\tS2-9,S3-5\n")
        cell = pd.DataFrame({"s1": ["S1-1", "S1-2", "S1-3", "S1-3"], "sx": ["S3-2", "S2-3", "S2-9", "S3-7"], "p": [.5, .4, .3, .2], "w": ["fils", "groupe", "cie", "cie"]})
        assert build(b, Path(d) / "o", "strict", cell) == 1      # S2-9 already owned, S3-7 not a candidate, groupe not strict
        assert build(b, Path(d) / "o2", "all", cell) == 2
    print("selftest OK")

if __name__ == "__main__":
    {"analyse": lambda: analyse(), "build": lambda: build(*sys.argv[2:5]), "selftest": lambda: selftest()}[sys.argv[1]]()
