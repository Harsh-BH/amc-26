# LOCO: France-like chain (t_profile + hn-extra 0.8 + gc 0.7 + rescue 0.9), N54 (any hn) applied; then test candidate
# REMOVAL cells (name relation x gained-word band x hn relation) — ΔF0.5 per half, both directions.
import sys; from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa
R = Path(".").resolve()
sys.path[:0] = [str(R / "probes/P46-noise-word-tier"), str(R / "probes/P28-addr-anchor"), str(R / "probes/P20-france-patterns")]
import p46, r05 as RR, features as F
from patterns import name_rel
from ho import half
from loco import CI, OFF1, OFF2, TS, fbeta
W = R / "data/work"; SIB3 = {3, 4, 5, 7, 9, 11, 13, 21}
def chain(src, tgt):
    ci, t = CI[tgt], TS[(src, tgt)][0]
    S = pd.read_parquet(W / f"p6/S_holdout_{src}.parquet", columns=["s1", "sx", "country", "p", "is_true", "am_all"]); S = S[S.country.to_numpy() == ci]
    id1, idx = F.ids("train", tgt)
    d = pd.DataFrame({"s1": id1.take(pa.array(S.s1.to_numpy() - ci * OFF1)).to_numpy(zero_copy_only=False), "sx": idx.take(pa.array(S.sx.to_numpy() - ci * OFF2)).to_numpy(zero_copy_only=False),
                      "p": S.p.to_numpy(), "y": S.is_true.to_numpy(), "am": S.am_all.to_numpy()}); del S
    d = d[d.am].reset_index(drop=True); d["src"] = d.sx.str[:2]
    fo = pd.read_parquet(W / "s1_folds.parquet"); fo = fo[(fo.fold == 0) & fo.entity_id.isin(set(id1.to_numpy(zero_copy_only=False)))].entity_id.to_numpy()
    ntrue = pd.read_parquet(W / "gt_pairs.parquet", columns=["s1"]).s1.value_counts().reindex(fo).fillna(0).to_numpy()
    pos, hv = pd.Index(fo).get_indexer(d.s1), half(fo)
    y = d.y.to_numpy()
    per = lambda pred: fbeta(np.bincount(pos[pred & y], minlength=len(fo)), np.bincount(pos[pred], minlength=len(fo)), ntrue)
    r1 = RR._rf(W / f"rf_train_s1_{tgt}.parquet", pd.unique(d.s1), ["hn", "core"]); rx = RR._rf(W / f"rf_train_sx_{tgt}.parquet", pd.unique(d.sx), ["hn", "core", "is_web"])
    h1, hx = r1.hn.reindex(d.s1).fillna("").to_numpy(), rx.hn.reindex(d.sx).fillna("").to_numpy()
    both = (h1 != "") & (hx != ""); dl = np.full(len(d), np.nan)
    dl[both] = pd.to_numeric(pd.Series(hx[both]), errors="coerce").to_numpy(float) - pd.to_numeric(pd.Series(h1[both]), errors="coerce").to_numpy(float)
    sib = np.isin(dl, list(SIB3)) | np.isin(-dl, list(SIB3)); hn_eq = (h1 != "") & (h1 == hx)
    c1, cx = r1.core.reindex(d.s1).fillna("").to_numpy(), rx.core.reindex(d.sx).fillna("").to_numpy(); wx = rx.is_web.reindex(d.sx).fillna(False).to_numpy()
    g = [[w for w in b.split() if w not in set(a.split())] for a, b in zip(c1, cx)]; dr = [[w for w in a.split() if w not in set(b.split())] for a, b in zip(c1, cx)]
    d["gain"] = [x[0] if len(x) == 1 else "" for x in g]
    lt = p46.lens_table(d[["s1", "sx"]], "train", tgt)
    D = set(lt[(lt.n >= 20) & (lt.s > 0.25)].index)
    one = np.array([len(x) == 1 and len(z) <= 1 for x, z in zip(g, dr)])
    n54 = one & d.gain.isin(D).to_numpy()
    p = d.p.to_numpy(); acc = (p >= t) & ~n54
    grp = set(d.s1[acc])
    acc |= ~acc & ~n54 & hn_eq & ~sib & (p >= 0.8) & d.s1.isin(grp).to_numpy()
    gs = d[acc].groupby("s1").src.agg(set)
    miss = np.array([(s not in gs.get(a, set())) and len(gs.get(a, set())) == 1 for a, s in zip(d.s1, d.src)])
    acc |= ~acc & ~n54 & ~sib & (p >= 0.7) & miss
    filled = set(d.s1[acc]); rc = ~acc & ~n54 & ~sib & (p >= 0.9) & ~d.s1.isin(filled).to_numpy()
    bi = d[rc].sort_values("p", ascending=False).drop_duplicates("s1").index; acc[bi] = True
    d["rel"] = [name_rel(a, b, w)[0] for a, b, w in zip(c1, cx, wx)]
    nn = lt.n.reindex(d.gain).fillna(0).to_numpy(); ss = lt.s.reindex(d.gain).to_numpy()
    d["band"] = np.select([d.gain == "", nn < 20, ss <= 0.05, ss <= 0.25], ["none", "rare", "s<=.05", ".05-.25"], ">.25")
    d["hrel"] = np.where(hn_eq, "hn_eq", np.where(both, "hn_diff", "hn_missing"))
    return d, acc, per, hv, y, t
cells = [("add1", "rare", None), ("sub1", "rare", None), ("add1", "rare", "hn_missing"), ("sub1", "rare", "hn_missing"),
         ("add1", ".05-.25", None), ("sub1", ".05-.25", None), ("typo", "rare", "hn_missing"), ("eq", "none", "hn_diff")]
for src, tgt in (("US", "India"), ("India", "US")):
    d, acc, per, hv, y, t = chain(src, tgt); f0 = per(acc)
    print(f"== {src}>{tgt} chain F {f0.mean():.5f}, accepted {acc.sum()}")
    for rel, band, hrel in cells:
        m = (d.rel == rel).to_numpy() & (d.band == band).to_numpy() & ((d.hrel == hrel).to_numpy() if hrel else True)
        for tag, extra in (("all", True), ("p<t", d.p.to_numpy() < t)):
            rm = acc & m & extra
            if rm.sum() == 0: continue
            dF = per(acc & ~rm) - f0
            print(f"  remove {rel:5s} {band:8s} {str(hrel):10s} {tag:4s}: n {rm.sum():5d} true {y[rm].mean():.3f}  dF h0 {dF[hv == 0].mean():+.6f} h1 {dF[hv == 1].mean():+.6f}")
