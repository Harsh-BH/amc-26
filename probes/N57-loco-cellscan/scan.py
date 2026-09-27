import sys, time, pickle; from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa
R = Path(".").resolve()
sys.path[:0] = [str(R / "probes/N46-noise-word-tier"), str(R / "probes/P46-noise-word-tier"), str(R / "probes/P28-addr-anchor"), str(R / "probes/P20-france-patterns")]
import p46, r05 as RR, features as F
from patterns import name_rel
from ho import half
from loco import CI, OFF1, OFF2, TS, fbeta

W = R / "data/work"; SIB3 = {3, 4, 5, 7, 9, 11, 13, 21}

def get_chain(src, tgt, hn_extra=True):
    ci, t = CI[tgt], TS[(src, tgt)][0]
    S = pd.read_parquet(W / f"p6/S_holdout_{src}.parquet", columns=["s1", "sx", "country", "p", "is_true", "am_all"])
    S = S[S.country.to_numpy() == ci]
    id1, idx = F.ids("train", tgt)
    d = pd.DataFrame({"s1": id1.take(pa.array(S.s1.to_numpy() - ci * OFF1)).to_numpy(zero_copy_only=False), 
                      "sx": idx.take(pa.array(S.sx.to_numpy() - ci * OFF2)).to_numpy(zero_copy_only=False),
                      "p": S.p.to_numpy(), "y": S.is_true.to_numpy(), "am": S.am_all.to_numpy()}); del S
    d = d[d.am].reset_index(drop=True); d["src"] = d.sx.str[:2]
    
    fo = pd.read_parquet(W / "s1_folds.parquet")
    fo = fo[(fo.fold == 0) & fo.entity_id.isin(set(id1.to_numpy(zero_copy_only=False)))].entity_id.to_numpy()
    ntrue = pd.read_parquet(W / "gt_pairs.parquet", columns=["s1"]).s1.value_counts().reindex(fo).fillna(0).to_numpy()
    pos, hv = pd.Index(fo).get_indexer(d.s1), half(fo)
    y = d.y.to_numpy()
    per = lambda pred: fbeta(np.bincount(pos[pred & y], minlength=len(fo)), np.bincount(pos[pred], minlength=len(fo)), ntrue)
    
    r1 = RR._rf(W / f"rf_train_s1_{tgt}.parquet", pd.unique(d.s1), ["hn", "core"])
    rx = RR._rf(W / f"rf_train_sx_{tgt}.parquet", pd.unique(d.sx), ["hn", "core", "is_web"])
    h1, hx = r1.hn.reindex(d.s1).fillna("").to_numpy(), rx.hn.reindex(d.sx).fillna("").to_numpy()
    both = (h1 != "") & (hx != ""); dl = np.full(len(d), np.nan)
    dl[both] = pd.to_numeric(pd.Series(hx[both]), errors="coerce").to_numpy(float) - pd.to_numeric(pd.Series(h1[both]), errors="coerce").to_numpy(float)
    sib = np.isin(dl, list(SIB3)) | np.isin(-dl, list(SIB3)); hn_eq = (h1 != "") & (h1 == hx)
    c1, cx = r1.core.reindex(d.s1).fillna("").to_numpy(), rx.core.reindex(d.sx).fillna("").to_numpy()
    wx = rx.is_web.reindex(d.sx).fillna(False).to_numpy()
    
    g = [[w for w in b.split() if w not in set(a.split())] for a, b in zip(c1, cx)]
    dr = [[w for w in a.split() if w not in set(b.split())] for a, b in zip(c1, cx)]
    d["gain"] = [x[0] if len(x) == 1 else "" for x in g]
    lt = p46.lens_table(d[["s1", "sx"]], "train", tgt)
    D = set(lt[(lt.n >= 20) & (lt.s > 0.25)].index)
    one = np.array([len(x) == 1 and len(z) <= 1 for x, z in zip(g, dr)])
    n54 = one & d.gain.isin(D).to_numpy()
    
    p = d.p.to_numpy()
    if hn_extra:
        acc = (p >= t) & ~n54
        grp = set(d.s1[acc])
        acc |= ~acc & ~n54 & hn_eq & ~sib & (p >= 0.8) & d.s1.isin(grp).to_numpy()
        gs = d[acc].groupby("s1").src.agg(set)
        miss = np.array([(s not in gs.get(a, set())) and len(gs.get(a, set())) == 1 for a, s in zip(d.s1, d.src)])
        acc |= ~acc & ~n54 & ~sib & (p >= 0.7) & miss
        filled = set(d.s1[acc]); rc = ~acc & ~n54 & ~sib & (p >= 0.9) & ~d.s1.isin(filled).to_numpy()
        bi = d[rc].sort_values("p", ascending=False).drop_duplicates("s1").index; acc[bi] = True
    else:
        acc = (p >= t) & ~n54
        gs = d[acc].groupby("s1").src.agg(set)
        miss = np.array([(s not in gs.get(a, set())) and len(gs.get(a, set())) == 1 for a, s in zip(d.s1, d.src)])
        acc |= ~acc & ~n54 & ~sib & (p >= 0.7) & miss
        filled = set(d.s1[acc]); rc = ~acc & ~n54 & ~sib & (p >= 0.9) & ~d.s1.isin(filled).to_numpy()
        bi = d[rc].sort_values("p", ascending=False).drop_duplicates("s1").index; acc[bi] = True

    d["rel"] = [name_rel(a, b, w)[0] for a, b, w in zip(c1, cx, wx)]
    nn = lt.n.reindex(d.gain).fillna(0).to_numpy(); ss = lt.s.reindex(d.gain).to_numpy()
    d["band"] = np.select([d.gain == "", nn < 20, ss <= 0.05, ss <= 0.25], ["none", "rare", "s<=.05", ".05-.25"], ">.25")
    d["hrel"] = np.where(hn_eq, "hn_eq", np.where(both, "hn_diff", "hn_missing"))
    
    conds = [p < 0.3, (p >= 0.3) & (p < 0.7), (p >= 0.7) & (p < 0.9), (p >= 0.9) & (p < t), p >= t]
    choices = ["<0.3", "0.3-0.7", "0.7-0.9", "0.9-t", ">=t"]
    d["pband"] = np.select(conds, choices, default="other")
    
    # Cap calculation logic for adds
    ne = np.zeros(len(fo), bool); ne[pos[acc]] = True
    owned = set(d.sx.to_numpy()[acc])
    free = ~d.sx.isin(owned).to_numpy()
    
    return d, acc, per, hv, y, t, pos, free, ne

if __name__ == "__main__":
    for hn_extra, tag in [(False, "gc70"), (True, "hn-extra")]:
        res = {}
        for src, tgt in (("US", "India"), ("India", "US")):
            print(f"Running {src}>{tgt} {tag}...")
            d, acc, per, hv, y, t, pos, free, ne = get_chain(src, tgt, hn_extra)
            f0 = per(acc)
            d["acc"] = acc
            d["free"] = free
            d["y"] = y
            res[(src, tgt)] = {"d": d, "per": per, "hv": hv, "f0": f0, "pos": pos, "acc": acc, "free": free, "y": y}
        
        with open(f"probes/N57-loco-cellscan/scan_res_{tag}.pkl", "wb") as f:
            # We don't pickle 'per' to avoid issues, we will compute stats here.
            pass
