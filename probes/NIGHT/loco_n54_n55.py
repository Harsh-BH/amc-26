# LOCO analogue for N54 (remove decoy-word same-hn swaps) and N55 (add same-address noise-word variants).
# Frame: P6 dense holdout scored by the OTHER country's booster (data/work/p6/S_holdout_<src>.parquet), France-like
# base = argmax & p >= t_profile, plus a simulated P42 hn-extra (free argmax, hn equal, non-sibling, p >= 0.8, S1 non-empty).
import sys, time; from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa
R = Path(".").resolve()
sys.path[:0] = [str(R / "probes/N55-fr-noiseword-fill"), str(R / "probes/P46-noise-word-tier"), str(R / "probes/P28-addr-anchor")]
import n55, p46, r05 as RR
import features as F
from ho import half
from loco import CI, OFF1, OFF2, TS, fbeta
W = R / "data/work"; SIB3 = {3, 4, 5, 7, 9, 11, 13, 21}
H = n55._helpers()
def run(src, tgt):
    t0, ci, t = time.time(), CI[tgt], TS[(src, tgt)][0]
    S = pd.read_parquet(W / f"p6/S_holdout_{src}.parquet", columns=["s1", "sx", "country", "p", "is_true", "am_all"])
    S = S[S.country.to_numpy() == ci].reset_index(drop=True)
    id1, idx = F.ids("train", tgt)
    d = pd.DataFrame({"s1": id1.take(pa.array(S.s1.to_numpy() - ci * OFF1)).to_numpy(zero_copy_only=False),
                      "sx": idx.take(pa.array(S.sx.to_numpy() - ci * OFF2)).to_numpy(zero_copy_only=False), "p": S.p.to_numpy(np.float64)})
    y, am = S.is_true.to_numpy(), S.am_all.to_numpy(); del S
    fo = pd.read_parquet(W / "s1_folds.parquet"); fo = fo[(fo.fold == 0) & fo.entity_id.isin(set(id1.to_numpy(zero_copy_only=False)))].entity_id.to_numpy()
    ntrue = pd.read_parquet(W / "gt_pairs.parquet", columns=["s1"]).s1.value_counts().reindex(fo).fillna(0).to_numpy()
    pos, hv = pd.Index(fo).get_indexer(d.s1), half(fo)
    per = lambda pred: fbeta(np.bincount(pos[pred & y], minlength=len(fo)), np.bincount(pos[pred], minlength=len(fo)), ntrue)
    p = d.p.to_numpy()
    base = am & (p >= t)
    # hn relation for argmax rows
    r1 = RR._rf(W / f"rf_train_s1_{tgt}.parquet", pd.unique(d.s1[am]), ["hn", "core"]); rx = RR._rf(W / f"rf_train_sx_{tgt}.parquet", pd.unique(d.sx[am]), ["hn", "core"])
    h1 = r1.hn.reindex(d.s1).fillna("").to_numpy(); hx = rx.hn.reindex(d.sx).fillna("").to_numpy()
    both = (h1 != "") & (hx != "")
    dl = np.full(len(d), np.nan); dl[both] = pd.to_numeric(pd.Series(hx[both]), errors="coerce").to_numpy(float) - pd.to_numeric(pd.Series(h1[both]), errors="coerce").to_numpy(float)
    hn_eq = (h1 != "") & (h1 == hx); sib = np.isin(dl, list(SIB3)) | np.isin(-dl, list(SIB3))
    ne = np.zeros(len(fo), bool); ne[pos[base]] = True
    owned = set(d.sx.to_numpy()[base])
    free = ~pd.Series(d.sx).isin(owned).to_numpy()
    hnx = am & ~base & free & hn_eq & ~sib & (p >= 0.8) & ne[pos]
    ext = base | hnx
    f0, fe = per(base), per(ext)
    out = {"t": t, "base": round(float(f0.mean()), 5), "hnx_added": int(hnx.sum()), "hnx_prec": round(float(y[hnx].mean()), 3) if hnx.any() else None,
           "hnx_dF": round(float((fe - f0).mean()), 6)}
    # lens on the target's argmax pairs (label-free)
    lt = p46.lens_table(d.loc[am, ["s1", "sx"]].reset_index(drop=True), "train", tgt)
    import os; SM = float(os.environ.get("SMIN", "0.2")); D = set(lt[(lt.n >= 20) & (lt.s > SM)].index); NZ = set(lt[(lt.n >= 20) & (lt.s <= 0.05) & (lt.e >= 0.8)].index)
    c1 = r1.core.reindex(d.s1).fillna("").to_numpy(); cx = rx.core.reindex(d.sx).fillna("").to_numpy()
    ia = np.flatnonzero(am & hn_eq)
    g = [[w for w in b.split() if w not in set(a.split())] for a, b in zip(c1[ia], cx[ia])]
    dr = [[w for w in a.split() if w not in set(b.split())] for a, b in zip(c1[ia], cx[ia])]
    sub1 = np.array([len(x) == 1 and len(z) == 1 for x, z in zip(g, dr)])
    wD = np.array([len(x) == 1 and x[0] in D for x in g])
    n54 = np.zeros(len(d), bool); n54[ia[sub1 & wD]] = True
    def rep(name, pred):
        dF = per(pred) - fe; flip = pred ^ ext
        r = {f"d_h{q}": round(float(dF[hv == q].mean()), 6) for q in (0, 1)} | {f"se_h{q}": round(float(dF[hv == q].std() / np.sqrt((hv == q).sum())), 6) for q in (0, 1)}
        r |= {"flips": int(flip.sum()), "flip_true": round(float(y[flip].mean()), 3) if flip.any() else None}
        print(f"{src}>{tgt} {name}: {r}", flush=True); out[name] = r
    rep("N54_ruled", ext & ~(n54 & (p < t)))
    rep("N54_all", ext & ~n54)
    sub = d.iloc[np.flatnonzero(am & free & ~ext)].reset_index()
    m, w = n55.cell_mask(sub[["s1", "sx"]], "train", tgt, NZ, H)
    n55m = np.zeros(len(d), bool); n55m[sub["index"].to_numpy()[m]] = True
    rep("N55_add", ext | n55m)
    rep("N54r+N55", (ext & ~(n54 & (p < t))) | n55m)
    out["D_words"], out["NZ_words"] = len(D), len(NZ); out["runtime_s"] = round(time.time() - t0)
    print(out, flush=True)
for src, tgt in (("US", "India"), ("India", "US")):
    run(src, tgt)
