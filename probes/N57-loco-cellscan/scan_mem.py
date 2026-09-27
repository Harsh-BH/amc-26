import sys, time; from pathlib import Path
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
    
    owned = set(d.sx[acc])
    d["free"] = ~d.sx.isin(owned).to_numpy()
    d["acc"] = acc
    d["y"] = y
    return d, acc, per, hv, y, t, pos

def eval_cells(d, acc, per, hv, y, t, pos):
    f0 = per(acc)
    # create combinations
    cols = ['rel', 'band', 'hrel', 'pband']
    cells = d[cols].drop_duplicates().to_numpy()
    
    res = []
    for r, b, h, pb in cells:
        m = (d.rel == r) & (d.band == b) & (d.hrel == h) & (d.pband == pb)
        # Remove
        rm = m & acc
        nrm = rm.sum()
        dF_rm = None
        if nrm >= 50:
            df = per(acc & ~rm) - f0
            d0, d1 = df[hv==0], df[hv==1]
            m0, s0 = d0.mean(), d0.std()/np.sqrt(len(d0))
            m1, s1 = d1.mean(), d1.std()/np.sqrt(len(d1))
            dF_rm = (nrm, m0, s0, m1, s1)
            
        # Add
        add = m & d.free & ~acc
        nadd = add.sum()
        dF_add = None
        if nadd >= 50:
            sub = d[add].copy()
            sub["rn"] = sub.groupby(["s1", "src"]).cumcount()
            cap = ((sub.src == "s2") & (sub.rn < 5)) | ((sub.src == "s3") & (sub.rn < 6))
            add_capped_idx = sub[cap].index
            add_mask = np.zeros(len(d), bool)
            add_mask[add_capped_idx] = True
            
            nadd_capped = add_mask.sum()
            if nadd_capped >= 50:
                df = per(acc | add_mask) - f0
                d0, d1 = df[hv==0], df[hv==1]
                m0, s0 = d0.mean(), d0.std()/np.sqrt(len(d0))
                m1, s1 = d1.mean(), d1.std()/np.sqrt(len(d1))
                dF_add = (nadd_capped, m0, s0, m1, s1)
                
        res.append(((r, b, h, pb), dF_rm, dF_add))
    return res

all_out = []
for hn_extra, tag in [(False, "gc70"), (True, "hn-extra")]:
    print(f"Starting tag: {tag}")
    res1, res2 = None, None
    for src, tgt in (("US", "India"), ("India", "US")):
        d, acc, per, hv, y, t, pos = get_chain(src, tgt, hn_extra)
        cres = eval_cells(d, acc, per, hv, y, t, pos)
        if src == "US": res1 = {k: (rm, add) for k, rm, add in cres}
        else: res2 = {k: (rm, add) for k, rm, add in cres}
        del d, acc, per, hv, y, pos # free mem
        import gc; gc.collect()
        
    # intersect cells
    cells = set(res1.keys()) | set(res2.keys())
    for c in cells:
        r1 = res1.get(c, (None, None))
        r2 = res2.get(c, (None, None))
        
        # Remove
        if r1[0] is not None and r2[0] is not None:
            n1, m1_0, s1_0, m1_1, s1_1 = r1[0]
            n2, m2_0, s2_0, m2_1, s2_1 = r2[0]
            if m1_0 > 2*s1_0 and m1_1 > 2*s1_1 and m2_0 > 2*s2_0 and m2_1 > 2*s2_1:
                all_out.append({
                    "tag": tag, "action": "remove", "cell": c,
                    "n1": n1, "n2": n2,
                    "m1_0": m1_0, "s1_0": s1_0, "m1_1": m1_1, "s1_1": s1_1,
                    "m2_0": m2_0, "s2_0": s2_0, "m2_1": m2_1, "s2_1": s2_1
                })
        
        # Add
        if r1[1] is not None and r2[1] is not None:
            n1, m1_0, s1_0, m1_1, s1_1 = r1[1]
            n2, m2_0, s2_0, m2_1, s2_1 = r2[1]
            if m1_0 > 2*s1_0 and m1_1 > 2*s1_1 and m2_0 > 2*s2_0 and m2_1 > 2*s2_1:
                all_out.append({
                    "tag": tag, "action": "add", "cell": c,
                    "n1": n1, "n2": n2,
                    "m1_0": m1_0, "s1_0": s1_0, "m1_1": m1_1, "s1_1": s1_1,
                    "m2_0": m2_0, "s2_0": s2_0, "m2_1": m2_1, "s2_1": s2_1
                })

print(f"Total survivor rows: {len(all_out)}")

if all_out:
    print("Loading France pairs...")
    d_fr = pd.read_parquet(W / "p31/pairs_France.parquet", columns=["s1", "sx", "p", "argmax", "acc", "name_rel", "core_1", "core_x", "hn_1", "hn_x", "is_web", "src"])
    
    n54_cells = set()
    if (W / "p54/cells.parquet").exists():
        n54_df = pd.read_parquet(W / "p54/cells.parquet")
        n54_cells = set(zip(n54_df.s1, n54_df.sx))
    
    d_fr = d_fr[d_fr.argmax].reset_index(drop=True)
    c1, cx = d_fr.core_1.fillna("").to_numpy(), d_fr.core_x.fillna("").to_numpy()
    wx = d_fr.is_web.fillna(False).to_numpy()
    h1, hx = d_fr.hn_1.fillna("").to_numpy(), d_fr.hn_x.fillna("").to_numpy()
    
    both = (h1 != "") & (hx != "")
    hn_eq = (h1 != "") & (h1 == hx)
    hrel = np.where(hn_eq, "hn_eq", np.where(both, "hn_diff", "hn_missing"))
    
    rel = np.array([name_rel(a, b, w)[0] for a, b, w in zip(c1, cx, wx)])
    g = [[w for w in b.split() if w not in set(a.split())] for a, b in zip(c1, cx)]
    gain = np.array([x[0] if len(x) == 1 else "" for x in g])
    
    lt = p46.lens_table(d_fr[["s1", "sx"]], "test", "France")
    nn = lt.n.reindex(gain).fillna(0).to_numpy()
    ss = lt.s.reindex(gain).to_numpy()
    band = np.select([gain == "", nn < 20, ss <= 0.05, ss <= 0.25], ["none", "rare", "s<=.05", ".05-.25"], ">.25")
    
    p = d_fr.p.to_numpy()
    t_fr = 0.9868
    conds = [p < 0.3, (p >= 0.3) & (p < 0.7), (p >= 0.7) & (p < 0.9), (p >= 0.9) & (p < t_fr), p >= t_fr]
    choices = ["<0.3", "0.3-0.7", "0.7-0.9", "0.9-t", ">=t"]
    pband = np.select(conds, choices, default="other")
    
    is_n54 = np.array([(a,b) in n54_cells for a,b in zip(d_fr.s1, d_fr.sx)])
    acc_fr = d_fr.acc.to_numpy()
    
    print("Evaluating survivors...")
    for row in all_out:
        r, b, h, pb = row["cell"]
        action = row["action"]
        m = (rel == r) & (band == b) & (hrel == h) & (pband == pb) & ~is_n54
        if action == "remove":
            fr_count = (m & acc_fr).sum()
        else:
            fr_free_argmax = m & ~acc_fr
            sub = pd.DataFrame({"s1": d_fr.s1[fr_free_argmax], "src": d_fr.src[fr_free_argmax]})
            sub["rn"] = sub.groupby(["s1", "src"]).cumcount()
            cap = ((sub.src == "s2") & (sub.rn < 5)) | ((sub.src == "s3") & (sub.rn < 6))
            fr_count = cap.sum()
            
        print(f"[{row['tag']}] {action} cell {r} {b} {h} {pb}: US>IN {row['n1']} IN>US {row['n2']} | FR count {fr_count}")
        avg_dF = (row['m1_0'] + row['m1_1'] + row['m2_0'] + row['m2_1']) / 4
        avg_loco = (row['n1'] + row['n2']) / 2
        exp_lb = (avg_dF * 0.15) * (fr_count / avg_loco) if avg_loco > 0 else 0
        print(f"  dF US>IN: h0 {row['m1_0']:.6f} h1 {row['m1_1']:.6f} | IN>US: h0 {row['m2_0']:.6f} h1 {row['m2_1']:.6f}")
        print(f"  Exp LB: {exp_lb:.6f}")
