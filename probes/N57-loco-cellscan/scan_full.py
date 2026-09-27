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

results = {}
for hn_extra, tag in [(False, "gc70"), (True, "hn-extra")]:
    results[tag] = {}
    for src, tgt in (("US", "India"), ("India", "US")):
        d, acc, per, hv, y, t, pos = get_chain(src, tgt, hn_extra)
        results[tag][(src, tgt)] = (d, per, hv, fbeta)

# evaluate cells
cells_res = []

# To enumerate cells, we can group by rel, band, hrel, pband
def evaluate(tag, results_tag):
    d1, per1, hv1, _ = results_tag[("US", "India")]
    d2, per2, hv2, _ = results_tag[("India", "US")]
    
    cells = pd.concat([
        d1[['rel', 'band', 'hrel', 'pband']].drop_duplicates(),
        d2[['rel', 'band', 'hrel', 'pband']].drop_duplicates()
    ]).drop_duplicates().to_numpy()
    
    f0_1 = per1(d1["acc"].to_numpy())
    f0_2 = per2(d2["acc"].to_numpy())
    
    out = []
    
    for r, b, h, pb in cells:
        m1 = (d1.rel == r) & (d1.band == b) & (d1.hrel == h) & (d1.pband == pb)
        m2 = (d2.rel == r) & (d2.band == b) & (d2.hrel == h) & (d2.pband == pb)
        
        # Remove
        rm1 = m1 & d1.acc; nrm1 = rm1.sum()
        rm2 = m2 & d2.acc; nrm2 = rm2.sum()
        
        if nrm1 >= 50 and nrm2 >= 50:
            df1 = per1(d1.acc & ~rm1) - f0_1
            df2 = per2(d2.acc & ~rm2) - f0_2
            
            d1_0, d1_1 = df1[hv1==0], df1[hv1==1]
            d2_0, d2_1 = df2[hv2==0], df2[hv2==1]
            
            m1_0, s1_0 = d1_0.mean(), d1_0.std()/np.sqrt(len(d1_0))
            m1_1, s1_1 = d1_1.mean(), d1_1.std()/np.sqrt(len(d1_1))
            m2_0, s2_0 = d2_0.mean(), d2_0.std()/np.sqrt(len(d2_0))
            m2_1, s2_1 = d2_1.mean(), d2_1.std()/np.sqrt(len(d2_1))
            
            if m1_0 > 2*s1_0 and m1_1 > 2*s1_1 and m2_0 > 2*s2_0 and m2_1 > 2*s2_1:
                out.append({
                    "tag": tag, "action": "remove", "cell": (r, b, h, pb),
                    "n1": nrm1, "n2": nrm2,
                    "m1_0": m1_0, "s1_0": s1_0, "m1_1": m1_1, "s1_1": s1_1,
                    "m2_0": m2_0, "s2_0": s2_0, "m2_1": m2_1, "s2_1": s2_1
                })
        
        # Add
        add1 = m1 & d1.free & ~d1.acc
        nadd1_raw = add1.sum()
        
        if nadd1_raw > 0:
            sub = d1[add1].copy()
            sub["rn"] = sub.groupby(["s1", "src"]).cumcount()
            cap = ((sub.src == "s2") & (sub.rn < 5)) | ((sub.src == "s3") & (sub.rn < 6))
            add1_capped_idx = sub[cap].index
            add1 = np.zeros(len(d1), bool)
            add1[add1_capped_idx] = True
            
        add2 = m2 & d2.free & ~d2.acc
        nadd2_raw = add2.sum()
        if nadd2_raw > 0:
            sub = d2[add2].copy()
            sub["rn"] = sub.groupby(["s1", "src"]).cumcount()
            cap = ((sub.src == "s2") & (sub.rn < 5)) | ((sub.src == "s3") & (sub.rn < 6))
            add2_capped_idx = sub[cap].index
            add2 = np.zeros(len(d2), bool)
            add2[add2_capped_idx] = True
            
        nadd1 = add1.sum()
        nadd2 = add2.sum()
        
        if nadd1 >= 50 and nadd2 >= 50:
            df1 = per1(d1.acc | add1) - f0_1
            df2 = per2(d2.acc | add2) - f0_2
            
            d1_0, d1_1 = df1[hv1==0], df1[hv1==1]
            d2_0, d2_1 = df2[hv2==0], df2[hv2==1]
            
            m1_0, s1_0 = d1_0.mean(), d1_0.std()/np.sqrt(len(d1_0))
            m1_1, s1_1 = d1_1.mean(), d1_1.std()/np.sqrt(len(d1_1))
            m2_0, s2_0 = d2_0.mean(), d2_0.std()/np.sqrt(len(d2_0))
            m2_1, s2_1 = d2_1.mean(), d2_1.std()/np.sqrt(len(d2_1))
            
            if m1_0 > 2*s1_0 and m1_1 > 2*s1_1 and m2_0 > 2*s2_0 and m2_1 > 2*s2_1:
                out.append({
                    "tag": tag, "action": "add", "cell": (r, b, h, pb),
                    "n1": nadd1, "n2": nadd2,
                    "m1_0": m1_0, "s1_0": s1_0, "m1_1": m1_1, "s1_1": s1_1,
                    "m2_0": m2_0, "s2_0": s2_0, "m2_1": m2_1, "s2_1": s2_1
                })
    return out

all_out = []
for tag in ["gc70", "hn-extra"]:
    all_out.extend(evaluate(tag, results[tag]))

# Print results and match France analogue for survivors
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
            # For removal, we want France accepted pairs
            fr_count = (m & acc_fr).sum()
        else:
            # For add, we want France free argmax pairs
            # Since pairs_France is from P31 and they are all argmax pairs
            # unaccepted free argmax pairs
            # "acc" in P31 is accepted by base
            fr_free_argmax = m & ~acc_fr
            sub = pd.DataFrame({"s1": d_fr.s1[fr_free_argmax], "src": d_fr.src[fr_free_argmax]})
            sub["rn"] = sub.groupby(["s1", "src"]).cumcount()
            cap = ((sub.src == "s2") & (sub.rn < 5)) | ((sub.src == "s3") & (sub.rn < 6))
            fr_count = cap.sum()
            
        print(f"[{row['tag']}] {action} cell {r} {b} {h} {pb}: US>IN {row['n1']} IN>US {row['n2']} | FR count {fr_count}")
        # Expected LB
        avg_dF = (row['m1_0'] + row['m1_1'] + row['m2_0'] + row['m2_1']) / 4
        avg_loco = (row['n1'] + row['n2']) / 2
        exp_lb = (avg_dF * 0.15) * (fr_count / avg_loco) if avg_loco > 0 else 0
        print(f"  dF US>IN: h0 {row['m1_0']:.6f} h1 {row['m1_1']:.6f} | IN>US: h0 {row['m2_0']:.6f} h1 {row['m2_1']:.6f}")
        print(f"  Exp LB: {exp_lb:.6f}")
