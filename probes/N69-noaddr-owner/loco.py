"""N69 LOCO gate: on the France-like chain (US->India, India->US), add unaccepted argmax pairs whose SX is address-less,
core_x == core_1, and core_1 is unique among the target country's S1 (optionally lfc != 4, p floor). dF per half."""
import sys, numpy as np, pandas as pd
sys.path[:0] = ["probes/P28-addr-anchor", "probes/P64-legal-form-feat", "code/business_entity_resolution/src", "probes/N63-floor-probes"]
from loco import fbeta
import lf
from cells import capped_add
for s_, t_ in (("US", "India"), ("India", "US")):
    d = pd.read_parquet(f"data/work/n63/chain_{s_}_{t_}.parquet", columns=["s1","sx","p","y","src","acc","pos","noaddr","c1","cx","name_1","name_x","sib"])
    m = np.load(f"data/work/n63/meta_{s_}_{t_}.npz", allow_pickle=True); ntrue, hv = m["ntrue"], m["hv"]; n = len(ntrue)
    nc = pd.read_parquet(f"data/work/rf_train_s1_{t_}.parquet", columns=["core"]).core.value_counts()
    d["ncore"] = nc.reindex(d.c1).fillna(0).astype(int).to_numpy()
    y = d.y.to_numpy(); pos = d.pos.to_numpy(); acc = d.acc.to_numpy()
    owned = set(d.sx[acc])
    per = lambda pr: fbeta(np.bincount(pos[pr & y], minlength=n), np.bincount(pos[pr], minlength=n), ntrue)
    f0 = per(acc)
    base = ~acc & d.noaddr.to_numpy() & (d.c1 == d.cx).to_numpy() & (d.c1 != "").to_numpy() & ~d.sx.isin(owned).to_numpy()
    lfc = np.full(len(d), -1); lfc[base] = lf.pair_class(lf.sig(d.name_1[base]), lf.sig(d.name_x[base]))
    print(f"== {s_}>{t_}: chain F {f0.mean():.5f}; addr-less eq-core unaccepted {base.sum():,} (true {y[base].mean():.3f})")
    for tag, mk in (("uniq", base & (d.ncore == 1).to_numpy()), ("uniq lf!=4", base & (d.ncore == 1).to_numpy() & (lfc != 4)),
                    ("uniq lf!=4 p>=.3", base & (d.ncore == 1).to_numpy() & (lfc != 4) & (d.p >= .3).to_numpy()),
                    ("ncore2", base & (d.ncore == 2).to_numpy() & (lfc != 4)), ("any ncore lf!=4", base & (lfc != 4))):
        add = capped_add(d, mk)
        dF = per(acc | add) - f0
        se = lambda x: x.std() / np.sqrt(len(x))
        print(f"   {tag:18s}: cand {mk.sum():5d} added {add.sum():5d} prec {y[add].mean():.3f}  dF h0 {dF[hv==0].mean():+.6f}±{se(dF[hv==0]):.6f}  h1 {dF[hv==1].mean():+.6f}±{se(dF[hv==1]):.6f}")
