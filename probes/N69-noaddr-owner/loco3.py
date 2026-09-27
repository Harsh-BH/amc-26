import sys, numpy as np, pandas as pd
sys.path[:0] = ["probes/P28-addr-anchor", "probes/P64-legal-form-feat", "code/business_entity_resolution/src", "probes/N63-floor-probes"]
from loco import fbeta
import lf
from cells import capped_add
for s_, t_ in (("US", "India"), ("India", "US")):
    d = pd.read_parquet(f"data/work/n63/chain_{s_}_{t_}.parquet", columns=["s1","sx","p","y","src","acc","pos","noaddr","c1","cx","name_1","name_x","rel"])
    m = np.load(f"data/work/n63/meta_{s_}_{t_}.npz", allow_pickle=True); ntrue, hv = m["ntrue"], m["hv"]; n = len(ntrue)
    nc = pd.read_parquet(f"data/work/rf_train_s1_{t_}.parquet", columns=["core"]).core.value_counts()
    u1 = nc.reindex(d.c1).fillna(0).to_numpy() == 1
    y = d.y.to_numpy(); pos = d.pos.to_numpy(); acc = d.acc.to_numpy(); owned = set(d.sx[acc])
    per = lambda pr: fbeta(np.bincount(pos[pr & y], minlength=n), np.bincount(pos[pr], minlength=n), ntrue)
    f0 = per(acc); se = lambda x: x.std() / np.sqrt(len(x))
    base = ~acc & d.noaddr.to_numpy() & ~d.sx.isin(owned).to_numpy() & u1
    print(f"== {s_}>{t_}")
    for rel in ("typo", "drop1", "sub1", "add1", "disjoint", "web"):
        for pf in (0.3, 0.6):
            mk = base & (d.rel == rel).to_numpy() & (d.p >= pf).to_numpy()
            if mk.sum() < 20: continue
            add = capped_add(d, mk); dF = per(acc | add) - f0
            print(f"   {rel:8s} p>={pf}: added {add.sum():5d} prec {y[add].mean():.3f}  dF h0 {dF[hv==0].mean():+.6f}±{se(dF[hv==0]):.6f} h1 {dF[hv==1].mean():+.6f}±{se(dF[hv==1]):.6f}")
