"""N70: same name, house number replaced (not sibling offset, not digit drop), same street key — LOCO gate."""
import sys, numpy as np, pandas as pd
sys.path[:0] = ["probes/P28-addr-anchor", "code/business_entity_resolution/src", "probes/N63-floor-probes"]
from loco import fbeta
import france as FR
from cells import capped_add
SIB = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}
def other(h1, hx):
    o = np.zeros(len(h1), bool)
    for i, (a, b) in enumerate(zip(h1, hx)):
        if a and b and a != b:
            try: d = int(b) - int(a)
            except ValueError: continue
            if abs(d) in SIB: continue
            if len(a) != len(b) and (a.startswith(b) or b.startswith(a) or a.endswith(b) or b.endswith(a)): continue
            o[i] = True
    return o
for s_, t_ in (("US", "India"), ("India", "US")):
    d = pd.read_parquet(f"data/work/n63/chain_{s_}_{t_}.parquet", columns=["s1","sx","p","y","src","acc","pos","rel","h1","hx","addr_1","addr_x","c1","cx"])
    m = np.load(f"data/work/n63/meta_{s_}_{t_}.npz", allow_pickle=True); ntrue, hv = m["ntrue"], m["hv"]; n = len(ntrue)
    y = d.y.to_numpy(); pos = d.pos.to_numpy(); acc = d.acc.to_numpy(); owned = set(d.sx[acc])
    per = lambda pr: fbeta(np.bincount(pos[pr & y], minlength=n), np.bincount(pos[pr], minlength=n), ntrue)
    f0 = per(acc); se = lambda x: x.std() / np.sqrt(len(x))
    oth = other(d.h1.fillna(""), d.hx.fillna(""))
    sel = oth & (d.rel == "eq").to_numpy() & ~acc & ~d.sx.isin(owned).to_numpy()
    st1 = FR.street_key(d.addr_1[sel]).to_numpy(); stx = FR.street_key(d.addr_x[sel]).to_numpy()
    st = np.zeros(len(d), bool); st[np.where(sel)[0]] = (st1 != "") & (st1 == stx)
    print(f"== {s_}>{t_}: eq-name hn-other unaccepted {sel.sum():,} true {y[sel].mean():.3f}; street-eq {st.sum():,} true {y[st].mean():.3f}; street-diff true {y[sel & ~st].mean():.3f}")
    for pf in (0.3, 0.5, 0.7):
        mk = st & (d.p >= pf).to_numpy(); add = capped_add(d, mk); dF = per(acc | add) - f0
        print(f"   street-eq p>={pf}: added {add.sum():5d} prec {y[add].mean():.3f}  dF h0 {dF[hv==0].mean():+.6f}±{se(dF[hv==0]):.6f} h1 {dF[hv==1].mean():+.6f}±{se(dF[hv==1]):.6f}")
