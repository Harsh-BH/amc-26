"""P66 diagnostic: why do exact-unique-name address-less adds fail in India->US? For the xf pool pairs (unaccepted by the
France-chain analogue), relate the SX to its TRUE owner: owner in file? owner's key vs SX key (legal-form drop, core equal)?
Does the argmax S1 share its core with other S1 (core count)? Prints aggregates only.
    .venv/bin/python probes/P66-fr-noaddr/diag.py [India|US]
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import loco66 as L  # noqa: E402
from loco66 import P, W  # noqa: E402

TGT = sys.argv[1] if len(sys.argv) > 1 else "US"
SRC = {"US": "India", "India": "US"}[TGT]


def main():
    dens = L.fr_dens()
    A = P.loco_load(SRC, TGT)
    s1, sx, is2, p, am, y, sib, n = (A[k] for k in ("s1", "sx", "is2", "p", "am", "y", "sib", "n_s1"))
    tb = P.TS[(SRC, TGT)]
    tm = {t: P.inv_dens(p, am, tb, g, n) for t, g in dens.items()}
    pred = am & (p >= tb)
    pred |= P.rescue(s1, p, am, pred, sib, tm[P.T_RESCUE])
    pred |= P.gc_add(s1, sx, is2, p, am, pred, sib, tm[0.7], "oneall")
    na = pd.Series(A["sx_ids"]).isin(set(L.addrless_ids("train", TGT))).to_numpy()
    w = np.flatnonzero(am & na & ~pred)
    fc = L.fk_counts("train", TGT)
    f = L.feats("train", TGT, A["s1_ids"][w], A["sx_ids"][w], fc)
    xf = f["fk_eq"] & f["fk_uniq"]
    w, yy = w[xf], y[w[xf]]
    ids1, idsx = A["s1_ids"][w], A["sx_ids"][w]
    core1 = pd.read_parquet(W / f"rf_train_s1_{TGT}.parquet", columns=["id", "core"]).set_index("id").core
    cc = core1.value_counts()
    c1 = core1.reindex(ids1).fillna("").to_numpy()
    cu = cc.reindex(c1).fillna(0).to_numpy()
    print(f"{SRC}->{TGT}: xf pool {len(w)}, prec {yy.mean():.3f}")
    for k in (1, 2, 3):
        m = (cu == k) if k < 3 else (cu >= 3)
        print(f"  S1 core count {'>=3' if k == 3 else k}: n {m.sum()}, prec {yy[m].mean() if m.any() else float('nan'):.3f}")
    gt = pd.read_parquet(W / "gt_pairs.parquet")
    own = gt.set_index("sx").s1.reindex(idsx[~yy]).to_numpy()
    print(f"  false: {(~yy).sum()}, owned {pd.notna(own).mean():.3f}")
    n1 = L.names("train", [o for o in own if isinstance(o, str)], (1,))
    kx = [L.fkey(x) for x in L.names("train", idsx[~yy], (2, 3)).reindex(idsx[~yy]).fillna("")]
    ko = [L.fkey(n1.get(o, "")) if isinstance(o, str) else None for o in own]
    rel = Counter()
    for a, b in zip(kx, ko):
        if b is None:
            rel["unowned"] += 1
        elif a == b:
            rel["owner same key (twin missed?)"] += 1
        elif b.startswith(a + " ") or b.endswith(" " + a):
            rel["owner key = SX key + extra word(s)"] += 1
        elif set(a.split()) < set(b.split()):
            rel["SX tokens subset of owner"] += 1
        else:
            rel["other"] += 1
    print("  false-pair relation to true owner:", dict(rel))
    oc = core1.reindex([o for o in own if isinstance(o, str)]).fillna("").to_numpy()
    print(f"  owner core == argmax S1 core: {np.mean(oc == c1[~yy][pd.notna(own)]):.3f}")
    # the true owner: is it in the file for this SX, and what p did it get?
    key = pd.MultiIndex.from_arrays([A["s1_ids"], A["sx_ids"]])
    inf = key.isin(pd.MultiIndex.from_arrays([pd.Series(own).fillna("-"), idsx[~yy]]))
    print(f"  owner pair in the LOCO file (fold-0 owner): {inf.sum()} of {(~yy).sum()}")


if __name__ == "__main__":
    main()
