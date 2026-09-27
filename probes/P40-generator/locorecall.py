"""P40: which LOCO direction is France-like per op class? LOCO base (gc70 recipe) recall on the target's true
address-less / gib pairs (holdout truth, fold 0) vs France's label-free recall (accepted per S1 / class SX per S1 x owned share)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "P37-fr-complete"),
                str(Path(__file__).resolve().parents[1] / "P28-addr-anchor")]
import p37 as P
from cat import OUT
from loco40 import T_GC, fr_dens
from loss import bits

dens = fr_dens()
tp = pd.read_parquet(OUT / "tp.parquet")
B = bits(tp.ops.to_numpy())
tp["emp"], tp["gib"] = B.A_EMPTY.to_numpy(), B.N_GIB_NAME.to_numpy()
for src, tgt in (("US", "India"), ("India", "US")):
    a = P.loco_load(src, tgt)
    s1, sx, is2, p, am, y, sib, n = a["s1"], a["sx"], a["is2"], a["p"], a["am"], a["y"], a["sib"], a["n_s1"]
    tb = P.TS[(src, tgt)]
    tmap = {t: P.inv_dens(p, am, tb, g, n) for t, g in dens.items()}
    pred = am & (p >= tb)
    pred |= P.rescue(s1, p, am, pred, sib, tmap[P.T_RESCUE])
    pred |= P.gc_add(s1, sx, is2, p, am, pred, sib, tmap[T_GC], "oneall")
    acc = set(zip(a["s1_ids"][pred], a["sx_ids"][pred]))
    t = tp[tp.c == tgt]
    hit = np.array([(u, v) in acc for u, v in zip(t.s1, t.sx)])
    print(f"{src}->{tgt}: recall all {hit.mean():.3f}; address-less {hit[t.emp.to_numpy()].mean():.3f}; gib {hit[t.gib.to_numpy()].mean():.3f}; "
          f"per S1 accepted {pred.sum() / n:.3f}")
