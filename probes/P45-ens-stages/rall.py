"""r_all diagnostics on the holdout: a10 + [sx_psum, n_same_src_acc] (base) vs base + l_r_all, both fit directions.
(1) per-S1 gain split by 'the S1 has an argmax SX whose exact text is in ce_bge_pool's training pairs' (leak.py);
(2) pair flips on the report half by #8c-p region (< 0.02 / band / >= 0.98): adds, drops and their precision."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zoo as Z

BASE = ["sx_psum", "n_same_src_acc"]
S = Z.load(["r_all"])
for s in S:
    lk = pd.read_parquet(Z.W / f"leak_sx_{s['c']}.parquet")
    assert (lk.s1.to_numpy() == s["a"].s1.to_numpy()).all()
    s["sx_in"] = np.bincount(s["a"].g.to_numpy(), lk.sx_in.to_numpy(), minlength=s["n"]) > 0
for fh in (0, 1):
    rep = 1 - fh
    ref, _ = Z.run_arm(S, BASE, fh)
    new, _ = Z.run_arm(S, BASE + ["l_r_all"], fh)
    for s, q, p in zip(S, ref, new):
        a = s["a"]
        d = (Z.H.f_of(s, p >= Z.A.T) - Z.H.f_of(s, q >= Z.A.T))
        hm = s["h"] == rep
        for nm, m in (("sx_in_ce_train", hm & s["sx_in"]), ("clean", hm & ~s["sx_in"])):
            x = d[m]
            print(f"fit h{fh} {s['c']:5s} {nm:15s} n={m.sum():6d} gain/S1 {x.mean():+.6f} ± {x.std(ddof=1) / np.sqrt(len(x)):.6f}", flush=True)
        rh = hm[a.g.to_numpy()]
        p8 = a.p.to_numpy()
        reg = np.select([p8 < 0.02, p8 >= 0.98], ["lt.02", "ge.98"], "band")
        y = a.y.to_numpy()
        b0, b1 = q >= Z.A.T, p >= Z.A.T
        for r in ("lt.02", "band", "ge.98"):
            m = rh & (reg == r)
            ad, dr = m & b1 & ~b0, m & b0 & ~b1
            print(f"   {s['c']:5s} {r:6s} pairs {m.sum():7d} | adds {ad.sum():5d} (true {y[ad].mean() if ad.any() else np.nan:.3f}) | "
                  f"drops {dr.sum():5d} (false {1 - y[dr].mean() if dr.any() else np.nan:.3f})", flush=True)
