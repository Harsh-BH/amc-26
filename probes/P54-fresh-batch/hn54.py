"""P54 idea 1 check: LOCO rescue adds (chain + m2, stacked on the veto, scope any) split by hn class (hn_rel 1 = hn equal,
0 = other / missing), with precision and dF per class; France's hn_eq share of adds is 0.37 vs 0.92 for the recipe."""
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import loco54  # noqa: E402

L, S = loco54.L, loco54.S
for dr in L.DIRS:
    z = L.frame(dr)
    g, y = z["g"], z["y"]
    s1, is2, n = z["s1"].astype(np.int64), z["is2"], len(z["ntrue"])
    r = np.nan_to_num(z["r"], nan=0.0)
    band = (g > S.LO) & (g < S.HI)
    for an in ("chain", "m2"):
        p = S.loco_p(g, z["r"], L.CUR["tau"], an)
        st, tm = L.recipe(z, dr, p, an=an)
        ref = st["hn"] & ~(st["gc"] & ~st["base"] & band & (np.nan_to_num(z["r"], nan=1.0) < 0.2))
        fref = L.F(z, ref)
        free = L.free_mask(z, st, p, tm)
        n2, n3 = S.counts(s1, is2, ref, n)
        m2, m3 = n2[s1], n3[s1]
        print(dr, an, "ref hn_eq share", round(float((z["hn_rel"][ref] == 1).mean()), 3))
        for u in (0.99, 0.995):
            for nm, keep in (("all", np.ones(len(g), bool)), ("hn_eq", z["hn_rel"] == 1), ("not_hn_eq", z["hn_rel"] != 1)):
                c = np.flatnonzero(free & ~ref & band & keep & (r >= u))
                new = L.add_capped(z, ref, c, r, m2, m3)
                v = L.cmp(z, ref, new, fref)
                add = new & ~ref
                print(f"  u{u} {nm}: add {v['add']} prec {v['add_prec']} dF {v['dF']:+.5f} h1 {v['dF_h1']:+.5f}±{v['se_h1']:.5f} hn_eq_share {float((z['hn_rel'][add] == 1).mean()):.3f}", flush=True)
