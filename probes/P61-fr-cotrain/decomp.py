"""Where does the co-trained gain come from? Split changed pairs by Qwen coverage (pseudo-labelable vs not), and the
hybrid (P55's veto + co-trained adds)."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p61  # noqa: E402

L, D, W = p61.L, p61.D, p61.W
for nm in sys.argv[1:]:
    for name, trn, tst in p61.p55.DIRN:
        b = p61.load(tst)
        cb = np.load(W / f"p55/cb_plain_{tst}.npy")
        c = np.load(D / f"c_{nm}_{tst}.npy").astype(np.float64)
        cov = np.isfinite(b["q"])
        base, new = p61.arm_a(b, cb), p61.arm_a(b, c)
        fb = L.F(b["z"], base)
        only_cov = np.where(cov, new, base)          # co-trained decisions on Qwen-covered pairs only
        only_unc = np.where(cov, base, new)
        vb = p61.arm_a(b, cb, u=None)
        hyb = p61.p55.decide(dict(b, ref=vb), c, None, 0.95)   # P55's veto, co-trained adds
        bnd = b["rg"] == 1
        au = lambda s, m: p61.p55.auc(b["z"]["y"][m], s[m])
        out = {k: L.cmp(b["z"], base, v, fb) for k, v in (("all", new), ("cov_only", only_cov), ("unc_only", only_unc), ("hybrid", hyb))}
        print(nm, name, " | ".join(f"{k} {v['dF']:+.5f} h1 {v['dF_h1']:+.5f} (+{v['add']} {v['add_prec']} -{v['rm']} {v['rm_false']})" for k, v in out.items()))
        print("   band AUC p55/cot: covered", au(cb, bnd & cov), au(c, bnd & cov), " uncovered", au(cb, bnd & ~cov), au(c, bnd & ~cov),
              " qwen on covered band", au(np.nan_to_num(b["q"]), bnd & cov))
