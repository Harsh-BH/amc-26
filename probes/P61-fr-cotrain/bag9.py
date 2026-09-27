"""9-seed bags (logit mean of three 3-seed runs) for P55 (cb + null_s345 + null_s678) and the co-trained combiner, compared
full and hybrid; saves c_<name>9_<tgt>.npy for comp.py / est.py."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p61  # noqa: E402

L, D, W, S = p61.L, p61.D, p61.W, p61.S
nm = sys.argv[1]                                    # e.g. c70q70_c30q30_share
lg = lambda x: S.logit(np.clip(np.asarray(x, np.float64), 1e-9, 1 - 1e-9))
bag = lambda fs: 1 / (1 + np.exp(-np.mean([lg(np.load(f)) for f in fs], 0)))
out = {}
for name, trn, tst in p61.p55.DIRN:
    b = p61.load(tst)
    cb = np.load(W / f"p55/cb_plain_{tst}.npy")
    p9 = bag([W / f"p55/cb_plain_{tst}.npy", D / f"c_null_s345_{tst}.npy", D / f"c_null_s678_{tst}.npy"])
    c9 = bag([D / f"c_{nm}_{tst}.npy", D / f"c_{nm}_s345_{tst}.npy", D / f"c_{nm}_s678_{tst}.npy"])
    np.save(D / f"c_{nm}9_{tst}.npy", c9.astype(np.float32))
    np.save(D / f"c_null9_{tst}.npy", p9.astype(np.float32))
    R = out[name] = {}
    for bn, bc in (("p55_3", cb), ("p55_9", p9)):
        A = p61.arm_a(b, bc)
        fA = L.F(b["z"], A)
        hyb = p61.p55.decide(dict(b, ref=p61.arm_a(b, bc, u=None)), c9, None, 0.95)
        for k, new in (("full", p61.arm_a(b, c9)), ("hybrid", hyb), ("p55_9" if bn == "p55_3" else "-", p61.arm_a(b, p9) if bn == "p55_3" else A)):
            if k == "-":
                continue
            d = L.cmp(b["z"], A, new, fA)
            R[f"{k}_vs_{bn}"] = d
            print(f"{name:10s} {k:7s} vs {bn}: dF {d['dF']:+.5f} h0 {d['dF_h0']:+.5f} h1 {d['dF_h1']:+.5f}±{d['se_h1']:.5f} (+{d['add']} {d['add_prec']} -{d['rm']} {d['rm_false']})")
(HERE / f"bag9_{nm}.json").write_text(json.dumps(out, indent=1))
