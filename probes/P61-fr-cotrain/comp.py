"""Components of the swap P55 -> P61 (veto re-admits, new vetoes, adds gained / lost): LOCO dF + precision, and the
label-free profile (per accepted pair) on LOCO and France."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p61  # noqa: E402

L, D, W = p61.L, p61.D, p61.W
nm = sys.argv[1]
out = {}
for name, trn, tst in p61.p55.DIRN:
    b = p61.load(tst)
    cb = np.load(W / f"p55/cb_plain_{tst}.npy")
    c = np.load(D / f"c_{nm}_{tst}.npy").astype(np.float64)
    A, B = p61.arm_a(b, cb), p61.arm_a(b, c)
    vA, vB = p61.arm_a(b, cb, u=None), p61.arm_a(b, c, u=None)
    fA = L.F(b["z"], A)
    nref = b["ref"].sum()
    comp = {"readmit": B & ~A & b["ref"], "newveto": A & ~B & b["ref"], "add_gain": B & ~A & ~b["ref"], "add_lost": A & ~B & ~b["ref"]}
    R = out[name] = {}
    for k, m in comp.items():
        new = (A | m) if k in ("readmit", "add_gain") else (A & ~m)
        d = L.cmp(b["z"], A, new, fA)
        R[k] = {"n": int(m.sum()), "per_ref": round(float(m.sum() / nref), 5), "true": round(float(b["z"]["y"][m].mean()), 3) if m.any() else None,
                "dF": d["dF"], "dF_h1": d["dF_h1"]}
    print(name, json.dumps(R))
fz = dict(np.load(D / "france.npz", allow_pickle=True))
rg = fz["X"][:, list(fz["cols"]).index("region")]
cf, cn = fz["cf"], np.load(D / f"cf_{nm}.npy")
ref, own, free = fz["ref"], fz["own"], fz["free"]
sup = ~((rg == 2) & (fz["p"] > 0))
vp, vn = ref & own & (rg == 1) & (cf < 0.6), ref & own & (rg == 1) & (cn < 0.6)
ap, an = free & ~ref & own & sup & (cf > 0.95), free & ~ref & own & sup & (cn > 0.95)
nref = ref.sum()
F = out["France"] = {k: {"n": int(m.sum()), "per_ref": round(float(m.sum() / nref), 5)} for k, m in
                     (("readmit", vp & ~vn), ("newveto", vn & ~vp), ("add_gain_cand", an & ~ap), ("add_lost_cand", ap & ~an))}
q = fz["q"]
for k, m in (("readmit", vp & ~vn), ("newveto", vn & ~vp), ("add_gain_cand", an & ~ap), ("add_lost_cand", ap & ~an)):
    F[k]["qwen_ge_.5"] = round(float((q[m] >= 0.5).mean()), 3) if m.any() else None
    F[k]["qwen_cov"] = round(float(np.isfinite(q[m]).mean()), 3) if m.any() else None
print("France", json.dumps(F))
(HERE / f"comp_{nm}.json").write_text(json.dumps(out, indent=1))
