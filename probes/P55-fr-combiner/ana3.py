"""Adds by region (LOCO per-pair value, France-reweighted estimate) and the veto(nobasehi / all) + add combinations."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p55  # noqa: E402

W = p55.W / "p55"
frames = {dr: p55.loco_frame(dr) for dr in p55.L.DIRS}
out, cell = {}, {}
for name, trn, tst in p55.DIRN:
    b = frames[tst]
    cb = np.load(W / f"cb_plain_{tst}.npy")
    fref = p55.L.F(b["z"], b["ref"])
    st, rg = b["X"]["stage"].to_numpy(), b["X"]["region"].to_numpy()
    R = out[name] = {}
    for u in (0.95, 0.97, 0.98):
        new = p55.decide(b, cb, None, u)
        add = new & ~b["ref"]
        for r_ in (0, 1, 2):
            m = add & (rg == r_)
            if m.any():
                c = p55.L.cmp(b["z"], b["ref"], b["ref"] | m, fref)
                R[f"add:{u}:rg{r_}"] = c
                cell[(name, u, r_)] = c["dF"] * b["n"] / m.sum()
        for sc in ("nobasehi", "all"):
            for v in (0.4, 0.5, 0.6):
                veto = b["ref"] & (cb < v) & p55.SCOPES[sc](st, rg)
                base = b["ref"] & ~veto
                bb = dict(b, ref=base)
                R[f"A:{sc}:{v}:{u}"] = p55.L.cmp(b["z"], b["ref"], p55.decide(bb, cb, None, u), fref)
F = json.loads((HERE / "ana2_plain.json").read_text())["France"]
for u in (0.95, 0.97, 0.98):
    k = F[f"add:{u}"]["by_region"]
    out.setdefault("France", {})[f"add:{u}"] = {"by_region": k, "est": {nm: round(sum(k[r_] * cell.get((nm, u, r_), 0.0) for r_ in range(3)) / 259452, 6) for nm, _, _ in p55.DIRN}}
(HERE / "ana3.json").write_text(json.dumps(out, indent=1))
I, U = out["India->US"], out["US->India"]
for k in I:
    print(f"{k:22s} {I[k]['dF']:+.5f} h1 {I[k]['dF_h1']:+.5f} (+{I[k]['add']} {I[k]['add_prec']} -{I[k]['rm']}) | {U[k]['dF']:+.5f} h1 {U[k]['dF_h1']:+.5f} (+{U[k]['add']} {U[k]['add_prec']} -{U[k]['rm']})")
print(json.dumps(out["France"]))
