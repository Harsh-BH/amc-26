"""P69 LOCO: P42's machinery (solution_p42: loco_p / fr_to_loco / decide) for France t .9868 -> .975 / .95 on the
#37-style chain (tau .1, caps 5/6, rescue .90, gc .70), both directions, chain + M2 analogues; the France density map
is re-measured on sub66_final's own France frame (p69.py's dump). Also B's per-bin precision of unassigned argmax
pairs. -> loco69.json
    .venv/bin/python probes/P69-fr-loosen/loco69.py"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = ["probes/P42-fr-joint-tune", "."]
import solution_p42 as S  # noqa: E402

W, HERE = Path("data/work/p42"), Path("probes/P69-fr-loosen")
TS = (0.9868, 0.975, 0.95)
EDGES = (0.0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.9868)
f = np.load(HERE / "work/t9868/p69_frame.npz")
p = f["p"].astype(np.float64)
am = S.argmax_sx(f["K"] & 0xFFFFFFFF, p)
n = int(np.load(W / "france.npz")["n_s1"])
xs = sorted(set(TS) | set(EDGES[1:]) | {0.9, 0.7})
G = {"fr_dens": {"0.1": {str(x): S.sdens(p, am, S.T_FR, x, n) for x in xs}}}
out = {"fr_dens_sub66": G["fr_dens"]["0.1"]}
del f, p, am
for an in ("chain", "m2"):
    for dr in ("US_India", "India_US"):
        z = S._frame(W, dr)
        s1, is2, y, nl = z["s1"].astype(np.int64), z["is2"], z["y"], len(z["ntrue"])
        pl = S.loco_p(z["g"], z["r"], 0.1, an)
        one = np.ones(len(s1), bool)
        tm = S.fr_to_loco(G, an, dr, 0.1, pl, one, nl)
        sib, hq = z["hn_rel"] == 2, z["hn_rel"] == 1
        pred, F = {}, {}
        for t in TS:
            pred[t] = S.decide(s1, is2, pl, one, sib, hq, nl, tm[str(t)], (5, 6), tm["0.9"], tm["0.7"], None)
            F[t] = S.fbeta(np.bincount(s1[pred[t] & y], minlength=nl), np.bincount(s1[pred[t]], minlength=nl),
                           z["ntrue"].astype(np.int64))
        r = {"t_loco": {str(t): round(tm[str(t)], 4) for t in TS}}
        for t in TS[1:]:
            a, rm = pred[t] & ~pred[TS[0]], pred[TS[0]] & ~pred[t]
            d = F[t] - F[TS[0]]
            r[str(t)] = {"dF": round(float(d.mean()), 6), "dF_h": [round(float(d[z["hv"] == h].mean()), 6) for h in (0, 1)],
                         "added": int(a.sum()), "removed": int(rm.sum()), "prec_added": round(float(y[a].mean()), 4) if a.any() else None,
                         "added_per_s1": round(a.sum() / nl, 5)}
        base = pred[TS[0]]
        owned = np.zeros(nl, bool)
        n2, n3 = S.counts(s1, is2, base, nl)
        lack = np.where(is2, n2[s1] == 0, n3[s1] == 0)             # the argmax S1 has no member of this SX's source
        un = ~base                                                  # rows = SX-argmax pairs: SX unassigned
        bins = {}
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            tl, th = (0.0 if lo == 0 else tm[str(lo)]), tm[str(hi)]
            m = un & (pl >= tl) & (pl < th)
            bins[f"{lo}-{hi}"] = {"t_loco": [round(tl, 4), round(th, 4)], "n_per_s1": round(m.sum() / nl, 5),
                                  "prec": round(float(y[m].mean()), 4) if m.any() else None,
                                  "prec_lacking_src": round(float(y[m & lack].mean()), 4) if (m & lack).any() else None,
                                  "share_lacking_src": round(float(lack[m].mean()), 4) if m.any() else None}
        r["bins_unassigned"] = bins
        out[f"{an}:{dr}"] = r
        print(an, dr, json.dumps(r), flush=True)
        del z, s1, is2, y, pl, pred, F
(HERE / "loco69.json").write_text(json.dumps(out, indent=1))
