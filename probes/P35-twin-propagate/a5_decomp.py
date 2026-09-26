"""P35 a5: where does the group-decision gain come from? Split the changed address-less SX into
 A = same argmax S1, accept/reject flips (a10's space) vs R = re-assigned to another S1; and by e5 exposure
 (SX with a band row of a competing train S1 without a bge r: holdout-only artefact, P23 a14). Fit h0 -> report h1."""
import sys
sys.path.insert(0, "probes/P35-twin-propagate")
import numpy as np, pandas as pd
from fit import load_all, fit, ARMS, T, LBW
from ev import f_of
from gdec import decide
from twins import argmax_mask

S = load_all()
for s in S:
    e = pd.read_parquet(f"data/work/p35/ho_{s['c']}.parquet", columns=["p_m2", "r"])
    e5 = ((e.p_m2 > 0.02) & (e.p_m2 < 0.98) & e.r.isna()).to_numpy()
    s["e5sx"] = pd.Series(e5).groupby(pd.factorize(s["d"].sx)[0]).transform("any").to_numpy()
for arm in sys.argv[1:] or ["full"]:
    for fh in (0, 1):
        b, ps = fit(S, ARMS[arm], fh)
        rep = 1 - fh
        out = {}
        for s, p in zip(S, ps):
            d = s["d"]; base = d.pred.to_numpy()
            new = decide(d, s["x"], p, T)
            pn = d.p.to_numpy().copy(); pn[s["x"].index.to_numpy()] = p
            am0, am1 = argmax_mask(d.sx.to_numpy(), d.p.to_numpy()), argmax_mask(d.sx.to_numpy(), pn)
            q = pd.factorize(d.sx)[0]
            moved = pd.Series(am0 != am1).groupby(q).transform("any").to_numpy()       # SX whose argmax S1 changed
            changed = pd.Series(new != base).groupby(q).transform("any").to_numpy()
            res = {}
            for name, m in (("all", changed), ("A_accrej", changed & ~moved), ("R_reassign", changed & moved),
                            ("no_e5", changed & ~s["e5sx"]), ("e5", changed & s["e5sx"]),
                            ("R_no_e5", changed & moved & ~s["e5sx"])):
                pr = np.where(m, new, base)
                dl = (f_of(s, pr) - s["f0"])[s["h"] == rep]
                res[name] = (dl.mean(), dl.std(ddof=1) / np.sqrt(len(dl)), int(len(np.unique(q[m & (s["g"] >= 0)]))))
            out[s["c"]] = res
            print(f"{arm} fit h{fh} {s['c']}: " + " | ".join(f"{k} {v[0]:+.6f} ± {v[1]:.6f} ({v[2]} SX)" for k, v in res.items()), flush=True)
        print(f"{arm} fit h{fh} LB-eq: " + " | ".join(f"{k} {sum(LBW[c] * out[c][k][0] for c in LBW):+.6f}" for k in out['India']), flush=True)
