# LOCO for N54 on a gc70-style France chain: t_profile base + gc (0.7, one-sided) + rescue (0.9, empty) — NO hn-extra.
import sys; from pathlib import Path
import numpy as np, pandas as pd
sys.path[:0] = ["probes/NIGHT"]
import importlib.util
spec = importlib.util.spec_from_file_location("lc", "probes/NIGHT/loco_chain.py")
src = open("probes/NIGHT/loco_chain.py").read().split("cells = [")[0]
src = src.replace("    acc |= ~acc & ~n54 & hn_eq & ~sib & (p >= 0.8) & d.s1.isin(grp).to_numpy()\n", "")   # drop hn-extra
src = src.replace("p = d.p.to_numpy(); acc = (p >= t) & ~n54", "p = d.p.to_numpy(); acc = (p >= t)")                 # base keeps N54 cells
src = src.replace("acc |= ~acc & ~n54 & ~sib & (p >= 0.7) & miss", "acc |= ~acc & ~sib & (p >= 0.7) & miss")
src = src.replace("rc = ~acc & ~n54 & ~sib & (p >= 0.9)", "rc = ~acc & ~sib & (p >= 0.9)")
src = src.replace("    return d, acc, per, hv, y, t", "    d['n54'] = n54\n    return d, acc, per, hv, y, t")
exec(src)
for s_, t_ in (("US", "India"), ("India", "US")):
    d, acc, per, hv, y, t = chain(s_, t_); f0 = per(acc); n54 = d.n54.to_numpy()
    for tag, rm in (("all", acc & n54), ("ruled(p<t)", acc & n54 & (d.p.to_numpy() < t))):
        dF = per(acc & ~rm) - f0
        print(f"{s_}>{t_} gc70-chain N54 {tag:10s}: removed {rm.sum():5d} true {y[rm].mean() if rm.any() else float('nan'):.3f} | dF h0 {dF[hv==0].mean():+.6f} ± {dF[hv==0].std()/np.sqrt((hv==0).sum()):.6f}  h1 {dF[hv==1].mean():+.6f} ± {dF[hv==1].std()/np.sqrt((hv==1).sum()):.6f}")
