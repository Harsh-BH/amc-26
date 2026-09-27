"""P35 a10+: do P35's twin / load columns add to P34's shipped stacker (P23 a10 'twin_p2mask' arm, a13 MASK)?
Same frames, same recipe, same halves; P35 columns are NaN on SX with an address. Fit h0 -> h1 and the swap, t 0.71."""
import sys
sys.path.insert(0, "/home/harsh1/github-repos/amc-26/probes/P23-error-analysis/work/lowp-fn")
import numpy as np, pandas as pd
from a10_stack import load, arm, report, CTX, TWIN

P35 = ["tw_p", "tw_sim", "n_tw", "tw_x", "tw_same", "n_sup", "n_conf_src", "n_hungry"]
S = [load(c) for c in ("India", "US")]
for s in S:
    pp = pd.read_parquet(f"data/work/p23/pairs_{s['c']}.parquet", columns=["sx", "am", "p", "p_m2", "r"])
    pp = pp[~pp.am].sort_values("p", ascending=False).drop_duplicates("sx")
    e5 = ((pp.p_m2 > 0.02) & (pp.p_m2 < 0.98) & pp.r.isna()).to_numpy()
    m = s["a"].sx.map(pd.Series(e5, index=pp.sx.to_numpy())).fillna(False).to_numpy().astype(bool)
    s["a"]["p2m"] = np.where(m, np.nan, s["a"].p2); s["a"]["marginm"] = np.where(m, np.nan, s["a"].margin)
    ho = pd.read_parquet(f"data/work/p35/ho_{s['c']}.parquet", columns=["s1", "sx"])
    gx = pd.read_parquet(f"data/work/p35/gx_{s['c']}.parquet", columns=P35)
    gx = gx.join(ho.loc[gx.index])
    s["a"] = s["a"].merge(gx, on=["s1", "sx"], how="left")
    print(s["c"], "argmax rows with P35 columns:", int(s["a"].tw_p.notna().sum()), flush=True)
MASK = [c for c in CTX if c not in ("p2", "margin")] + ["p2m", "marginm"]
for fh in (0, 1):
    for name, cols in (("twin_p2mask", MASK + TWIN), ("twin_p2mask+P35", MASK + TWIN + P35)):
        ps, _ = arm(S, cols, fh)
        report(S, ps, fh, name)
