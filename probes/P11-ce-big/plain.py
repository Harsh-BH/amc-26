"""The shipped decision rule (submit_probs: argmax per S2/S3, p >= t; = D3 with t0 = t), t tuned on half 0, reported
on half 1, paired vs #5's CE -- does a CE's D3 gain survive the rule actually submitted?   plain.py <tag>"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "probes/P11-ce-big")
import ce

def plain(B, d, S):
    import decide as D
    p = d.p.to_numpy(np.float64)
    g, x = D._codes(d.s1), D._codes(d.sx)
    keep = p >= D._gmax(x, p, x.max() + 1)[x]
    s1, y, n = d.s1.to_numpy(), d.is_true.to_numpy(), len(S)
    nt, m0 = S.nt.to_numpy(), S.half.to_numpy() == 0
    def f(t):
        sel = keep & (p >= t)
        tp, npred = np.bincount(s1[sel & y], minlength=n), np.bincount(s1[sel], minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(nt == 0, npred == 0, 1.25 * tp / (0.25 * nt + npred))[m0].mean()
    t = max(np.round(np.arange(0.50, 0.91, 0.01), 2), key=f)
    return {"excl": True, "t": float(t), "t0": float(t)}

big = ce._big()
ce.rr.d3 = plain
ctl = big._blend_eval(big.PROB)
rc = pd.read_parquet(ce.OUT / f"r_{sys.argv[1]}_holdout.parquet")
big._r = lambda ids: ids.merge(rc, on=["s1", "sx"], how="left").r.to_numpy(np.float32)
new = big._blend_eval(big.PROB)
h1 = new[4].half.to_numpy() == 1
dif = (new[3] - ctl[3])[h1]
for nm, v in (("ce5", ctl), (sys.argv[1], new)):
    print(f"{nm}: plain t={v[1]['t']:.2f} f05={v[0]['f05']:.5f} India {v[0]['f05_India']:.5f} US {v[0]['f05_US']:.5f} "
          f"no-CE {v[0]['f05_no_ce']:.5f}")
print(f"{sys.argv[1]} vs ce5, plain rule: {dif.mean():+.5f} +- {dif.std(ddof=1) / np.sqrt(len(dif)):.5f}, "
      f"{(dif > 0).sum()} up {(dif < 0).sum()} down")
