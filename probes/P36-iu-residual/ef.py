"""P36 lever (b): per-S1 expected-F0.5 decoding on the a10 p' (out of fold), with P23's decision-theory bench
(probes/P23-error-analysis/work/decision-theory/s3_bench.py: ef_E / ef_mix, P5's ef_select) run on p' instead of
#8c's p. Base = #34 (am & p' >= 0.71). Tune on half fh, report half 1-fh, both directions; + the train caps.
    PYTHONPATH=probes/P36-iu-residual probes/P36-iu-residual/py.sh probes/P36-iu-residual/ef.py
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from base import A, LB_W, OUT, ROOT, paired, state

sys.path[:0] = [str(ROOT / "probes/P23-error-analysis/work/decision-theory"), str(ROOT / "probes/P5-decide")]
from s3_bench import ef_mix  # noqa: E402
from decide import _rank, ef_select  # noqa: E402

TT = np.round(np.arange(0.50, 0.95, 0.01), 2)


def frames(c: str):
    s = state(c)
    pr = s["pairs"]
    df = pd.DataFrame({"g": pr.g.to_numpy(np.int64), "y": pr.y.to_numpy(), "am": pr.am.to_numpy(),
                       "p": np.where(pr.am, pr.pa.fillna(0), 0.0), "src": pr.sx.str[:2].to_numpy()})
    s1 = pd.DataFrame({"n_true": s["n_true"].astype(np.int32), "half": s["h"].astype(np.int64)})
    return s, df, s1


def f_of(sel, df, s1):
    g, y, n = df.g.to_numpy(), df.y.to_numpy(), len(s1)
    return A.fb(np.bincount(g[sel & y], minlength=n), np.bincount(g[sel], minlength=n), s1.n_true.to_numpy())


def capped(sel, df, s1, caps=(5, 6)):
    """At most caps[0] S2 and caps[1] S3 per S1 (the train truth never exceeds them): keep the highest p."""
    out = sel.copy()
    for src, k in zip(("S2", "S3"), caps):
        m = np.flatnonzero(sel & (df.src.to_numpy() == src))
        r = _rank(df.g.to_numpy()[m], df.p.to_numpy()[m])
        out[m[r >= k]] = False
    return out


def variants(df, s1, fh: int) -> dict:
    """{name: selection mask}, every knob tuned on S1 half fh."""
    am, p, y, g = df.am.to_numpy(), df.p.to_numpy().astype(np.float64), df.y.to_numpy(), df.g.to_numpy()
    ng, hf = len(s1), s1.half.to_numpy() == fh
    rh = hf[g]
    tune = lambda sel: f_of(sel, df, s1)[hf].mean()
    V = {"base": am & (p >= A.T)}
    V["cap56"] = capped(V["base"], df, s1)
    f0 = [tune(am & (p >= t)) for t in TT]
    V["t_retune"] = am & (p >= TT[int(np.argmax(f0))])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p[am & rh], y[am & rh])
    for nm, q in (("raw", np.where(am, p, 0.0)), ("iso", np.where(am, iso.predict(p), 0.0))):
        tp_am = np.bincount(g[am & y], minlength=ng)
        lam = float((s1.n_true.to_numpy() - tp_am)[hf].mean())
        best = max(((tune(ef_select(g, q, 12, m, empty=True) & am & (q > 0)), m) for m in (0.0, 0.02, lam, 0.15)))
        V[f"ef_indep_{nm}"] = ef_select(g, q, 12, best[1], empty=True) & am & (q > 0)
        ek = ef_select(g, q, 12, lam, empty=False) & am & (q > 0)
        bq = np.zeros(ng)
        np.maximum.at(bq, g, q)
        t0 = max(((tune(ek & (bq[g] >= t)), t) for t in np.round(np.arange(0.40, 0.90, 0.02), 2)))[1]
        V[f"ef_gate_{nm}"] = ek & (bq[g] >= t0)
        single = (s1.n_true.to_numpy() == 0) * 1.0
        s_hat = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(bq[hf], single[hf]).predict(bq)
        rq = _rank(g, q)
        for sc in (False, True):
            best = max(((tune(ef_mix(g, q, rq, ng, m, s_hat, scale=sc) & am), m) for m in (0.0, lam)))
            V[f"ef_mix{'_scaled' if sc else ''}_{nm}"] = ef_mix(g, q, rq, ng, best[1], s_hat, scale=sc) & am
    V["ef_gate_iso+cap56"] = capped(V["ef_gate_iso"], df, s1)
    return V


if __name__ == "__main__":
    res = {}
    F = {c: frames(c) for c in LB_W}
    for fh in (0, 1):
        rep = 1 - fh
        per = {}
        for c, (s, df, s1) in F.items():
            V = variants(df, s1, fh)
            base = f_of(V["base"], df, s1)
            assert np.allclose(base, s["f34"])
            per[c] = {k: paired(f_of(v, df, s1) - base, s["h"], rep) for k, v in V.items()}
        for k in per["India"]:
            lb = sum(LB_W[c] * per[c][k][0] for c in LB_W)
            se = np.sqrt(sum((LB_W[c] * per[c][k][1]) ** 2 for c in LB_W))
            res[f"{k}_tune{fh}"] = dict(lb=lb, se=se, **{c: per[c][k] for c in LB_W})
            print(f"tune h{fh} -> h{rep} {k:24s} LB-eq {lb:+.6f} ± {se:.6f} | "
                  + " ".join(f"{c} {per[c][k][0]:+.6f} ± {per[c][k][1]:.6f}" for c in LB_W), flush=True)
    json.dump(res, open(OUT / "ef.json", "w"), indent=1, default=float)
