"""Could the stacker run on France? Leave-one-country-out proxy in France's regime: base = m2_big p on the pruned list
(France has no CE), t = the target's own base optimum on its half 0 (France's t is fixed before any stack, by the LB).
Fit on source S's holdout half 0 (lambda_l2 1, the shipped recipe), apply unchanged to target T, report T's half 1
with paired SE vs T's base; in-domain control = fit on T's half 0. Then the label-free France test diagnostic: what
the two LOCO models would do to France at t 0.95 and at the t that keeps #7's France accept count (flips by hn class). Aggregates only.
    .venv/bin/python probes/P17-hn-stack/loco.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stack as S  # noqa: E402

S.BASE = "m2"
S.PRM["lambda_l2"] = 1.0
CLS = {0: "none/eq", 1: "+SIB", 2: "digit drop/add", 3: "small other", 4: "far"}


def main():
    t0 = time.time()
    s1, t = S.holdout_s1()
    W, models = {}, {}
    for c in ("India", "US"):
        d = S.frame("holdout", c)
        pos, y, ntrue, hv = S.labels(d, c, s1, t)
        X = S.X_of(d, "sup")
        W[c] = dict(X=X, y=y, pos=pos, ntrue=ntrue, hv=hv, qx=d["qx"], p=d["p"])
        models[c] = S.fit(X, y, (pos >= 0) & (hv[np.maximum(pos, 0)] == 0))
        models[c].save_model(str(S.OUT / f"stack_sup-m2-l21.0_{c}.txt"))      # France candidates (no-CE base)
        del d
        print(f"{c}: fit, {time.time() - t0:.0f}s peak {S.peak():.2f} GB", flush=True)
    out = {}
    for tc, w in W.items():
        a0 = S.M.argmax_sx(w["qx"], w["p"])
        f = lambda am, s, t_: S.f05(am, s, t_, w["pos"], w["y"], w["ntrue"])
        h0, h1 = w["hv"] == 0, w["hv"] == 1
        tt = max(S.TS, key=lambda t_: f(a0, w["p"], t_)[h0].mean())
        fb = f(a0, w["p"], tt)
        for sc, m in models.items():
            s = S.stacked(m, w["X"])
            dl = (f(S.M.argmax_sx(w["qx"], s), s, tt) - fb)[h1]
            out[f"{sc}>{tc}"] = (round(dl.mean(), 5), round(dl.std() / np.sqrt(len(dl)), 5), float(tt))
            print(f"stack fit on {sc}, applied to {tc} (m2 base, t {tt}): half 1 d {dl.mean():+.5f} ± "
                  f"{dl.std() / np.sqrt(len(dl)):.5f}", flush=True)
    del W
    d = S.frame("test", "France")
    X = S.X_of(d, "sup")
    a0 = S.M.argmax_sx(d["qx"], d["p"]) & (d["p"] >= 0.95)
    cls = X.cls.to_numpy()
    for sc, m in models.items():
        s = S.stacked(m, X)
        am = S.M.argmax_sx(d["qx"], s)
        tm = float(np.sort(s[am])[::-1][int(a0.sum()) - 1])     # label-free: t that keeps #7's France accept count
        for tag, tt in ((f"{sc}>France-test", 0.95), (f"{sc}>France-test-countmatched", tm)):
            a = am & (s >= tt)
            g = pd.DataFrame({"cls": pd.Series(cls).map(CLS), "new_acc": a & ~a0, "new_rej": a0 & ~a}).groupby("cls").sum()
            out[tag] = {"t": round(tt, 4), "acc": int(a0.sum()), "acc_new": int(a.sum()), "flips": int((a != a0).sum()),
                        "s1_changed": int(len(np.unique(d["q1"][a != a0]))), "by_cls": g.astype(int).to_dict("index")}
            print(f"France test, {sc} model at t {tt:.4f}: accepted {int(a0.sum()):,} -> {int(a.sum()):,}; "
                  f"S1 changed {out[tag]['s1_changed']:,}\n{g.to_string()}", flush=True)
    return out, time.time() - t0


if __name__ == "__main__":
    r, rt = main()
    sys.path.insert(0, str(S.ROOT))
    from ledger import log
    log("exp", probe="P17-hn-stack", subproblem="france", name="hn-stack-loco-m2base",
        world="holdout-dense half1 cross-country (m2 p, pruned list) + France test (label-free)",
        metrics={k: v for k, v in r.items() if ">" in k and "France" not in k} | {
            f"{k}_flips": v["flips"] for k, v in r.items() if "France" in k},
        config={"stacker": "sup, lambda_l2 1, fit on source half 0", "t": "target's base optimum on its half 0",
                "france": "t 0.95, #7's France p (m2_big, pruned)"},
        hypothesis="a stacker fit on one labelled country helps another in the no-CE regime -> a France candidate",
        verdict="inconclusive", runtime_s=round(rt), peak_gb=round(S.peak(), 2))
    (S.OUT / "loco.json").write_text(__import__("json").dumps(r, indent=1, default=str))
