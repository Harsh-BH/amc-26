"""P79: LOCO value of an arm's NEW part on top of the shipped France state, the question sub79 asks (P73's gate measures the
arm over P55's base, but sub77 already carries 67q = P67-column adds, Qwen-gated). Per LOCO direction, on P61's frames:
  old  = 67q's cell: P55 veto (base combiner c < .6, band) + P67-column combiner adds c > .95, minus Qwen q < .5
  ship = the arm's shipped cell (p61.arm_a: veto band c < .6 + add free c > .95)
  new  = (old - (ref - ship)) | (ship - ref)            = build79's 73v then 73 (owner / cap limits not re-applied)
-> dF(new - old) with se (also: new adds Qwen-gated q < .5 out; vetoes Qwen-protected q >= .5 kept), split into veto part / add part, the new adds' precision by Qwen band, the new vetoes' truth share.
    .venv/bin/python probes/P79-p73-takeover/inc79.py p67+ele4n [JOB ...]  -> probes/P79-p73-takeover/inc79_<fs>.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "probes/P73-fr-independent"), str(ROOT)]
import ev73  # noqa: E402  (puts P55 / P61 on sys.path)
import p55  # noqa: E402
import p61  # noqa: E402
from ledger import log  # noqa: E402


def main(fs, *extra):
    t0 = time.time()
    jobs = ["p67-bge-loco-0927-1405", "p73b-both", *extra]
    fr = {dr: p61.load(dr) for dr in ("US_India", "India_US")}
    cols = {dr: pd.DataFrame(ev73.columns(jobs, dr, len(fr[dr]["X"]))) for dr in fr}
    L, out = p55.L, {}
    for name, trn, tst in p55.DIRN:
        a, b = fr[trn], fr[tst]
        c = {k: p55.bag_score(p55.bag(ev73.design(a["X"], cols[trn], k), a["z"]["y"], None),
                              ev73.design(b["X"], cols[tst], k), None) for k in dict.fromkeys(("base", "p67", fs))}
        z, ref, q, y = b["z"], b["ref"], b["q"], b["z"]["y"]
        dec = p61.arm_a(b, c["base"], u=None)
        add67 = p55.decide(dict(b, ref=dec), c["p67"], None, 0.95) & ~dec
        old = dec | (add67 & ~(q < .5))                                   # unscored adds kept (build67.qgate)
        ship = p61.arm_a(b, c[fs])
        veto, adds = ref & ~ship, ship & ~ref
        new = (old & ~veto) | adds
        fo = L.F(z, old)
        na, nv = adds & ~old, veto & old
        he = b["X"]["hn_rel"].to_numpy() == 1                            # hn-equal (P39 hn_rel 1)
        r3 = lambda m: round(float(y[m].mean()), 3) if m.any() else None
        prec = lambda m: {"pairs": int(m.sum()), "prec": r3(m), "hn_eq_share": round(float(he[m].mean()), 3) if m.any() else None,
                          "prec_hn_eq": r3(m & he), "prec_hn_other": r3(m & ~he)}
        out[name] = {"new_minus_old": L.cmp(z, old, new, fo), "veto_part": L.cmp(z, old, old & ~veto, fo),
                     "add_part": L.cmp(z, old, old | adds, fo),
                     "new_qgated": L.cmp(z, old, (old & ~veto) | (adds & ~(q < .5)), fo),
                     "new_qgated_qprot": L.cmp(z, old, (old & ~(veto & ~(q >= .5))) | (adds & ~(q < .5)), fo),
                     "new_qprot": L.cmp(z, old, (old & ~(veto & ~(q >= .5))) | adds, fo),
                     "new_adds": {"all": prec(na), "q>=.5": prec(na & (q >= .5)), "q<.5": prec(na & (q < .5)),
                                  "unscored": prec(na & np.isnan(q))},
                     "new_vetoes": {"pairs": int(nv.sum()), "true_share": round(float(y[nv].mean()), 3) if nv.any() else None,
                                    "q>=.5": prec(nv & (q >= .5)), "q<.5": prec(nv & (q < .5))},
                     "old_vs_P55": L.cmp(z, dec, old, L.F(z, dec))}
        print(name, json.dumps(out[name]), f"{time.time() - t0:.0f}s", flush=True)
    (HERE / f"inc79_{fs}.json").write_text(json.dumps(out, indent=1))
    g = {n: out[n]["new_minus_old"]["dF"] for n in out}
    log("exp", probe="P79-p73-takeover", subproblem="france-combiner", name=f"p79-inc-over-67q-{fs}", world="holdout-loco",
        metrics={f"{n}|{k}": out[n][k]["dF"] for n in out for k in ("new_minus_old", "veto_part", "add_part", "new_qgated", "new_qgated_qprot", "new_qprot")} |
                {f"{n}|new_adds_prec": out[n]["new_adds"]["all"]["prec"] for n in out} |
                {f"{n}|new_adds_q<.5_prec": out[n]["new_adds"]["q<.5"]["prec"] for n in out},
        config={"jobs": jobs, "arm": fs, "old": "P55 veto + P67 adds > .95 Qwen-gated (67q)", "new": "old - arm vetoes + arm adds"},
        hypothesis=f"{fs}'s change set adds France-F0.5 on top of the 67q state already in sub77",
        verdict="keep" if g["India->US"] > 0 and g["US->India"] > 0 else "drop", runtime_s=round(time.time() - t0))
    print("INC", json.dumps(g))


if __name__ == "__main__":
    main(*sys.argv[1:])
