"""P54 idea 1: France in-band CE rescue on LOCO (mirror of P48's veto). After the current recipe (t .99 / r .9 / gc .7 /
hn .8, tau .1, caps 5/6), add free (argmax, non-sibling, not capped out) unaccepted pairs with g > 0.02 and p < t whose
source-only bge r >= u, best r first, up to the caps. Region: band = 0.02 < g < 0.98 (p = OOD blend), hi = g >= 0.98
and p < t; scope: any / empty (best one) / onesided (missing source) / nonempty. Chain and M2-only analogues, both
directions; u picked on h0, reported on h1.

    probes/P54-fresh-batch/py.sh probes/P54-fresh-batch/loco54.py      # -> probes/P54-fresh-batch/loco54.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "probes/P48-fr-ce-all")]
import loco48 as L  # noqa: E402

S = L.S
US = [0.8, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99]


def main():
    t0 = time.time()
    out = {}
    for dr in L.DIRS:
        z = L.frame(dr)
        g, y = z["g"], z["y"]
        s1, is2, n = z["s1"].astype(np.int64), z["is2"], len(z["ntrue"])
        r = np.nan_to_num(z["r"], nan=0.0)
        for an in ("chain", "m2"):
            p = S.loco_p(g, z["r"], L.CUR["tau"], an)
            st, tm = L.recipe(z, dr, p, an=an)
            ref, fref = st["hn"], L.F(z, st["hn"])
            free = L.free_mask(z, st, p, tm)
            t = tm[str(L.CUR["t"])]
            reg = {"band": (g > S.LO) & (g < S.HI), "hi": (g >= S.HI) & (p < t)}
            reg["band+hi"] = reg["band"] | reg["hi"]
            n2, n3 = S.counts(s1, is2, ref, n)
            m2, m3 = n2[s1], n3[s1]
            scope = {"any": np.ones(len(g), bool), "empty": m2 + m3 == 0,
                     "onesided": np.where(is2, (m2 == 0) & (m3 > 0), (m3 == 0) & (m2 > 0)), "nonempty": m2 + m3 > 0}
            R = out[f"{an}:{dr}"] = {"t_map": round(t, 4), "ref_F": round(float(fref.mean()), 5),
                                     "pool": {k: [int((v & free & ~ref).sum()), round(float(y[v & free & ~ref].mean()), 3),
                                                  int((v & free & ~ref & (r >= 0.9)).sum()),
                                                  round(float(y[v & free & ~ref & (r >= 0.9)].mean()), 3) if (v & free & ~ref & (r >= 0.9)).any() else None]
                                              for k, v in reg.items()}}
            for rg, rm in reg.items():
                for sc, sm in scope.items():
                    for u in US:
                        c = np.flatnonzero(free & ~ref & rm & sm & (r >= u))
                        R[f"{rg}:{sc}:{u}"] = L.cmp(z, ref, L.add_capped(z, ref, c, r, m2, m3, one=sc == "empty"), fref)
            print(an, dr, R["t_map"], R["pool"], flush=True)
    # u picked on h0 (per analogue x region x scope, both directions separately), reported on h1
    best = {}
    for an in ("chain", "m2"):
        for rg in ("band", "hi", "band+hi"):
            for sc in ("any", "empty", "onesided", "nonempty"):
                row = {}
                for dr in L.DIRS:
                    R = out[f"{an}:{dr}"]
                    u = max(US, key=lambda x: R[f"{rg}:{sc}:{x}"]["dF_h0"])
                    v = R[f"{rg}:{sc}:{u}"]
                    row[dr] = {"u": u, "dF": v["dF"], "dF_h1": v["dF_h1"], "se_h1": v["se_h1"], "add": v["add"], "prec": v["add_prec"]}
                best[f"{an}:{rg}:{sc}"] = row
                print(an, rg, sc, row, flush=True)
    out["best_h0"] = best
    out["runtime_s"] = round(time.time() - t0)
    (HERE / "loco54.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__" and not sys.argv[1:]:
    main()


def stack():
    """The same rescue on top of P48's shipped veto (rescue+gc stage band pairs with r < 0.2 removed), i.e. on sub48's
    France recipe; add rate per accepted pair for the France comparison."""
    out = {}
    for dr in L.DIRS:
        z = L.frame(dr)
        g, y = z["g"], z["y"]
        s1, is2, n = z["s1"].astype(np.int64), z["is2"], len(z["ntrue"])
        r = np.nan_to_num(z["r"], nan=0.0)
        band = (g > S.LO) & (g < S.HI)
        for an in ("chain", "m2"):
            p = S.loco_p(g, z["r"], L.CUR["tau"], an)
            st, tm = L.recipe(z, dr, p, an=an)
            veto = st["gc"] & ~st["base"] & band & (np.nan_to_num(z["r"], nan=1.0) < 0.2)
            ref = st["hn"] & ~veto
            fref = L.F(z, ref)
            free = L.free_mask(z, st, p, tm)
            n2, n3 = S.counts(s1, is2, ref, n)
            m2, m3 = n2[s1], n3[s1]
            scope = {"any": np.ones(len(g), bool), "nonempty": m2 + m3 > 0, "empty": m2 + m3 == 0,
                     "onesided": np.where(is2, (m2 == 0) & (m3 > 0), (m3 == 0) & (m2 > 0))}
            R = out[f"{an}:{dr}"] = {"ref_pairs": int(ref.sum()), "veto": int(veto.sum()),
                                     "veto_dF_vs_recipe": L.cmp(z, st["hn"], ref, L.F(z, st["hn"]))["dF"]}
            for sc, sm in scope.items():
                for u in (0.95, 0.97, 0.98, 0.99, 0.995, 0.999):
                    c = np.flatnonzero(free & ~ref & band & sm & (r >= u))
                    v = L.cmp(z, ref, L.add_capped(z, ref, c, r, m2, m3, one=sc == "empty"), fref)
                    v["rate"] = round(v["add"] / R["ref_pairs"], 5)
                    R[f"{sc}:{u}"] = v
                    print(an, dr, sc, u, v, flush=True)
    (HERE / "loco54_stack.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__" and sys.argv[1:] == ["stack"]:
    stack()
