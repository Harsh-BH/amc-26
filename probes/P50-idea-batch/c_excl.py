"""P50 C: exclusivity veto for France. other = the SX's p mass over its OTHER candidate S1 (raw GBDT p: P6's source-only
M2 in LOCO, c_hn's g on France; P44 sx_psum minus own p). Rule: drop an accepted pair when other >= m and its p is below
the France bar t_hi (LOCO: density-mapped, P42 fr_to_loco). Evaluated on P47's LOCO frames (P42 recipe analogue,
t .99 / rescue .9 / gc .7 / hn .8, veto .1), both analogues and directions; France: label-free removal counts on sub44r.
    .venv/bin/python probes/P50-idea-batch/c_excl.py"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(".").resolve()
sys.path[:0] = [str(ROOT / "probes/P47-fr-shape-match"), str(ROOT / "probes/P42-fr-joint-tune"), str(ROOT)]
import p47  # noqa: E402
import solution_p42 as S  # noqa: E402

CI = {"India": 0, "US": 1}
MS = (0.1, 0.3, 0.5, 0.8)
TH = (0.8, 0.9, 0.95, 0.97, 0.99, None)          # None = any p


def other_mass(sx, g):
    tot = np.bincount(sx, weights=g)
    return tot[sx] - g


def loco_other(dr, g_frame):
    src, tgt = dr.split("_")
    d = pd.read_parquet(f"data/work/p27/p6/S_holdout_{src}.parquet", columns=["sx", "country", "p", "am_all"])
    d = d[d.country.to_numpy() == CI[tgt]].reset_index(drop=True)
    sx = pd.factorize(d.sx)[0]
    o = other_mass(sx, d.p.to_numpy(np.float64))[d.am_all.to_numpy()]
    assert len(o) == len(g_frame) and (d.p.to_numpy(np.float32)[d.am_all.to_numpy()] == g_frame.astype(np.float32)).all()
    return o


if __name__ == "__main__":
    from ledger import log
    t0 = time.time()
    res = []
    for an in ("chain", "m2"):
        for dr in ("US_India", "India_US"):
            fr = p47.loco_frame(an, dr)
            z = S._frame(ROOT / "data/work/p42", dr)
            o = loco_other(dr, z["g"])
            s1, n, y, p, pred = fr["s1"], fr["n"], fr["y"], fr["p"], fr["pred"]
            print(f"== {an} {dr}: pred {int(pred.sum()):,}, prec {y[pred].mean():.4f}; other>=.3 among pred {(pred & (o >= .3)).mean() / pred.mean():.4f}", flush=True)
            for m in MS:
                for th in TH:
                    bar = np.inf if th is None else fr["tm"][str(th)]
                    rm = pred & (o >= m) & (p < bar)
                    new = pred & ~rm
                    f = S.fbeta(np.bincount(s1[new & y], minlength=n), np.bincount(s1[new], minlength=n), fr["ntrue"])
                    dd = f - fr["f"]
                    r = dict(an=an, dir=dr, m=m, t_hi=th, removed=int(rm.sum()), prec_removed=round(float(y[rm].mean()), 4) if rm.any() else None,
                             dF=round(float(dd.mean()), 6), se=round(float(dd.std() / np.sqrt(n)), 6),
                             dF_h0=round(float(dd[fr["hv"] == 0].mean()), 6), dF_h1=round(float(dd[fr["hv"] == 1].mean()), 6))
                    res.append(r)
                    print("  ", json.dumps({k: v for k, v in r.items() if k not in ("an", "dir")}), flush=True)
            del fr, z, o
    # France, label-free: what each rule would remove from sub44r's France lists
    st = p47.fr_state("sub44r")
    o = other_mass(st["sx"], st["g"])
    fr_counts = {}
    for m in MS:
        for th in TH:
            bar = np.inf if th is None else th
            rm = st["pred"] & (o >= m) & (st["p"] < bar)
            fr_counts[f"m{m}_t{th}"] = {"removed": int(rm.sum()), "s1": int(len(np.unique(st["s1"][rm]))),
                                       "per_pred": round(float(rm.sum() / st["pred"].sum()), 5)}
    print("France", json.dumps(fr_counts), flush=True)
    json.dump({"loco": res, "france": fr_counts}, open("data/work/p50/c_excl.json", "w"), indent=0)
    print(f"{time.time() - t0:.0f}s", flush=True)
