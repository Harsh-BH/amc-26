"""P73 veto arm (user, 27 Sep 19:00): the France-independent CE scores France's CONFIDENT accepted pairs (g >= .98, never
seen by an India/US-free model) and vetoes the ones it rejects. Gate = LOCO on P61's frames, both directions:
  incumbent sel0 = P55 bag + (p67, ele4n) columns at the shipped cell (= sub79q's recipe)
  veto          = sel0 rows with g >= .98 whose seed-averaged independent logit r < tau  (r from the fc_<frame> scores)
  tau is picked on the SOURCE frame (max dF there) and applied to the TARGET frame; report dF vs sel0 with per-S1 se.
    .venv/bin/python probes/P73-fr-independent/vt73.py gate JOB [JOB ...]     # the p73s-* seed jobs
    .venv/bin/python probes/P73-fr-independent/vt73.py build TAU JOB [JOB ...] # France: sub79q minus vetoed pairs -> output/sub81v
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT)]
import ev73  # noqa: E402  (P55 / P61 on sys.path)

W = ROOT / "data/work/p73"
INC = ["p67-bge-loco-0927-1405", "p73b-both"]           # incumbent columns (p67 + ele4n)
FB = {"US_India": "fc_US_India", "India_US": "fc_India_US", "France": "fc_France"}
TAUS = np.round(np.arange(-8.0, 2.01, 0.5), 2)


def fc_logit(jobs, frame, n):
    """Mean logit over every seed's r_ind_*_fc_<frame>.parquet (s1 = frame row); NaN where unscored."""
    acc, cnt = np.zeros(n), np.zeros(n)
    for j in jobs:
        for f in (W / "r" / j).rglob(f"r_ind_*_{FB[frame]}.parquet"):
            d = pd.read_parquet(f)
            r = np.clip(d.r.to_numpy(np.float64), 1e-6, 1 - 1e-6)
            i = d.s1.to_numpy(np.int64)
            acc[i] += np.log(r / (1 - r)); cnt[i] += 1
    out = np.full(n, np.nan)
    out[cnt > 0] = acc[cnt > 0] / cnt[cnt > 0]
    return out, int(cnt.max())


def gate(*jobs):
    import p55
    import p61
    for j in jobs:
        ev73.fetch(j)
    L = p55.L
    fr = {dr: p61.load(dr) for dr in ("US_India", "India_US")}
    inc = {dr: pd.DataFrame(ev73.columns(INC, dr, len(fr[dr]["X"]))) for dr in fr}
    sel0, rv, curves = {}, {}, {}
    for name, trn, tst in p55.DIRN:                     # incumbent selection on each TARGET frame
        a, b = fr[trn], fr[tst]
        bs = p55.bag(ev73.design(a["X"], inc[trn], "p67+ele4n"), a["z"]["y"], None)
        sel0[tst] = p61.arm_a(b, p55.bag_score(bs, ev73.design(b["X"], inc[tst], "p67+ele4n"), None))
    for dr, b in fr.items():
        r, ns = fc_logit(jobs, dr, len(b["X"]))
        rv[dr] = r
        conf = sel0[dr] & (b["z"]["g"] >= 0.98) & ~np.isnan(r)
        f0 = L.F(b["z"], sel0[dr])
        cur = {}
        for t in TAUS:
            veto = conf & (r < t)
            cur[float(t)] = {**L.cmp(b["z"], sel0[dr], sel0[dr] & ~veto, f0), "veto": int(veto.sum()),
                             "veto_true": round(float(b["z"]["y"][veto].mean()), 3) if veto.any() else None}
        curves[dr] = cur
        print(dr, f"seeds {ns}, confident accepted scored {int(conf.sum()):,}", flush=True)
    out = {}
    for name, trn, tst in p55.DIRN:                     # tau from the source frame, applied to the target
        t = max(curves[trn], key=lambda k: curves[trn][k]["dF"])
        out[name] = {"tau_from_source": t, "source_dF": curves[trn][t]["dF"], "target": curves[tst][t]}
        print(name, json.dumps(out[name]), flush=True)
    ok = all(out[n]["target"]["dF"] > 0 and out[n]["target"]["dF_h0"] > 0 and out[n]["target"]["dF_h1"] > 0 for n in out)
    out["pass"] = ok
    out["curves"] = curves
    (HERE / "vt73.json").write_text(json.dumps(out, indent=1))
    from ledger import log
    log("exp", probe="P73-fr-independent", subproblem="france-veto", name="p73-veto-confident-indep-ce", world="holdout-loco",
        metrics={f"{n}|{k}": out[n]["target"][k] for n in ("India->US", "US->India") for k in ("dF", "se", "veto", "veto_true")}
        | {f"{n}|tau": out[n]["tau_from_source"] for n in ("India->US", "US->India")},
        config={"jobs": list(jobs), "incumbent": "P55 + p67 + ele4n shipped cell", "veto": "g>=.98 accepted & seed-mean logit < tau",
                "tau": "argmax dF on the source frame"},
        hypothesis="a CE free of India/US labels spots decoys among the unseen country's confident accepted pairs",
        verdict="keep" if ok else "drop", runtime_s=0, peak_gb=0)
    print("PASS" if ok else "FAIL")


def build(tau, *jobs):
    """France: output/sub79q minus its France pairs with g >= .98 and seed-mean independent logit < tau -> output/sub81v."""
    import shutil
    for j in jobs:
        ev73.fetch(j)
    f = pd.read_parquet(ROOT / "data/work/p65/sm/frame_France.parquet", columns=["row", "s1", "sx", "g"])
    r, ns = fc_logit(jobs, "France", len(f))
    v = f[(f.g.to_numpy() >= 0.98) & (r < float(tau))]
    veto = set(zip(v.s1, v.sx))
    src, out = ROOT / "output/sub79q", ROOT / "output/sub81v"
    out.mkdir(exist_ok=True)
    n = 0
    with open(src / "matching_results.tsv") as fi, open(out / "matching_results.tsv", "w") as fo:
        fo.write(next(fi))
        for line in fi:
            a, _, b = line.rstrip("\n").partition("\t")
            ids = [x for x in b.split(",") if x]
            keep = [x for x in ids if (a, x) not in veto]
            n += len(ids) - len(keep)
            fo.write(a + "\t" + ",".join(keep) + "\n")
    cp = out / "candidate_pairs.tsv"
    if not cp.exists():
        shutil.copyfile(src / "candidate_pairs.tsv", cp)
    print(json.dumps({"tau": float(tau), "seeds": ns, "veto_candidates": len(veto), "removed_from_sub79q": n}))


if __name__ == "__main__":
    {"gate": lambda: gate(*sys.argv[2:]), "build": lambda: build(*sys.argv[2:])}[sys.argv[1]]()
