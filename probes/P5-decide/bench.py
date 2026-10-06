"""Benchmark decision strategies on out-of-fold pair probabilities, dense holdout: tune on hash half 0 of the
holdout S1, report on half 1. Every holdout S1 counts -- one with no candidates scores 1 if a singleton, else 0.

    .venv/bin/python probes/P5-decide/bench.py proxy      # data/work/p5/prob_proxy.parquet
    .venv/bin/python probes/P5-decide/bench.py p4         # data/work/p4/prob_holdout.parquet
    .venv/bin/python probes/P5-decide/bench.py blend      # data/work/p4b/prob_holdout_blend.parquet
"""
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1]), str(HERE.parents[1] / "code/business_entity_resolution/src")]
from decide import select, decide
from proxy import half, peak
from worlds import WORK

SRC = {"proxy": WORK / "p5/prob_proxy.parquet", "p4": WORK / "p4/prob_holdout.parquet",
       "blend": WORK / "p4b/prob_holdout_blend.parquet"}      # MERGE-2: P4 M2 + P4b cross-encoder blend


def load(src: str):
    s1 = pd.read_parquet(WORK / "worlds/holdout/s1.parquet", columns=["entity_id", "country"])
    ids = pd.Index(s1.entity_id)
    tr = pd.read_parquet(WORK / "worlds/holdout/truth.parquet")
    nt = np.bincount(ids.get_indexer(tr.s1), minlength=len(ids))
    cols = pq.read_schema(SRC[src]).names
    d = pd.read_parquet(SRC[src], columns=["s1", "sx", "p", "is_true"] + [c for c in ("rule_acc", "am") if c in cols])
    d["s1"] = ids.get_indexer(d.s1)
    assert (d.s1 >= 0).all(), "every scored S1 is a holdout S1"
    d["sx"] = pd.factorize(d.sx)[0]
    d["country"] = s1.country.to_numpy()[d.s1]
    S = pd.DataFrame({"country": s1.country.to_numpy(), "nt": nt, "half": half(s1.entity_id)})
    return d, S


def per_s1(d, S, sel):
    n = len(S)
    tp = np.bincount(d.s1.to_numpy()[sel & d.is_true.to_numpy()], minlength=n)
    npred = np.bincount(d.s1.to_numpy()[sel], minlength=n)
    nt = S.nt.to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(nt == 0, (npred == 0).astype(float), (1 + 0.25) * tp / (0.25 * nt + npred))
    return f, tp, npred


def metrics(d, S, sel, h):
    f, tp, npred = per_s1(d, S, sel)
    m = S.half.to_numpy() == h
    nt = S.nt.to_numpy()
    out = {"f05": f[m].mean(), "singleton_acc": f[m & (nt == 0)].mean(),
           "pair_precision": tp[m].sum() / max(npred[m].sum(), 1), "pair_recall": tp[m].sum() / nt[m].sum(),
           "pred_per_s1": npred[m].mean()}
    for c in sorted(S.country.unique()):
        out[f"f05_{c}"] = f[m & (S.country.to_numpy() == c)].mean()
    return out


def sweep(d, S, base: dict, grid: dict, h: int = 0):
    """Exhaustive over the grid on half h; returns (best cfg, best f05)."""
    keys, best = list(grid), (None, -1)
    for vals in np.array(np.meshgrid(*grid.values(), indexing="ij")).reshape(len(keys), -1).T:
        cfg = {**base, **{k: float(v) for k, v in zip(keys, vals)}}
        f = metrics(d, S, select(d, cfg), h)["f05"]
        if f > best[1]:
            best = (cfg, f)
    return best


def iso(d, S, cfg_excl: bool):
    """Isotonic p -> P(true) fitted on the tune half (on the exclusivity survivors if excl)."""
    from sklearn.isotonic import IsotonicRegression
    m = S.half.to_numpy()[d.s1.to_numpy()] == 0
    if cfg_excl:
        m &= select(d, {"excl": True, "t": 0.0})
    r = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(d.p.to_numpy()[m], d.is_true.to_numpy()[m])
    return [r.X_thresholds_.tolist(), r.y_thresholds_.tolist()]


if __name__ == "__main__":
    src = sys.argv[1]
    sys.path.insert(0, str(HERE.parents[1]))
    from ledger import log
    t0 = time.time()
    d, S = load(src)
    print(f"{len(d):,} pairs, {len(S):,} S1, peak {peak():.2f} GB", flush=True)
    res = {}

    def report(name, cfg, hyp, extra=None):
        tune, rep = metrics(d, S, select(d, cfg), 0), metrics(d, S, select(d, cfg), 1)
        res[name] = (cfg, tune["f05"], rep)
        print(f"{name:28s} tune {tune['f05']:.4f}  report {rep['f05']:.4f}  "
              + "  ".join(f"{k}={v:.4f}" for k, v in rep.items() if k != "f05"), flush=True)
        log("exp", probe="P5-decide", subproblem="decision", name=f"{src}-{name}", world="holdout-dense-half1",
            metrics={"f05_tune": round(tune["f05"], 4), **{k: round(float(v), 4) for k, v in rep.items()}},
            config={k: v for k, v in cfg.items() if k != "calib"} | ({"calib": "isotonic(tune half)"}
                                                                  if cfg.get("calib") else {}),
            hypothesis=hyp, verdict="inconclusive", runtime_s=round(time.time() - t0), peak_gb=round(peak(), 2))

    if "rule_acc" in d:                       # the rules-only answer on the same pairs and truth
        tune, rep = (metrics(d, S, d.rule_acc.to_numpy(bool), h) for h in (0, 1))
        res["R-rules"] = ({"rules": True}, tune["f05"], rep)
        print(f"{'R-rules':28s} tune {tune['f05']:.4f}  report {rep['f05']:.4f}  "
              + "  ".join(f"{k}={v:.4f}" for k, v in rep.items() if k != "f05"), flush=True)
    T = np.round(np.arange(0.05, 0.96, 0.05), 2)
    c, _ = sweep(d, S, {}, {"t": T})
    c, _ = sweep(d, S, {}, {"t": np.clip(c["t"] + np.arange(-0.04, 0.041, 0.01), 0.01, 0.99)})
    report("D0-global-t", c, "a single global threshold is the reference")
    c1, _ = sweep(d, S, {"excl": True}, {"t": T})
    c1, _ = sweep(d, S, {"excl": True}, {"t": np.clip(c1["t"] + np.arange(-0.04, 0.041, 0.01), 0.01, 0.99)})
    report("D1-excl+t", c1, "each SX only to its argmax S1 removes wrong claims on shared SX")
    c3, _ = sweep(d, S, {"excl": True}, {"t": T, "t0": T})
    c3, _ = sweep(d, S, {"excl": True}, {"t": np.clip(c3["t"] + np.arange(-0.03, 0.031, 0.01), 0.01, 0.99),
                                         "t0": np.clip(c3["t0"] + np.arange(-0.03, 0.031, 0.01), 0.01, 0.99)})
    report("D3-excl+t+gate", c3, "separate thresholds for 'S1 has a match' and 'add another pair'")
    cr, _ = sweep(d, S, {k: v for k, v in c3.items()}, {"rel": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})
    report("D3r-excl+t+gate+rel", cr, "relative threshold to the S1's best pair")
    tc, t0c = {}, {}
    for cn in sorted(S.country.unique()):
        # other countries' S1 have no pairs in dm: a constant offset, so the argmax is the per-country optimum
        cc, _ = sweep(d[d.country == cn], S, {"excl": True},
                      {"t": np.clip(c3["t"] + np.arange(-0.1, 0.101, 0.02), 0.01, 0.99),
                       "t0": np.clip(c3["t0"] + np.arange(-0.1, 0.101, 0.02), 0.01, 0.99)})
        tc[cn], t0c[cn] = cc["t"], cc["t0"]
    print("per-country t", tc, "t0", t0c)
    report("D4-per-country", {**c3, "t_country": tc}, "per-country t (t0 global); unseen countries use global t")
    cm, _ = sweep(d, S, c3, {"m": [1, 2, 3, 4, 5, 6, 8, 10]})
    report("D5-topm-cap", cm, "cap answer size per S1")
    for ex in (False, True):
        cal = iso(d, S, ex)
        for miss in (0.0, 0.02):
            report(f"D2-ef{'-excl' if ex else ''}-iso-miss{miss}",
                   {"excl": ex, "mode": "ef", "calib": cal, "miss": miss}, "expected-F0.5-optimal top-k per S1")
        report(f"D2-ef{'-excl' if ex else ''}-raw", {"excl": ex, "mode": "ef"}, "EF on raw (uncalibrated) p")
    cg, _ = sweep(d, S, {"excl": True, "mode": "ef"}, {"t0": np.round(np.arange(0.3, 0.86, 0.05), 2)})
    report("D2g-ef-gate-raw", cg, "EF picks k>=1 once a gate on the best p takes the empty decision")
    best = max(res, key=lambda k: res[k][1])
    print("best on tune:", best, json.dumps({k: v for k, v in res[best][0].items() if k != "calib"}))
    json.dump({k: [v[0], v[1], v[2]] for k, v in res.items()}, open(WORK / f"p5/bench_{src}.json", "w"))
    print(f"done in {time.time() - t0:.0f}s, peak {peak():.2f} GB")
