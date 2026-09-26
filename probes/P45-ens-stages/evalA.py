"""P45 stage A (pruner): a pruner zoo + combiners at P10's operating point, on the holdout.

Rows = data/work/p10/hold_<c>.parquet (every blocked pair of a fold-0 S1 plus every competing pair of the same SX; P10's
30 pruner features, m2_big p). Keep rule = combiner score >= tau | rule_acc (P10's guard). Every combiner's tau is
set on holdout half 0 so its fold-0 kept count equals the shipped pruner's at tau 0.03 (equal size), or its kept TRUE
count (equal recall); half 1 reports:
  size    = kept fold-0 pairs / fold-0 S1 (P10: 4.30 holdout, 4.83 test)
  recall  = kept true pairs / true pairs in the blocking list
  F0.5    = paired per-S1 vs the shipped pruner, two decisions:
            m2  : argmax S1 per SX of m2_big p over the kept pairs, accept p >= 0.69 (P10's t; symmetric in both arms)
            p34 : as m2, but a pair of #6's pruned set carries its shipped final p (#34 p' on argmax pairs, #8c p else), t .71
    evalA.py score <job>    # the pruner job's hold_q files (members + P10 ship / big) -> data/work/p45/hq_<c>.parquet
    evalA.py eval           # combiners -> table + ledger rows
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P34-a10-stack"), str(ROOT)]
W = ROOT / "data/work"
OUT = W / "p45"
C = ("India", "US")
TAU = 0.03
LB_W = {"India": 0.47, "US": 0.38}
lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))


def score(job: str):
    """hold_<c> ids / labels / rule_acc / m2 p + the pruner job's hold_q_<c> (row-aligned, P10 prep.hold order; its
    'ship' column must equal P10's hold_q) -> data/work/p45/hq_<c>.parquet."""
    import subprocess
    for c in C:
        f = OUT / f"hold_q_{c}.parquet"
        subprocess.run(["aws", "s3", "cp", "--only-show-errors", "--profile", "harsh",
                        f"s3://sagemaker-114872709824-ap-south-1/amc26-p45/results/{job}/{f.name}", str(f)], check=True)
        d = pd.read_parquet(W / f"p10/hold_{c}.parquet", columns=["q1", "qx", "fold", "is_true", "rule_acc", "p"])
        d["rule_acc"] = d.rule_acc.to_numpy() > 0
        q = pd.read_parquet(f)
        assert len(q) == len(d) and np.array_equal(q.ship.to_numpy(), pd.read_parquet(W / f"p10/hold_q_{c}.parquet").q.to_numpy())
        pd.concat([d, q], axis=1).to_parquet(OUT / f"hq_{c}.parquet", index=False)
        f.unlink()
        print(c, len(d), list(q.columns), flush=True)


# ---------------------------------------------------------------------------------------------------------------- eval
def frame(c: str) -> dict:
    import features as F
    import ho as H
    d = pd.read_parquet(OUT / f"hq_{c}.parquet")
    s1, sx = F.ids("train", c)
    ctx = pd.read_parquet(W / f"p23/s1ctx_{c}.parquet", columns=["n_true"])
    ids = ctx.index.to_numpy()
    pos = pd.Series(np.arange(len(ids)), index=ids)
    s1ids = s1.to_numpy(zero_copy_only=False)
    g = pos.reindex(s1ids[d.q1.to_numpy()]).to_numpy()              # scored-S1 index, NaN for competitor S1
    f0 = d.fold.to_numpy() == 0
    assert not np.isnan(g[f0]).any(), "every fold-0 row's S1 is a scored S1"
    g = np.where(np.isnan(g), -1, g).astype(np.int64)
    # #6's pruned rows carry their shipped final p: #34 p' (h0 fit, guard) on its argmax pairs, #8c p elsewhere
    k = pq.read_table(W / f"p9/m2_holdout_{c}_d2b6569f_6a23c2ec_0.03.parquet", columns=["q1", "qx"])
    key6 = (k.column("q1").to_numpy().astype(np.int64) << 32) | k.column("qx").to_numpy()
    pp = pd.read_parquet(W / f"p23/pairs_{c}.parquet", columns=["s1", "sx", "p"])
    ho = pd.read_parquet(W / f"p34/ho_{c}.parquet", columns=["s1", "sx"])
    ho["p34"] = np.load(W / f"p34/ho_p_{c}_h0.npy")
    pfin = pp.merge(ho, on=["s1", "sx"], how="left").p34.fillna(pp.p).to_numpy()
    assert len(pfin) == len(pp)
    key = (d.q1.to_numpy().astype(np.int64) << 32) | d.qx.to_numpy()
    fin = pd.Series(pfin, index=key6).reindex(key).to_numpy()        # NaN = outside #6's pruned set
    return dict(c=c, d=d, g=g, f0=f0, n=len(ids), n_true=ctx.n_true.to_numpy(), h=H.half(ids).astype(np.int64), fin=fin,
                sxg=pd.factorize(d.qx.to_numpy())[0])


def decide(s: dict, keep: np.ndarray, p: np.ndarray, t: float) -> np.ndarray:
    """Argmax S1 per SX over kept rows (ties -> first), accept p >= t; returns per-S1 F0.5 of the scored S1."""
    from a10 import fb
    pk = np.where(keep, p, -1.0)
    o = np.lexsort((-pk, s["sxg"]))
    first = np.r_[True, s["sxg"][o][1:] != s["sxg"][o][:-1]]
    am = np.zeros(len(p), bool)
    am[o[first]] = True
    acc = am & keep & (p >= t) & (s["g"] >= 0)
    y = s["d"].is_true.to_numpy()
    return fb(np.bincount(s["g"][acc & y], minlength=s["n"]), np.bincount(s["g"][acc], minlength=s["n"]), s["n_true"])


def evaluate():
    from sklearn.linear_model import LogisticRegression
    import os
    from ledger import log
    if os.environ.get("P45_NOLOG"):                     # code checks stay out of the ledger
        log = lambda *a, **k: None
    S = [frame(c) for c in C]
    mem = [m for m in S[0]["d"].columns if m not in ("q1", "qx", "fold", "is_true", "rule_acc", "p")]
    print("members:", mem, flush=True)
    # member error correlation on fold-0 pairs near the cut (ship q in [0.005, 0.2]), pooled
    near = pd.concat([s["d"][s["f0"] & (s["d"].ship > 0.005) & (s["d"].ship < 0.2)] for s in S])
    E = pd.DataFrame({m: near.is_true.to_numpy() - near[m].to_numpy() for m in mem})
    print("error corr near the cut (pooled):\n" + E.corr().round(3).to_string(), flush=True)
    res = {"err_corr_near_cut": E.corr().round(3).to_dict()}
    del near, E
    for s in S:                                         # one float32 logit matrix per country; raw member q dropped
        s["L"] = lg(s["d"][mem].to_numpy(np.float32)).astype(np.float32)
        s["d"] = s["d"].drop(columns=[m for m in mem if m != "ship"])
    col = {m: i for i, m in enumerate(mem)}
    for fh in (0, 1):
        rep = 1 - fh
        rowh = [np.where(s["g"] >= 0, s["h"][np.maximum(s["g"], 0)], -1) for s in S]
        fit = [s["f0"] & (r == fh) for s, r in zip(S, rowh)]
        # LR stacker on member logits, fold-0 pairs of the fit half, pooled
        X = np.vstack([s["L"][m] for s, m in zip(S, fit)])
        y = np.concatenate([s["d"].is_true.to_numpy()[m] for s, m in zip(S, fit)])
        lr = LogisticRegression(C=1.0, max_iter=3000).fit(X, y)
        others = [m for m in mem if m not in ("ship", "big")]
        sub = lambda L, ms: L[:, [col[m] for m in ms]]              # all combiners work in logit space (monotone)
        combos = {m: (lambda m: lambda L: L[:, col[m]])(m) for m in mem}
        combos["mean_all"] = lambda L: L.mean(1)
        if others:
            combos["mean_zoo"] = lambda L: sub(L, ["ship"] + others).mean(1)
            data = [m for m in ("f45", "f67", "f89") if m in others]
            if data:
                combos["mean_data"] = lambda L: sub(L, ["ship"] + data).mean(1)
            combos["max_zoo"] = lambda L: sub(L, ["ship"] + others).max(1)
        combos["lr"] = lambda L: lr.decision_function(L)
        base_keep = [(s["d"].ship.to_numpy() >= TAU) | s["d"].rule_acc.to_numpy() for s in S]
        n_base = sum(int((k & m).sum()) for k, m in zip(base_keep, fit))
        nt_base = sum(int((k & m & s["d"].is_true.to_numpy()).sum()) for k, m, s in zip(base_keep, fit, S))
        f_base = {k2: [decide(s, k, s["d"].p.to_numpy() if k2 == "m2" else np.where(np.isnan(s["fin"]), s["d"].p.to_numpy(), s["fin"]),
                              0.69 if k2 == "m2" else 0.71) for s, k in zip(S, base_keep)] for k2 in ("m2", "p34")}
        for name, fn in combos.items():
            sc = [fn(s["L"]) for s in S]
            ra = [s["d"].rule_acc.to_numpy() for s in S]
            cand = np.concatenate([x[m & ~r] for x, m, r in zip(sc, fit, ra)])
            candt = np.concatenate([x[m & ~r & s["d"].is_true.to_numpy()] for x, m, r, s in zip(sc, fit, ra, S)])
            n_ra = sum(int((m & r).sum()) for m, r in zip(fit, ra))
            nt_ra = sum(int((m & r & s["d"].is_true.to_numpy()).sum()) for m, r, s in zip(fit, ra, S))
            for mode in ("size", "recall"):
                if mode == "size":
                    tau = np.sort(cand)[::-1][n_base - n_ra - 1]
                else:
                    tau = np.sort(candt)[::-1][nt_base - nt_ra - 1]
                keep = [(x >= tau) | r for x, r in zip(sc, ra)]
                o = dict(tau=float(tau))
                for s, k, r, bk in zip(S, keep, rowh, base_keep):
                    m = s["f0"] & (r == rep)
                    yt = s["d"].is_true.to_numpy()
                    ns1 = int((s["h"] == rep).sum())
                    o[s["c"]] = dict(size=float((k & m).sum() / ns1), size_base=float((bk & m).sum() / ns1),
                                     recall=float((k & m & yt).sum() / (m & yt).sum()),
                                     recall_base=float((bk & m & yt).sum() / (m & yt).sum()))
                    for k2 in ("m2", "p34"):
                        p = s["d"].p.to_numpy() if k2 == "m2" else np.where(np.isnan(s["fin"]), s["d"].p.to_numpy(), s["fin"])
                        dd = (decide(s, k, p, 0.69 if k2 == "m2" else 0.71) - f_base[k2][S.index(s)])[s["h"] == rep]
                        o[s["c"]][k2] = dict(d=float(dd.mean()), se=float(dd.std(ddof=1) / np.sqrt(len(dd))))
                for k2 in ("m2", "p34"):
                    o[f"lb_{k2}"] = dict(d=sum(LB_W[c] * o[c][k2]["d"] for c in C),
                                         se=float(np.sqrt(sum((LB_W[c] * o[c][k2]["se"]) ** 2 for c in C))))
                o["size"] = sum(o[c]["size"] * (s["h"] == rep).sum() for c, s in zip(C, S)) / sum((s["h"] == rep).sum() for s in S)
                o["size_base"] = sum(o[c]["size_base"] * (s["h"] == rep).sum() for c, s in zip(C, S)) / sum((s["h"] == rep).sum() for s in S)
                res[f"h{fh}/{mode}/{name}"] = o
                print(f"h{fh} {mode:6s} {name:9s} size {o['size']:.3f} (ship {o['size_base']:.3f}) | recall In {o['India']['recall']:.5f} "
                      f"US {o['US']['recall']:.5f} (ship {o['India']['recall_base']:.5f} / {o['US']['recall_base']:.5f}) | "
                      f"F m2 {o['lb_m2']['d']:+.6f} ± {o['lb_m2']['se']:.6f} | F p34 {o['lb_p34']['d']:+.6f} ± {o['lb_p34']['se']:.6f}", flush=True)
    json.dump(res, open(OUT / "evalA.json", "w"), indent=1, default=float)
    for name in {k.split("/")[2] for k in res if k.startswith("h0/")}:
        o, o1 = res[f"h0/size/{name}"], res.get(f"h1/size/{name}", {})
        r = res[f"h0/recall/{name}"]
        log("exp", probe="P45-ens-stages", subproblem="pruner-zoo", name=f"A-{name}", world="holdout",
            metrics={"size_h1": round(o["size"], 4), "size_ship_h1": round(o["size_base"], 4),
                     **{f"recall_{c}_h1": round(o[c]["recall"], 5) for c in C},
                     "lb_m2_eqsize_h0h1": round(o["lb_m2"]["d"], 6), "se_m2": round(o["lb_m2"]["se"], 6),
                     "lb_p34_eqsize_h0h1": round(o["lb_p34"]["d"], 6), "se_p34": round(o["lb_p34"]["se"], 6),
                     "lb_p34_eqsize_h1h0": round(o1.get("lb_p34", {}).get("d", np.nan), 6),
                     "size_eqrecall_h1": round(r["size"], 4)},
            config={"combiner": name, "members": mem, "tau_matched_on": "half 0 fold-0 kept count (size) / kept true (recall)"},
            hypothesis="a pruner ensemble beats P10's pruner at equal candidate-set size (recall / F0.5) or is smaller at equal recall",
            verdict="keep" if o["lb_p34"]["d"] >= 3 * o["lb_p34"]["se"] and o["lb_p34"]["d"] > 0 and o1.get("lb_p34", {}).get("d", -1) > 0
            else "drop", runtime_s=None, peak_gb=None)


if __name__ == "__main__":
    score(sys.argv[2]) if sys.argv[1] == "score" else evaluate()
