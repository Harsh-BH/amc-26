"""P26 Phase C (laptop, light): a Phase-C job's France p -> bge CE re-blend -> label-free profile -> France file.

    .venv/bin/python probes/P26-france-da/phase_c.py <tag> [<tag> ...]      # tag = data/work/p26/job_<tag>.txt
    .venv/bin/python probes/P26-france-da/phase_c.py 9ab                    # profile #9ab's France file (reference)

1. fetch s3 .../results/<job>/pred/test_France.parquet (q1, qx, p, p_m2; P14's IDF world, #6-pruner keys)
2. p' = rerank blend (P13's out-of-domain France weights [0.9245, 0.2595, 0.0515], band 0.02 < p < 0.98) with the bge r of
   P13 (r_bge_pool_test_France) + P19 (the IDF-world pairs P13 lacked); band pairs without r keep p (counted)
3. profile over all 259,452 France S1: t_eq_d4 (empty rate = D4's 6.85%), P6's rule t (empty = India/US's 5.75% at
   t* 0.71, clipped [0.71, 0.91]), matches per S1, and label-free hn diagnostics on the accepted pairs: share whose hn is
   equal (France siblings keep the hn only 2-6%, P20) and share at a sibling offset hx - h1 in {3,4,5,7,9,11,13,21};
   churn vs #9ab's France (P19's file at t 0.985)
-> data/work/p26/prob_test_France_<tag>_bge.parquet (s1, sx, p, p_gbdt) + data/work/p26/profile_<tag>.json
"""
import json, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "data/work"
OUT = W / "p26"
B = "s3://sagemaker-114872709824-ap-south-1/amc26-p26/results"
WOOD, LO, HI = [0.9245, 0.2595, 0.0515], 0.02, 0.98
N_S1, D4_EMPTY, REF_EMPTY, T_STAR = 259_452, 0.0685, 0.0575, 0.71
SIB3 = [3, 4, 5, 7, 9, 11, 13, 21]
REF9AB = W / "p19/prob_test_France_idf144_bge.parquet"


def lg(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def blend(p, r):
    b = (p > LO) & (p < HI) & ~np.isnan(r)
    out = p.copy()
    out[b] = 1 / (1 + np.exp(-(WOOD[0] * lg(p[b]) + WOOD[1] * lg(r[b]) + WOOD[2])))
    return out, b


def ids():
    t = W / "p14/w_idfx/p3/txt_test"
    s1 = pq.read_table(t / "France_s1.parquet", columns=["entity_id"]).column(0).combine_chunks()
    sx = pa.concat_arrays([pq.read_table(f, columns=["entity_id"]).column(0).combine_chunks()
                           for f in sorted(t.glob("France_sx_*.parquet"))])
    return s1, sx


def argmax(sx_codes, p):
    o = np.lexsort((-p, sx_codes))
    first = np.r_[True, sx_codes[o][1:] != sx_codes[o][:-1]]
    m = np.zeros(len(p), bool)
    m[o[first]] = True
    return m


def profile(d: pd.DataFrame, hn: pd.DataFrame, ref: pd.DataFrame | None, t_fixed=None) -> dict:
    q1 = pd.Categorical(d.s1).codes.astype(np.int64)
    qx = pd.factorize(d.sx)[0]
    p = d.p.to_numpy(np.float64)
    am = argmax(qx, p)
    top = np.full(N_S1, -1.0)
    np.maximum.at(top, q1[am], p[am])
    emp = lambda t: float((top < t).mean())
    grid = np.unique(np.r_[np.round(np.arange(0.50, 0.95, 0.005), 3), np.round(np.arange(0.95, 0.9981, 0.001), 3)])
    e = pd.Series([emp(t) for t in grid], index=grid)
    t_d4 = float((e - D4_EMPTY).abs().idxmin())
    eg = e[(e.index >= T_STAR) & (e.index <= T_STAR + 0.2)]
    t_p6 = float((eg - REF_EMPTY).abs().idxmin())
    pa_ = p[am]
    out = {"pairs": len(d), "band_03_09": float(((pa_ > 0.3) & (pa_ < 0.9)).mean()), "p_ge_095": float((pa_ >= 0.95).mean()),
           "t_eq_d4": t_d4, "empty_t_eq_d4": float(e[t_d4]), "t_p6": t_p6, "empty_t_p6": float(e[t_p6])}
    h = d[["s1", "sx"]].merge(hn, on=["s1", "sx"], how="left")
    both = (h.h1.to_numpy() >= 0) & (h.hx.to_numpy() >= 0)
    dd = h.hx.to_numpy() - h.h1.to_numpy()
    eq, sib = both & (dd == 0), both & np.isin(dd, SIB3)
    out["curve"] = {}
    for t in (0.9, 0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995):
        a = am & (p >= t)
        out["curve"][t] = {"empty": round(emp(t), 4), "mps": round(float(a.sum() / N_S1), 4),
                           "sib3_per_1k": round(float(1000 * sib[a].mean()), 3), "hn_eq_share": round(float(eq[a].mean()), 4)}
    for tag, t in (("d4", t_d4), ("p6", t_p6)) + ((("fixed", t_fixed),) if t_fixed else ()):
        a = am & (p >= t)
        out[f"mps_{tag}"] = float(a.sum() / N_S1)
        out[f"acc_hn_eq_share_{tag}"] = float(eq[a].mean())
        out[f"acc_sib3_per_1k_{tag}"] = float(1000 * sib[a].mean())
        out[f"hn_eq_accepted_rate_{tag}"] = float(a[am & eq].mean())        # of argmax hn-equal pairs, share accepted
        out[f"sib3_accepted_rate_{tag}"] = float(a[am & sib].mean())
        if ref is not None:
            A = set(zip(d.s1.to_numpy()[a], d.sx.to_numpy()[a]))
            out[f"pairs_only_new_{tag}"], out[f"pairs_only_9ab_{tag}"] = len(A - ref.attrs["acc"]), len(ref.attrs["acc"] - A)
            s_new = pd.Series(1, index=pd.MultiIndex.from_tuples(list(A))).groupby(level=0).size() if A else pd.Series(dtype=int)
            ch = {s for s, _ in A ^ ref.attrs["acc"]}
            out[f"s1_changed_vs_9ab_{tag}"] = len(ch)
    return out


def hn_table():
    s1, sx = ids()
    M = np.load(OUT / "sm/meta/test_France.npz")
    return s1, sx, M


def main(tag: str):
    t0 = time.time()
    s1, sx, M = hn_table()
    if tag == "9ab":
        d = pd.read_parquet(REF9AB)
        nb = int((d.p != d.p_gbdt).sum())
        miss = 0
    else:
        f = OUT / f"fr_{tag}.parquet"
        if not f.exists():
            job = (OUT / f"job_{tag}.txt").read_text().strip()
            subprocess.run(["aws", "s3", "cp", f"{B}/{job}/pred/test_France.parquet", str(f), "--profile", "harsh",
                            "--region", "ap-south-1", "--only-show-errors"], check=True)
        k = pd.read_parquet(f)
        d = pd.DataFrame({"s1": s1.take(pa.array(k.q1.to_numpy())).to_numpy(zero_copy_only=False),
                          "sx": sx.take(pa.array(k.qx.to_numpy())).to_numpy(zero_copy_only=False),
                          "p_gbdt": k.p.to_numpy(np.float32)})
        r = pd.concat([pd.read_parquet(W / "p13/r/r_bge_pool_test_France.parquet"),
                       pd.read_parquet(W / "p19/r/r_bge_pool_test_France.parquet"),
                       pd.read_parquet(W / "p26/bge/r/r_bge_pool_test_France.parquet")]).drop_duplicates(["s1", "sx"])
        rr = d[["s1", "sx"]].merge(r, on=["s1", "sx"], how="left").r.to_numpy(np.float64)
        p, b = blend(d.p_gbdt.to_numpy(np.float64), rr)
        band = (d.p_gbdt > LO) & (d.p_gbdt < HI)
        miss, nb = int((band & np.isnan(rr)).sum()), int(b.sum())
        d["p"] = p.astype(np.float32)
        d = d[["s1", "sx", "p", "p_gbdt"]]
        d.to_parquet(OUT / f"prob_test_France_{tag}_bge.parquet", index=False)
    q1 = pd.Series(np.arange(len(s1)), index=s1.to_numpy(zero_copy_only=False))
    qx = pd.Series(np.arange(len(sx)), index=sx.to_numpy(zero_copy_only=False))
    hn = pd.DataFrame({"s1": d.s1, "sx": d.sx, "h1": M["h1"][q1.reindex(d.s1).to_numpy()],
                       "hx": M["hx"][qx.reindex(d.sx).to_numpy()]})
    ref = None
    if tag != "9ab":
        ref = pd.read_parquet(REF9AB)
        c = pd.factorize(ref.sx)[0]
        a = argmax(c, ref.p.to_numpy(np.float64)) & (ref.p.to_numpy() >= 0.985)
        ref.attrs["acc"] = set(zip(ref.s1.to_numpy()[a], ref.sx.to_numpy()[a]))
    out = {"tag": tag, "band_blended": nb, "band_missing_r": miss} | profile(d, hn, ref, 0.985)
    if tag != "9ab":
        out["corr_logit_p_vs_9ab_gbdt"] = None
    out["runtime_s"] = round(time.time() - t0)
    (OUT / f"profile_{tag}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in out.items() if k != "curve"}), flush=True)
    print("curve", json.dumps(out["curve"]), flush=True)
    return out


if __name__ == "__main__":
    for t in sys.argv[1:]:
        main(t)
