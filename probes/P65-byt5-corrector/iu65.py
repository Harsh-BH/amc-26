"""P65 India/US arm: corrector columns on sub54's India/US combiner (P57's full holdout frame, half split).

  feats C JOB    A_full_C rows (P57) -> data/work/p65/iu_C.parquet: raw + corrected sims (ev65's), the SX's decode
                 taken from JOB's decode of the LOCO frame that holds C's fold-0 SX (C's own corrector: in-domain,
                 trained on folds != 0). ~2% of rows have no decode (NaN).
  eval           sub54 (P57's cached boosters b_sub54_h*) vs sub54 + CORR cols refit on the same half, 5 seeds,
                 scored on the other half. Paired per-S1 F0.5 diff on the report half, LB-eq = 0.468 India + 0.383 US.
                 Gate: >= +0.00015 LB-eq at >= 3 SE, both directions (h0->h1, h1->h0) positive. Columns on IU band rows only.

    .venv/bin/python probes/P65-byt5-corrector/iu65.py feats India <India job> | feats US <US job> | eval
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ["P44_VARIANT"] = "r"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "probes/P57-iu-transfer"), str(ROOT / "probes/P44-meta-stack"),
                str(ROOT / "probes/P50-idea-batch"), str(ROOT), str(ROOT / "code/business_entity_resolution/src")]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import ev65 as V  # noqa: E402

W = ROOT / "data/work"
CI = {"India": 0, "US": 1}
FRAME = {"India": "US_India", "US": "India_US"}     # the LOCO frame holding C's fold-0 SX
COLS = V.CORR


def feats(c, job):
    import features as F
    from prep65 import txt
    t0 = time.time()
    ci = CI[c]
    A = pq.read_table(W / f"p57/A_full_{c}.parquet", columns=["q1", "qx", "p"]).to_pandas()
    fr = FRAME[c]
    S = pd.read_parquet(W / f"p27/p6/S_holdout_{'US' if c == 'India' else 'India'}.parquet", columns=["sx", "country", "am_all"])
    S = S[(S.country.to_numpy() == ci) & S.am_all.to_numpy()].reset_index(drop=True)
    qx_row = pd.Series(np.arange(len(S)), index=S.sx.to_numpy() - ci * 100_000_000)   # frame row by qx
    del S
    V.fetch(job)
    dj = V.W / "dec" / job
    dd = pd.concat([pd.read_parquet(p) for p in sorted(dj.glob(f"dec_frame_{fr}_[0-9]*.parquet")) + sorted(dj.glob(f"dec_frame_{fr}B_[0-9]*.parquet"))],
                   ignore_index=True)                           # LOCO-frame decodes: row = frame row
    by_qx = pd.Series(dd.out.to_numpy(), index=qx_row.index.to_numpy()[dd.row.to_numpy()])
    iu = [pd.read_parquet(p) for p in sorted(dj.glob(f"dec_frame_IU{c}_[0-9]*.parquet"))]     # IU-band decodes: row = qx
    if iu:
        iu = pd.concat(iu, ignore_index=True)
        by_qx = pd.concat([by_qx, pd.Series(iu.out.to_numpy(), index=iu.row.to_numpy())])
    by_qx = by_qx[~by_qx.index.duplicated()]
    out = by_qx.reindex(A.qx.to_numpy()).to_numpy()
    id1, idx = F.ids("train", c)
    e1 = id1.take(pa.array(A.q1.to_numpy(np.int64))).to_numpy(zero_copy_only=False)
    ex = idx.take(pa.array(A.qx.to_numpy(np.int64))).to_numpy(zero_copy_only=False)
    del id1, idx
    s1t = txt(["train_source1.parquet"], e1).reindex(e1).to_numpy()
    sxt = txt(["train_source2.parquet", "train_source3.parquet"], ex).reindex(ex).to_numpy()
    s1n, s1a = V.split(pd.Series(s1t))
    sxn, sxa = V.split(pd.Series(sxt))
    s1f, sxf = (s1n + " " + s1a).str.strip(), (sxn + " " + sxa).str.strip()
    r = pd.DataFrame(index=np.arange(len(A)))
    r["nm_r"], r["ad_r"], r["ts_r"] = V.sims(sxn, sxa, s1n, s1a, sxf, s1f)
    for k in V.CORR:
        r[k] = np.float32(np.nan)
    ok = pd.notna(out)
    on, oa = V.split(pd.Series(out[ok]))
    of = (on + " " + oa).str.strip()
    i = np.flatnonzero(ok)
    take = lambda s: s.iloc[i].reset_index(drop=True)
    a, b, cc = V.sims(on, oa, take(s1n), take(s1a), of, take(s1f))
    from rapidfuzz import fuzz, process
    cd = 1 - process.cpdist(take(sxf).to_numpy(), of.to_numpy(), scorer=fuzz.ratio, workers=4).astype(np.float32) / 100
    r.loc[ok, "nm_c"], r.loc[ok, "ad_c"], r.loc[ok, "ts_c"], r.loc[ok, "cd"] = a, b, cc, cd
    r.loc[ok, "ex_c"] = (on.to_numpy() == take(s1n).to_numpy()).astype(np.float32)
    r["dnm"], r["dad"] = r.nm_c - r.nm_r, r.ad_c - r.ad_r
    band = ((A.p > 0.02) & (A.p < 0.98)).to_numpy()            # IU band only (#8c p), as the France arm's band rule
    r.loc[~band, :] = np.nan
    print(f"  band rows {band.sum():,}, band decoded {(band & ok).mean() / max(band.mean(), 1e-9):.4f}", flush=True)
    r.astype(np.float32).to_parquet(V.W / f"iu_{c}.parquet", index=False)
    print(f"iu feats {c}: {len(r):,} rows, decoded {ok.mean():.4f}; {time.time() - t0:.0f}s peak {V.peak():.2f} GB", flush=True)


def evaluate():
    import ev57 as E7
    from ledger import log
    t0 = time.time()
    S = E7.load_s()
    D = E7.load("full")
    base_cols, seeds = E7.ARMS["sub54"]
    for c in E7.E.C:
        D[c] = pd.concat([D[c][sorted(set(E7.KEEP) | set(base_cols))].reset_index(drop=True),
                          pd.read_parquet(V.W / f"iu_{c}.parquet")[COLS]], axis=1)
    print(f"loaded {time.time() - t0:.0f}s peak {V.peak():.2f} GB", flush=True)
    res, tot = {}, {}
    for fit_h in (0, 1):
        rep = 1 - fit_h
        b0 = E7.boosters("sub54", fit_h)
        b1 = E7.fit(D, fit_h, base_cols + COLS, seeds)
        for c in E7.E.C:
            A, nt, h = D[c], S[c].n_true.to_numpy(), S[c].half.to_numpy()
            f0 = E7.per_s1(A, nt, E7.score(b0, A, base_cols) >= E7.E.T)
            f1 = E7.per_s1(A, nt, E7.score(b1, A, base_cols + COLS) >= E7.E.T)
            x = (f1 - f0)[h == rep]
            res[f"{c}|h{fit_h}"] = {"dF": float(x.mean()), "se": float(x.std(ddof=1) / np.sqrt(len(x))),
                                    "up": int((x > 0).sum()), "down": int((x < 0).sum())}
            print(c, f"h{fit_h}->h{rep}", res[f"{c}|h{fit_h}"], f"{time.time() - t0:.0f}s", flush=True)
        imp = pd.Series(b1[0].feature_importance("gain"), index=base_cols + COLS)
        res[f"imp_h{fit_h}"] = {k: round(float(imp[k] / imp.sum()), 4) for k in COLS}
        lb = sum(E7.E.LB_W[c] * res[f"{c}|h{fit_h}"]["dF"] for c in E7.E.C)
        se = float(np.sqrt(sum((E7.E.LB_W[c] * res[f"{c}|h{fit_h}"]["se"]) ** 2 for c in E7.E.C)))
        tot[fit_h] = res[f"lb|h{fit_h}"] = {"dF": lb, "se": se}
        print("LB-eq", f"h{fit_h}", tot[fit_h], flush=True)
        del b0, b1
    m = (tot[0]["dF"] + tot[1]["dF"]) / 2
    se = float(np.sqrt(tot[0]["se"] ** 2 + tot[1]["se"] ** 2) / 2)
    ok = m >= 1.5e-4 and m >= 3 * se and tot[0]["dF"] > 0 and tot[1]["dF"] > 0     # lead's IU gate (27 Sep 14:27)
    res["gate"] = {"mean": m, "se": se, "pass": bool(ok)}
    (HERE / "iu65.json").write_text(json.dumps(res, indent=1))
    log("exp", probe="P65-byt5-corrector", subproblem="iu-combiner", name="p65-iu-sub54-plus-corr", world="holdout-full-halfsplit",
        metrics={k: (round(v["dF"], 7) if isinstance(v, dict) and "dF" in v else v) for k, v in res.items() if not k.startswith("imp")} | {"gate_se": se},
        config={"base": "sub54 (P57 cached boosters)", "cols": COLS, "seeds": list(seeds), "t": E7.E.T},
        hypothesis="ByT5 corrector sims add India/US F0.5 on top of sub54's combiner", verdict="keep" if ok else "drop",
        runtime_s=round(time.time() - t0), peak_gb=round(V.peak(), 2))
    print("IU GATE", "PASS" if ok else "FAIL", res["gate"], f"{time.time() - t0:.0f}s peak {V.peak():.2f} GB")


if __name__ == "__main__":
    cmd, pos = sys.argv[1], sys.argv[2:]
    {"feats": lambda: feats(*pos), "eval": evaluate}[cmd]()
