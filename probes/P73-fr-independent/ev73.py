"""P73 gate: France-independent CE scores as extra columns of P55's combiner, leave-one-country-out
(ev67.loco's procedure on P61's cached frames: fit P55's 3-seed bag on the source frame, score the target frame, shipped
cell = veto band c < .6 + add free c > .95). Columns are band-only logit r. Each frame's column comes from the run
that trained on THAT frame's country (synthetic + pseudo-labels from the other country's models), the France situation.
  fetch JOB                 s3 results/JOB/**/r/*.parquet -> data/work/p73/r/JOB/
  loco JOB [JOB ...]        -> probes/P73-fr-independent/loco73.json + ledger row
  self                      reproduce P67's ledgered v2 gate from P67's column (harness check)
  frscore FS JOB [JOB ...]  pooled bag (both LOCO frames) with FSETS[FS] -> France scores data/work/p55/cf_p73<FS>.npy
    .venv/bin/python probes/P73-fr-independent/ev73.py loco p73-bge-loco-0927-1516 p73-sm-0927-1516
"""
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "probes/P61-fr-cotrain"), str(ROOT / "probes/P55-fr-combiner"), str(ROOT)]
W = ROOT / "data/work/p73"
BUCKET = "s3://sagemaker-114872709824-ap-south-1/amc26-{}/results"   # p67* jobs: P67's synthetic-tuned CE (r_syn)
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
# France has no English text: the native-language slot (electra on India/US LOCO) is filled by camembert-mmarcoFR
FR_ALIAS = {"r_ele": "r_cam", "r_eleh": "r_cam"}
FB = {"US_India": "fb_US_India", "India_US": "fb_India_US", "France": "fb_France"}
# r_<m>: P73 CE m (h = strict v2h synthetic); r_p67: P67's synthetic-tuned CE; swap_*: the column REPLACES the transfer lr
FSETS = {"base": [], "bge": ["r_bge"], "ele": ["r_ele"], "cam": ["r_cam"], "bgeh": ["r_bgeh"], "eleh": ["r_eleh"],
         "both": ["r_bge", "r_ele"], "swap_bge": ["r_bge"], "swap_ele": ["r_ele"], "p67": ["r_p67"],
         "p67+bge": ["r_p67", "r_bge"], "p67+ele": ["r_p67", "r_ele"], "p67+cam": ["r_p67", "r_cam"],
         "p67+bgeh": ["r_p67", "r_bgeh"], "p67+eleh": ["r_p67", "r_eleh"],
         "p70": ["kind", "w_s", "w_e"], "p67+p70": ["r_p67", "kind", "w_s", "w_e"],       # P70's label-free columns (LOCO only)
         "p67+ele+p70": ["r_p67", "r_ele", "kind", "w_s", "w_e"],
         "p67+ele+bge": ["r_p67", "r_ele", "r_bge"], "p67+eleh+bgeh": ["r_p67", "r_eleh", "r_bgeh"],
         "ele4": ["r_ele4"], "ele4n": ["r_ele4n"], "p67+ele4": ["r_p67", "r_ele4"], "p67+ele4n": ["r_p67", "r_ele4n"]}


def fetch(job):
    d = W / "r" / job
    d.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "sync", f"{BUCKET.format(job[:3])}/{job}/", str(d), "--exclude", "*", "--include", "*/r/*.parquet",
                    "--include", "*log.jsonl", "--profile", "harsh", "--region", "ap-south-1", "--only-show-errors"], check=True)
    return d


def columns(jobs, frame, n):
    """{r_<m>: logit r on the frame's band rows (s1 = frame row), NaN elsewhere} for every model m found."""
    out = {}
    for j in jobs:
        if j == "p70":                                  # data/work/p70/roles_<dr>.parquet, row-aligned with the frame
            if frame != "France":
                d = pd.read_parquet(ROOT / f"data/work/p70/roles_{frame}.parquet", columns=["kind", "w_s", "w_e"])
                assert len(d) == n
                out |= {c: d[c].to_numpy() for c in d.columns}
            continue
        if j.startswith("p67"):
            fs = [f for f in (W / "r" / j).rglob(f"r_syn_{FB[frame]}.parquet")]
            assert len(fs) <= 1, fs
            if fs:
                d = pd.read_parquet(fs[0])
                x = np.full(n, np.nan, np.float32)
                r = np.clip(d.r.to_numpy(np.float64), 1e-6, 1 - 1e-6)
                x[d.s1.to_numpy(np.int64)] = np.log(r / (1 - r))
                out["r_p67"] = x
            continue
        for f in sorted((W / "r" / j).rglob(f"r_ind_*_{FB[frame]}.parquet")):
            m = f.name.split("_")[2]
            d = pd.read_parquet(f)
            x = np.full(n, np.nan, np.float32)
            r = np.clip(d.r.to_numpy(np.float64), 1e-6, 1 - 1e-6)
            x[d.s1.to_numpy(np.int64)] = np.log(r / (1 - r))
            out[f"r_{m}"] = x
    return out


def design(X, C, fs):
    cs = FSETS[fs]
    if fs.startswith("swap_"):
        return pd.concat([X.drop(columns="lr"), C[cs].rename(columns={cs[0]: "lr"})], axis=1)
    return pd.concat([X, C[cs]], axis=1) if cs else X


def loco(*jobs, tag=""):
    import p55
    import p61
    from ledger import log
    t0 = time.time()
    for j in jobs:
        if j != "p70":
            fetch(j)
    L = p55.L
    fr = {dr: p61.load(dr) for dr in ("US_India", "India_US")}
    cols = {dr: pd.DataFrame(columns(jobs, dr, len(fr[dr]["X"]))) for dr in fr}
    FS = [fs for fs in FSETS if all(c in cols["US_India"] and c in cols["India_US"] for c in FSETS[fs])]
    print("feature sets", FS, f"{time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)
    out = {}
    for name, trn, tst in p55.DIRN:
        a, b = fr[trn], fr[tst]
        rg, y = b["rg"], b["z"]["y"]
        band = lambda v: (rg == 1) & ~np.isnan(v)
        lr = b["X"]["lr"].to_numpy()
        R = out[name] = {"band_auc_tgt": {c: round(p55.auc(y[band(v)], v[band(v)]), 4) for c, v in cols[tst].items()}
                         | {"lr_transfer": round(p55.auc(y[band(lr)], lr[band(lr)]), 4)}}
        fref = L.F(b["z"], b["ref"])
        F = {}
        for fs in FS:
            bs = p55.bag(design(a["X"], cols[trn], fs), a["z"]["y"], None)
            cb = p55.bag_score(bs, design(b["X"], cols[tst], fs), None)
            sel = p61.arm_a(b, cb)
            F[fs] = L.F(b["z"], sel)
            imp = pd.Series(bs[0].feature_importance("gain"), index=design(a["X"].iloc[:1], cols[trn].iloc[:1], fs).columns)
            R[fs] = {"auc_band": round(p55.auc(y[rg == 1], cb[rg == 1]), 4), "ship_vs_ref": L.cmp(b["z"], b["ref"], sel, fref),
                     "gain_share": {k: round(float(imp[k] / imp.sum()), 4) for k in (["lr"] if fs.startswith("swap") else FSETS[fs])}}
            print(name, fs, R[fs], f"{time.time() - t0:.0f}s", flush=True)
        hv = b["z"]["hv"]
        for fs in FS:
            if fs == "base":
                continue
            d_ = F[fs] - F["base"]
            R[f"{fs}-base"] = {"dF": round(float(d_.mean()), 6), "se": round(float(d_.std() / np.sqrt(len(d_))), 6),
                               "dF_h0": round(float(d_[hv == 0].mean()), 6), "dF_h1": round(float(d_[hv == 1].mean()), 6)}
            print(name, f"{fs}-base", R[f"{fs}-base"], flush=True)
    gate = {}
    for fs in FS:
        if fs == "base":
            continue
        g = {n: out[n][f"{fs}-base"]["dF"] for n in ("India->US", "US->India")}
        gate[fs] = {"dF": g, "pass": bool(g["India->US"] >= 0.0003 and g["US->India"] >= 0)}
    out["gate"] = {"rule": "<fs>-base at the shipped cell: India->US (US target) >= +0.0003 and US->India >= 0", **gate}
    out["runtime_s"], out["peak_gb"] = round(time.time() - t0), round(peak(), 2)
    (HERE / f"loco73{tag}.json").write_text(json.dumps(out, indent=1))
    log("exp", probe="P73-fr-independent", subproblem="france-combiner", name=f"p73-loco-independent-ce{tag}", world="holdout-loco",
        metrics={f"{n}|{k}": v["dF"] for n in ("India->US", "US->India") for k, v in out[n].items() if k.endswith("-base")} |
                {f"{n}|auc|{c}": v for n in ("India->US", "US->India") for c, v in out[n]["band_auc_tgt"].items()},
        config={"jobs": list(jobs), "fsets": {k: FSETS[k] for k in FS}, "combiner": "P55 bag 3 seeds, 200 rounds",
                "train": "fresh public ckpt, 2 epochs on P67 syn_v2 of the target (120k) + P61-rule agreement pseudo-labels "
                         "of the target band from other-country models; no labelled India/US pair",
                "cells": "ship = veto band c<.6 + add free c>.95"},
        hypothesis="a CE trained without any India/US labels (fresh public ckpt, target synthetic + pseudo-labels) adds "
                   "France-analogue F0.5 over P55's combiner, whose CE column was trained on the other country's labels",
        verdict="keep" if any(v["pass"] for v in gate.values()) else "drop", runtime_s=out["runtime_s"], peak_gb=out["peak_gb"])
    print("GATE", json.dumps(gate), f"{time.time() - t0:.0f}s peak {peak():.2f} GB")


def frscore(fs, *jobs):
    """P55's pooled bag (both LOCO frames) with FSETS[fs], applied to P55's France frame -> cf_p73<fs>.npy."""
    import p55
    import p61
    t0 = time.time()
    for j in jobs:
        if j != "p70":
            fetch(j)
    Xs, ys = [], []
    for dr in ("US_India", "India_US"):
        f = p61.load(dr)
        Xs.append(design(f["X"], pd.DataFrame(columns(jobs, dr, len(f["X"]))), fs))
        ys.append(f["z"]["y"])
        del f
    bs = p55.bag(pd.concat(Xs, ignore_index=True), np.concatenate(ys), None)
    del Xs, ys
    z = np.load(ROOT / "data/work/p61/france.npz", allow_pickle=True)
    X = pd.DataFrame(z["X"], columns=list(z["cols"]))
    C = pd.DataFrame(columns(jobs, "France", len(X)))
    for c in FSETS[fs]:
        if c not in C and FR_ALIAS.get(c) in C:
            C[c] = C[FR_ALIAS[c]]
            print(f"France column {c} <- {FR_ALIAS[c]}", flush=True)
    cf = p55.bag_score(bs, design(X, C, fs), None)
    np.save(ROOT / f"data/work/p55/cf_p73{fs}.npy", cf)
    c0, ref, free, rg = z["cf"], z["ref"], z["free"], X["region"].to_numpy()
    print(f"frscore {fs}: France {len(cf):,} pairs, band covered {C[FSETS[fs][0]].notna()[rg == 1].mean():.3f}; "
          f"veto<.6 p73 {int((ref & (cf < .6) & (rg == 1)).sum())} vs P55 {int((ref & (c0 < .6) & (rg == 1)).sum())}; "
          f"add>.95 cand p73 {int((free & ~ref & (cf > .95)).sum())} vs P55 {int((free & ~ref & (c0 > .95)).sum())}; "
          f"{time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)


def selfcheck():
    """The harness must reproduce P67's ledgered v2 gate (p67-loco-bge-syn: +0.001608 / +0.002057) from P67's column."""
    import ledger
    ledger.log = lambda *a, **k: None
    global FSETS
    keep, FSETS = FSETS, {k: v for k, v in FSETS.items() if k in ("base", "p67")}
    loco("p67v2-bge-loco-0927-1432")
    FSETS = keep
    g = json.loads((HERE / f"loco73{tag}.json").read_text())["gate"]["p67"]["dF"]
    assert abs(g["India->US"] - 0.001608) < 2e-6 and abs(g["US->India"] - 0.002057) < 2e-6, g
    print("selfcheck OK", g)


def decomp(*args):
    """Split fs-vs-base at the shipped cell into its parts, per direction: 'add' = base veto (old c) + adds on the new c
    (build67 noswap, what sub67 ships), 'veto' = the new c's veto alone, 'ship' = both (build67 swap).
        ev73.py decomp FS JOB [JOB ...]"""
    import p55
    import p61
    fs, jobs = args[0], args[1:]
    L = p55.L
    fr = {dr: p61.load(dr) for dr in ("US_India", "India_US")}
    cols = {dr: pd.DataFrame(columns(jobs, dr, len(fr[dr]["X"]))) for dr in fr}
    out = {}
    for name, trn, tst in p55.DIRN:
        a, b = fr[trn], fr[tst]
        c = {k: p55.bag_score(p55.bag(design(a["X"], cols[trn], k), a["z"]["y"], None), design(b["X"], cols[tst], k), None)
             for k in ("base", fs)}
        base = p61.arm_a(b, c["base"])
        Fb = L.F(b["z"], base)
        cells = {"ship": p61.arm_a(b, c[fs]),
                 "add": p55.decide(dict(b, ref=p61.arm_a(b, c["base"], u=None)), c[fs], None, 0.95),
                 "veto": p55.decide(dict(b, ref=p61.arm_a(b, c[fs], u=None)), c["base"], None, 0.95)}
        out[name] = {k: L.cmp(b["z"], base, v, Fb) for k, v in cells.items()}
        print(name, fs, json.dumps(out[name]), flush=True)
    (HERE / f"decomp73_{fs}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    cmd, pos = sys.argv[1], sys.argv[2:]
    {"fetch": lambda: fetch(*pos), "loco": lambda: loco(*[p for p in pos if not p.startswith("tag=")],
                                                 **{k: v for k, v in (p.split("=", 1) for p in pos if p.startswith("tag="))}), "frscore": lambda: frscore(*pos), "self": selfcheck, "decomp": lambda: decomp(*pos)}[cmd]()
