"""P67 evaluation: synthetic-tuned CE scores as extra columns of P55's country-invariant combiner, LOCO-gated
(P65's ev65.loco procedure: fit P55's 3-seed bag on the source frame, score the target frame, shipped cell = veto band
c < .6 + add free c > .95). Columns are band-only (0.02 < g < 0.98, pre-declared) and each frame's column comes from
the run that targets that frame's country (frame US_India <- run India: US-trained CE + synthetic India pairs), i.e.
the same situation France is in (pooled CE + synthetic France pairs).

  fetch JOB                 s3 results/JOB/**/r/*.parquet -> data/work/p67/r/JOB/
  loco ARM JOB_A [JOB_B]    ARM bge: JOB_A holds runs India + US ('pre' + 'syn'); ARM qwen: JOB_A = run India job,
                            JOB_B = run US job ('qpre' + 'qsyn') -> probes/P67-fr-synth/loco67_<ARM>.json + ledger row
  frscore ARM JOB_A [JOB_B] JOB_F   France combiner scores with the gate's feature set -> data/work/p55/cf_p67<ARM>.npy
    .venv/bin/python probes/P67-fr-synth/ev67.py loco bge p67-bge-loco-0927-1530
"""
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "probes/P55-fr-combiner"), str(ROOT)]
W = ROOT / "data/work/p67"
BUCKET = "s3://" + os.environ.get("BUCKET", "your-bucket") + "/amc26-p67/results"
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
FRAME_RUN = {"US_India": "India", "India_US": "US", "France": "France"}


def fetch(job):
    d = W / "r" / job
    d.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "sync", f"{BUCKET}/{job}/", str(d), "--exclude", "*", "--include", "*/r/*.parquet",
                    "--include", "*log.jsonl", "--profile", os.environ.get("AWS_PROFILE", "default"), "--region", "ap-south-1", "--only-show-errors"], check=True)
    return d


def col(job, frame, tag, n):
    """logit r of <tag> on the frame's band rows (s1 = frame row), NaN elsewhere."""
    fs = sorted((W / "r" / job).rglob(f"r_{tag}*fb_{frame}.parquet"))
    fs = [f for f in fs if "smoke" not in f.name]
    assert len(fs) == 1, (job, frame, tag, fs)
    d = pd.read_parquet(fs[0])
    x = np.full(n, np.nan, np.float32)
    r = np.clip(d.r.to_numpy(np.float64), 1e-6, 1 - 1e-6)
    x[d.s1.to_numpy(np.int64)] = np.log(r / (1 - r))
    return x


def columns(arm, jobs, frame, n):
    """{column: array} for one frame. bge: pre_/syn_ files of the one job; qwen: qpre_/qsyn_ of the frame's run job."""
    k = FRAME_RUN[frame]
    if arm == "wf":
        d = pd.read_parquet(W / f"wf_{frame}.parquet")
        assert len(d) == n
        return {c: d[c].to_numpy() for c in d.columns}
    if arm == "bge":
        j = jobs[0] if frame != "France" else jobs[-1]
        out = {"r_pre": col(j, frame, "pre_", n) if frame != "France" else None, "r_syn": col(j, frame, "syn_", n)}
    else:
        j = {"India": jobs[0], "US": jobs[1] if len(jobs) > 1 else jobs[0], "France": jobs[-1]}[k]
        out = {"r_pre": col(j, frame, f"qpre_{k}_", n) if frame != "France" else None, "r_syn": col(j, frame, f"qsyn_{k}_", n)}
    return {c: v for c, v in out.items() if v is not None}


FSETS = {"base": [], "pre": ["r_pre"], "syn": ["r_syn"], "both": ["r_pre", "r_syn"]}
FSETS_WF = {"base": [], "wf": ["wf_jac", "wf_shared", "wf_sx_miss", "wf_s1_miss"]}
PRIMARY = {"bge": "syn", "qwen": "syn", "wf": "wf"}


def _new(p55, b, cb, k):
    rg = b["X"]["region"].to_numpy()
    v = b["ref"] & ~(b["ref"] & (cb < 0.6) & (rg == 1))
    return p55.decide({**b, "ref": v}, cb, None, 0.95) if k == "ship" else v


def loco(arm, *jobs):
    import p55
    from ledger import log
    t0 = time.time()
    for j in jobs:
        fetch(j)
    FS, prim = (FSETS_WF if arm == "wf" else FSETS), PRIMARY[arm]
    frames = {dr: p55.loco_frame(dr) for dr in p55.L.DIRS}
    cols = {dr: pd.DataFrame(columns(arm, jobs, dr, len(frames[dr]["X"]))) for dr in frames}
    print(f"frames {time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)
    out = {}
    for name, trn, tst in p55.DIRN:
        a, b = frames[trn], frames[tst]
        rg = b["X"]["region"].to_numpy()
        y = b["z"]["y"]
        R = out[name] = {"band_auc_tgt": {c: p55.auc(y[(rg == 1) & ~np.isnan(v)], v[(rg == 1) & ~np.isnan(v)])
                                          for c, v in cols[tst].items() if not c.startswith("wf")} |
                         {"lr_shipped": p55.auc(y[(rg == 1) & b["X"]["lr"].notna().to_numpy()],
                                                b["X"]["lr"].to_numpy()[(rg == 1) & b["X"]["lr"].notna().to_numpy()])},
                         "cover_tgt_band": float(cols[tst][FS[prim][0]].notna()[rg == 1].mean())}
        fref = p55.L.F(b["z"], b["ref"])
        C = {}
        for fs, cs in FS.items():
            Xa = pd.concat([a["X"], cols[trn][cs]], axis=1) if cs else a["X"]
            Xb = pd.concat([b["X"], cols[tst][cs]], axis=1) if cs else b["X"]
            bs = p55.bag(Xa, a["z"]["y"], None)
            cb = C[fs] = p55.bag_score(bs, Xb, None)
            S = R[fs] = {"auc_ref": p55.auc(y[b["ref"]], cb[b["ref"]]), "auc_band": p55.auc(y[rg == 1], cb[rg == 1])}
            S["ship"] = p55.L.cmp(b["z"], b["ref"], _new(p55, b, cb, "ship"), fref)
            S["veto06"] = p55.L.cmp(b["z"], b["ref"], _new(p55, b, cb, "veto06"), fref)
            imp = pd.Series(bs[0].feature_importance("gain"), index=Xa.columns)
            S["imp_new"] = {k: round(float(imp[k] / imp.sum()), 4) for k in cs}
            print(name, fs, {k: S[k] for k in ("auc_ref", "auc_band", "ship", "veto06", "imp_new")}, f"{time.time() - t0:.0f}s", flush=True)
        hv = b["z"]["hv"]
        for k in ("ship", "veto06"):
            F = {fs: p55.L.F(b["z"], _new(p55, b, C[fs], k)) for fs in FS}
            for x, y_ in [(f, "base") for f in FS if f != "base"] + ([("syn", "pre")] if "pre" in FS else []):
                d_ = F[x] - F[y_]
                R[f"{x}-{y_}_{k}"] = {"dF": round(float(d_.mean()), 6), "se": round(float(d_.std() / np.sqrt(len(d_))), 6),
                                      "dF_h0": round(float(d_[hv == 0].mean()), 6), "dF_h1": round(float(d_[hv == 1].mean()), 6)}
                print(name, k, f"{x}-{y_}", R[f"{x}-{y_}_{k}"], flush=True)
    g = {n: out[n][f"{prim}-base_ship"]["dF"] for n in ("India->US", "US->India")}
    ok = g["India->US"] >= 0.0003 and g["US->India"] >= 0
    out["gate"] = {"rule": f"{prim}-base at the shipped cell: India->US >= +0.0003 and US->India >= 0", "pass": bool(ok), "dF": g}
    out["runtime_s"], out["peak_gb"] = round(time.time() - t0), round(peak(), 2)
    (HERE / f"loco67_{arm}.json").write_text(json.dumps(out, indent=1))
    log("exp", probe="P67-fr-synth", subproblem="france-combiner", name=f"p67-loco-{arm}-{prim}", world="holdout-loco",
        metrics={f"{n}|{k}": v["dF"] for n in ("India->US", "US->India") for k, v in out[n].items() if "-" in k and isinstance(v, dict)} |
                {f"{n}|auc|{c}": v for n in ("India->US", "US->India") for c, v in out[n]["band_auc_tgt"].items()},
        config={"arm": arm, "jobs": list(jobs), "fsets": FS, "combiner": "P55 bag 3 seeds, 200 rounds",
                "columns": "band-only logit r; r_pre = transfer CE as shipped, r_syn = + 1 epoch synthetic target pairs + source replay",
                "cells": "ship = veto band c<.6 + add free c>.95"},
        hypothesis=("wordfreq name-token rarity overlap (4 cols)" if arm == "wf" else f"{arm} CE fine-tuned on synthetic target-country pairs (generated from the target's own S1)") + " adds France-analogue F0.5 over P55's combiner",
        verdict="keep" if ok else "drop", runtime_s=out["runtime_s"], peak_gb=out["peak_gb"])
    print("GATE", "PASS" if ok else "FAIL", g, f"{time.time() - t0:.0f}s peak {peak():.2f} GB")


def frscore(arm, *jobs, fs="syn"):
    """P55's pooled bag (both LOCO frames) with FSETS[fs] columns, applied to P55's France frame -> cf_p67<arm>.npy,
    so `p55.py build <base> <out> mode=p67<arm>,v=0.6,scope=band,u=0.95` uses it."""
    import p55
    t0 = time.time()
    for j in jobs:
        fetch(j)
    cs = (FSETS_WF if arm == "wf" else FSETS)[fs]
    Xs, ys = [], []
    for dr in p55.L.DIRS:
        fr = p55.loco_frame(dr)
        c = pd.DataFrame(columns(arm, jobs, dr, len(fr["X"])))
        Xs.append(pd.concat([fr["X"], c[cs]], axis=1))
        ys.append(fr["z"]["y"])
        del fr
    bs = p55.bag(pd.concat(Xs, ignore_index=True), np.concatenate(ys), None)
    del Xs, ys
    f = p55.france_frame()
    c = pd.DataFrame(columns(arm, jobs, "France", len(f["X"])))
    cf = p55.bag_score(bs, pd.concat([f["X"], c[cs]], axis=1), None)
    np.save(ROOT / f"data/work/p55/cf_p67{arm}.npy", cf)
    c0 = np.load(ROOT / "data/work/p55/cf_plain.npy")
    rg = f["X"]["region"].to_numpy()
    print(f"frscore {arm}/{fs}: France {len(cf):,} pairs, band covered {c[cs[0]].notna()[rg == 1].mean():.3f}; "
          f"veto<.6 p67 {int((f['ref'] & (cf < .6) & (rg == 1)).sum())} vs P55 {int((f['ref'] & (c0 < .6) & (rg == 1)).sum())}; "
          f"add>.95 cand p67 {int((f['free'] & ~f['ref'] & (cf > .95)).sum())} vs P55 {int((f['free'] & ~f['ref'] & (c0 > .95)).sum())}; "
          f"{time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)


if __name__ == "__main__":
    cmd, pos = sys.argv[1], sys.argv[2:]
    {"fetch": lambda: fetch(*pos), "loco": lambda: loco(*pos),
     "frscore": lambda: frscore(*pos[:-1], fs=pos[-1]) if pos[-1] in FSETS or pos[-1] in FSETS_WF else frscore(*pos)}[cmd]()
