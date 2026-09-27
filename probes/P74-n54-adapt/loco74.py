"""P74 LOCO: N54's decoy-word removal (france.decoy_mask, s > .25, the src port itself) on P69's chain analogue
(P42 frames: #37-style base + rescue .90 + gc .70, caps 5/6, France density map of sub66_final), both directions,
chain + M2 analogues. The frame rows get their ids back from P6's S_holdout (prep.py's row order); the lens runs on
the target's own argmax rows, label-free, with rf_train_* standing in for rf_test_*.  -> loco74.json
    .venv/bin/python probes/P74-n54-adapt/loco74.py"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

sys.path[:0] = ["probes/P42-fr-joint-tune", "code/business_entity_resolution/src"]
import solution_p42 as S  # noqa: E402
import france as FR  # noqa: E402
import features as F  # noqa: E402

W, HERE = Path("data/work"), Path("probes/P74-n54-adapt")
CI, OFF1, OFF2 = {"India": 0, "US": 1}, 10_000_000, 100_000_000        # P28 loco.py
G = {"fr_dens": {"0.1": json.loads((Path("probes/P69-fr-loosen/loco69.json")).read_text())["fr_dens_sub66"]}}
S_MIN = float(os.environ.get("S_MIN", "0.25"))
tmp = Path(tempfile.mkdtemp(prefix="p74rf", dir="data/work/p74"))
out = {}
for dr in ("US_India", "India_US"):
    t0 = time.time()
    src, tgt = dr.split("_")
    h = pd.read_parquet(W / f"p27/p6/S_holdout_{src}.parquet", columns=["s1", "sx", "country", "am_all", "is_true"])
    h = h[(h.country.to_numpy() == CI[tgt]) & h.am_all.to_numpy()]
    z = S._frame(W / "p42", dr)
    assert len(h) == len(z["y"]) and (h.is_true.to_numpy() == z["y"]).all(), "prep.py's row order"
    id1, idx = F.ids("train", tgt)
    s1i = id1.take(pa.array(h.s1.to_numpy() - CI[tgt] * OFF1)).to_numpy(zero_copy_only=False)
    sxi = idx.take(pa.array(h.sx.to_numpy() - CI[tgt] * OFF2)).to_numpy(zero_copy_only=False)
    del h
    for k in ("s1", "sx"):
        (tmp / f"rf_test_{k}_{tgt}.parquet").unlink(missing_ok=True)
        (tmp / f"rf_test_{k}_{tgt}.parquet").symlink_to((W / f"rf_train_{k}_{tgt}.parquet").resolve())
    FR.WORK, FR.C = tmp, tgt
    one = np.ones(len(s1i), bool)
    n54 = FR.decoy_mask(s1i, sxi, one, one, S_MIN)
    st = FR.lens_table(s1i, sxi)
    words = st[(st.n >= 20) & (st.s > S_MIN)].sort_values("n", ascending=False)
    s1, is2, y, nl = z["s1"].astype(np.int64), z["is2"], z["y"], len(z["ntrue"])
    sib, hq, ntrue = z["hn_rel"] == 2, z["hn_rel"] == 1, z["ntrue"].astype(np.int64)
    fb = lambda pred: S.fbeta(np.bincount(s1[pred & y], minlength=nl), np.bincount(s1[pred], minlength=nl), ntrue)
    r_dr = {"flagged_rows": int(n54.sum()), "flagged_true": round(float(y[n54].mean()), 4) if n54.any() else None,
            "decoy_words": len(words), "top_words": {w: [int(r.n), round(float(r.s), 3)] for w, r in words.head(12).iterrows()}}
    for an in ("chain", "m2"):
        pl = S.loco_p(z["g"], z["r"], 0.1, an)
        tm = S.fr_to_loco(G, an, dr, 0.1, pl, one, nl)
        st_ = S.decide(s1, is2, pl, one, sib, hq, nl, tm["0.9868"], (5, 6), tm["0.9"], tm["0.7"], None, steps=True)
        pred = st_["gc"]
        rm = pred & n54
        d = fb(pred & ~rm) - fb(pred)
        stage = {"base": rm & st_["base"], "rescue": rm & st_["rescue"] & ~st_["base"], "gc": rm & st_["gc"] & ~st_["rescue"]}
        r = {"removed": int(rm.sum()), "removed_true": round(float(y[rm].mean()), 4) if rm.any() else None,
             "dF": round(float(d.mean()), 6),
             "dF_h": [round(float(d[z["hv"] == q].mean()), 6) for q in (0, 1)],
             "se_h": [round(float(d[z["hv"] == q].std() / np.sqrt((z["hv"] == q).sum())), 6) for q in (0, 1)],
             "by_stage": {k: [int(v.sum()), round(float(y[v].mean()), 3) if v.any() else None] for k, v in stage.items()},
             "base_F": round(float(fb(pred).mean()), 5), "pred": int(pred.sum()),
             "bge_r_removed": {"q10_50_90": [round(float(x), 3) for x in np.nanquantile(z["r"][rm], [.1, .5, .9])] if rm.any() else None,
                               "ge.9": round(float(np.nanmean(z["r"][rm] >= .9)), 3) if rm.any() else None}}
        r_dr[an] = r
        print(dr, an, json.dumps(r), flush=True)
    r_dr["runtime_s"] = round(time.time() - t0)
    out[dr] = r_dr
    print(dr, {k: v for k, v in r_dr.items() if k not in ("chain", "m2")}, flush=True)
(HERE / f"loco74{'' if S_MIN == 0.25 else f'_s{S_MIN}'}.json").write_text(json.dumps(out, indent=1))
