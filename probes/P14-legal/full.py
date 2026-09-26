"""P14-legal step 4: the full #7 pipeline on the full-corpus holdout, with vs without the 7 legal-form features.
Both arms run run.model_parts("holdout") unchanged: pruner (tau 0.03) -> M2 -> the 2-epoch CE on 0.02 < p < 0.98
(India, US) -> blend -> each S2/S3 at its argmax S1 -> p >= t.
  base-shipped  m2_big with #7's own w and t (must reproduce #7's holdout 0.98881)
  base          m2_big,       blend w refit + t tuned on holdout half 0
  legal         m2_legal_big, blend w refit + t tuned on holdout half 0
F0.5 (macro, singletons included) on half 1, legal vs base paired per S1; also over all fold 0.

    .venv/bin/python probes/P14-legal/full.py      # CE band scoring on the GPU; M2 + CE scores cached in data/work/p9
"""
import json
import sys

sys.path[:0] = ["code/business_entity_resolution/src", "."]

import numpy as np

import model as M
import rerank as RR
import run
from train_matcher import f05_per_s1
from worlds import WORK

TS = np.round(np.arange(0.50, 0.91, 0.01), 2)


def arm(booster: str) -> list:
    run.use("sub7")
    M.BOOSTER, M._M = WORK / "p4" / booster, None
    return run.model_parts("holdout")


def flat(parts: list) -> list:
    """Per country: pairs of every S2/S3 a holdout S1 competes for, with s1 = the holdout-S1 index within the country
    (-1 for a competing non-holdout S1), y = true pair; nt = true matches per holdout S1, hs = its half."""
    out = []
    for d in parts:
        ids = d["ids"]
        pos = np.full(len(d["id1"]), -1)
        pos[run.codes(d["id1"], ids)] = np.arange(len(ids))
        tr = [(s, x) for s in ids.to_pylist() for x in d["truth"][s]]
        tk = (run.codes(d["id1"], [s for s, _ in tr]) << 32) | run.codes(d["idx"], [x for _, x in tr])
        key = (d["q1"].astype(np.int64) << 32) | d["qx"].astype(np.int64)
        out.append({"c": d["c"], "s1": pos[d["q1"]], "qx": d["qx"], "pm": d["p_m2"], "r": d["r"],
                    "y": np.isin(key, tk), "nt": np.array([len(d["truth"][s]) for s in ids.to_pylist()]),
                    "hs": run.half(ids.to_pylist())})
    return out


def per_s1(fl: list, w, t, ce: bool = True) -> np.ndarray:
    """Macro-F0.5 term of every holdout S1 (countries in order) under blend w and threshold t."""
    lo, hi = RR.CFG["band"]
    fs = []
    for d in fl:
        p = RR.blend(d["pm"], d["r"], w, lo, hi) if ce else d["pm"]
        sel = M.argmax_sx(d["qx"], p) & (p >= t) & (d["s1"] >= 0)
        n, s = len(d["nt"]), d["s1"][sel]
        fs.append(f05_per_s1(np.bincount(s[d["y"][sel]], minlength=n), np.bincount(s, minlength=n), d["nt"]))
    return np.concatenate(fs)


def tune(fl: list, w=None, ce=True) -> dict:
    """w refit on the band pairs of half-0 S1 (RR.fit_blend, as run.refit_w) unless given; t swept on half 0."""
    hs = np.concatenate([d["hs"] for d in fl])
    if w is None and ce:
        P, R, Y = [], [], []
        for d in fl:
            b = ~np.isnan(d["r"]) & (d["s1"] >= 0)
            b[b] &= d["hs"][d["s1"][b]] == 0
            P.append(d["pm"][b]), R.append(d["r"][b]), Y.append(d["y"][b])
        w = [round(x, 4) for x in RR.fit_blend(*(np.concatenate(z) for z in (P, R, Y)))]
    t = max(TS, key=lambda t: per_s1(fl, w, t, ce)[hs == 0].mean())
    f = per_s1(fl, w, t, ce)
    grp = np.concatenate([[d["c"]] * len(d["nt"]) for d in fl])
    return {"w": w, "t": float(t), "f": f, "f05_half1": float(f[hs == 1].mean()), "f05_all": float(f.mean()),
            **{f"f05_half1_{c}": float(f[(hs == 1) & (grp == c)].mean()) for c in ("India", "US")}}


if __name__ == "__main__":
    from ledger import log
    res = {}
    for name, booster in (("base", "m2_big.txt"), ("legal", "m2_legal_big.txt")):
        fl = flat(arm(booster))
        hs = np.concatenate([d["hs"] for d in fl])
        if name == "base":
            shipped = per_s1(fl, run.RECIPES["sub7"]["w"], run.RECIPES["sub7"]["t"])
            res["base_shipped"] = {"f05_all": float(shipped.mean()), "f05_half1": float(shipped[hs == 1].mean())}
        res[name] = tune(fl)
        res[name + "_m2only"] = {k: v for k, v in tune(fl, ce=False).items() if k != "f"}
        print(name, {k: v for k, v in res[name].items() if k != "f"}, flush=True)
    d = (res["legal"]["f"] - res["base"]["f"])[hs == 1]
    res["ab"] = {"diff_half1": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))),
                 "s1_up": int((d > 0).sum()), "s1_down": int((d < 0).sum()), "n_half1": int(len(d))}
    for k in ("base", "legal"):
        res[k].pop("f")
    (WORK / "p14" / "full.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)
    for k in ("base", "legal"):
        log("pipe", version=f"p14-{k}-sub7", world="holdout-dense",
            metrics={q: v for q, v in res[k].items() if isinstance(v, float)} | (res["ab"] if k == "legal" else {}),
            config={"recipe": "sub7 (pruner 0.03 + 2-epoch CE)", "booster": {"base": "m2_big", "legal": "m2_legal_big"}[k],
                    "w": res[k]["w"], "t": res[k]["t"], "tuning": "w refit + t on holdout half 0, F0.5 on half 1",
                    "t_m2": res[k + "_m2only"]["t"]})
