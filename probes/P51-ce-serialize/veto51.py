"""P48's France veto re-applied to a rebuilt France chain (P48 build48.veto_mask, with this chain's own r and blend).
    veto51.py POST BLEND R ANCHOR OUT [v=0.2]
POST / BLEND: build38 post / blend files of the chain; R: the band r file(s) the blend used ('bge' = build38.BGE_R);
ANCHOR: the tpick t. Drops pairs that on43's P42 decide (t by P42's density rule, caps 5/6, rescue .90, gc .70, hn .80)
added in its rescue or gc stage, sit in the band (0.02 < p_gbdt < 0.98), were not moved by a tier / bxneg rule
(post p == blend p) and have r < v. Writes the vetoed (s1, sx) to OUT; splice.py removes them from the France lines."""
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P38-qwen4b-fr"), str(ROOT / "probes/P42-fr-joint-tune")]
import build38 as B


def main(post, blend, rf, anchor, out, v="0.2"):
    t0, anchor, v = time.time(), float(anchor), float(v)
    import model as M
    import solution_p42 as S
    zb = pd.read_parquet(B.W / "p31/fr31v0.1_t9868.parquet", columns=["sx", "p"])      # = build38.on43's t42 rule
    pb = zb.p.to_numpy(np.float64)
    g42 = S.sdens(pb, M.argmax_sx(pd.factorize(zb.sx)[0].astype(np.int64), pb), 0.9868, 0.99, 259_452)
    del zb, pb
    d, a, _ = B.fr_frame(post)
    t42 = S.inv_sdens(a["p"], a["am"], anchor, g42, a["n_s1"])
    st = S.decide(a["s1"], a["is2"], a["p"], a["am"], a["sib"], a["hn_eq"], a["n_s1"], t42, (5, 6), 0.90, 0.70, 0.80,
                  steps=True)
    bl = pd.read_parquet(blend, columns=["s1", "sx", "p", "p_gbdt"])
    assert (bl.s1.to_numpy() == d.s1.to_numpy()).all() and (bl.sx.to_numpy() == d.sx.to_numpy()).all()
    g = bl.p_gbdt.to_numpy()
    fs = B.BGE_R if rf == "bge" else [Path(f) for f in rf.split(",")]
    rk = pd.concat([pd.read_parquet(f, columns=["s1", "sx", "r"]) for f in fs]).drop_duplicates(["s1", "sx"])
    r = d[["s1", "sx"]].merge(rk, on=["s1", "sx"], how="left").r.to_numpy()
    added = (st["rescue"] & ~st["base"]) | (st["gc"] & ~st["rescue"])
    veto = added & (g > 0.02) & (g < 0.98) & (np.abs(a["p"] - bl.p.to_numpy()) <= 1e-4) & (np.nan_to_num(r, nan=1.0) < v)
    d[veto][["s1", "sx"]].to_parquet(out, index=False)
    info = {"t42": round(float(t42), 5), "veto_pairs": int(veto.sum()),
            "by_stage": {"rescue": int((veto & st["rescue"] & ~st["base"]).sum()), "gc": int((veto & st["gc"] & ~st["rescue"]).sum())},
            "v": v, "runtime_s": round(time.time() - t0)}
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
