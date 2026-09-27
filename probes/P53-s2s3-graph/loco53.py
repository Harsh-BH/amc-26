"""P53 France: triangle evidence as an extra veto / acceptance rule on P48's LOCO chain analogue of the France recipe
(P42's frames, p = chain blend at tau 0.1, thresholds density-mapped from France; t .99 / rescue .9 / gc .7 / hn .8).
Triangle features on the LOCO rows come from tri.feats with the LOCO p (so they see only the source-only models).
  veto    drop accepted pairs whose SX has a competitor S1 with stronger triangle support: c_pmax >= v and
          c_pmax - t_max >= m
  accept  add free, unaccepted, under-cap argmax pairs of non-empty S1 with p >= the mapped France bar b and a
          confident same-S1 twin: t_best >= u and t_ncs >= 1 (the hn-extras stage with triangle support instead of hn)
    py.sh loco53.py       -> data/work/p53/loco53.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "P48-fr-ce-all")]
import loco48 as L  # noqa: E402
import tri  # noqa: E402

S = L.S


def main():
    out = {}
    for dr, (src, tgt) in L.DIRS.items():
        z = L.frame(dr)
        ids = pd.read_parquet(L.W / f"p48/loco_{dr}_ids.parquet", columns=["qx"])
        p = S.loco_p(z["g"], z["r"], L.CUR["tau"], "chain")
        st, tm = L.recipe(z, dr, p)
        ref = st["hn"]
        fref = L.F(z, ref)
        f = tri.feats("train", tgt, z["s1"].astype(np.int64), ids.qx.to_numpy(np.int64), p)
        free = L.free_mask(z, st, p, tm)
        R = out[dr] = {"ref_F": round(float(fref.mean()), 5), "ref_pairs": int(ref.sum())}
        tmx, cpm = np.nan_to_num(f["t_max"], nan=0.0), np.nan_to_num(f["c_pmax"], nan=0.0)
        for v in (0.3, 0.5, 0.7, 0.9):
            for m in (0.0, 0.1, 0.3):
                veto = ref & (cpm >= v) & (cpm - tmx >= m)
                R[f"veto:v{v}:m{m}"] = L.cmp(z, ref, ref & ~veto, fref)
        s1, is2, n = z["s1"].astype(np.int64), z["is2"], len(z["ntrue"])
        n2, n3 = S.counts(s1, is2, ref, n)
        nonempty = (n2 + n3)[s1] > 0
        tb = np.nan_to_num(f["t_best"], nan=0.0)
        for b in (0.5, 0.7, 0.8, 0.9):
            for u in (0.9, 0.95, 0.99):
                cand = free & ~ref & nonempty & (p >= tm[str(b)] if str(b) in tm else p >= b) & (tb >= u) & (f["t_ncs"] >= 1)
                c = np.flatnonzero(cand)
                R[f"accept:b{b}:u{u}"] = L.cmp(z, ref, L.add_capped(z, ref, c, p, n2[s1], n3[s1]), fref)
        print(dr, json.dumps({k: v for k, v in R.items() if isinstance(v, dict) and (v["add"] or v["rm"])}), flush=True)
    (tri.OUT / "loco53.json").write_text(json.dumps(out, indent=0))


if __name__ == "__main__":
    main()
