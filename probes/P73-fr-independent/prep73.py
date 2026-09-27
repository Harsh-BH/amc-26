"""P73 prep: France-independent CE training sets -> data/work/p73/ch/ (then s3 amc26-p73/ch/).
No labelled India/US pair enters any train file. Per target country T (India, US as LOCO stand-ins; France):
  syn    P67's generator v2 on T's own S1 (data/work/p67/sm/syn_T_v2.parquet, 120k, y from the generator op)
  pseudo T's band pairs (the CE frame fb_<frame>) where P55's combiner c and Qwen agree (P61's rule: c, q >= .7 -> 1,
         c, q <= .3 -> 0), both scored by models trained on the OTHER country (France: the pooled ones).
         France is subsampled to 20% of its band (LOCO's agreement sets cover 16% / 21% of the band, because P38's
         LOCO Qwen scored only 28% / 37% of it), keeping France's own pos/neg ratio, so the recipe matches the gate.
Frames/val files are P67's ch2 copies (P55 band rows, s1 = frame row); val_* is only the Job's post-train monitor.
    .venv/bin/python probes/P73-fr-independent/prep73.py [v2h]   # v2h: India/US synthetic from gen73.py (other country's
                                                                # op rates) -> train_ind<T>h, the strict LOCO check
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "data/work"
OUT = W / "p73/ch"
FRAME = {"India": "US_India", "US": "India_US", "France": "France"}   # frame holding T's pairs
FR_SHARE = 0.2


def pseudo(t):
    fr = FRAME[t]
    fb = pd.read_parquet(W / f"p67/sm/ch2/fb_{fr}.parquet", columns=["s1", "ta", "tb"])
    i = fb.s1.to_numpy()
    if t == "France":
        z = np.load(W / "p61/france.npz", allow_pickle=True)
        c, q = z["cf"][i], z["q"][i]
    else:
        c, q = np.load(W / f"p55/cb_plain_{fr}.npy")[i], np.load(W / f"p61/f_{fr}.npz", allow_pickle=True)["q"][i]
    pos, neg = (c >= .7) & (q >= .7), (c <= .3) & (q <= .3)       # NaN q -> neither
    keep = pos | neg
    if t == "France":
        k = np.flatnonzero(keep)
        keep = np.zeros(len(i), bool)
        keep[np.random.default_rng(73).choice(k, int(FR_SHARE * len(i)), replace=False)] = True
    d = fb[keep].assign(y=pos[keep].astype(np.int64), p=np.float32(0.5), kind=9)
    print(f"{t}: band {len(i):,}, pseudo {len(d):,} (pos {int(d.y.sum()):,}, neg {int((1 - d.y).sum()):,})", flush=True)
    return d[["ta", "tb", "y", "p", "kind"]]


def main(ver="v2"):
    OUT.mkdir(parents=True, exist_ok=True)
    for t in FRAME if ver == "v2" else ("India", "US"):
        src = W / (f"p67/sm/syn_{t}_v2.parquet" if ver == "v2" else f"p73/syn_{t}_v2h.parquet")
        s = pd.read_parquet(src, columns=["ta", "tb", "y", "p", "kind"])
        d = pd.concat([s, pseudo(t)], ignore_index=True).sample(frac=1, random_state=73).reset_index(drop=True)
        d["y"] = d.y.astype(np.int64)
        assert d.ta.notna().all() and d.tb.notna().all() and set(d.y.unique()) <= {0, 1}
        t += "" if ver == "v2" else "h"
        d.to_parquet(OUT / f"train_ind{t}.parquet", index=False)
        print(f"train_ind{t}: {len(d):,} pairs, true {d.y.mean():.3f}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
