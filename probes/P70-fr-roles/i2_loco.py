"""LOCO analogue for Fable I2 / related-name same-key extras / I5 byte-equal address: on P55's LOCO frames (target pairs under
source-trained models), cells by (name relation, S1 and SX at the same hn|street key, key unique among S1, gib), with precision
overall and among FREE pairs (not accepted by the reference), plus the dF of adding the free cell pairs (caps, best-first)."""
import gc, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/home/harsh1/github-repos/amc-26")
sys.path.insert(0, str(ROOT / "probes/P70-fr-roles"))
import p70
p55, L = p70.p55, p70.L
W = ROOT / "data/work"
out = {}
for dr in L.DIRS:
    t0 = time.time()
    tgt = L.DIRS[dr][1]
    fr = p55.loco_frame(dr)
    R = pd.read_parquet(W / f"p70/roles_{dr}.parquet", columns=["kind", "key_n", "gib_x"])
    ids = pd.read_parquet(W / f"p48/loco_{dr}_ids.parquet", columns=["s1", "sx"])
    a1 = p70._raw("train", ids.s1, [1], ["business_address", "country"]).business_address
    ax = p70._raw("train", ids.sx, [2, 3], ["business_address", "country"]).business_address
    r1, rx = a1.reindex(ids.s1).fillna("").reset_index(drop=True), ax.reindex(ids.sx).fillna("").reset_index(drop=True)
    cc = pd.Series([tgt] * len(ids))
    k1, kx = p70._street(r1, cc), p70._street(rx, cc)
    same = ((k1 == kx) & (kx != "")).to_numpy()
    byte = ((r1 == rx) & (rx.str.strip() != "")).to_numpy()
    y, ref, free = fr["z"]["y"].astype(bool), fr["ref"], fr["free"]
    kind, kn, gib = R.kind.to_numpy(), R.key_n.to_numpy(), R.gib_x.to_numpy() > 0
    rel, dis = (kind >= 0) & (kind <= 7), kind == 8
    cells = {"related|same key|unique": rel & same & (kn == 1), "related|same key|shared": rel & same & (kn >= 2),
             "disjoint gib|same key|unique": dis & gib & same & (kn == 1), "disjoint gib|same key|shared": dis & gib & same & (kn >= 2),
             "disjoint plain|same key|unique": dis & ~gib & same & (kn == 1), "disjoint plain|same key|shared": dis & ~gib & same & (kn >= 2),
             "any|byte-equal addr": byte, "any|same key not byte-equal": same & ~byte}
    fref = L.F(fr["z"], ref)
    R_ = {}
    for name, m in cells.items():
        f_ = m & free & ~ref
        d = {"n": int(m.sum()), "prec": round(float(y[m].mean()), 3) if m.any() else None, "accepted": round(float(ref[m].mean()), 3) if m.any() else None,
             "free_n": int(f_.sum()), "free_prec": round(float(y[f_].mean()), 3) if f_.any() else None}
        if f_.any() and not name.startswith("any"):
            new = p55.decide(fr, f_.astype(float), None, 0.5)
            d["add_dF"] = {k: v for k, v in L.cmp(fr["z"], ref, new, fref).items() if k in ("dF", "dF_h1", "se_h1", "add", "add_prec")}
        R_[name] = d
        print(f"{dr} {name:32s} {json.dumps(d)}", flush=True)
    out[dr] = R_
    print(f"{dr} done {time.time()-t0:.0f}s peak {p70.peak():.2f} GB", flush=True)
    del fr, R, ids, a1, ax; gc.collect()
(W / "p70/i2_loco.json").write_text(json.dumps(out, indent=1))
