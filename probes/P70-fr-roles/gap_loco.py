"""LOCO gates for the France gap after sub77: missing-source fill, empty-S1 fill and copy-of-an-accepted-copy extras,
all at the S1's unique hn|street key. Precision on FREE pairs (what a rule would add) + add dF, on P55's LOCO frames."""
import gc, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "probes/P70-fr-roles"))
import p70
p55, L = p70.p55, p70.L
W = ROOT / "data/work"
out = {}
for dr in L.DIRS:
    t0, tgt = time.time(), L.DIRS[dr][1]
    fr = p55.loco_frame(dr)
    R = pd.read_parquet(W / f"p70/roles_{dr}.parquet", columns=["kind", "key_n"])
    ids = pd.read_parquet(W / f"p48/loco_{dr}_ids.parquet", columns=["s1", "sx"])
    a1 = p70._raw("train", ids.s1, [1], ["business_address", "country"]).business_address
    ax = p70._raw("train", ids.sx, [2, 3], ["business_address", "country"]).business_address
    cc = pd.Series([tgt] * len(ids))
    k1 = p70._street(a1.reindex(ids.s1).fillna("").reset_index(drop=True), cc).to_numpy()
    kx = p70._street(ax.reindex(ids.sx).fillna("").reset_index(drop=True), cc).to_numpy()
    cx = pd.read_parquet(W / f"rf_train_sx_{tgt}.parquet", columns=["id", "core"]).set_index("id").core.reindex(ids.sx).fillna("").to_numpy()
    z, ref, free, y = fr["z"], fr["ref"], fr["free"], fr["z"]["y"].astype(bool)
    s1, is2 = z["s1"].astype(np.int64), z["is2"]
    n2 = np.bincount(s1[ref & is2], minlength=fr["n"]); n3 = np.bincount(s1[ref & ~is2], minlength=fr["n"])
    kind, kn = R.kind.to_numpy(), R.key_n.to_numpy()
    rel = (kind >= 0) & (kind <= 7)
    key_u = (k1 == kx) & (kx != "") & (kn == 1)
    miss = np.where(is2, (n2[s1] == 0) & (n3[s1] > 0), (n3[s1] == 0) & (n2[s1] > 0))     # SX fills the S1's missing source
    empty = (n2[s1] + n3[s1]) == 0
    acc = pd.DataFrame({"s1": s1[ref], "core": cx[ref], "key": kx[ref]})
    ck = set(zip(acc.s1, acc.core, acc.key)); c_ = set(zip(acc.s1, acc.core))
    copy_ck = np.array([(a, b, c) in ck for a, b, c in zip(s1, cx, kx)]) & (cx != "")
    copy_c = np.array([(a, b) in c_ for a, b in zip(s1, cx)]) & (cx != "")
    cells = {"missing-source | related | unique key": miss & rel & key_u, "missing-source | related | any key": miss & rel,
             "empty S1 | related | unique key": empty & rel & key_u,
             "copy of accepted SX (core + key)": copy_ck, "copy of accepted SX (core only)": copy_c & ~copy_ck,
             "extra | related | unique key (not missing-source)": rel & key_u & ~miss & ~empty}
    fref = L.F(z, ref)
    res = {}
    for name, m in cells.items():
        f_ = m & free & ~ref
        d = {"free_n": int(f_.sum()), "free_prec": round(float(y[f_].mean()), 3) if f_.any() else None}
        if f_.any():
            new = p55.decide(fr, f_.astype(float), None, 0.5)
            d |= {k: v for k, v in L.cmp(z, ref, new, fref).items() if k in ("dF", "dF_h1", "se_h1", "add", "add_prec")}
        res[name] = d
        print(f"{dr} {name:50s} {json.dumps(d)}", flush=True)
    out[dr] = res
    print(f"{dr} {time.time()-t0:.0f}s peak {p70.peak():.2f} GB", flush=True)
    del fr, R, ids, a1, ax, acc, ck, c_; gc.collect()
(W / "p70/gap_loco.json").write_text(json.dumps(out, indent=1))
