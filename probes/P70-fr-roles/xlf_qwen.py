import sys
import numpy as np, pandas as pd
sys.path[:0] = ["probes/P70-fr-roles"]
import xlf_loco as X
L6, P = X.L6, X.P
for src, tgt in (("US", "India"), ("India", "US")):
    A = P.loco_load(src, tgt)
    na = pd.Series(A["sx_ids"]).isin(set(L6.addrless_ids("train", tgt))).to_numpy()
    w = np.flatnonzero(A["am"] & na)
    f = L6.feats("train", tgt, A["s1_ids"][w], A["sx_ids"][w], L6.fk_counts("train", tgt))
    cx = pd.read_parquet(L6.W / f"rf_train_sx_{tgt}.parquet", columns=["id", "core"]).set_index("id").core.reindex(A["sx_ids"][w]).fillna("").to_numpy()
    c1 = pd.read_parquet(L6.W / f"rf_train_s1_{tgt}.parquet", columns=["id", "core"]).set_index("id").core.reindex(A["s1_ids"][w]).fillna("").to_numpy()
    base = (c1 == cx) & (c1 != "") & (f["core_n"] == 1)
    sp = X.lf_split("train", A["s1_ids"][w], A["sx_ids"][w], f["fk_eq"])
    d = pd.DataFrame({"s1": A["s1_ids"][w], "sx": A["sx_ids"][w], "y": A["y"][w]})
    q = pd.read_parquet(L6.W / f"p38/r/r_qwen_loco{src}_holdout_{tgt}.parquet")
    d = d.merge(q[["s1", "sx", "r"]], on=["s1", "sx"], how="left")
    for k in ("a", "b"):
        g = d[base & sp[k]]
        r = g.dropna(subset=["r"])
        print(f"{src}->{tgt} {k}: n {len(g)}, qwen scored {len(r)}; true: r>=.5 {(r.r[r.y] >= .5).mean():.2f} (n {int(r.y.sum())}); false: r>=.5 {(r.r[~r.y] >= .5).mean() if (~r.y).any() else float('nan'):.2f} (n {int((~r.y).sum())})", flush=True)
    del A
