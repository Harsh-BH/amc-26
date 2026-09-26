# Holdout truth: argmax sub1 swap pairs by swapped-in word lens band x house-number relation. Precision + accepted share.
import sys; import numpy as np, pandas as pd
sys.path[:0] = ["probes/P46-noise-word-tier"]
import p46, r05 as R
W = p46.W
for c in ["US", "India"]:
    d = pd.read_parquet(W / f"p23/pairs_{c}.parquet"); d = d[d.holdout & d.am].reset_index(drop=True)
    lt = p46.lens_table(d[["s1", "sx"]], "train", c)
    x = d.copy(); x["w"] = p46.swap_words(x[["s1", "sx"]], "train", c); x = x[x.w != ""]
    r1 = R._rf(W / f"rf_train_s1_{c}.parquet", pd.unique(x.s1), ["hn"]); rx = R._rf(W / f"rf_train_sx_{c}.parquet", pd.unique(x.sx), ["hn"])
    h1, hx = r1.hn.reindex(x.s1).fillna("").to_numpy(), rx.hn.reindex(x.sx).fillna("").to_numpy()
    x["hrel"] = np.where((h1 != "") & (h1 == hx), "hn_eq", np.where((h1 != "") & (hx != ""), "hn_diff", "hn_missing"))
    x = x.merge(lt[["s", "e", "n"]], left_on="w", right_index=True, how="left")
    x["band"] = pd.cut(x.s, [-1, .05, .1, .2, .4, 1.01], labels=["<=.05", ".05-.1", ".1-.2", ".2-.4", ">.4"])
    g = x.groupby(["band", "hrel"], observed=True).agg(n=("y", "size"), prec=("y", "mean"), acc=("pred", "mean"),
                                                      acc_prec=("y", lambda s: s[x.loc[s.index, "pred"]].mean()))
    print(c); print(g.round(3).to_string())
    print(lt[lt.n >= 20].sort_values("s", ascending=False).head(8)[["n", "s", "e"]].round(3).to_string())
