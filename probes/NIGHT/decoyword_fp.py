# Do France same-hn pairs that SWAP IN a decoy word (P46 lens s high) get accepted (base p >= t, or hn-extra p >= 0.8)?
import sys; from pathlib import Path
import numpy as np, pandas as pd
sys.path[:0] = ["probes/P46-noise-word-tier"]
import p46
fr = pd.read_parquet("data/work/p31/pairs_France.parquet", columns=["s1", "sx", "p", "acc", "argmax", "hn_cls", "hn_1", "hn_x", "name_rel", "name_1", "name_x", "addr_1", "addr_x"])
am = fr.argmax.astype(bool).to_numpy()
lt = p46.lens_table(fr.loc[am, ["s1", "sx"]].reset_index(drop=True), "test", "France")
print(lt.sort_values("n", ascending=False).head(25)[["n", "s", "e"]].round(3).to_string())
x = fr[am & (fr.name_rel == "sub1") & (fr.hn_cls == 0) & (fr.hn_1 != "")].copy()
x["w"] = p46.swap_words(x[["s1", "sx"]].reset_index(drop=True), "test", "France")
x = x.merge(lt[["s", "e", "n"]], left_on="w", right_index=True, how="left")
taken = set(fr.sx[fr.acc]); nonemp = fr.groupby("s1").acc.transform("any")
x["nonemp"] = nonemp[x.index]; x["free"] = ~x.sx.isin(taken)
x["band"] = pd.cut(x.s, [-1, .05, .1, .2, .4, 1.01], labels=["<=.05", ".05-.1", ".1-.2", ".2-.4", ">.4"])
x["state"] = np.where(x.acc, "acc_base", np.where(x.free & (x.p >= 0.8) & x.nonemp, "hn_extra_adds", "rejected"))
print(pd.crosstab(x.band, x.state, dropna=False))
x.to_parquet("data/work/p54_sub1_hneq.parquet", index=False)
