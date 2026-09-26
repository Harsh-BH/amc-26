"""Expected precision of sub45's TEST flips, from holdout strata (P34 shift.py's idea): holdout h1 flips of the h0-fit
booster, binned by region (#8c p < .02 / band / >= .98) x l_r_all quintile x p' band; each test flip gets its stratum's
holdout P(true) (adds) or P(false) (drops). Also the per-bin flip rate test / holdout.
    P45_VAR=psumoff? shift45.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import te45 as T

Z, A, W = T.Z, T.A, T.W


def strata(p8, lr, pn):
    reg = np.select([p8 < 0.02, p8 >= 0.98], [0, 2], 1)
    rb = np.digitize(lr, [-4, -1, 1, 4])                 # l_r_all bins (logit)
    pb = np.digitize(pn, [0.3, 0.71, 0.9])               # new p' bins
    return reg * 100 + rb * 10 + pb


def main():
    import lightgbm as lgb
    cols = json.load(open(W / f"{T.TAG}_cols.json"))["cols"]
    b, b34 = lgb.Booster(model_file=str(W / f"{T.TAG}_h0.txt")), lgb.Booster(model_file=str(T.P34 / "a10_h0.txt"))
    S = Z.load(["r_all"])
    out = {}
    for s in S:
        c, a = s["c"], s["a"]
        rh = s["h"][a.g.to_numpy()] == 1
        base = list(A.COLS)
        q = A.stacked(b34, a, True)
        try:
            A.COLS = cols
            p = A.stacked(b, a, True)
        finally:
            A.COLS = base
        y, p8, lr = a.y.to_numpy(), a.p.to_numpy(), a.l_r_all.to_numpy()
        add, drop = rh & (p >= A.T) & (q < A.T), rh & (q >= A.T) & (p < A.T)
        st = strata(p8, lr, p)
        H = pd.DataFrame({"st": st, "add": add, "drop": drop, "y": y})
        pa = H[H["add"]].groupby("st").y.agg(["mean", "size"])
        pdr = H[H["drop"]].groupby("st").y.agg(["mean", "size"])
        # test flips
        pr = pd.read_parquet(W / f"prob_test_{c}_{T.TAG}.parquet", columns=["p", "p34", "p8c", "am"])
        r = pd.read_parquet(T.P34 / f"te_pairs_{c}.parquet", columns=["s1", "sx", "r"])
        ra = pd.read_parquet(W / "oof" / f"r_all_test_{c}.parquet").rename(columns={"r": "ra"})
        m = pr.am.to_numpy()
        r = r[m].merge(ra, on=["s1", "sx"], how="left")
        rr = np.where(r.r.isna(), r.ra, r.r).astype(np.float64)
        assert not np.isnan(rr).any()
        tp, tq, t8 = pr.p.to_numpy()[m], pr.p34.to_numpy()[m], pr.p8c.to_numpy()[m]
        tadd, tdrop = (tp >= A.T) & (tq < A.T), (tq >= A.T) & (tp < A.T)
        tst = strata(t8, Z.lg(rr), tp)
        ea = pd.Series(tst[tadd]).map(pa["mean"])
        ed = 1 - pd.Series(tst[tdrop]).map(pdr["mean"])
        out[c] = {"holdout_h1": {"adds": int(add.sum()), "add_prec": float(y[add].mean()), "drops": int(drop.sum()),
                                 "drop_false": float(1 - y[drop].mean())},
                  "test": {"adds": int(tadd.sum()), "drops": int(tdrop.sum()),
                           "exp_add_prec": float(ea.mean()), "add_unmatched_strata": int(ea.isna().sum()),
                           "exp_drop_false": float(ed.mean()), "drop_unmatched_strata": int(ed.isna().sum())}}
        # veto stratum mix: test vs holdout share of drops by l_r_all bin
        out[c]["drop_rbin_share"] = {"holdout": pd.Series((st[drop] // 10) % 10).value_counts(normalize=True).round(3).to_dict(),
                                     "test": pd.Series((tst[tdrop] // 10) % 10).value_counts(normalize=True).round(3).to_dict()}
        print(c, json.dumps(out[c]), flush=True)
    json.dump(out, open(W / f"{T.TAG}_shift.json", "w"), indent=1)


if __name__ == "__main__":
    main()
