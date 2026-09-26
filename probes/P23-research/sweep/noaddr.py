import pandas as pd, numpy as np
D = "/home/harsh1/github-repos/amc-26/data/work/p23"
for c in ("India", "US"):
    e = pd.read_parquet(f"{D}/err_{c}.parquet")
    pr = pd.read_parquet(f"{D}/pairs_{c}.parquet", columns=["holdout", "y", "pred"])
    tp_total = int((pr.holdout & pr.y & pr.pred).sum())
    ctx = pd.read_parquet(f"{D}/s1ctx_{c}.parquet")
    n_true = int(ctx.n_true.sum())
    a = e.sx_addr.fillna("").str.strip().str.upper()
    raw_noaddr = a.isin(["", "NULL", "<NULL>", "N/A", "NA", "NONE", "-"])
    nouse = e.sx_hn.fillna("").eq("") & e.sx_city.fillna("").eq("") & e.sx_places.fillna("").eq("")
    e = e.assign(raw=raw_noaddr, nouse=nouse)
    tp = e[e.kind == "tp"]
    out = {}
    for flag in ("raw", "nouse"):
        tp_cls = tp[flag].mean() * tp_total
        fn_cls = e[e.kind.isin(["missed", "fn_lowp", "fn_stolen"]) & e[flag]].shape[0]
        fn_all = e[e.kind.isin(["missed", "fn_lowp", "fn_stolen"])].shape[0]
        out[flag] = dict(share_true=round((tp_cls + fn_cls) / n_true, 4), recall=round(tp_cls / (tp_cls + fn_cls), 3),
                         share_of_fn=round(fn_cls / fn_all, 3), fp_share=round(e[(e.kind == "fp")][flag].mean(), 3))
    print(c, "n_true", n_true, "per S1", round(n_true / len(ctx), 3), "tp", tp_total, out)
    # other-class recall (addressed)
    fn = e[e.kind.isin(["missed", "fn_lowp", "fn_stolen"]) & ~e.raw]
    tpa = (1 - tp.raw.mean()) * tp_total
    print("   addressed-SX recall", round(tpa / (tpa + len(fn)), 4), "addressed FN by kind", fn.kind.value_counts().to_dict())
