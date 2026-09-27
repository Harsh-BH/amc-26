import numpy as np, pandas as pd
for s_,t_ in (("US","India"),("India","US")):
    d=pd.read_parquet(f"data/work/n63/ctx_{s_}_{t_}.parquet")
    m=np.load(f"data/work/n63/meta_{s_}_{t_}.npz",allow_pickle=True); ntrue=m["ntrue"]; n=len(ntrue)
    acc=d.acc.to_numpy(); y=d.y.to_numpy()
    print(f"=== {s_}>{t_}: S1 {n}, truth/S1 {ntrue.mean():.3f}, acc/S1 {acc.sum()/n:.3f}, TP/S1 {(acc&y).sum()/n:.3f}, FP/S1 {(acc&~y).sum()/n:.4f}, "
          f"FN in frame/S1 {(~acc&y).sum()/n:.3f}, FN outside frame/S1 {(ntrue.sum()-y.sum())/n:.3f}")
    fn=d[~acc & y]; un=d[~acc]
    for col in ["pb","hrel","rel","extra","nsrc","src","noaddr","hedit"]:
        g=un.groupby(col).agg(n=("y","size"),true=("y","mean"),tp=("y","sum"))
        g["fn_share"]=g.tp/len(fn); print(col); print(g.sort_values("tp",ascending=False).head(8).round(3).to_string())
