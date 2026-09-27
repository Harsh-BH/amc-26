import sys, numpy as np, pandas as pd
sys.path.insert(0,"probes/P28-addr-anchor"); from loco import fbeta
for s_,t_ in (("US","India"),("India","US")):
    d=pd.read_parquet(f"data/work/n63/ctx_{s_}_{t_}.parquet",columns=["y","acc","pos","p","s1"])
    m=np.load(f"data/work/n63/meta_{s_}_{t_}.npz",allow_pickle=True); ntrue=m["ntrue"]; n=len(ntrue)
    y=d.y.to_numpy(); acc=d.acc.to_numpy(); pos=d.pos.to_numpy()
    per=lambda pr: fbeta(np.bincount(pos[pr&y],minlength=n),np.bincount(pos[pr],minlength=n),ntrue)
    f=per(acc).mean(); fp=per(acc&y).mean(); fn=per(acc|y).mean(); fb=per(y).mean()
    print(f"{s_}>{t_}: chain F {f:.5f} | no FP {fp:.5f} (+{fp-f:.5f}) | +all in-frame FN {per(acc|y).mean():.5f} | oracle frame {fb:.5f}")
    # emptiness
    pred_empty=(np.bincount(pos[acc],minlength=n)==0); tr_empty=ntrue==0
    print(f"   pred empty {pred_empty.mean():.4f} truth empty {tr_empty.mean():.4f}; wrong-empty {(pred_empty&~tr_empty).mean():.4f} wrong-nonempty {(~pred_empty&tr_empty).mean():.4f}")
