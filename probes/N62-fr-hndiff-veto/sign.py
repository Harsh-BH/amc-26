import numpy as np, pandas as pd
SIB={3,4,5,7,9,11,13,21}; W="data/work"
def dnum(h1,hx):
    a=pd.to_numeric(pd.Series(h1).astype(str).str.extract(r'^(\d+)$')[0],errors='coerce').to_numpy(float)
    b=pd.to_numeric(pd.Series(hx).astype(str).str.extract(r'^(\d+)$')[0],errors='coerce').to_numpy(float)
    return b-a
def tab(d,lab):
    s=pd.Series(d[np.isin(np.abs(d),list(SIB))]); print(lab, 'n',len(s),' +:',int((s>0).sum()),' -:',int((s<0).sum()), ' by |d|:', s.abs().value_counts().sort_index().to_dict())
    nn=pd.Series(d[np.isin(np.abs(d),[1,2])]); print('    near |d|<=2 n',len(nn),'+',int((nn>0).sum()),'-',int((nn<0).sum()), ' |d|=6,8,10:', int(np.isin(np.abs(d),[6,8,10]).sum()))
gt=pd.read_parquet(f"{W}/gt_pairs.parquet")
for c in ("US","India"):
    s1=pd.read_parquet(f"{W}/rf_train_s1_{c}.parquet",columns=["id","hn"]).set_index("id").hn
    sx=pd.read_parquet(f"{W}/rf_train_sx_{c}.parquet",columns=["id","hn"]).set_index("id").hn
    g=gt[gt.s1.isin(s1.index)]
    tab(dnum(s1.reindex(g.s1).to_numpy(), sx.reindex(g.sx).to_numpy()), f"{c} TRUE")
f=pd.read_parquet(f"{W}/p31/pairs_France.parquet",columns=["s1","hn_1","hn_x","acc","p"])
a=f[f.acc.astype(bool)]
d=dnum(a.hn_1.to_numpy(),a.hn_x.to_numpy()); tab(d,"France ACCEPTED")
u=f[~f.acc.astype(bool)]; tab(dnum(u.hn_1.to_numpy(),u.hn_x.to_numpy()),"France candidates NOT accepted")
m=np.isin(np.abs(d),list(SIB)); print('France accepted sib pairs:',m.sum(),'on S1',a.s1[m].nunique(),' p quantiles',np.quantile(a.p[m],[.1,.25,.5,.75,.9]).round(4))
