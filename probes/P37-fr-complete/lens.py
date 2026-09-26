import sys, numpy as np, pandas as pd
sys.argv=['x']; sys.path.insert(0,'probes/P37-fr-complete')
import p37 as P
d,a=P.fr_load()
rf1=pd.read_parquet('data/work/rf_test_s1_France.parquet',columns=['id','core','city']).set_index('id')
rfx=pd.read_parquet('data/work/p29/w_fix/rf_test_sx_France.parquet',columns=['id','core','city']).set_index('id')
tok=lambda s: set(w for w in str(s).split() if len(w)>=3)
def lens(m,tag):
    s1,sx=d.s1[m],d.sx[m]
    c1,cx=rf1.core.reindex(s1).fillna('').to_numpy(),rfx.core.reindex(sx).fillna('').to_numpy()
    ov=np.mean([len(tok(x)&tok(y))>0 for x,y in zip(c1,cx)])
    ceq=np.mean(c1==cx)
    ci=np.mean(rf1.city.reindex(s1).fillna('').to_numpy()==rfx.city.reindex(sx).fillna('').to_numpy())
    hc=np.bincount(a['hc'][m],minlength=4)/m.sum()
    print(f"{tag:28s} n {int(m.sum()):7,d} hn eq/both-miss/one-miss/diff {hc.round(3).tolist()} name-token overlap {ov:.3f} core equal {ceq:.3f} same city {ci:.3f}")
lens(a['pred'],'accepted (sub34v_r90)')
lens(a['pred']&(a['p']<0.995),'accepted, p<0.995')
for t in (0.9,0.8,0.7):
    add=P.gc_add(a["s1"],a["sx"],a["is2"],a["p"],a["am"],a["pred"],a["sib"],t,"oneall")
    lens(add,f'gc oneall add t {t}')
    lens(add&(a['p']<t+0.1),f'  of which p<{t+0.1:.1f}')
ex=P.gc_add(a["s1"],a["sx"],a["is2"],a["p"],a["am"],a["pred"],a["sib"],0.9,"allhn",a['hc']==0)&~P.gc_add(a["s1"],a["sx"],a["is2"],a["p"],a["am"],a["pred"],a["sib"],0.9,"oneall")
lens(ex,'allhn extra (two-sided) 0.9')
