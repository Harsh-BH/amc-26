import sys, json, numpy as np, pandas as pd
sys.argv=['x']; sys.path.insert(0,'probes/P37-fr-complete')
import p37 as P
d,a=P.fr_load()
add=P.gc_add(a["s1"],a["sx"],a["is2"],a["p"],a["am"],a["pred"],a["sib"],0.7,"oneall")
pred=a["pred"]|add
miss,n2,n3=P.one_sided(a["s1"],a["is2"],pred)
s1ids=d.s1.to_numpy()
# per S1: is one-sided missing S2 / S3, and has candidate of missing source in kept file
g=pd.DataFrame({'s1':s1ids,'is2':a["is2"],'miss':miss,'n2':n2,'n3':n3})
st=g.groupby('s1').agg(n2=('n2','first'),n3=('n3','first'),c2=('is2','sum'),c=('is2','size'))
st['c3']=st.c-st.c2
st['miss_src']=np.where((st.n2==0)&(st.n3>0),'S2',np.where((st.n3==0)&(st.n2>0),'S3',''))
st['nocand']=np.where(st.miss_src=='S2',st.c2==0,np.where(st.miss_src=='S3',st.c3==0,False))
print('one-sided after gc70:',(st.miss_src!='').sum(),' no missing-source cand in kept file:',st.nocand.sum(), st[st.nocand].miss_src.value_counts().to_dict())
tgt=st[st.nocand].miss_src
assigned=set(d.sx[pred])
full=pd.read_parquet('data/work/p14/w_idfx/p3/cands_test_France.parquet',columns=['s1','sx','block_rank_s1','via'])
full=full[full.s1.isin(tgt.index)]
full=full[full.sx.str[:2].to_numpy()==tgt.reindex(full.s1).to_numpy()]
print('full-list missing-source pairs for them:',len(full),' S1 covered:',full.s1.nunique())
full['free']=~full.sx.isin(assigned)
rf1=pd.read_parquet('data/work/rf_test_s1_France.parquet',columns=['id','hn','city','core']).set_index('id')
rfx=pd.read_parquet('data/work/p29/w_fix/rf_test_sx_France.parquet',columns=['id','hn','city','core']).set_index('id')
x1=rf1.reindex(full.s1); xx=rfx.reindex(full.sx)
h1,hx=x1.hn.fillna('').to_numpy(),xx.hn.fillna('').to_numpy()
full['hc']=P.hn_cat(h1,hx); full['sib']=P.sib_mask(h1,hx)
full['city_eq']=(x1.city.fillna('').to_numpy()==xx.city.fillna('').to_numpy())&(x1.city.fillna('').to_numpy()!='')
full['core_eq']=(x1.core.fillna('').to_numpy()==xx.core.fillna('').to_numpy())&(x1.core.fillna('').to_numpy()!='')
full['in_kept']=pd.MultiIndex.from_arrays([full.s1,full.sx]).isin(pd.MultiIndex.from_arrays([d.s1,d.sx]))
print('in kept file (should be 0):',full.in_kept.sum())
print('free share',round(full.free.mean(),3),' hc dist',full.hc.value_counts().sort_index().to_dict(),' sib',full.sib.sum())
k=full[full.free&~full.sib]
strong=k[(k.hc==0)&k.city_eq]
print('free non-sib pairs',len(k),'S1',k.s1.nunique(),' hn-eq & same city:',len(strong),'S1',strong.s1.nunique(),' +core eq:',int(strong.core_eq.sum()))
print('block_rank_s1 of strong:',strong.block_rank_s1.describe().round(1).to_dict())
print('via of strong:',strong.via.value_counts().to_dict())
tok=lambda s: set(w for w in str(s).split() if len(w)>=3)
ov=np.array([len(tok(a_)&tok(b_))>0 for a_,b_ in zip(rf1.core.reindex(strong.s1).fillna(''),rfx.core.reindex(strong.sx).fillna(''))])
print('strong pool: share with >=1 shared name-core token:',round(ov.mean(),3),' pairs',int(ov.sum()),'S1',strong.s1[ov].nunique())
allk=k.copy()
ov2=np.array([len(tok(a_)&tok(b_))>0 for a_,b_ in zip(rf1.core.reindex(allk.s1).fillna(''),rfx.core.reindex(allk.sx).fillna(''))])
print('all free non-sib pruned pairs: shared-token share',round(ov2.mean(),3),' pairs',int(ov2.sum()),'S1',allk.s1[ov2].nunique(),
      ' of which hn-eq',int((ov2&(allk.hc.to_numpy()==0)).sum()),' hn-missing',int((ov2&(allk.hc.to_numpy()==2)).sum()))
