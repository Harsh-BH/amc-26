import sys, time, pickle, json; sys.path[:0]=['probes/P3-block','code/business_entity_resolution/src']
import numpy as np, pandas as pd, block as B, evalb as E
from worlds import load_world
s1,sx,truth=load_world('dev')
fitsx=pd.read_parquet('data/work/p3/fit300k.parquet')
tp=E.truth_df(truth); tp["country"]=tp.s1.map(s1.set_index("entity_id").country); tp["cls"]=E.classify(tp,s1,sx)
COMBOS={'f:na20':({'na':20},{}),'r:na1':({},{'na':1}),'r:na2':({},{'na':2}),'f:na10+r:na2':({'na':10},{'na':2}),
        'r:na2,name1,addr1':({},{'na':2,'name':1,'addr':1}),
        'f:na10,name5,addr5+r:na2,name1,addr1':({'na':10,'name':5,'addr':5},{'na':2,'name':1,'addr':1})}
variants=json.loads(sys.argv[1])
rows=[]
for name,over in variants.items():
    cfg={**B.CFG,"views":{"na":20,"name":10,"addr":10},"rev":{"na":3,"name":2,"addr":2},**over}
    t=time.time(); vecs=B.fit_views(fitsx,cfg,sample=300_000); tfit=time.time()-t
    t=time.time(); h=E.hits_world(s1,sx,cfg,vecs,log=lambda *_:None); dt=time.time()-t
    for n,(f,r) in COMBOS.items():
        m=E.metrics(E.select(h,f,r),tp,s1); rows.append(dict(var=name,combo=n,fit_s=round(tfit),secs=round(dt,1),**{k:round(v,4) for k,v in m.items() if k in('pair_recall','cands_per_s1','s1_full_recall','recall_India','recall_US','recall_cross_script','recall_no_token_overlap','recall_website','recall_token_overlap')}))
    print(pd.DataFrame(rows[-len(COMBOS):]).to_string(),flush=True)
import resource; print('peak GB',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6)
pd.DataFrame(rows).to_csv(f'data/work/p3/var_{sys.argv[2]}.csv',index=False)
