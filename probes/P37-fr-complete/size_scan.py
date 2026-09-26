import sys, json, numpy as np, pandas as pd
sys.argv=['x']; sys.path.insert(0,'probes/P37-fr-complete'); sys.path.insert(0,'probes/P28-addr-anchor')
import p37 as P
fr_res=json.load(open('probes/P37-fr-complete/fr.json'))
for src,tgt in (("US","India"),("India","US")):
    a=P.loco_load(src,tgt); s1,sx,is2,p,am,y,sib,hc,n=(a[k] for k in ("s1","sx","is2","p","am","y","sib","hc","n_s1"))
    tb=P.TS[(src,tgt)]; tmap={t:P.inv_dens(p,am,tb,g,n) for t,g in fr_res["dens"].items()}
    pred=am&(p>=tb); pred|=P.rescue(s1,p,am,pred,sib,tmap["0.9"])
    t=tmap["0.8"]
    ex=P.gc_add(s1,sx,is2,p,am,pred,sib,t,"all")&~P.gc_add(s1,sx,is2,p,am,pred,sib,t,"oneall")
    _,n2,n3=P.one_sided(s1,is2,pred); same=np.where(is2,n2,n3); tot=n2+n3
    df=pd.DataFrame({'same':np.minimum(same,4),'tot':np.minimum(tot,6),'y':y,'hc':hc,'pb':pd.cut(p,[0,t+(tb-t)/2,tb]).astype(str)})[ex]
    print(src,'->',tgt,'two-sided adds',len(df),'prec',round(df.y.mean(),3))
    print(' by same-source count:',df.groupby('same').y.agg(['size','mean']).round(3).T.to_dict())
    print(' by total size:',df.groupby('tot').y.agg(['size','mean']).round(3).T.to_dict())
    print(' by p half:',df.groupby('pb').y.agg(['size','mean']).round(3).T.to_dict())
    oa=P.gc_add(s1,sx,is2,p,am,pred,sib,t,"oneall")
    d2=pd.DataFrame({'same_other':np.minimum(np.where(is2,n3,n2),4),'y':y,'hc':hc})[oa]
    print(' oneall adds by other-source count:',d2.groupby('same_other').y.agg(['size','mean']).round(3).T.to_dict())
    print(' oneall adds by hc:',d2.groupby('hc').y.agg(['size','mean']).round(3).T.to_dict())
    del a
