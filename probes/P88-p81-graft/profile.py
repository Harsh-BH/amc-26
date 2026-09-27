import csv, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path("/home/harsh1/github-repos/amc-26"); W=ROOT/"data/work"; S="/tmp/claude-1001/-home-harsh1-github-repos-amc-26/b9a894ea-d7e4-441a-9f08-e7799e3d80b9/scratchpad"
sys.path[:0]=[str(ROOT/"probes/P70-fr-roles")]
import p70, street_fr
R=dict(sep="\t",quoting=csv.QUOTE_NONE,dtype=str,keep_default_na=False)
cs=pd.read_parquet(f"{S}/p81/cs81A.parquet"); cs12=set(zip(*pd.read_parquet(f"{S}/p81/cs81A12.parquet").query("op=='add'")[["s1","sx"]].values.T))
t1=pd.read_parquet(W/"test_source1.parquet",columns=["entity_id","country"]); cty=dict(zip(t1.entity_id,t1.country))
cs["c"]=cs.s1.map(cty); print(cs.groupby(["c","op"]).size().to_dict())
a=cs[(cs.op=="add")&(cs.c=="France")].copy(); a["in12"]=[(s,x) in cs12 for s,x in zip(a.s1,a.sx)]
def owners(sub):
    m=pd.read_csv(ROOT/f"output/{sub}/matching_results.tsv",**R); m.columns=["s1","ids"]; m=m[m.ids!=""]
    m=m.assign(sx=m.ids.str.split(",")).explode("sx"); return dict(zip(m.sx,m.s1))
o84=owners("sub84"); print("SX already owned in sub84:",int(a.sx.isin(o84).sum()),"same S1:",int((a.sx.map(o84)==a.s1).sum())); a=a[~a.sx.isin(o84)].copy()
for sub in ("sub66_final","sub77"):
    o=owners(sub); a[f"was_{sub}"]=a.sx.map(o).notna(); a[f"same_{sub}"]=a.sx.map(o)==a.s1
d=pd.read_parquet(W/"p31/fr31v0.1_t9868.parquet",columns=["s1","sx","p"])
best=d.sort_values("p",ascending=False).drop_duplicates("sx").set_index("sx")
pm=dict(zip(zip(d.s1,d.sx),d.p)); a["p"]=[pm.get((s,x),np.nan) for s,x in zip(a.s1,a.sx)]
a["argmax"]=[best.s1.get(x)==s for s,x in zip(a.s1,a.sx)]
h=pd.read_parquet(W/"p70/changes_sub77_hnst.parquet"); hs=set(zip(h.s1,h.sx)); a["hnst"]=[(s,x) in hs for s,x in zip(a.s1,a.sx)]
# sub85 removals (xlf a + sub68fr)
o85=owners("sub85"); rem85={(s,x) for x,s in o84.items() if x not in o85}
m84=pd.read_csv(ROOT/"output/sub84/matching_results.tsv",**R)
a1=p70._raw("test",set(a.s1),[1],["business_address","country"]).business_address
ax=p70._raw("test",set(a.sx),[2,3],["business_address","country"]).business_address
A=a.reset_index(drop=True); fr_=lambda s: pd.Series(["France"]*len(s),index=s.index)
k1=p70._street(a1.reindex(A.s1).fillna("").reset_index(drop=True),fr_(A)); kx=p70._street(ax.reindex(A.sx).fillna("").reset_index(drop=True),fr_(A))
A["cls"]=[street_fr.classify(u,v) if u and v else ("no_street_s1" if not u else "no_street_sx") for u,v in zip(k1,kx)]
print("France adds",len(A),"| in 81A12",int(A.in12.sum()))
print("was owned in sub66_final:",int(A.was_sub66_final.sum()),"(same S1",int(A.same_sub66_final.sum()),") | in sub77:",int(A.was_sub77.sum()),"(same S1",int(A.same_sub77.sum()),")")
print("hnst overlap:",int(A.hnst.sum()),"| p>=.9868 (vetoed high-p):",int((A.p>=.9868).sum()),"| p NaN (not in frame):",int(A.p.isna().sum()),"| argmax share",round(A.argmax.mean(),3))
print("p quartiles",A.p.quantile([.1,.25,.5,.75,.9]).round(3).tolist())
print(A.groupby("cls").agg(n=("sx","size"),p_med=("p","median"),in12=("in12","sum")).round(3).to_string())
A.to_parquet(f"{S}/p81/fr_adds.parquet")
