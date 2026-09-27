# aggregates only: effect of a France key tier (argmax pairs with a unique exact (core,hn) S1 key + same city accepted
# at t_key instead of t) on the #9ab France file: pairs added, S1 changed, empty S1 filled, new profile.
import sys, numpy as np, pandas as pd, pyarrow.parquet as pq
W = '/home/harsh1/github-repos/amc-26/data/work/'
f, t = 'p19/prob_test_France_idf144_bge.parquet', 0.985
s1 = pq.read_table(W+'rf_test_s1_France.parquet', columns=['id','core','hn','city']).to_pandas()
sx = pq.read_table(W+'rf_test_sx_France.parquet', columns=['id','core','hn','city']).to_pandas()
n1 = len(s1)
k1 = s1[(s1.core != '') & (s1.hn != '')].groupby(['core','hn']).id.agg(['first','size']).reset_index()
k1 = k1[k1['size'] == 1].rename(columns={'first': 's1'})
m = sx[(sx.core != '') & (sx.hn != '')].merge(k1[['core','hn','s1']], on=['core','hn']).rename(columns={'id':'sx','city':'cix'})
m = m.merge(s1[['id','city']].rename(columns={'id':'s1'}), on='s1')
key = set(zip(m.s1[(m.cix == m.city) & (m.city != '')], m.sx[(m.cix == m.city) & (m.city != '')]))
d = pd.read_parquet(W+f, columns=['s1','sx','p'])
d = d[d.p.to_numpy() == d.groupby('sx').p.transform('max').to_numpy()]
base = d[d.p >= t]; bs1 = set(base.s1)
isk = np.array([(a, b) in key for a, b in zip(d.s1, d.sx)])
for tk in (0.95, 0.90, 0.71):
    add = d[isk & (d.p >= tk) & (d.p < t)]
    new_nonempty = len(set(add.s1) - bs1)
    tot = len(base) + len(add)
    print(f"t_key {tk}: +{len(add):,} pairs, {add.s1.nunique():,} S1 changed ({add.s1.nunique()/n1:.4f}), "
          f"{new_nonempty:,} empty S1 filled; profile {tot/n1:.3f}/S1, empty {1-(len(bs1)+new_nonempty)/n1:.4f}")
