# aggregates only: share of argmax "unique exact (core,hn) S1 key + same city" test pairs that a country's t rejects
#   keyrej.py <country> <prob parquet (under data/work)> <t>
import sys, numpy as np, pandas as pd, pyarrow.parquet as pq
W = '/home/harsh1/github-repos/amc-26/data/work/'
c, f, t = sys.argv[1], sys.argv[2], float(sys.argv[3])
s1 = pq.read_table(W+f'rf_test_s1_{c}.parquet', columns=['id','core','hn','city']).to_pandas()
k1 = s1[(s1.core != '') & (s1.hn != '')].groupby(['core','hn']).id.agg(['first','size']).reset_index()
k1 = k1[k1['size'] == 1].rename(columns={'first': 's1'})
city1 = s1[['id','city']].rename(columns={'id':'s1'}); del s1
sx = pq.read_table(W+f'rf_test_sx_{c}.parquet', columns=['id','core','hn','city']).to_pandas()
m = sx[(sx.core != '') & (sx.hn != '')].merge(k1[['core','hn','s1']], on=['core','hn']).rename(columns={'id':'sx','city':'cix'}); del sx
m = m.merge(city1, on='s1'); m = m[(m.cix == m.city) & (m.city != '')][['s1','sx']]
d = pd.read_parquet(W+f, columns=['s1','sx','p'])
d = d[d.p.to_numpy() == d.groupby('sx').p.transform('max').to_numpy()]
k = d.merge(m, on=['s1','sx'])
print(f"{c} {f} t {t}: argmax key+city pairs {len(k):,}; rejected (p < t) {(k.p < t).mean():.4f}; p < 0.95 {(k.p < 0.95).mean():.4f}")
