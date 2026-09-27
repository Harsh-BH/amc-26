# aggregates only: accepted France pair sets of staged variants, their overlaps and empty-S1 rates
import numpy as np, pandas as pd, pyarrow.parquet as pq
W = '/home/harsh1/github-repos/amc-26/data/work/'
n1 = pq.ParquetFile(W+'rf_test_s1_France.parquet').metadata.num_rows
V = {'m2': ('p10/prob_test_France.parquet', .95), 'idf': ('p14/prob_test_France_idf144.parquet', .975),
     'bge': ('p13/prob_test_France_blend.parquet', .98), 'idfbge': ('p19/prob_test_France_idf144_bge.parquet', .985)}
S = {}
for k, (f, t) in V.items():
    d = pd.read_parquet(W+f, columns=['s1','sx','p'])
    am = d.p.to_numpy() == d.groupby('sx').p.transform('max').to_numpy()
    d = d[am & (d.p.to_numpy() >= t)]
    S[k] = set(zip(d.s1.to_numpy(), d.sx.to_numpy()))
    print(f"{k:7s} t {t}: {len(S[k]):,} pairs, {len(S[k])/n1:.3f}/S1, empty S1 {1-len({a for a,_ in S[k]})/n1:.4f}")
def rep(name, s):
    print(f"{name:22s}: {len(s):,} pairs, {len(s)/n1:.3f}/S1, empty S1 {1-len({a for a,_ in s})/n1:.4f}")
rep('idf & bge', S['idf'] & S['bge'])
rep('idfbge & m2', S['idfbge'] & S['m2'])
rep('idfbge & idf', S['idfbge'] & S['idf'])
vote = [p for p in S['idf'] | S['bge'] | S['idfbge'] if (p in S['idf']) + (p in S['bge']) + (p in S['idfbge']) >= 2]
rep('2-of-3 idf/bge/idfbge', set(vote))
print('only-in idfbge vs idf:', len(S['idfbge'] - S['idf']), ' only-in idf vs idfbge:', len(S['idf'] - S['idfbge']))
