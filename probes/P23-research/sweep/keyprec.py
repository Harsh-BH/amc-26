# aggregates only: on the labelled holdout (#8c p, P23 dump), precision of "unique exact (core,hn) S1 key + same city"
# pairs, overall and in the band a France-style high threshold would cut; plus how often the key S1 loses the argmax.
import sys, numpy as np, pandas as pd, pyarrow.parquet as pq
W = '/home/harsh1/github-repos/amc-26/data/work/'
for c in sys.argv[1:]:
    h = pd.read_parquet(W+f'p23/pairs_{c}.parquet', columns=['s1','sx','holdout','y','p','am'])
    h = h[h.holdout]
    s1 = pq.read_table(W+f'rf_train_s1_{c}.parquet', columns=['id','core','hn','city']).to_pandas()
    k1 = s1[(s1.core != '') & (s1.hn != '')].groupby(['core','hn']).id.agg(['first','size']).reset_index()
    k1 = k1[k1['size'] == 1]
    key1 = dict(zip(k1['first'], zip(k1.core, k1.hn)))
    city1 = dict(zip(s1.id, s1.city)); del s1, k1
    sxids = set(h.sx)
    sx = pq.read_table(W+f'rf_train_sx_{c}.parquet', columns=['id','core','hn','city']).to_pandas()
    sx = sx[sx.id.isin(sxids)]
    kx = dict(zip(sx.id, zip(sx.core, sx.hn))); cx = dict(zip(sx.id, sx.city)); del sx
    isk = np.array([key1.get(a) is not None and key1.get(a) == kx.get(b) and city1.get(a, '') != '' and city1.get(a) == cx.get(b)
                    for a, b in zip(h.s1, h.sx)])
    k = h[isk]
    print(f"{c}: holdout key+city pairs {len(k):,}; true share {k.y.mean():.4f}; argmax share {k.am.mean():.4f}")
    for lo, hi in [(0, .3), (.3, .71), (.71, .9), (.9, .95), (.95, .985), (.985, 1.01)]:
        b = k[(k.p >= lo) & (k.p < hi)]
        print(f"   p [{lo},{hi}): n {len(b):,} true {b.y.mean() if len(b) else float('nan'):.4f}; argmax&true {(b.am & b.y).mean() if len(b) else float('nan'):.4f}")
    na = k[~k.am]
    print(f"   key S1 not argmax: {len(na):,}, of which true {na.y.mean() if len(na) else float('nan'):.4f}")
