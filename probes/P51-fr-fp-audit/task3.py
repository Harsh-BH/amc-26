import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "data/work"

print("--- France ---")
d = pd.read_parquet(W / "p31/pairs_France.parquet", columns=["s1", "sx", "acc", "p"])
acc = d[d.acc == True].copy()
acc['is_s2'] = acc.sx.str.startswith('S2')
acc['is_s3'] = acc.sx.str.startswith('S3')

g = acc.groupby('s1').agg(n_s2=('is_s2', 'sum'), n_s3=('is_s3', 'sum'))

n_s1 = 259452

s2_shares = [(g.n_s2 == i).sum() / n_s1 for i in range(6)]
s2_shares.append((g.n_s2 >= 6).sum() / n_s1)
print(f"France S2 shares: {np.round(s2_shares, 3)}")

s3_shares = [(g.n_s3 == i).sum() / n_s1 for i in range(7)]
s3_shares.append((g.n_s3 >= 7).sum() / n_s1)
print(f"France S3 shares: {np.round(s3_shares, 3)}")

# US/India truth shares
print("\n--- US / India (predicted / truth) ---")
for c in ["US", "India"]:
    d_c = pd.read_parquet(W / f"p23/pairs_{c}.parquet", columns=["s1", "sx", "holdout", "pred", "y"])
    d_c = d_c[d_c.holdout == True]
    n_s1_c = d_c.s1.nunique()
    
    for split, mask in [('pred', d_c.pred == True), ('truth', d_c.y == True)]:
        sub = d_c[mask].copy()
        sub['is_s2'] = sub.sx.str.startswith('S2')
        sub['is_s3'] = sub.sx.str.startswith('S3')
        g_c = sub.groupby('s1').agg(n_s2=('is_s2', 'sum'), n_s3=('is_s3', 'sum'))
        s2_sh = [(g_c.n_s2 == i).sum() / n_s1_c for i in range(6)]
        s2_sh.append((g_c.n_s2 >= 6).sum() / n_s1_c)
        print(f"{c} {split} S2 shares: {np.round(s2_sh, 3)}")

# which pairs in over-full groups are the weakest?
# get pairs in S1 with >= 5 S2
bad_s1 = g[g.n_s2 >= 5].index
bad_pairs = acc[(acc.s1.isin(bad_s1)) & (acc.is_s2)]
print("\nFrance weakest S2 in >= 5 groups:")
print(bad_pairs.sort_values('p').head(10))

bad_s1_s3 = g[g.n_s3 >= 6].index
bad_pairs_s3 = acc[(acc.s1.isin(bad_s1_s3)) & (acc.is_s3)]
print("\nFrance weakest S3 in >= 6 groups:")
print(bad_pairs_s3.sort_values('p').head(10))
