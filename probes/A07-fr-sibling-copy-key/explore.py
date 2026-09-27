import sys, numpy as np, pandas as pd
sys.path[:0] = ["probes/P28-addr-anchor", "probes/N63-floor-probes", "probes/N76-hn-edit-ops"]
from loco import fbeta
from cells import capped_add
import math
from scipy.stats import norm

def wilson_lb(hits, n, conf=0.95):
    if n == 0: return 0.0
    z = norm.ppf(1 - (1 - conf) / 2)
    p = hits / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (center - spread) / denom

for s_, t_ in (('US', 'India'), ('India', 'US')):
    print(f"== {s_}>{t_} (target {t_})")
    d = pd.read_parquet(f"data/work/n63/chain_{s_}_{t_}.parquet", columns=["s1","sx","p","y","src","acc","pos","am", "hn_eq", "hrel", "rel", "sib"])
    m = np.load(f"data/work/n63/meta_{s_}_{t_}.npz", allow_pickle=True)
    ntrue, hv = m["ntrue"], m["hv"]
    n = len(ntrue)
    
    sx_df = pd.read_parquet(f"data/work/rf_train_sx_{t_}.parquet", columns=["id", "core", "hn", "city"])
    sx_df = sx_df[sx_df.core != '']
    
    acc_mask = d.acc.to_numpy()
    acc_df = d[acc_mask]
    sx_to_acc_s1 = dict(zip(acc_df.sx, acc_df.s1))
    
    sx_df['acc_s1'] = sx_df.id.map(sx_to_acc_s1)
    sx_df['core'] = sx_df['core'].fillna('')
    sx_df['hn'] = sx_df['hn'].fillna('')
    sx_df['city'] = sx_df['city'].fillna('')
    
    group_acc_s1 = sx_df.dropna(subset=['acc_s1']).groupby(['core', 'hn', 'city'])['acc_s1'].unique()
    owner_groups = group_acc_s1[group_acc_s1.str.len() == 1].apply(lambda x: x[0])
    owner_groups_df = owner_groups.reset_index().rename(columns={'acc_s1': 'ow'})
    
    sx_df = sx_df.merge(owner_groups_df, on=['core', 'hn', 'city'], how='inner')
    
    owned = d.sx.isin(set(d.sx[acc_mask])).to_numpy()
    d['free'] = d.am.to_numpy() & ~acc_mask & ~owned
    
    # R0 candidate mask
    owner_map = dict(zip(sx_df.id, sx_df.ow))
    d['r0_ow'] = d.sx.map(owner_map)
    d['is_r0'] = (d.s1 == d.r0_ow) & d.free
    
    r0_d = d[d.is_r0]
    y_r0 = r0_d.y.to_numpy()
    p_r0 = r0_d.p.to_numpy()
    
    print(f"frame chain_{s_}_{t_} target {t_}:")
    print(f"free argmax pairs: {len(r0_d)} ({len(r0_d)/n:.3f}/S1)")
    print(f"precision: {y_r0.mean():.3f}")
    
    b1 = p_r0 < 0.1
    b2 = (p_r0 >= 0.1) & (p_r0 < 0.3)
    b3 = (p_r0 >= 0.3) & (p_r0 < 0.5)
    b4 = p_r0 >= 0.5
    for tag, b in zip(['p<.1', 'p .1-.3', 'p .3-.5', 'p>=.5'], [b1, b2, b3, b4]):
        print(f"{tag}: {b.sum()} @{y_r0[b].mean() if b.sum() else 0:.3f}")
