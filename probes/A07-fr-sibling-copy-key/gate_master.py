import sys, numpy as np, pandas as pd
sys.path[:0] = ["probes/P28-addr-anchor", "probes/N63-floor-probes", "probes/N76-hn-edit-ops"]
from loco import fbeta
from cells import capped_add
import math
from scipy.stats import norm
import os

GOOD={'eq','typo','drop1','web'}

def wilson_lb(hits, n, conf=0.95):
    if n == 0: return 0.0
    z = norm.ppf(1 - (1 - conf) / 2)
    p = hits / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (center - spread) / denom

def evaluate():
    res = {}
    gt = pd.read_parquet("data/work/gt_pairs.parquet")
    sx_to_true_s1 = dict(zip(gt.sx, gt.s1))

    for s_, t_ in (('US', 'India'), ('India', 'US')):
        print(f"\n== {s_}>{t_} (target {t_})")
        d = pd.read_parquet(f"data/work/n63/chain_{s_}_{t_}.parquet", 
                            columns=["s1","sx","p","y","src","acc","pos","am", "hn_eq", "hrel", "rel", "sib"])
        m = np.load(f"data/work/n63/meta_{s_}_{t_}.npz", allow_pickle=True)
        ntrue, hv = m["ntrue"], m["hv"]
        n = len(ntrue)
        
        sx_df = pd.read_parquet(f"data/work/rf_train_sx_{t_}.parquet", columns=["id", "core", "hn", "city"])
        
        chain_s1_set = set(d.s1)
        # restricting the SX universe to SX whose true S1 is a chain S1 or none
        sx_df['true_s1'] = sx_df.id.map(sx_to_true_s1)
        sx_df = sx_df[sx_df.true_s1.isna() | sx_df.true_s1.isin(chain_s1_set)]
        
        # R0: core != ''
        sx_r0 = sx_df[sx_df.core != ''].copy()
        sx_r0['core'] = sx_r0['core'].fillna('')
        sx_r0['hn'] = sx_r0['hn'].fillna('')
        sx_r0['city'] = sx_r0['city'].fillna('')
        
        acc_mask = d.acc.to_numpy()
        acc_df = d[acc_mask]
        
        # owner groups for R0
        sx_to_acc_s1 = dict(zip(acc_df.sx, acc_df.s1))
        sx_r0['acc_s1'] = sx_r0.id.map(sx_to_acc_s1)
        
        group_acc_s1 = sx_r0.dropna(subset=['acc_s1']).groupby(['core', 'hn', 'city'])['acc_s1'].unique()
        owner_groups = group_acc_s1[group_acc_s1.str.len() == 1].apply(lambda x: x[0])
        owner_groups_df = owner_groups.reset_index().rename(columns={'acc_s1': 'ow'})
        
        sx_r0 = sx_r0.merge(owner_groups_df, on=['core', 'hn', 'city'], how='inner')
        owner_map_r0 = dict(zip(sx_r0.id, sx_r0.ow))
        
        km_counts = sx_r0.dropna(subset=['acc_s1']).groupby(['core', 'hn', 'city']).size()
        km_df = km_counts.reset_index().rename(columns={0: 'km'})
        sx_r0 = sx_r0.merge(km_df, on=['core', 'hn', 'city'], how='left')
        km_map = dict(zip(sx_r0.id, sx_r0.km))
        
        # R1: core != '', K2 = (core, city), restricted to groups where every member's hn is empty
        sx_r1 = sx_df.copy()
        sx_r1['core'] = sx_r1['core'].fillna('')
        sx_r1['hn'] = sx_r1['hn'].fillna('')
        sx_r1['city'] = sx_r1['city'].fillna('')
        sx_r1 = sx_r1[sx_r1.core != '']
        
        hn_nonempty_k2 = set(sx_r1[sx_r1.hn != ''][['core', 'city']].itertuples(index=False, name=None))
        sx_r1['k2'] = list(zip(sx_r1.core, sx_r1.city))
        sx_r1 = sx_r1[~sx_r1.k2.isin(hn_nonempty_k2)]
        
        sx_r1['acc_s1'] = sx_r1.id.map(sx_to_acc_s1)
        group_acc_s1_r1 = sx_r1.dropna(subset=['acc_s1']).groupby(['core', 'city'])['acc_s1'].unique()
        owner_groups_r1 = group_acc_s1_r1[group_acc_s1_r1.str.len() == 1].apply(lambda x: x[0])
        owner_groups_r1_df = owner_groups_r1.reset_index().rename(columns={'acc_s1': 'ow'})
        sx_r1 = sx_r1.merge(owner_groups_r1_df, on=['core', 'city'], how='inner')
        owner_map_r1 = dict(zip(sx_r1.id, sx_r1.ow))
        
        d['free'] = d.am.to_numpy() & ~acc_mask & ~d.sx.isin(set(d.sx[acc_mask])).to_numpy()
        d['r0_ow'] = d.sx.map(owner_map_r0)
        d['is_r0'] = (d.s1 == d.r0_ow) & d.free
        d['r1_ow'] = d.sx.map(owner_map_r1)
        d['is_r1'] = (d.s1 == d.r1_ow) & d.free
        
        d['km'] = d.sx.map(km_map).fillna(0)
        
        src_map = acc_df.groupby('sx')['src'].first().to_dict()
        sx_r0['acc_src'] = sx_r0.id.map(src_map)
        acc_src_groups = sx_r0.dropna(subset=['acc_src']).groupby(['core', 'hn', 'city'])['acc_src'].unique()
        group_to_acc_srcs = acc_src_groups.to_dict()
        sx_r0['k_tuple'] = list(zip(sx_r0.core, sx_r0.hn, sx_r0.city))
        d_k_tuples = d.sx.map(dict(zip(sx_r0.id, sx_r0.k_tuple)))
        d['acc_srcs'] = d_k_tuples.map(group_to_acc_srcs)
        d['src_in_acc'] = [src in srcs if isinstance(srcs, np.ndarray) else False for src, srcs in zip(d.src, d.acc_srcs)]
        
        acc_counts = acc_df.groupby(['s1', 'src']).size().unstack(fill_value=0)
        if 'S2' not in acc_counts.columns: acc_counts['S2'] = 0
        if 'S3' not in acc_counts.columns: acc_counts['S3'] = 0
        gc70_s1 = set(acc_counts[(acc_counts['S2'] > 0) ^ (acc_counts['S3'] > 0)].index)
        d['gc70'] = d.s1.isin(gc70_s1) & (d.p >= 0.70) & d.free
        d['hn_eq_80'] = d.hn_eq & (d.p >= 0.80) & d.free
        d['is_resid'] = ~d.gc70 & ~d.hn_eq_80
        
        y = d.y.to_numpy(); pos = d.pos.to_numpy(); p = d.p.to_numpy()
        per = lambda pr: fbeta(np.bincount(pos[pr & y], minlength=n), np.bincount(pos[pr], minlength=n), ntrue)
        f0 = per(acc_mask)
        
        print(f"R0 base stats (like master): free argmax pairs: {d.is_r0.sum()} ({d.is_r0.sum()/len(chain_s1_set):.3f}/S1) precision: {y[d.is_r0].mean():.3f}")
        for p_thresh in [0.0, 0.1, 0.3, 0.5, 0.8]:
            m = d.is_r0 & (p >= p_thresh)
            print(f"  p>={p_thresh}: cand {m.sum()} prec {y[m].mean() if m.any() else 0:.3f}")
            
        print("\n--- Tiers ---")
        def eval_tier(tag, m):
            add = capped_add(d, m)
            if not add.any():
                print(f"{tag:40s}: cand {m.sum():4d} add {add.sum():4d} prec 0")
                return
            prec = y[add].mean()
            wlb = wilson_lb(y[m].sum(), m.sum())
            dF = per(acc_mask | add) - f0
            se = lambda x: x.std()/np.sqrt(len(x))
            print(f"{tag:40s}: cand {m.sum():4d} add {add.sum():4d} prec {prec:.3f} (pool WLB {wlb:.3f}) dF {dF.mean():+.6f} h0 {dF[hv==0].mean():+.6f}±{se(dF[hv==0]):.6f} h1 {dF[hv==1].mean():+.6f}±{se(dF[hv==1]):.6f}")
        
        hne = d.sx.map(dict(zip(sx_r0.id, sx_r0.hn == ''))).fillna(True).astype(bool).to_numpy()
        r0=d.is_r0.to_numpy(); ns=(d.sib==0).to_numpy(); rs=d.is_resid.to_numpy()
        for tag,m in [("R0 all",r0),("R0 hn nonempty",r0&~hne),("R0 hn empty",r0&hne),("R0 nonempty nosib",r0&~hne&ns),
                      ("R0 nonempty nosib resid",r0&~hne&ns&rs),("R0 nonempty nosib p>=.1 resid",r0&~hne&ns&rs&(p>=.1)),
                      ("R0 nonempty nosib p>=.3 resid",r0&~hne&ns&rs&(p>=.3)),("R0 nonempty nosib p>=.5 resid",r0&~hne&ns&rs&(p>=.5)),
                      ("R0 p>=.5 resid",r0&rs&(p>=.5)),("R0 nonempty hnEQ",r0&~hne&d.hn_eq.to_numpy()),("R0 nonempty hnEQ resid",r0&~hne&d.hn_eq.to_numpy()&rs),("R0 nonempty hnEQ p<.1",r0&~hne&d.hn_eq.to_numpy()&(p<.1)),("R0 nonempty km>=2 nosib",r0&~hne&ns&(d.km.to_numpy()>=2))]:
            eval_tier(tag, m)
        res[t_] = {
            'r0': d.is_r0.sum(),
            'r0_resid': (d.is_r0 & d.is_resid).sum(),
            'S1_count': len(chain_s1_set)
        }
    return res

if __name__ == '__main__':
    res = evaluate()
    print(res)
