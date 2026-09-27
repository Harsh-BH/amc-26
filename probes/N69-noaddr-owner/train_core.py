"""N69: address-less SX are 97.7% true in train, yet test assigns only 35-62% of them. For the ones the #8c holdout
leaves unassigned, does a country-wide exact-core lookup find the true S1? (labels only score)"""
import duckdb, numpy as np, pandas as pd
con = duckdb.connect(); con.execute("SET memory_limit='3500MB'")
for c in ("India", "US"):
    ex = con.execute(f"""select t.entity_id sx, r.core, r.state from (select entity_id, business_address from 'data/work/train_source2.parquet' where country='{c}'
        union all select entity_id, business_address from 'data/work/train_source3.parquet' where country='{c}') t
        join 'data/work/rf_train_sx_{c}.parquet' r on r.id=t.entity_id where trim(t.business_address) in ('','NULL','<NULL>')""").df()
    s1 = con.execute(f"select id s1, core, state from 'data/work/rf_train_s1_{c}.parquet'").df()
    gt = con.execute("select * from 'data/work/gt_pairs.parquet'").df().set_index("sx").s1
    ex["true_s1"] = gt.reindex(ex.sx).fillna("").to_numpy()
    d = pd.read_parquet(f"data/work/p23/pairs_{c}.parquet", columns=["s1", "sx", "holdout", "pred", "p", "am"])
    ho = set(d.s1[d.holdout]); pr = d[d.pred].drop_duplicates("sx").set_index("sx").s1
    ex["pred_s1"] = pr.reindex(ex.sx).fillna("").to_numpy(); ex["in_dump"] = ex.sx.isin(set(d.sx))
    cnt = s1.groupby("core").s1.agg(["first", "size"])
    k = cnt.reindex(ex.core)
    ex["n_core"] = k["size"].fillna(0).astype(int).to_numpy(); ex["key_s1"] = np.where(ex.n_core == 1, k["first"].fillna("").to_numpy(), "")
    # restrict evaluation to SX whose true S1 or key S1 is a holdout S1
    tr_ho = ex.true_s1.isin(ho)
    print(f"== {c}: empty-addr SX {len(ex):,}, true {(ex.true_s1!='').mean():.3f}; true S1 in holdout {tr_ho.sum():,}: "
          f"assigned correctly {(ex.pred_s1==ex.true_s1)[tr_ho].mean():.3f}, wrong S1 {((ex.pred_s1!='')&(ex.pred_s1!=ex.true_s1))[tr_ho].mean():.3f}, "
          f"free {(ex.pred_s1=='')[tr_ho].mean():.3f} (in dump {ex.in_dump[tr_ho & (ex.pred_s1=='')].mean():.3f})")
    f = ex[(ex.pred_s1 == "")]
    for tag, m in (("unique core", f.n_core == 1),):
        z = f[m & f.key_s1.isin(ho)]
        print(f"   free & {tag} (key S1 in holdout): {len(z):,}  precision {(z.key_s1==z.true_s1).mean():.3f}  same-state {(z.state.to_numpy()==s1.set_index('s1').state.reindex(z.key_s1).to_numpy()).mean():.3f}")
    ft = f[f.true_s1.isin(ho)]
    print(f"   free true (holdout): {len(ft):,}  n_core dist {ft.n_core.clip(0,5).value_counts().sort_index().to_dict()}")
    ex.to_parquet(f"data/work/n69/ex_train_{c}.parquet", index=False)
