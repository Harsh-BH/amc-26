"""N69 frame: France argmax pairs whose SX has an empty/NULL address, with base-file status, S1 core multiplicity, P64 lf class.
    .venv/bin/python probes/N69-noaddr-owner/fr2.py <base matching_results.tsv>   -> data/work/n69/fr_noaddr.parquet"""
import duckdb, numpy as np, pandas as pd, sys
sys.path[:0]=["probes/P64-legal-form-feat","code/business_entity_resolution/src"]
import lf
BASE = sys.argv[1] if len(sys.argv) > 1 else "data/work/n63/sub76_matching.tsv"
con = duckdb.connect(); con.execute("SET memory_limit='3500MB'")
con.execute(f"""create temp table m as select source1_entity_id s1, unnest(string_split(matched_entity_ids,',')) sx from read_csv('{BASE}',delim='\t',header=true,quote='',all_varchar=true) where coalesce(matched_entity_ids,'')<>''""")
con.execute("create temp table k as select s1, count(*) k, sum((sx like 'S2%')::int) k2, sum((sx like 'S3%')::int) k3 from m group by 1")
con.execute("create temp table nc as select core, count(*) ncore from 'data/work/rf_test_s1_France.parquet' group by 1")
d = con.execute("""select p.s1, p.sx, p.src, p.p, p.r, p.p_best_x, p.p_2nd_x, p.n_x, p.name_rel, p.name_1, p.name_x, p.core_1, p.core_x, m.s1 own,
   coalesce(k.k,0) k1, coalesce(k.k2,0) k2, coalesce(k.k3,0) k3, nc.ncore
  from 'data/work/p31/pairs_France.parquet' p left join m on m.sx=p.sx left join k on k.s1=p.s1 left join nc on nc.core=p.core_1
  where p.argmax and trim(p.addr_x) in ('','NULL','<NULL>')""").df()
d["st"] = np.select([d.own.isna(), d.own == d.s1], ["free", "acc"], "other")
d["lfc"] = lf.pair_class(lf.sig(d.name_1), lf.sig(d.name_x))
d["p2"] = d.p_2nd_x.fillna(0)
d.to_parquet("data/work/n69/fr_noaddr.parquet", index=False)
f = d[(d.st == "free") & (d.p >= .2) & (d.p2 < .05)]
print("free p>=.2 p2<.05:", len(f))
print("ncore:", f.ncore.clip(0, 6).value_counts().sort_index().to_dict())
print("n_x (cands of SX):", f.n_x.clip(0, 6).value_counts().sort_index().to_dict())
print("lfc:", f.lfc.value_counts().to_dict(), "| name_rel", f.name_rel.value_counts().head(5).to_dict())
cap = np.where(f.src == "S2", f.k2 >= 5, f.k3 >= 6); print("caps full:", cap.sum(), "| S1 empty:", (f.k1 == 0).sum())
a = d[(d.st == "acc") & (d.p2 < .05)]; print("ACCEPTED same shape: ncore", a.ncore.clip(0, 6).value_counts().sort_index().to_dict(), "p<.9868:", (a.p < .9868).sum())
