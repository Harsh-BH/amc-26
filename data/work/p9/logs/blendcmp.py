import sys
sys.path[:0] = ["code/business_entity_resolution/src"]
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
import run, model as M, rerank as RR, features as F
run.use("sub8c"); RR.CFG["w"] = [0.5901490903244407, 0.6319357246516834, 0.29732167912424273]
for c in ("India", "US"):
    d = pd.read_parquet(M.probs_path("test", c))
    q1, qx, pm = d.q1.to_numpy(), d.qx.to_numpy(), d.p.to_numpy(np.float32)
    id1, idx = F.ids("test", c)
    cache = run.r_cache("test", c, id1, idx)
    p, r = run.rerank(c, q1, qx, pm, None, cache)
    ref = pq.read_table(f"data/work/p13/prob_test_{c}_blend.parquet")
    s1 = id1.take(pa.array(q1)).to_numpy(zero_copy_only=False); sx = idx.take(pa.array(qx)).to_numpy(zero_copy_only=False)
    same_order = (s1 == ref.column("s1").to_numpy()).all() and (sx == ref.column("sx").to_numpy()).all()
    rp, rg = ref.column("p").to_numpy(), ref.column("p_gbdt").to_numpy()
    print(c, "rows", len(p), ref.num_rows, "order", same_order, "p_gbdt==pm", bool((rg == pm).all()),
          "p diff rows", int((rp != p).sum()), "max |dp|", float(np.abs(rp - p).max()))
    k = np.flatnonzero(rp != p)[:5]
    print("  examples", [(float(pm[i]), float(r[i]), float(p[i]), float(rp[i])) for i in k])
