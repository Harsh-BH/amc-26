"""P57: competitor density / mass on test argmax pairs (India/US), the same stats p57.build prints for each world."""
import numpy as np, pandas as pd, pyarrow.parquet as pq
W = "data/work"
for c in ("India", "US"):
    t = pq.read_table(f"{W}/p34/te_pairs_{c}.parquet", columns=["sx", "p"])
    kx = pd.factorize(t.column("sx").to_pandas())[0]
    p = t.column("p").to_numpy(); del t
    am = pd.read_parquet(f"{W}/p34/prob_test_{c}_a10.parquet", columns=["am"]).am.to_numpy()
    n = np.bincount(kx)[kx] - 1
    mass = np.bincount(kx, p)[kx] - p
    a = am
    print(c, f"test: {a.sum():,} argmax pairs; mean n_comp_rows {n[a].mean():.3f}, sx_psum-p>=.02 {(mass[a] >= .02).mean():.4f}, "
             f">=.5 {(mass[a] >= .5).mean():.4f}; accepted(p>=.71) share with mass>=.5 {(mass[a & (p >= .71)] >= .5).mean():.4f}", flush=True)
