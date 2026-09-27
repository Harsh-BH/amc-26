import pandas as pd
for c in ("India","US"):
    ex = pd.read_parquet(f"data/work/n69/ex_train_{c}.parquet")
    u = ex[ex.n_core == 1]
    print(f"{c}: addr-less SX {len(ex):,} | unique exact core {len(u):,} ({len(u)/len(ex):.3f}) key==truth {(u.key_s1==u.true_s1).mean():.3f} "
          f"| SX is a distractor {(u.true_s1=='').mean():.3f} | true but other S1 {((u.true_s1!='')&(u.key_s1!=u.true_s1)).mean():.3f}")
    for n in (2,3):
        v = ex[ex.n_core == n]; print(f"   n_core {n}: {len(v):,}")
