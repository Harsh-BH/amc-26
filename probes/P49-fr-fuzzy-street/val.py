import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow as pa
from p49 import rule_p49, W

def val_country(c):
    print(f"\n--- Validation on {c} holdout ---")
    t = pq.read_table(W / f"p23/pairs_{c}.parquet")
    t = t.filter(pc.equal(t.column("holdout"), pa.scalar(True)))
    d = t.to_pandas()
    
    t_val = d[d.pred].p.min() if d.pred.any() else 0.71
    print(f"Inferred threshold: {t_val:.4f}")
    
    masks = rule_p49(d, "holdout", c, t_val)
    
    for arm in ["arm_eq", "arm_miss"]:
        m = masks[arm]
        print(f"\nArm: {arm}")
        arm_df = d[m]
        if len(arm_df) == 0:
            print("0 pairs")
            continue
        
        prec = arm_df.y.mean()
        print(f"Overall precision: {prec:.4f} ({arm_df.y.sum()}/{len(arm_df)})")
        
        arm_pred_false = arm_df[~arm_df.pred]
        if len(arm_pred_false) > 0:
            prec_f = arm_pred_false.y.mean()
            print(f"Among pred==False: {prec_f:.4f} ({arm_pred_false.y.sum()}/{len(arm_pred_false)})")
        else:
            print("Among pred==False: 0 pairs")

for c in ["US", "India"]:
    val_country(c)
