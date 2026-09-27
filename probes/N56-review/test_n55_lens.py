import pandas as pd
import sys
from pathlib import Path
ROOT = Path("../../").resolve()
sys.path[:0] = [str(ROOT / "probes/P46-noise-word-tier"), str(ROOT / "probes/N55-fr-noiseword-fill")]
import n55
import p46

W = ROOT / "data/work"
for c in ("US", "India"):
    if not (W / f"p23/pairs_{c}.parquet").exists():
        print(f"Skipping {c}")
        continue
    d_all = pd.read_parquet(W / f"p23/pairs_{c}.parquet", columns=["s1", "sx", "holdout", "am", "y", "pred", "p"])
    a_all = d_all[d_all.am].reset_index(drop=True)
    lt_all = p46.lens_table(a_all[["s1", "sx"]], "train", c)
    noise_all = set(lt_all[(lt_all.n >= 20) & (lt_all.s <= 0.05) & (lt_all.e >= 0.8)].index)
    
    a_ho = d_all[d_all.holdout & d_all.am].reset_index(drop=True)
    lt_ho = p46.lens_table(a_ho[["s1", "sx"]], "train", c)
    noise_ho = set(lt_ho[(lt_ho.n >= 20) & (lt_ho.s <= 0.05) & (lt_ho.e >= 0.8)].index)
    
    print(f"{c}: all_argmax words={len(noise_all)}, holdout_argmax words={len(noise_ho)}")
