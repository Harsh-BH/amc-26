import sys; sys.path.insert(0, "code/business_entity_resolution/src")
import numpy as np, pandas as pd
import france as FR
FR.C = "France"
fr = pd.read_parquet("data/work/p31/pairs_France.parquet", columns=["s1", "sx", "acc", "argmax"])
s1, sx = fr.s1.to_numpy(object), fr.sx.to_numpy(object)
m = FR.decoy_mask(s1, sx, fr.argmax.to_numpy(bool), fr.acc.to_numpy(bool), 0.25)
c = pd.read_parquet("data/work/p54/cells.parquet")
ref = fr.reset_index().merge(c[["s1", "sx"]], on=["s1", "sx"]); ref = ref[ref.acc]
got = set(zip(s1[m], sx[m])); want = set(zip(ref.s1, ref.sx))
print("src decoy_mask removes", m.sum(), "| p54 cells ∩ acc", len(want), "| both", len(got & want), "| only src", len(got - want), "| only p54", len(want - got))
