"""src/block.py == probes/P3-block/block.py on the dev world (same CFG, same train vectorizers)."""
import importlib.util, sys, time
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT)]
import block as S
from worlds import load_world
spec = importlib.util.spec_from_file_location("p3block", ROOT / "probes/P3-block/block.py")
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
s1, sx, truth = load_world("dev")
v = S.vecs()
t0 = time.time(); a = S.candidates(s1, sx, S.CFG, v); ta = time.time() - t0
b = P.candidates(s1, sx, P.CFG, v)
ka, kb = set(zip(a.s1, a.sx)), set(zip(b.s1, b.sx))
n_true = sum(map(len, truth.values()))
rec = sum(1 for s, x in ka if x in truth.get(s, ()))/n_true
print(f"src {len(ka):,} pairs, p3 {len(kb):,}, identical={ka == kb}, dev pair recall {rec:.4f}, {ta:.0f}s, peak {S.peak_gb():.2f} GB")
a.to_parquet(ROOT / "data/work/p8/cands_dev.parquet", index=False)
