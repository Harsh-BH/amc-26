import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P20-france-patterns")]
from stack import hn_cls
from patterns import name_rel

W = ROOT / "data/work"

def get_feats(c, s1_arr, sx_arr):
    import pyarrow.parquet as pq
    r1 = pq.read_table(W / f"rf_test_s1_{c}.parquet", columns=["id", "core", "hn"]).to_pandas().set_index("id")
    rx = pq.read_table(W / f"rf_test_sx_{c}.parquet", columns=["id", "core", "hn", "is_web"]).to_pandas().set_index("id")
    c1 = r1.core.reindex(s1_arr).fillna("").to_numpy()
    h1 = r1.hn.reindex(s1_arr).fillna("").to_numpy()
    cx = rx.core.reindex(sx_arr).fillna("").to_numpy()
    hx = rx.hn.reindex(sx_arr).fillna("").to_numpy()
    wx = rx.is_web.reindex(sx_arr).fillna(False).to_numpy()
    return c1, h1, cx, hx, wx

d = pd.read_parquet(W / "p31/pairs_France.parquet", columns=["s1", "sx", "argmax"])
am_mask = d.argmax == True
d_am = d[am_mask].reset_index(drop=True)

c1, h1, cx, hx, wx = get_feats("France", d_am.s1, d_am.sx)

kw = [name_rel(a, b, x) for a, b, x in zip(c1, cx, wx)]
kinds = np.array([k for k, w in kw])
words = np.array([w for k, w in kw])
cls = hn_cls(np.array(h1, object), np.array(hx, object))
both_hn = (h1 != "") & (hx != "")

df = pd.DataFrame({'kind': kinds, 'w': words, 'sib': (cls == 1) & both_hn, 'eq': (cls == 0) & both_hn, 'both': both_hn})
df = df[df.w == 'developpement']
print(df.groupby('kind').sum())
