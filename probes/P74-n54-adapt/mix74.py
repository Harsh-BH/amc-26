"""P74 risk bound: per decoy word, France's lens (s = sibling-offset share, e = same-hn share over hn-present swaps) read
as a two-population mix of pure decoys (the top-s words' s_d / e_d) and true noise (noise words' s_n / e_n), then the
decoy share among the removed pairs' own hn relation (hn-equal: f e_d / (f e_d + (1 - f) e_n)).  -> mix74.json"""
import json
import sys

import numpy as np
import pandas as pd
import pyarrow as pa

sys.path.insert(0, "code/business_entity_resolution/src")
import france as FR  # noqa: E402
import features as F  # noqa: E402

FR.C = "France"
ka = np.load("probes/P74-n54-adapt/work/n54/p74_frame.npz")["ka"]
id1, idx = F.ids("test", "France")
ids = lambda a, q: a.take(pa.array(q)).to_numpy(zero_copy_only=False)
st = FR.lens_table(ids(id1, ka >> 32), ids(idx, ka & 0xFFFFFFFF))
st = st[st.n >= 20]
noise, pure = st[st.s <= 0.05], st[st.s > 0.25].nlargest(10, "s")
s_n, e_n = float(np.average(noise.s, weights=noise.n)), float(np.average(noise.e, weights=noise.n))
s_d, e_d = float(np.average(pure.s, weights=pure.n)), float(np.average(pure.e, weights=pure.n))
r = pd.read_parquet("probes/P74-n54-adapt/work/n54/removed.parquet")
w = st.reindex(r.w)
f_s = ((w.s - s_n) / (s_d - s_n)).clip(0, 1).to_numpy()
f_e = ((e_n - w.e) / (e_n - e_d)).clip(0, 1).to_numpy()
out = {"s_n": round(s_n, 3), "e_n": round(e_n, 3), "s_d": round(s_d, 3), "e_d": round(e_d, 3)}
for tag, f in (("f_from_s", f_s), ("f_from_e", f_e)):
    dec = f * e_d / (f * e_d + (1 - f) * e_n)            # the decoy share at hn equal (1,619 of 1,666 removed pairs)
    out[tag] = {"removed_weighted_decoy_share_hn_equal": round(float(np.nanmean(dec)), 3),
                "word_f_median": round(float(np.nanmedian(f)), 3),
                "pairs_in_words_below_0.5_decoy": int(np.nansum(dec < 0.5))}
out["per_word"] = {k: [int(v.n), round(float(v.s), 3), round(float(v.e), 3)] for k, v in st.reindex(r.w.value_counts().index[:15]).iterrows()}
print(json.dumps(out, indent=1))
open("probes/P74-n54-adapt/mix74.json", "w").write(json.dumps(out, indent=1))
