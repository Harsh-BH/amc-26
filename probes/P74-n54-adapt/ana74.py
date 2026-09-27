"""P74: what sub66_final_n54's decoy step removed (p74.py's frame): pairs, S1 emptied, Qwen / bge r and p of the removed
pairs, their words, the stage each came from, dup_last after it, and the port check against p54.py's cell rule (P46's
lens_table / swap_words, as test_port.py compares).  -> ana74.json
    .venv/bin/python probes/P74-n54-adapt/ana74.py probes/P74-n54-adapt/work/n54"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

sys.path[:0] = ["code/business_entity_resolution/src", "probes/P46-noise-word-tier"]
import france as FR  # noqa: E402
import features as F  # noqa: E402

out = Path(sys.argv[1])
f = np.load(out / "p74_frame.npz")
K, din, dout = f["K"], f["decoy_in"], f["decoy_out"]
rm = np.setdiff1d(din, dout)
assert np.isin(dout, din).all() and len(rm) == len(din) - len(dout), "decoy only removes"
id1, idx = F.ids("test", "France")
ids = lambda a, q: a.take(pa.array(q)).to_numpy(zero_copy_only=False)
s1, sx = ids(id1, rm >> 32), ids(idx, rm & 0xFFFFFFFF)
FR.C = "France"
r1 = FR._rf(FR.WORK / "rf_test_s1_France.parquet", pd.unique(s1), ["core", "hn"])
rx = FR._rf(FR.WORK / "rf_test_sx_France.parquet", pd.unique(sx), ["core", "hn", "is_web"])
rel = [FR.name_rel(a, b, w) for a, b, w in zip(r1.core.reindex(s1).fillna(""), rx.core.reindex(sx).fillna(""),
                                                rx.is_web.reindex(sx).fillna(False))]
h1, hx = r1.hn.reindex(s1).fillna("").to_numpy(), rx.hn.reindex(sx).fillna("").to_numpy()
pos = pd.Series(np.arange(len(K)), index=K).reindex(rm).to_numpy()
inK = ~np.isnan(pos)
pi = pos[inK].astype(np.int64)


def dist(v):
    v = v[~np.isnan(v)]
    return {"n": int(len(v)), "q10_50_90": [round(float(x), 3) for x in np.quantile(v, [.1, .5, .9])] if len(v) else None,
            "ge.9": round(float((v >= .9).mean()), 3) if len(v) else None, "lt.5": round(float((v < .5).mean()), 3) if len(v) else None}


rq = np.full(len(rm), np.nan); rq[inK] = f["rq"][pi]
rb = np.full(len(rm), np.nan); rb[inK] = f["rb"][pi]
p = np.full(len(rm), np.nan); p[inK] = f["p"][pi]
acc0, acc1, ro, vm = f["acc0"], f["acc1"], f["rescue_only"], f["vm"]
stage_sets = {"chain (decision + post-rules)": K[acc0], "rescue90": K[ro & ~acc0], "gc70": K[acc1 & ~ro],
              "acro (P62)": np.setdiff1d(f["dup1_in"], K[acc1 & ~vm]), "dup (P54, first)": np.setdiff1d(f["dup1_out"], f["dup1_in"]),
              "xfc (P66)": np.setdiff1d(f["xfc_out"], f["xfc_in"])}
by_stage = {k: int(np.isin(rm, v).sum()) for k, v in stage_sets.items()}
assert sum(by_stage.values()) == len(rm), by_stage
final_s1 = np.unique(dout >> 32)
emptied = np.setdiff1d(np.unique(rm >> 32), np.unique(f["dup2_out"] >> 32))
kinds = Counter(k for k, _ in rel)
words = Counter(w for _, w in rel)
hnrel = Counter("both empty" if a == "" and b == "" else "S1 only" if b == "" else "SX only" if a == "" else "equal" if a == b else "differ"
                for a, b in zip(h1, hx))
# port check: p54.py's cell rule (P46 lens_table + swap_words on the argmax pairs / the lines) vs france.decoy_mask
import p46  # noqa: E402
ka = f["ka"]
lt = p46.lens_table(pd.DataFrame({"s1": ids(id1, ka >> 32), "sx": ids(idx, ka & 0xFFFFFFFF)}), "test", "France")
dec = set(lt.index[(lt.n >= 20) & (lt.s > 0.25)])
x = pd.DataFrame({"s1": ids(id1, din >> 32), "sx": ids(idx, din & 0xFFFFFFFF)})
c1 = FR._rf(FR.WORK / "rf_test_s1_France.parquet", pd.unique(x.s1), ["core"]).core.reindex(x.s1).fillna("").to_numpy()
rxx = FR._rf(FR.WORK / "rf_test_sx_France.parquet", pd.unique(x.sx), ["core", "is_web"])
cx, wx = rxx.core.reindex(x.sx).fillna("").to_numpy(), rxx.is_web.reindex(x.sx).fillna(False).to_numpy()
kk = np.array([FR.name_rel(a, b, w)[0] for a, b, w in zip(c1, cx, wx)], object)
sub, add = kk == "sub1", kk == "add1"
w = np.full(len(x), "", object)
w[sub] = p46.swap_words(x[sub].reset_index(drop=True), "test", "France")
w[add] = [(g[0] if len(g) == 1 else "") for g in ([t for t in b.split() if t not in set(a.split())] for a, b in zip(c1[add], cx[add]))]
ref = din[(sub | add) & np.isin(w, list(dec))]
port = {"src_removes": int(len(rm)), "p54_rule": int(len(ref)), "both": int(np.isin(rm, ref).sum()),
        "only_src": int((~np.isin(rm, ref)).sum()), "only_p54": int((~np.isin(ref, rm)).sum()), "decoy_words_p46": len(dec)}
res = {"lines_in": int(len(din)), "removed": int(len(rm)), "s1_touched": int(len(np.unique(rm >> 32))), "s1_emptied": int(len(emptied)),
       "final_pairs": int(len(f["dup2_out"])), "dup_last_adds": int(len(f["dup2_out"]) - len(dout)),
       "dup_last_readds_removed": int(np.isin(f["dup2_out"], rm).sum()),
       "is2_share": round(float(np.char.startswith(sx.astype(str), "S2-").mean()), 3), "kind": dict(kinds), "hn": dict(hnrel),
       "by_stage": by_stage, "qwen_r": dist(rq), "bge_r": dist(rb), "p": dist(p), "not_in_frame": int((~inK).sum()),
       "top_words": dict(words.most_common(25)), "n_words": len(words), "port_check": port}
print(json.dumps(res, indent=1))
(Path("probes/P74-n54-adapt") / "ana74.json").write_text(json.dumps(res, indent=1))
pd.DataFrame({"s1": s1, "sx": sx, "kind": [k for k, _ in rel], "w": [w_ for _, w_ in rel], "rq": rq, "rb": rb, "p": p}).to_parquet(
    out / "removed.parquet", index=False)
