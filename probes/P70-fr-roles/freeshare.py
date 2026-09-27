"""Free-share diagnostic for the noise-word swap classes (lead, 19:05). LOCO (P55 frames, the target's own lens w_s / w_e):
identical-hn one-word swaps (kind sub1) into a word with lens s in (.05, .10] (groupe-like) or <= .05 (dev-like), e >= .8.
Accepted share, free share, unaccepted-pool precision. France: identical-hn sub1 swaps into 'groupe' (vs sub77) and into
'developpement' (vs sub66_final, before P71a), accepted vs free."""
import csv, gc, json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P70-fr-roles")]
import p70
from patterns import name_rel
p55, L = p70.p55, p70.L
W = ROOT / "data/work"
def wil(k, n):
    if n == 0: return None
    p, d = k / n, 1 + 3.8416 / n
    c, h = (p + 1.92 / n) / d, 1.96 * math.sqrt(p * (1 - p) / n + 0.9604 / n / n) / d
    return [round(c - h, 3), round(c + h, 3)]
out = {}
for dr in L.DIRS:
    tgt = L.DIRS[dr][1]
    fr = p55.loco_frame(dr)
    ids = pd.read_parquet(W / f"p48/loco_{dr}_ids.parquet", columns=["s1", "sx"])
    R = pd.read_parquet(W / f"p70/roles_{dr}.parquet", columns=["kind", "w_s", "w_e"])
    h1 = pd.read_parquet(W / f"rf_train_s1_{tgt}.parquet", columns=["id", "hn"]).set_index("id").hn.reindex(ids.s1).fillna("").to_numpy()
    hx = pd.read_parquet(W / f"rf_train_sx_{tgt}.parquet", columns=["id", "hn"]).set_index("id").hn.reindex(ids.sx).fillna("").to_numpy()
    eq = (h1 == hx) & (h1 != "")
    sub = R.kind.to_numpy() == 4
    s, e = R.w_s.to_numpy(), R.w_e.to_numpy()
    ref, free, y = fr["ref"], fr["free"], fr["z"]["y"].astype(bool)
    res = {}
    for name, m in {"groupe-like (s .05-.10, e>=.8)": eq & sub & (s > .05) & (s <= .10) & (e >= .8),
                    "dev-like (s<=.05, e>=.8)": eq & sub & (s <= .05) & (e >= .8)}.items():
        pool = m & free & ~ref
        k = int(y[pool].sum())
        res[name] = {"n": int(m.sum()), "prec_all": round(float(y[m].mean()), 3) if m.any() else None,
                     "accepted_share": round(float(ref[m].mean()), 3) if m.any() else None,
                     "free_share": round(float(pool.sum() / max(1, m.sum())), 3), "pool_n": int(pool.sum()),
                     "pool_prec": round(k / pool.sum(), 3) if pool.any() else None, "pool_wilson": wil(k, int(pool.sum()))}
        print(f"{dr} (target {tgt}) {name:32s} {json.dumps(res[name])}", flush=True)
    out[dr] = res
    del fr, ids, R; gc.collect()
# France
d = pd.read_parquet(W / "p31/fr31v0.1_t9868.parquet", columns=["s1", "sx", "p"]).sort_values("p", ascending=False).drop_duplicates("sx")
r1 = pd.read_parquet(W / "rf_test_s1_France.parquet", columns=["id", "core", "hn"]).set_index("id")
rx = pd.read_parquet(W / "p29/w_fix/rf_test_sx_France.parquet", columns=["id", "core", "hn", "is_web"]).set_index("id")
d["h1"], d["hx"] = r1.hn.reindex(d.s1).fillna("").to_numpy(), rx.hn.reindex(d.sx).fillna("").to_numpy()
d = d[(d.h1 == d.hx) & (d.h1 != "")].reset_index(drop=True)
c1, cx, wb = r1.core.reindex(d.s1).fillna("").to_numpy(), rx.core.reindex(d.sx).fillna("").to_numpy(), rx.is_web.reindex(d.sx).fillna(False).to_numpy()
rel = [name_rel(a, b, w) if a and b else ("", "") for a, b, w in zip(c1, cx, wb)]
d["kind"], d["word"] = [k for k, _ in rel], [w for _, w in rel]
for word, base in (("groupe", "sub77"), ("developpement", "sub66_final")):
    m = pd.read_csv(ROOT / "output" / base / "matching_results.tsv", sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
    m = m[m.source1_entity_id.isin(r1.index)]
    acc = {(s, x) for s, v in zip(m.source1_entity_id, m.matched_entity_ids) if v for x in v.split(",")}
    used = {x for _, x in acc}
    g = d[(d.kind == "sub1") & (d.word == word)]
    a_ = sum((s, x) in acc for s, x in zip(g.s1, g.sx)); f_ = int((~g.sx.isin(used)).sum())
    out[f"France {word} vs {base}"] = {"n": len(g), "accepted": int(a_), "free": f_, "accepted_share": round(a_ / max(1, len(g)), 3), "free_share": round(f_ / max(1, len(g)), 3)}
    print(f"France {word} swaps (identical hn) vs {base}: {json.dumps(out[f'France {word} vs {base}'])}", flush=True)
(Path(__file__).parent / "freeshare.json").write_text(json.dumps(out, indent=1))
