"""Target noise rates without target labels: run the SOURCE model on the target's folds >= 2, keep its confident
argmax pairs (p >= 0.98, P6's self-training rule) as pseudo-pairs, and measure the op rates on them with the same
detectors as synth.estimate. Pseudo-labels only feed rate estimates -- never a training label (P6: self-training on
them lost 3.5 pt). Structural constants (cluster sizes, orphan share, sibling shape) stay the source's: they agree
across US / India (orphan share 26.6 / 25.4% both, 4.67 / 4.68 SX per S1, identical sibling house-number offsets).

    .venv/bin/python probes/P12-synth/pseudo.py India US     # US model on India -> rates_psIndia.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "probes/P6-france")]
import loco_dense as LD
import synth as S

GEN = ["cluster_sizes", "orphan_share", "sx_per_s1", "orph_family", "orph_lookalike", "orph_sib_added_word",
       "orph_sib_added", "orph_hn_delta", "orph_hn_far"]


def estimate_pseudo(tgt: str, src: str, thr: float = 0.98, n: int = 15000, seed: int = 0) -> dict:
    import lightgbm as lgb
    import pyarrow as pa
    import pyarrow.parquet as pq
    T4, M = LD.T4, LD.M
    m = lgb.Booster(model_file=str(LD.MODELS / f"{src}.txt"))
    d = T4.load((2,), countries=[tgt], frac=LD.FRAC)
    p = M.predict(m, d)
    am = M.argmax_sx(d.q1.to_numpy(), d.qx.to_numpy(), p)
    sel = d.loc[am & (p >= thr), ["q1", "qx", "is_true"]]
    prec = float(sel.is_true.mean())                                # reported only (LOCO check), never used
    del d
    ids1 = pd.read_parquet(T4.TXT["train"] / f"{tgt}_s1.parquet", columns=["entity_id"]).entity_id
    files = sorted(T4.TXT["train"].glob(f"{tgt}_sx_*.parquet"))
    idx = pa.concat_arrays([pq.read_table(f, columns=["entity_id"]).column(0).combine_chunks() for f in files])
    g = pd.DataFrame({"s1": ids1.iloc[sel.q1.to_numpy()].to_numpy(),
                      "sx": idx.take(pa.array(sel.qx.to_numpy())).to_pylist()})
    del idx
    rng = np.random.default_rng(seed)
    keep = rng.choice(g.s1.unique(), min(n, g.s1.nunique()), replace=False)
    g = g[g.s1.isin(keep)]
    s1 = pd.read_parquet(S.WORK / "train_source1.parquet", filters=[("entity_id", "in", list(keep))]).set_index("entity_id")
    out = {k: v for k, v in json.loads((HERE / f"rates_{src}.json").read_text()).items() if k in GEN}
    out |= {"country": tgt, "pseudo_from": f"{src} model, argmax & p >= {thr}", "pseudo_pairs": len(sel),
            "pseudo_pair_precision_LOCO_only": round(prec, 4)}
    out |= S._summarise(S.pair_rows(tgt, s1, g))
    return out


if __name__ == "__main__":
    tgt, src = sys.argv[1:3]
    r = estimate_pseudo(tgt, src)
    (HERE / f"rates_ps{tgt}.json").write_text(json.dumps(r, indent=1))
    print({k: r[k] for k in ("pseudo_pairs", "pseudo_pair_precision_LOCO_only")})
    for s in ("S2", "S3"):
        print(s, {k: v for k, v in r[s]["addr"].items() if k not in ("hn_delta", "base_share")})
        print(s, {k: v for k, v in r[s]["name"].items() if k not in ("added", "prefix_tok")})
