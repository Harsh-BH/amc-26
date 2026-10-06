"""P47 shape: per-S1 group shape (S2 / S3 count shares, two-/one-sided shares, mean SX per S1) of a submission, per country.

    .venv/bin/python probes/P47-fr-shape-match/shape.py output/sub46_on43 [more dirs]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def shape(n2, n3):
    """n2 / n3: per-S1 S2 / S3 counts (int arrays, one row per S1)."""
    n2, n3 = np.asarray(n2), np.asarray(n3)
    return {"n": int(len(n2)), "mean": round(float((n2 + n3).mean()), 4),
            "s2": [round(float((np.minimum(n2, 5) == k).mean()), 4) for k in range(6)],
            "s3": [round(float((np.minimum(n3, 6) == k).mean()), 4) for k in range(7)],
            "both": round(float(((n2 > 0) & (n3 > 0)).mean()), 4), "s3only": round(float(((n2 == 0) & (n3 > 0)).mean()), 4),
            "s2only": round(float(((n2 > 0) & (n3 == 0)).mean()), 4), "empty": round(float(((n2 + n3) == 0).mean()), 4)}


def counts(sub_dir):
    """DataFrame s1, country, n2, n3 for every S1 row of sub_dir/matching_results.tsv."""
    m = pd.read_csv(Path(sub_dir) / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    ids = m.iloc[:, 1]
    n2 = ids.str.count("S2-").to_numpy()
    n3 = ids.str.count("S3-").to_numpy()
    c = pd.read_parquet(ROOT / "data/work/test_source1.parquet", columns=["entity_id", "country"]).set_index("entity_id").country
    return pd.DataFrame({"s1": m.iloc[:, 0].to_numpy(), "country": c.reindex(m.iloc[:, 0]).to_numpy(), "n2": n2, "n3": n3})


def joint(n2, n3):
    """Joint (min(n2, 5), min(n3, 6)) shares, a 6 x 7 list."""
    h = np.zeros((6, 7))
    np.add.at(h, (np.minimum(n2, 5), np.minimum(n3, 6)), 1)
    return (h / max(len(n2), 1)).round(6).tolist()


def targets(base="sub46_on43"):
    """Label-free targets -> probes/P47-fr-shape-match/targets.json: per seen country and pooled, the test-prediction shape
    of `base`, and the train-truth shape (train_ground_truth x train S1 country)."""
    out = {}
    df = counts(ROOT / "output" / base)
    seen = sorted(pd.read_parquet(ROOT / "data/work/train_source1.parquet", columns=["country"]).country.unique())
    for c in seen + ["seen"]:
        g = df[df.country.isin(seen)] if c == "seen" else df[df.country == c]
        out[f"pred_{c}"] = dict(shape(g.n2, g.n3), joint=joint(g.n2.to_numpy(), g.n3.to_numpy()))
    t = pd.read_parquet(ROOT / "data/work/train_ground_truth.parquet")
    c1 = pd.read_parquet(ROOT / "data/work/train_source1.parquet", columns=["entity_id", "country"]).set_index("entity_id").country
    t["country"] = c1.reindex(t.source1_entity_id).to_numpy()
    t2, t3 = t.matched_entity_ids.str.count("S2-").to_numpy(), t.matched_entity_ids.str.count("S3-").to_numpy()
    for c in seen + ["seen"]:
        m = np.ones(len(t), bool) if c == "seen" else (t.country == c).to_numpy()
        out[f"truth_{c}"] = dict(shape(t2[m], t3[m]), joint=joint(t2[m], t3[m]))
    (Path(__file__).parent / "targets.json").write_text(json.dumps(out))
    print(json.dumps({k: {kk: v[kk] for kk in ("mean", "both", "empty", "s2", "s3")} for k, v in out.items()}, indent=0))


if __name__ == "__main__":
    assert shape([0, 1, 2], [0, 0, 3])["both"] == round(1 / 3, 4) and shape([0, 1, 2], [0, 0, 3])["empty"] == round(1 / 3, 4)
    if sys.argv[1] == "targets":
        targets(*sys.argv[2:])
        sys.exit()
    for d in sys.argv[1:]:
        df = counts(ROOT / d)
        out = {c: shape(g.n2, g.n3) for c, g in df.groupby("country")}
        seen = set(pd.read_parquet(ROOT / "data/work/train_source1.parquet", columns=["country"]).country.unique())
        iu = df[df.country.isin(seen)]                        # the label-free target = the countries seen in training
        out["seen"] = shape(iu.n2, iu.n3)
        print(d, json.dumps(out, indent=1))
