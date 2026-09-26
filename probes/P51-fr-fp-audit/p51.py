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
    cx = rx.core.reorder(sx_arr).fillna("").to_numpy()
    hx = rx.hn.reorder(sx_arr).fillna("").to_numpy()
    wx = rx.is_web.reorder(sx_arr).fillna(False).to_numpy()
    return c1, h1, cx, hx, wx

def build_lens(c1, h1, cx, hx, wx):
    # Build lens statistics used for band assignment
    kw = [name_rel(a, b, x) for a, b, x in zip(c1, cx, wx)]
    kinds = np.array([k for k, w in kw])
    words = np.array([w for k, w in kw])
    cls = hn_cls(np.array(h1, object), np.array(hx, object))
    both_hn = (h1 != "") & (hx != "")
    df = pd.DataFrame({"kind": kinds, "w": words, "sib": (cls == 1) & both_hn, "eq": (cls == 0) & both_hn, "both": both_hn})
    g = df[df.w != ""].groupby(["kind", "w"]).sum()
    g["s"] = g["sib"] / g["both"].clip(lower=1)
    g = g[g["both"] >= 20]
    return g, kinds, words, cls, both_hn

def assign_band(kind, word, lens_df):
    if word == "":
        return "no_gained"
    idx = (kind, word)
    if idx not in lens_df.index:
        return "low_support"
    s = lens_df.loc[idx, "s"]
    if s <= 0.05:
        return "<=0.05"
    if s <= 0.25:
        return "0.05-0.25"
    if s <= 0.5:
        return "0.25-0.5"
    return ">0.5"

def load_cells():
    cells = pd.read_csv(ROOT / "probes/P51-fr-fp-audit/cells_fix.csv")
    suspects = cells[(cells["excess"] > 0.001) & (cells["us_ind_min_cell_prec"] < 0.7)]
    return suspects

def build_removal_mask(fr_df, suspects):
    # fr_df contains columns kind, hc, band, addr_less, and identifiers s1, sx
    mask = pd.Series(False, index=fr_df.index)
    for _, row in suspects.iterrows():
        cond = (
            (fr_df["kind"] == row["kind"]) &
            (fr_df["hc"] == row["hc"]) &
            (fr_df["band"] == row["band"]) &
            (fr_df["addr_less"] == row["addr_less"])
        )
        mask |= cond
    return mask

def selftest():
    # Tiny synthetic data to verify that removal works without errors
    base_dir = Path("/tmp/p51_selftest_base")
    out_dir = Path("/tmp/p51_selftest_out")
    base_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Create minimal matching_results.tsv with required columns
    mr = pd.DataFrame({
        "s1": ["S1A"],
        "sx": ["S2A"],
        "country": ["France"],
        "kind": ["add1"],
        "hc": ["equal"],
        "addr_less": [False],
        "band": ["0.05-0.25"],
        "acc": [True],
    })
    mr.to_csv(base_dir / "matching_results.tsv", sep="\t", index=False)
    # Candidate pairs (byte‑identical info not used in this tiny test)
    cp = pd.DataFrame({"s1": ["S1A"], "sx": ["S2A"]})
    cp.to_csv(base_dir / "candidate_pairs.tsv", sep="\t", index=False)
    # Run the removal script on synthetic data
    build(base_dir, out_dir)
    # Verify output exists
    assert (out_dir / "matching_results.tsv").exists()
    assert (out_dir / "candidate_pairs.tsv").exists()
    print("selftest passed")

def build(base_dir: Path, out_dir: Path):
    base_dir = Path(base_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load original data
    mr = pd.read_csv(base_dir / "matching_results.tsv", sep="\t")
    cp = pd.read_csv(base_dir / "candidate_pairs.tsv", sep="\t")

    # Keep only France rows for further analysis
    fr = mr[mr["country"] == "France"].copy()

    # Load suspect cells
    suspects = load_cells()

    # Build mask of France rows that belong to suspect cells
    removal_mask = build_removal_mask(fr, suspects)
    removed = fr[removal_mask]
    kept = fr[~removal_mask]

    # Write updated matching_results (keep non‑France rows unchanged, plus kept France rows)
    mr_updated = pd.concat([mr[mr["country"] != "France"], kept], ignore_index=True)
    mr_updated.to_csv(out_dir / "matching_results.tsv", sep="\t", index=False)
    # Candidate pairs remain unchanged (rule does not add/delete candidates)
    cp.to_csv(out_dir / "candidate_pairs.tsv", sep="\t", index=False)

    # Simple reporting for the RESULT.md
    print(f"Removed {len(removed)} France pairs across {len(suspects)} suspect cells")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: build <base_dir> <out_dir>")
        sys.exit(1)
    build(sys.argv[1], sys.argv[2])
