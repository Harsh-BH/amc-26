"""Submission #1 (sanity only): exact normalized name + first address segment, within country.

    python src/s0_exact.py holdout     # score on the holdout world, log a pipe row
    python src/s0_exact.py test        # write output/matching_results.tsv + output/candidate_pairs.tsv
"""
import sys
from pathlib import Path
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path[:0] = [str(SRC), str(SRC.parents[2])]          # src/ and the repo root (ledger.py, the submission log)
from worlds import WORK, load_world
from metric import score
from ledger import log

PUNCT = r"[\s,.;:'\"()\[\]{}\-&/|#!?*+_@%$~`^<>=]+"    # explicit class: RE2's \w would eat Indic/accents


def norm(s: pd.Series) -> pd.Series:
    """Lower case, punctuation runs -> one space, stripped."""
    return s.str.lower().str.replace(PUNCT, " ", regex=True).str.strip()


def key(df: pd.DataFrame) -> pd.Series:
    """Exact-match key: country | normalised name | normalised first address segment."""
    return df.country + "|" + norm(df.business_name) + "|" + norm(df.business_address.str.split(",").str[0])


def match(s1: pd.DataFrame, sx: pd.DataFrame) -> dict:
    """{s1 id: set(sx ids)} for every S1: each S2/S3 joins an S1 with the same key (the first one when
    several share it); an S1 with an empty normalised name never matches."""
    k1 = pd.DataFrame({"s1": s1.entity_id, "k": key(s1)})
    k1 = k1[norm(s1.business_name) != ""]                 # never match on an empty name
    kx = pd.DataFrame({"sx": sx.entity_id, "k": key(sx)})
    m = kx.merge(k1, on="k").drop_duplicates("sx")       # each S2/S3 goes to at most one S1
    out = {s: set() for s in s1.entity_id}
    for a, b in zip(m.s1, m.sx):
        out[a].add(b)
    return out


def write(sets: dict, path: Path, col: str):
    """{s1: ids} -> a submission TSV at path with second column `col` (ids sorted, comma-joined)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for s, ids in sets.items():
            f.write(f"{s}\t{','.join(sorted(ids))}\n")


if __name__ == "__main__":
    if sys.argv[1] == "holdout":
        s1, sx, truth = load_world("holdout")
        m = score(match(s1, sx), truth, dict(zip(s1.entity_id, s1.country)))
        print(m)
        log("pipe", version="s0-exact", world="holdout", metrics=m, config={"key": "country|name|addr_seg0"})
    else:
        s1 = pd.read_parquet(WORK / "test_source1.parquet")
        sx = pd.concat([pd.read_parquet(WORK / f"test_source{i}.parquet") for i in (2, 3)], ignore_index=True)
        sets = match(s1, sx)
        assert len(sets) == len(s1) == 1_732_544
        write(sets, Path("output/matching_results.tsv"), "matched_entity_ids")
        write(sets, Path("output/candidate_pairs.tsv"), "candidate_entity_ids")
