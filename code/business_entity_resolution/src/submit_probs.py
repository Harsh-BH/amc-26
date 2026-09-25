"""Scored candidate pairs -> the two submission TSVs. Decision: each S2/S3 goes to its argmax S1 (column `am`),
kept when p >= the country's threshold. Candidates = every scored pair (the exact set the model ran on).

    python src/submit_probs.py --probs data/work/p4/prob_test_{country}.parquet --out output/sub3 \
        --t 0.71 --t-country France=0.83
"""
import argparse, sys
from pathlib import Path
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path[:0] = [str(SRC)]
from worlds import WORK
from rules import _lines


def decide(p: pd.DataFrame, t: float) -> pd.DataFrame:
    """Argmax per S2/S3 (precomputed `am`), then p >= t. Many-to-one: an S1 keeps every record that chose it."""
    return p[p.am & (p.p >= t)][["s1", "sx"]]


def write(probs_tmpl: str, out: Path, t: float, t_country: dict) -> dict:
    s1_all = pd.read_parquet(WORK / "test_source1.parquet", columns=["entity_id", "country"])
    out.mkdir(parents=True, exist_ok=True)
    stats, rows = {}, 0
    with open(out / "matching_results.tsv", "w") as fm, open(out / "candidate_pairs.tsv", "w") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n"); fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for country in sorted(s1_all.country.unique()):             # open label set: every country in the file
            ids = s1_all.entity_id[s1_all.country == country].tolist()
            path = Path(probs_tmpl.format(country=country))
            p = pd.read_parquet(path, columns=["s1", "sx", "p", "am"]) if path.exists() else \
                pd.DataFrame({"s1": [], "sx": [], "p": [], "am": []}).astype({"am": bool})
            tc = t_country.get(country, t)
            acc = decide(p, tc)
            assert acc.sx.is_unique, "an S2/S3 record may go to one S1 only"
            assert set(acc.s1) <= set(ids) and set(p.s1) <= set(ids), "pairs reference S1 of another country/split"
            assert p.sx.str.startswith(("S2-", "S3-")).all()
            fm.write(_lines(acc, ids)); fc.write(_lines(p[["s1", "sx"]], ids))
            rows += len(ids)
            n = acc.groupby("s1").size().reindex(ids, fill_value=0)
            stats[country] = {"t": tc, "s1": len(ids), "pairs_scored": len(p), "matches": len(acc),
                              "matches_per_s1": round(n.mean(), 3), "s1_empty": round((n == 0).mean(), 4)}
            print(country, stats[country], flush=True)
    assert rows == len(s1_all), (rows, len(s1_all))
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True, help="path template with {country}")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--t", type=float, default=0.71)
    ap.add_argument("--t-country", nargs="*", default=[], help="Country=threshold overrides")
    a = ap.parse_args()
    write(a.probs, a.out, a.t, {k: float(v) for k, v in (x.split("=") for x in a.t_country)})
