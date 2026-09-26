"""Swap one country's rows of a staged submission for a probability file's decision: each S2/S3 at its argmax S1,
kept if p >= t; candidates = every pair of the file (what M2 scored). Other rows are copied byte for byte.

    .venv/bin/python probes/P14-france-shift/combine.py SRC_DIR PROB.parquet T OUT_DIR [--country France]
    e.g. SRC_DIR = /home/harsh1/github-repos/amc-26/output/sub6c, PROB = data/work/p14/prob_test_France_idf144.parquet
Then run the official validator on OUT_DIR (the script prints the command).
"""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src")]
import model as M


def lists(d: pd.DataFrame, t: float | None) -> dict:
    if t is not None:
        am = M.argmax_sx(pd.factorize(d.sx)[0], d.p.to_numpy())
        d = d[am & (d.p.to_numpy() >= t)]
    return d.groupby("s1").sx.agg(lambda s: ",".join(sorted(set(s)))).to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"), ap.add_argument("prob"), ap.add_argument("t", type=float), ap.add_argument("out")
    ap.add_argument("--country", default="France")
    a = ap.parse_args()
    s1 = pd.read_parquet(ROOT / "data/work/test_source1.parquet", columns=["entity_id", "country"])
    ids = set(s1.entity_id[s1.country == a.country])
    d = pd.read_parquet(a.prob)
    assert set(d.s1) <= ids, "the file holds only S1 of the country"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, t in (("matching_results.tsv", a.t), ("candidate_pairs.tsv", None)):
        new, seen = lists(d, t), 0
        with open(Path(a.src) / name, encoding="utf-8") as f, open(out / name, "w", encoding="utf-8") as g:
            g.write(next(f))
            for line in f:
                s = line.split("\t", 1)[0]
                if s in ids:
                    g.write(f"{s}\t{new.get(s, '')}\n")
                    seen += 1
                else:
                    g.write(line)
        assert seen == len(ids), (name, seen, len(ids))
        print(f"{name}: {seen:,} {a.country} rows replaced, {sum(1 for s in ids if s in new):,} non-empty", flush=True)
    print(f"validate: cd data/student_resource && python3 utils/validate_submission.py --matching {out.resolve()}/"
          f"matching_results.tsv --candidate {out.resolve()}/candidate_pairs.tsv --test-dir dataset/test")


if __name__ == "__main__":
    main()
