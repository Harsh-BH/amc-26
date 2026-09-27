"""End-to-end guard on both output files, for both stages, on a slice of the dev world plus two injected labels:
"Xland" (an unseen country with S1 and S2/S3: copies of 200 US S1 and their records under fresh ids) and "Yland"
(S1 only, no S2/S3 at all). Every S1 of every label gets exactly one row, matches ⊆ candidates, lists hold only
S2-/S3- ids of the input, of the S1's own country, without duplicates, spaces or quotes. The model stage runs the
real M2 + cross-encoder (GPU, under the lock) and gives Xland P6's unseen-country threshold. ~4 min.

    python code/business_entity_resolution/src/test_pipeline.py [--recipe R] [--ce PATH --ce-base e5|bge]
                                                                [--unseen France --france sib|bge|m2|idf144_bge]
                                                                [--cap 5,6] [--drop-sibling-cands]
--ce swaps the recipe's cross-encoder (e.g. #8c's bge CE needs its HF base weights and a GPU; the e5 CE runs anywhere),
so every other stage of the recipe (pruner, blend, P17's stacker, France's scorer) is still exercised. --unseen names
the injected unseen label (France: its fixed t and --france scorer apply to it).
"""
import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run
from worlds import load_world


def world(n: int = 1500, seed: int = 0, unseen: str = "Xland"):
    s1, sx, truth = load_world("dev")
    s1 = s1.groupby("country").sample(n=n, random_state=seed)
    kids = set().union(*(truth[s] for s in s1.entity_id))
    sx = pd.concat([sx[sx.entity_id.isin(kids)], sx[~sx.entity_id.isin(kids)].sample(n=2 * n, random_state=seed)])
    xs1 = s1[s1.country == "US"].head(200)
    xkids = set().union(*(truth[s] for s in xs1.entity_id))
    relabel = lambda d, tag, c: d.assign(entity_id=d.entity_id + tag, country=c)
    ys1 = relabel(s1.head(3), "y", "Yland")
    S1 = pd.concat([s1, relabel(xs1, "x", unseen), ys1], ignore_index=True)
    SX = pd.concat([sx, relabel(sx[sx.entity_id.isin(xkids)], "x", unseen)], ignore_index=True)
    T = ({s: truth[s] for s in s1.entity_id} | {s + "x": {k + "x" for k in truth[s]} for s in xs1.entity_id}
         | {s: set() for s in ys1.entity_id})
    return S1, SX, T


def check(out: Path, s1: pd.DataFrame, sx: pd.DataFrame):
    c1, cx = dict(zip(s1.entity_id, s1.country)), dict(zip(sx.entity_id, sx.country))
    got = []
    for name, col in (("matching_results.tsv", "matched_entity_ids"), ("candidate_pairs.tsv", "candidate_entity_ids")):
        text = (out / name).read_text(encoding="utf-8")
        assert text.endswith("\n") and not text.startswith("﻿"), name
        lines = text[:-1].split("\n")
        assert lines[0] == f"source1_entity_id\t{col}", lines[0]
        rows = [l.split("\t") for l in lines[1:]]
        assert all(len(r) == 2 for r in rows), "two tab-separated columns"
        ids = [r[0] for r in rows]
        assert len(ids) == len(set(ids)) == len(c1) and set(ids) == set(c1), "exactly one row per S1, every label"
        lists = {r[0]: r[1].split(",") if r[1] else [] for r in rows}
        for s, v in lists.items():
            assert len(v) == len(set(v)), f"duplicate id in {s}"
            assert all(x in cx and x[:3] in ("S2-", "S3-") and cx[x] == c1[s] for x in v), f"bad id for {s}: {v}"
            assert not any(ch in r for r in v for ch in ' "\''), "no spaces or quotes"
        got.append({s: set(v) for s, v in lists.items()})
    m, c = got
    assert all(m[s] <= c[s] for s in m), "matches ⊆ candidates"
    assert any(m[s] for s in m if s.endswith("x")), "the unseen label is matched, not only listed"
    assert not any(c[s] for s in c if c1[s] == "Yland"), "a label with no S2/S3 gets empty rows"
    return m


if __name__ == "__main__":          # forkserver workers (rules.feats) re-import this file
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default=run.RECIPE, choices=sorted(run.RECIPES))
    ap.add_argument("--ce", type=Path)
    ap.add_argument("--ce-base", choices=["e5", "bge"])
    ap.add_argument("--unseen", default="Xland")
    ap.add_argument("--france", choices=["m2", "bge", "sib", "idf144_bge"])
    ap.add_argument("--cap", help="S2,S3 matches per S1 (run.py --cap)")
    ap.add_argument("--drop-sibling-cands", action="store_true")
    a = ap.parse_args()
    run.use(a.recipe)
    if a.ce:
        run.RR.CE, run.RR.FAMILY = a.ce.resolve(), a.ce_base or run.RR.FAMILY
    run.FRANCE = a.france or run.FRANCE
    run.CAP = tuple(int(x) for x in a.cap.split(",")) if a.cap else None
    run.DROP_SIB_CANDS |= a.drop_sibling_cands
    print(f"recipe {a.recipe}: CE {run.RR.CE.name} ({run.RR.FAMILY}), stack {list(run.STACK)}, France {run.FRANCE}, "
          f"unseen label {a.unseen}", flush=True)
    s1, sx, truth = world(unseen=a.unseen)
    with tempfile.TemporaryDirectory() as d:
        for stage in ("rules", "model"):
            met = run.run("pipeline-test", stage, Path(d), world=(s1, sx, truth), log=False)
            m = check(Path(d) / "pipeline-test", s1, sx)
            x = [s for s in truth if s.endswith("x")]
            fx = sum(bool(m[s] & truth[s]) for s in x) / len(x)
            assert met["f05"] > 0.7 and fx > 0.5, (met, fx)
            print(f"pipeline test ok  stage={stage}  S1 rows={len(m):,}  f05={met['f05']:.4f}  Xland S1 with a hit={fx:.2f}")
