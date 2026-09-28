"""P20: label-free pattern table over every candidate pair of a country's test file. For each pair: how the SX name core
differs from the S1 core (eq / add1 w / sub1 w / drop1 / typo / disjoint / web), and the house-number relation
(eq / d in SIB3 = the generator's sibling offsets {3,4,5,7,9,11,13,21} / d in {1,2} / other / missing). The hn relation is
the label-free lens: true pairs keep the hn (US 90%), generated siblings take hn + {1..21} (P12), so the SIB3 share of a
name pattern estimates how sibling-heavy it is. Nothing is labelled; only aggregates are printed.
    .venv/bin/python probes/P20-france-patterns/patterns.py France       # -> data/work/p20/pairs_France.parquet + tables
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "data/work"
OUT = W / "p20"
SIB3 = {3, 4, 5, 7, 9, 11, 13, 21}


def hn_rel(h1: str, hx: str) -> str:
    if not h1 or not hx:
        return "miss"
    d = int(hx) - int(h1)
    return "eq" if d == 0 else "sib3" if d in SIB3 else "d12" if d in (1, 2) else "neg12" if d in (-1, -2) else "other"


def name_rel(c1: str, cx: str, webx: bool) -> tuple:
    """(kind, word): eq | add1 w | sub1 w (one S1 word replaced by w) | drop1 | typo | addN | subN | disjoint | web."""
    if webx:
        return "web", ""
    a, b = c1.split(), cx.split()
    s1, sx = set(a), set(b)
    add, drop = [t for t in b if t not in s1], [t for t in a if t not in sx]
    if not add and not drop:
        return "eq", ""
    if not s1 & sx:
        return "disjoint", ""
    if not add:
        return "drop1" if len(drop) == 1 else "dropN", ""
    if not drop:
        return ("add1", add[0]) if len(add) == 1 else ("addN", " ".join(add))
    if len(add) == 1 and len(drop) == 1:
        return ("typo", "") if ratio(add[0], drop[0]) >= 75 else ("sub1", add[0])
    typo = sum(max(ratio(x, y) for y in drop) >= 75 for x in add)
    rest = [x for x in add if max(ratio(x, y) for y in drop) < 75]
    if not rest:
        return "typo", ""
    return ("sub1" if len(rest) == 1 else "subN"), " ".join(rest)


def build(c: str, prob: str) -> pd.DataFrame:
    f = OUT / f"pairs_{c}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    d = pd.read_parquet(W / prob) if c == "France" else pd.read_parquet(W / prob, columns=["s1", "sx", "p"])
    if c != "France":
        d = d.rename(columns={"p": "p8c"})
    r1 = pd.read_parquet(W / f"rf_test_s1_{c}.parquet", columns=["id", "core", "hn"]).set_index("id")
    rx = pd.read_parquet(W / f"rf_test_sx_{c}.parquet", columns=["id", "core", "hn", "is_web"]).set_index("id")
    c1, h1 = r1.core.reindex(d.s1).fillna("").to_numpy(), r1.hn.reindex(d.s1).fillna("").to_numpy()
    cx, hx = rx.core.reindex(d.sx).fillna("").to_numpy(), rx.hn.reindex(d.sx).fillna("").to_numpy()
    wx = rx.is_web.reindex(d.sx).fillna(False).to_numpy()
    del r1, rx
    nr = [name_rel(a, b, w) for a, b, w in zip(c1, cx, wx)]
    d["kind"] = [k for k, _ in nr]
    d["word"] = [w for _, w in nr]
    d["hrel"] = [hn_rel(a, b) for a, b in zip(h1, hx)]
    d.to_parquet(f, index=False)
    return d


def table(d: pd.DataFrame, by: list, pcol: str = "p8c", acol: str | None = "a8c", top: int = 40) -> pd.DataFrame:
    g = d.assign(band=(d[pcol] >= 0.3) & (d[pcol] < 0.95), hi=d[pcol] >= 0.95)
    hr = pd.crosstab([g[b] for b in by], g.hrel, normalize="index").round(3)
    agg = {"n": (pcol, "size"), "p_mean": (pcol, "mean"), "band": ("band", "sum"), "p_ge95": ("hi", "sum")}
    if acol:
        agg["acc"] = (acol, "sum")
    t = g.groupby(by).agg(**agg).join(hr)
    return t.sort_values("n", ascending=False).head(top)


if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else "France"
    d = build(c, "p20/merged.parquet" if c == "France" else f"p10/prob_test_{c}.parquet")
    pd.set_option("display.width", 250), pd.set_option("display.max_columns", 30)
    d = d[d.p8c.notna()] if c == "France" else d
    print(table(d, ["kind"], acol="a8c" if c == "France" else None))
    print(table(d[d.kind.isin(["add1", "sub1"])], ["kind", "word"], acol="a8c" if c == "France" else None, top=60))
