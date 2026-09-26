"""P54: remove France pairs whose SX swaps one S1 word for a generator DECOY word at the same house number.

France decoys copy the S1 name, swap one org word (club / comite / ecole / amicale / parents / ...), and keep the house
number about half the time (P46 lens on France argmax pairs: s 0.29-0.42, e 0.47-0.62 — the same profile as India's
decoy words). In-domain the same cell (sub1 swap, swapped-in word with n_hn >= 20 and lens s > 0.2, hn equal, argmax) is
0/65 true (US holdout) and 1/759 (India). The India/US-trained matcher cannot see the swap (French org words are common,
low IDF) and P42's hn-equal extras (p >= 0.8) add these pairs wholesale.

Variants: 'ruled' removes only pairs with p < T_BASE (added by post-threshold rules: hn-extra, gc, rescue, ...);
          'all' also removes pairs the matcher accepted on its own (p >= T_BASE).
    .venv/bin/python probes/N54-fr-decoyword-removal/p54.py cells             # -> data/work/p54/cells.parquet
    .venv/bin/python probes/N54-fr-decoyword-removal/p54.py build <base> <out> [ruled|all]
    .venv/bin/python probes/N54-fr-decoyword-removal/p54.py selftest
"""
import csv, os, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P46-noise-word-tier")]
W = ROOT / "data/work"; OUT = W / "p54"
T_BASE, S_MIN, N_MIN = 0.9868, 0.25, 20   # 0.25 not 0.2: LOCO India>US flags "partners" (US noise word, s 0.217)
RD = dict(sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)

def cells() -> pd.DataFrame:
    import p46
    fr = pd.read_parquet(W / "p31/pairs_France.parquet", columns=["s1", "sx", "p", "argmax", "hn_cls", "hn_1", "name_rel"])
    am = fr.argmax.astype(bool).to_numpy()
    lt = p46.lens_table(fr.loc[am, ["s1", "sx"]].reset_index(drop=True), "test", "France")      # label-free, France argmax
    dec = lt[(lt.n >= N_MIN) & (lt.s > S_MIN)]
    x = fr[fr.name_rel.isin(["sub1", "add1"])].reset_index(drop=True)          # any hn relation (LOCO addendum 5)
    sub = (x.name_rel == "sub1").to_numpy()
    x["w"] = ""
    x.loc[sub, "w"] = p46.swap_words(x.loc[sub, ["s1", "sx"]].reset_index(drop=True), "test", "France")
    r1 = p46.R._rf(W / "rf_test_s1_France.parquet", pd.unique(x.s1[~sub]), ["core"])      # add1: the single added word
    rx = p46.R._rf(W / "rf_test_sx_France.parquet", pd.unique(x.sx[~sub]), ["core"])
    c1, cx = r1.core.reindex(x.s1[~sub]).fillna("").to_numpy(), rx.core.reindex(x.sx[~sub]).fillna("").to_numpy()
    x.loc[~sub, "w"] = [(g[0] if len(g) == 1 else "") for g in ([t for t in b.split() if t not in set(a.split())] for a, b in zip(c1, cx))]
    x = x[x.w.isin(dec.index)].merge(dec[["s", "e", "n"]], left_on="w", right_index=True)
    print(x.name_rel.value_counts().to_dict())
    OUT.mkdir(parents=True, exist_ok=True)
    x[["s1", "sx", "p", "w", "s", "e", "n", "name_rel"]].to_parquet(OUT / "cells.parquet", index=False)
    dec.to_csv(OUT / "decoy_words.csv")
    print(f"decoy words {len(dec)}; cell pairs {len(x)} (p >= {T_BASE}: {(x.p >= T_BASE).sum()})")
    return x

def build(base: str, out: str, variant: str = "ruled", cell: pd.DataFrame = None) -> int:
    base, out = Path(base), Path(out)
    c = cell if cell is not None else pd.read_parquet(OUT / "cells.parquet")
    if variant == "ruled":
        c = c[c.p < T_BASE]
    drop = {}
    for s1, sx in zip(c.s1, c.sx):
        drop.setdefault(s1, set()).add(sx)
    out.mkdir(parents=True, exist_ok=True)
    n, touched = 0, 0
    with open(base / "matching_results.tsv", encoding="utf-8") as f, open(out / "matching_results.tsv", "w", encoding="utf-8") as g:
        g.write(f.readline())
        for line in f:
            s1, _, ids = line.rstrip("\n").partition("\t")
            if s1 in drop and ids:
                keep = [x for x in ids.split(",") if x not in drop[s1]]
                n += len(ids.split(",")) - len(keep)
                touched += len(keep) < len(ids.split(","))
                g.write(f"{s1}\t{','.join(keep)}\n")
            else:
                g.write(line if line.endswith("\n") else line + "\n")
    for name in ("candidate_pairs.tsv", "code.zip"):
        if (base / name).exists() and not (out / name).exists():
            os.link(base / name, out / name)
    print(f"{variant}: removed {n} France pairs from {touched} S1 -> {out}")
    return n

def selftest():
    with tempfile.TemporaryDirectory() as d:
        b = Path(d) / "b"; b.mkdir()
        (b / "matching_results.tsv").write_text("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S3-2\nS1-2\tS2-3\nS1-3\t\n")
        (b / "candidate_pairs.tsv").write_text("source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-1,S3-2\nS1-2\tS2-3\nS1-3\t\n")
        cell = pd.DataFrame({"s1": ["S1-1", "S1-2"], "sx": ["S3-2", "S2-3"], "p": [0.5, 0.999]})
        assert build(b, Path(d) / "o1", "ruled", cell) == 1
        assert (Path(d) / "o1/matching_results.tsv").read_text().splitlines()[1:] == ["S1-1\tS2-1", "S1-2\tS2-3", "S1-3\t"]
        assert build(b, Path(d) / "o2", "all", cell) == 2
        assert (Path(d) / "o2/matching_results.tsv").read_text().splitlines()[2] == "S1-2\t"
    print("selftest OK")

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "cells": cells()
    elif cmd == "build": build(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "ruled")
    elif cmd == "selftest": selftest()
