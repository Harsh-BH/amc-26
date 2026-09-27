"""Name relation between an S1 and an SX core: 1 shared token (len>=3), 2 squashed containment (len>=4),
3 acronym (initials), 4 a common squashed substring of >= 5 letters; 0 = unrelated (e.g. an invented name at the same address).
Measures, on train truth (LOCO files), how true 'same hn + same city, unrelated name' pairs are, by p band.
    .venv/bin/python probes/P37-fr-complete/namerel.py
"""
import difflib, json, re, sys
import numpy as np
import pandas as pd
sys.path[:0] = ["probes/P37-fr-complete", "probes/P28-addr-anchor"]
import p37 as P

SQ = re.compile(r"[\W\d_]+")          # Python re (Unicode-aware), not RE2: keeps accented letters


def rel(c1: str, cx: str) -> int:
    t1, tx = c1.split(), cx.split()
    if {w for w in t1 if len(w) >= 3} & {w for w in tx if len(w) >= 3}:
        return 1
    q1, qx = SQ.sub("", c1), SQ.sub("", cx)
    if len(qx) >= 4 and len(q1) >= 4 and (qx in q1 or q1 in qx):
        return 2
    if len(qx) >= 2 and qx in ("".join(w[0] for w in t1 if w), "".join(w[0] for w in t1 if len(w) >= 3)):
        return 3
    if min(len(q1), len(qx)) >= 5 and difflib.SequenceMatcher(None, q1, qx, autojunk=False).find_longest_match(
            0, len(q1), 0, len(qx)).size >= 5:
        return 4
    return 0


def rel_arr(c1, cx):
    return np.fromiter((rel(a, b) for a, b in zip(c1, cx)), np.int8, len(c1))


if __name__ == "__main__":
    assert [rel("ehpad de marie", "em"), rel("maison de sante joseph", "maisonsant"), rel("roubaix club", "rc"),
            rel("bordeaux club", "orbikelo sys"), rel("car groupe", "car groupe"), rel("saint nazaire anciens", "asaintnazaire com")] \
        == [3, 4, 3, 0, 1, 4]
    W = P.W
    out = {}
    for src, tgt in (("India", "US"), ("US", "India")):
        a = P.loco_load(src, tgt)
        p, y, hc = a["p"], a["y"], a["hc"]
        c1 = pd.read_parquet(W / f"rf_train_s1_{tgt}.parquet", columns=["id", "core", "city"]).set_index("id")
        cx = pd.read_parquet(W / f"rf_train_sx_{tgt}.parquet", columns=["id", "core", "city"]).set_index("id")
        m = hc == 0
        o = np.flatnonzero(m)
        k1, kx = a["s1_ids"][o], a["sx_ids"][o]
        city = (c1.city.reindex(k1).fillna("").to_numpy() == cx.city.reindex(kx).fillna("").to_numpy())
        r = rel_arr(c1.core.reindex(k1).fillna("").to_numpy(), cx.core.reindex(kx).fillna("").to_numpy())
        df = pd.DataFrame({"p": pd.cut(p[o], [0, .05, .3, .6, .9, .97, 1.0001]).astype(str), "rel": r, "y": y[o], "city": city})
        df = df[df.city]
        tab = df.groupby(["rel", "p"]).y.agg(["size", "mean"]).round(3)
        print(f"{src}->{tgt}: hn-equal + same-city candidate pairs by name relation x p band (size, precision)\n{tab.to_string()}", flush=True)
        print("  true pairs' relation mix:", df[df.y].rel.value_counts(normalize=True).round(3).sort_index().to_dict(), flush=True)
        out[f"{src}->{tgt}"] = {f"{k[0]}|{k[1]}": v for k, v in tab.to_dict("index").items()}
        del a
    json.dump(out, open("probes/P37-fr-complete/namerel.json", "w"), indent=1)
