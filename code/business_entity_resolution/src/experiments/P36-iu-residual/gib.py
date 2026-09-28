"""P36: the generator's synthetic brand names ("gib"), learned label-free from one split's own names.

The data generator renames some S2/S3 records to a made-up brand glued from a closed set of syllables
(shapes like 'Zephcalonex', 'QÚOZETASOL', 'Ke1ozeph': 2-3 syllables, case / accent / digit noise), and keeps
(a perturbed copy of) the S1's address. Such a record is a true match 88% (US) / 99% (India) of the time in train,
but the name view of blocking cannot find its S1.

learn(split) finds the syllables with no labels and no word list: take SX name tokens that no S1 name uses (non-web
names, >= 6 letters), keep the 3-7 letter strings that start AND end >= 100 of them, link two strings when they
glue into one such token, and keep the k-core (every syllable glues with >= 25 others). Real-word glue ('global' +
'trading') never forms a core that dense. On train it returns exactly 44 syllables.
    probes/P36-iu-residual/py.sh probes/P36-iu-residual/gib.py [split]     # self-check (+ learn on split)
"""
import collections
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

W = Path(__file__).resolve().parents[2] / "data/work"
DIG = str.maketrans("0158", "olsb")                     # the generator's digit-for-letter typos
WEB = re.compile(r"www|\.[a-z]{2,4}\b|@|#|com\b", re.I)
MIN_END, CORE = 100, 25


def toks(s: str) -> list:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower().translate(DIG)
    return re.findall(r"[a-z]+", s)


def _names(split: str, src: int):
    f = pq.ParquetFile(W / f"{split}_source{src}.parquet")
    for rg in range(f.num_row_groups):
        yield from f.read_row_group(rg, columns=["business_name"]).column(0).to_pylist()


def core(oov) -> list:
    """The k-core syllables of an iterable of OOV tokens."""
    pre, suf = collections.Counter(), collections.Counter()
    for t in oov:
        for L in range(3, min(8, len(t) - 2)):
            pre[t[:L]] += 1
            suf[t[-L:]] += 1
    cand = {m for m in pre if pre[m] >= MIN_END and suf[m] >= MIN_END}
    edges = collections.defaultdict(set)
    for t in oov:
        for L in range(3, len(t) - 2):
            a, b = t[:L], t[L:]
            if a in cand and b in cand:
                edges[a].add(b)
                edges[b].add(a)
    M = set(cand)
    while drop := {m for m in M if len(edges[m] & M) < CORE}:
        M -= drop
    return sorted(M)


def learn(split: str) -> list:
    """Syllables of one split (cached at data/work/p36/morph_<split>.json)."""
    path = W / "p36" / f"morph_{split}.json"
    if path.exists():
        return json.loads(path.read_text())
    v1 = {t for n in _names(split, 1) for t in toks(n)}
    oov = {t for i in (2, 3) for n in _names(split, i) if not WEB.search(n) for t in toks(n) if len(t) >= 6 and t not in v1}
    M = core(oov)
    path.write_text(json.dumps(M))
    return M


def n_syll(t: str, M: set) -> int:
    """Fewest syllables that spell token t exactly (0 = it cannot be spelled)."""
    best = [0] + [99] * len(t)
    for i in range(len(t)):
        if best[i] < 99:
            for L in range(2, min(8, len(t) - i) + 1):
                if t[i:i + L] in M:
                    best[i + L] = min(best[i + L], best[i] + 1)
    return best[-1] if best[-1] < 99 else 0


def is_gib(names, M) -> np.ndarray:
    """A name with a token spelled by >= 2 syllables."""
    M = set(M)
    return np.fromiter((any(n_syll(t, M) >= 2 for t in toks(n)) for n in names), bool, len(names))


if __name__ == "__main__":
    M = {"zeph", "calo", "nex", "brix", "pyra", "lyra", "kelo", "iri", "quo", "sol"}
    got = is_gib(["Zephcalonex", "M/s Brixpyra Pvt Ltd", "QÚOLYRASOL", "Ke1ozeph", "Zeph", "Solar Nexus", "Brixton LLC",
                  "Calo Brix"], M)
    assert list(got) == [True, True, True, True, False, False, False, False], got
    assert n_syll("zephcalobrix", {"zeph", "calo", "brix"}) == 3
    print("gib self-check ok")
    if len(sys.argv) > 1:
        m = learn(sys.argv[1])
        print(sys.argv[1], len(m), m)
