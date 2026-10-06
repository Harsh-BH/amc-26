"""P72: corruption marks from RAW text (before normalize's lowercase / accent fold) for (S1, SX) pairs.

marks(n1, a1, nx, ax, vocab) -> DataFrame of MK columns (float32), one row per pair. Raw strings, '' for missing.
vocab: Counter of name tokens (tok()) over the country's S1 pool (test side: the test S1 pool -- unsupervised, allowed).
    .venv/bin/python probes/P72-corrupt-marks/marks.py        # self-check
"""
import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

MK = ["case_x", "case_1", "case_diff", "acc_add", "acc_rm", "ntok_x", "ntok_d", "punct_d", "brack_x", "lead_x", "dba_x",
      "dom_x", "public_x", "noaddr_x", "num_same", "num_trunc", "num_step", "num_other", "num_drop", "only_1", "only_x",
      "swap", "vnew_x", "vlost_1"]
DBA = re.compile(r"\b(dba|d/b/a|d\.b\.a\.?|aka|a\.k\.a\.?|a/k/a|fka|f/k/a|formerly|t/a|trading as)\b")
DOM = re.compile(r"(www\.|https?:|\.(com|net|org|in|co|biz|info|io|us|fr|shop|store)\b)")
NUM = re.compile(r"\d+")
WORD = re.compile(r"[^\W_]+")          # python re: unicode letters/digits (Indic + accents kept)
NULLS = {"", "null", "<null>", "none", "n/a", "na"}
F_VOCAB, SIM = 20, 0.5


def case(s: str) -> int:
    """0 lower, 1 upper, 2 title, 3 mixed, 4 no cased letters (Indic / digits only)."""
    if not any(ch.isalpha() and ch.lower() != ch.upper() for ch in s):
        return 4
    return 0 if s.islower() else 1 if s.isupper() else 2 if all(map(_title, s.split())) else 3


def _title(w: str) -> bool:
    """First cased letter upper, the rest lower; short all-caps words (LLC, SA) pass too."""
    c = [ch for ch in w if ch.lower() != ch.upper()]
    return not c or (c[0].isupper() and all(x.islower() for x in c[1:])) or (w.isupper() and len(c) <= 4)


def n_acc(s: str) -> int:
    """Latin letters carrying a diacritic (NFD has a combining mark after a Latin base)."""
    return sum(1 for ch in s if "À" <= ch <= "ɏ" and ch.isalpha() and len(unicodedata.normalize("NFD", ch)) > 1)


def fold(t: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", t.casefold()) if not unicodedata.combining(ch))


def tok(s: str) -> list:
    return [fold(t) for t in WORD.findall(s) if not t.isdigit()]


def punct(s: str) -> int:
    return sum(1 for ch in s if unicodedata.category(ch)[0] in "PS")     # not isalnum: Indic matras are Mn/Mc


def num_cls(n1: set, nx: set) -> tuple:
    same = trunc = step = other = 0
    for v in nx:
        if v in n1:
            same += 1
        elif any(len(u) != len(v) and (u.startswith(v) or v.startswith(u)) for u in n1):
            trunc += 1
        elif any(len(u) == len(v) and 0 < abs(int(u) - int(v)) <= 20 for u in n1 if len(u) < 12):
            step += 1
        else:
            other += 1
    return same, trunc, step, other, sum(1 for u in n1 if u not in nx)


def swap_cols(t1: list, tx: list, vocab: Counter) -> tuple:
    s1, sx = set(t1), set(tx)
    o1, ox = s1 - sx, sx - s1
    close = lambda t, pool: any(Levenshtein.normalized_similarity(t, u) >= 0.75 or u.startswith(t) or t.startswith(u)
                                for u in pool)
    real = lambda t: vocab.get(t, 0) >= F_VOCAB and len(t) > 2
    vnew = sum(1 for t in ox if real(t) and not close(t, o1))
    vlost = sum(1 for t in o1 if real(t) and not close(t, ox))
    sw = 0
    if len(o1) == 1 and len(ox) == 1:
        a, b = next(iter(o1)), next(iter(ox))
        sw = int(real(a) and real(b) and Levenshtein.normalized_similarity(a, b) < SIM
                 and not (a.startswith(b) or b.startswith(a)))
    return len(o1), len(ox), sw, vnew, vlost


def marks(n1, a1, nx, ax, vocab: Counter) -> pd.DataFrame:
    rows = []
    for u1, b1, ux, bx in zip(n1, a1, nx, ax):
        c1, cx = case(u1), case(ux)
        k1, kx = n_acc(u1), n_acc(ux)
        w1, wx = u1.split(), ux.split()
        lx = ux.lower()
        t1, tx = tok(u1), tok(ux)
        rows.append((cx, c1, float(c1 != cx and cx != 4 and c1 != 4), float(kx > 0 and k1 == 0), float(k1 > 0 and kx == 0),
                     len(wx), len(wx) - len(w1), punct(ux) - punct(u1),
                     float(any(ch in "()[]{}" for ch in ux) and not any(ch in "()[]{}" for ch in u1)),
                     float(bool(ux) and not ux[0].isalnum()), float(bool(DBA.search(lx))), float(bool(DOM.search(lx))),
                     float("public" in tx and "public" not in t1), float(bx.strip().lower() in NULLS),
                     *num_cls(set(NUM.findall(u1 + " " + b1)), set(NUM.findall(ux + " " + bx))),
                     *swap_cols(t1, tx, vocab)))
    return pd.DataFrame(rows, columns=MK, dtype=np.float32)


def vocab_of(names) -> Counter:
    v = Counter()
    for s in names:
        v.update(set(tok(s)))
    return v


if __name__ == "__main__":
    v = Counter({"bakery": 50, "garage": 50, "joes": 30, "cafe": 99})
    d = marks(["Joe's Bakery", "Café Rouge SARL", "ACME 4712 Inc"], ["12590 Main St", "3 rue X", "4712 Elm"],
              ["joe's garage", "CAFE ROUGE sarl (dba Rouge)", "www.acme.com"], ["12594 Main St", "NULL", "471 Elm"], v)
    r = d.iloc[0]
    assert (r.case_x, r.case_1, r.case_diff, r.swap, r.num_step, r.num_same) == (0, 2, 1, 1, 1, 0), r.to_dict()
    r = d.iloc[1]
    assert (r.acc_rm, r.brack_x, r.dba_x, r.noaddr_x, r.case_x) == (1, 1, 1, 1, 3), r.to_dict()
    r = d.iloc[2]
    assert (r.dom_x, r.num_trunc, r.num_drop) == (1, 1, 1), r.to_dict()
    assert case("सनराइज") == 4 and n_acc("Érwin") == 1 and tok("Café-Rouge 12") == ["cafe", "rouge"]
    print("marks self-check ok")
