"""P63 typo corrector: map each SX token that is not in the S1 vocabulary to its unique near S1 token.

    V = token counts of the S1 texts of one pool (one country's S1 file; label-free, test files only on test)
    t -> v  when t not in V, len(t) >= 4, t has a letter and no digit, and v is the only V token at the minimal
            Damerau-Levenshtein distance d <= 1 (<= 2 for len(t) >= 8); on a tie the most frequent v wins if it is
            >= 3x the runner-up, else t stays.  Candidates come from a deletes index (SymSpell): V keys = v and its
            1-deletes; query keys = t, its 1-deletes and, for len >= 8, its 2-deletes, then an exact DL check.
    ponytail: the index misses t = v minus 2 letters and 2 substitutions (V-side 2-deletes cost ~10x the RAM);
              add V-side 2-deletes for long tokens if the miss rate matters.

    .venv/bin/python probes/P63-fr-typo-correct/typo.py        # self-check
"""
import re

import numpy as np
import pandas as pd
from rapidfuzz.distance import DamerauLevenshtein
from rapidfuzz.process import cpdist

_OK = re.compile(r"^[^\d\s]{4,}$")


def counts(texts) -> pd.Series:
    """token -> frequency over an iterable of space-separated strings."""
    s = pd.Series(texts, dtype="string").fillna("")
    return s.str.split(" ").explode().loc[lambda x: x.str.len() > 0].value_counts()


def _dels(tokens: np.ndarray, twice: np.ndarray | None = None) -> pd.DataFrame:
    """(key, i) for every token i: the token itself, its 1-deletes and, where twice[i], its 2-deletes."""
    k, ix = [], []
    for i, t in enumerate(tokens):
        s = {t} | {t[:a] + t[a + 1:] for a in range(len(t))}
        if twice is not None and twice[i]:
            s |= {u[:a] + u[a + 1:] for u in list(s) for a in range(len(u))}
        k.extend(s)
        ix.extend([i] * len(s))
    return pd.DataFrame({"key": pd.array(k, dtype="string"), "i": np.asarray(ix, np.int32)})


def build_map(vocab: pd.Series, query: pd.Series, chunk: int = 20_000) -> pd.Series:
    """vocab = S1 token counts, query = SX token counts -> Series oov token -> corrected token (only the corrected)."""
    q = query.index.to_numpy(object)
    oov = q[~pd.Index(q).isin(vocab.index) & pd.Series(q).str.match(_OK.pattern).to_numpy()]
    if not len(oov):
        return pd.Series(dtype=object)
    vt = vocab.index.to_numpy(object)
    vl = np.fromiter(map(len, vt), np.int32, len(vt))
    keep = vl >= 3                                    # a len-4 t is within 1 of len >= 3 tokens only
    vt, vf = vt[keep], vocab.to_numpy()[keep]
    ql = np.fromiter(map(len, oov), np.int32, len(oov))
    VK = _dels(vt)
    parts = []
    for a in range(0, len(oov), chunk):                # query side in chunks (its 2-deletes dominate RAM)
        K = VK.merge(_dels(oov[a:a + chunk], ql[a:a + chunk] >= 8).rename(columns={"i": "j"}), on="key")[["i", "j"]]
        K = K.drop_duplicates()
        j = K.j.to_numpy() + a
        d = cpdist(list(oov[j]), list(vt[K.i.to_numpy()]), scorer=DamerauLevenshtein.distance, workers=-1)
        ok = d <= np.where(ql[j] >= 8, 2, 1)
        parts.append(pd.DataFrame({"j": j[ok], "i": K.i.to_numpy()[ok], "d": d[ok]}))
        del K
    C = pd.concat(parts, ignore_index=True)
    C["f"] = vf[C.i.to_numpy()]
    C = C[C.d == C.groupby("j").d.transform("min")].sort_values(["j", "f"], ascending=[True, False])
    top = C.groupby("j").head(2)
    first = top.groupby("j").nth(0).set_index("j")
    second = top.groupby("j").nth(1).set_index("j").f.reindex(first.index)
    ok = second.isna() | (first.f >= 3 * second.fillna(0))
    first = first[ok.to_numpy()]
    return pd.Series(vt[first.i.to_numpy()], index=oov[first.index.to_numpy()])


def apply(texts: pd.Series, m: pd.Series) -> tuple:
    """Rewrite each space-separated string token by token with map m -> (new texts, changed mask)."""
    if not len(m):
        return texts, np.zeros(len(texts), bool)
    md = m.to_dict()
    s = texts.fillna("").astype(str).tolist()
    out = [" ".join(md.get(w, w) for w in x.split(" ")) if x else x for x in s]
    ch = np.fromiter((o != x for o, x in zip(out, s)), bool, len(s))
    return pd.Series(out, index=texts.index, dtype=texts.dtype), ch


def _selfcheck():
    v = counts(["rue konrad adenauer", "jeanne dupont", "creation bois", "jeanne marie", "konrad", "jeanie"])
    m = build_map(v, counts(["rue ktnrad adenauer", "jaenne dupond", "crtation bois", "abcd", "rue konrad", "12ab"]))
    assert m.get("ktnrad") == "konrad" and m.get("crtation") == "creation", m
    assert m.get("jaenne") == "jeanne", m               # transposition; jeanie is d=2 > 1 for len 6
    assert "abcd" not in m and "12ab" not in m and "konrad" not in m
    assert m.get("dupond") == "dupont"
    t, ch = apply(pd.Series(["rue ktnrad adenauer", "abcd", ""]), m)
    assert t.tolist() == ["rue konrad adenauer", "abcd", ""] and ch.tolist() == [True, False, False]
    # tie: two V tokens at d = 1 with similar counts -> leave; a 3x-dominant one -> take it
    v2 = pd.Series({"marin": 5, "marie": 4, "paris": 30, "parts": 2})
    m2 = build_map(v2, pd.Series({"marix": 1, "parix": 1}))
    assert "marix" not in m2 and m2.get("parix") == "paris", m2
    print("typo self-check ok")


if __name__ == "__main__":
    _selfcheck()
