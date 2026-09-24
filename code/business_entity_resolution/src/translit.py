"""Indic script -> Latin for S2/S3 names and addresses (S1 is always Latin).

    to_latin(s)   Indic spans -> Latin: learned table first (S1 spelling), rule romanizer for the rest
    fold(s)       phonetic key for BOTH sides (lowercase, aa->a, w->v, sh->s, z->j, ...), for name n-grams
    oov_flag(s)   True where an Indic word is outside the learned table: on dev, only distractors do that

The generator writes Indic names as token-for-token transliterations of the S1 name from a closed
vocabulary (~1.5k word types, identical in train and test), and Indic addresses carry only the state.
So a table learned from train pairs gives the S1 spelling back. The rule romanizer covers the rest: all
nine Indic Unicode blocks are parallel (ISCII order), so every script shifts onto the Devanagari offsets
and one table romanizes it. Words are cached, so cost scales with the vocabulary, not the row count.
"""
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

INDIC = "[\u0900-\u0D7F]"                                # not raw: RE2 cannot parse a \u escape
IA = {0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00}            # Indo-Aryan blocks: final schwa is silent
CONS = dict(zip(range(0x15, 0x3A), "k kh g gh n ch chh j jh n t th d dh n t th d dh n n p ph b bh m y r r l l l v sh sh s h".split()))
VOWEL = dict(zip(range(0x04, 0x15), "a a a i i u u ri li e e e ai o o o au".split()))
SIGN = dict(zip(range(0x3E, 0x4D), "a i i u u ri ri e e e ai o o o au".split()))
NUKTA = {"k": "q", "g": "g", "j": "z", "d": "r", "dh": "rh", "ph": "f"}
PRECOMP = {0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f", 0x5F: "y"}
OTHER = {0x01: "n", 0x02: "n", 0x03: "h", 0x50: "om", 0x60: "ri", 0x61: "li", 0x64: " ", 0x65: " ", 0x70: "."}
OTHER.update({0x66 + d: str(d) for d in range(10)})
RAW = {0x09CE: "t", 0x09F0: "r", 0x09F1: "v", 0x0A70: "n", 0x0B71: "v",     # cps whose offset collides
       0x0D7A: "n", 0x0D7B: "n", 0x0D7C: "r", 0x0D7D: "l", 0x0D7E: "l", 0x0D7F: "k"}
NUK, VIRAMA, SEMI = 0x3C, 0x4D, {0x2F, 0x30, 0x32, 0x35}     # SEMI = y r l v


@lru_cache(maxsize=None)
def roman(tok: str) -> str:
    """One whitespace token -> Latin. Non-Indic characters are kept as they are."""
    out, schwa, akshara, block = [], -1, 0, 0x0900
    cps = [ord(c) for c in tok if c not in "\u200c\u200d"]
    i = 0
    while i < len(cps):
        cp = cps[i]
        if not 0x0900 <= cp <= 0x0D7F:
            out.append(chr(cp)); i += 1; continue
        block, off = cp & ~0x7F, cp & 0x7F
        if cp in RAW:
            out.append(RAW[cp])
        elif off in CONS or off in PRECOMP:
            c = CONS.get(off) or PRECOMP[off]
            nxt = cps[i + 1] & 0x7F if i + 1 < len(cps) and 0x0900 <= cps[i + 1] <= 0x0D7F else None
            if nxt == NUK:
                c = NUKTA.get(c, c); i += 1
                nxt = cps[i + 1] & 0x7F if i + 1 < len(cps) and 0x0900 <= cps[i + 1] <= 0x0D7F else None
            out.append(c); akshara += 1
            if nxt not in SIGN and nxt != VIRAMA:
                schwa = len(out); out.append("a")
        else:
            out.append(VOWEL.get(off) or SIGN.get(off) or OTHER.get(off, ""))
            akshara += off in VOWEL
        i += 1
    # final inherent 'a' is silent in Hindi/Marathi/Gujarati/Bengali/Punjabi (bharat, shivam), unless the
    # word ends in a semivowel conjunct (surya, mitra, shukla); English loans drop it (projects, modern).
    # Dravidian scripts write their final consonants with virama, so their final 'a' is real
    if block in IA and schwa == len(out) - 1 and akshara > 1 and not (
            len(cps) > 2 and cps[-2] & 0x7F == VIRAMA and cps[-1] & 0x7F in SEMI):
        out.pop()
    return "".join(out)


SPAN = re.compile("[\u0900-\u0D7F\u200c\u200d]+(?: [\u0900-\u0D7F\u200c\u200d]+)*")   # Indic run, may hold spaces
EDGE = re.compile(r"^\W+|\W+$")
TABLE = Path(os.environ.get("ER_WORK", "data/work")) / "p2_xlit_table.parquet"


def learn(pairs: pd.DataFrame) -> dict:
    """{Indic span: Latin} from true pairs (xn, xa = Indic S2/S3 name, address; sn, sa = their S1).
    Names are token-for-token transliterations of the S1 name, so tokens align by position. Addresses carry
    Indic only in the state segment: it maps to the S1 segment the Latin part of the S2/S3 address lacks."""
    names, addrs, seen = defaultdict(Counter), defaultdict(Counter), Counter()
    for xn, xa, sn, sa in zip(pairs.xn, pairs.xa, pairs.sn, pairs.sa):
        x, y = xn.split(" "), sn.split(" ")
        if len(x) == len(y):
            for t, l in zip(x, y):
                m = SPAN.findall(t)
                if len(m) == 1 and EDGE.sub("", l):
                    names[m[0]][EDGE.sub("", l).lower()] += 1
        xs = [s.strip() for s in xa.split(",")]
        ind = [s for s in xs if SPAN.fullmatch(s)]
        if ind:
            cand = {s.strip().lower() for s in sa.split(",")} - {s.lower() for s in xs}
            seen.update(ind)
            for s in ind:
                addrs[s].update(cand)
    out = {t: v.most_common(1)[0][0] for t, v in names.items() if v.total() >= 2}
    for s, v in addrs.items():
        l, n = v.most_common(1)[0]
        if n >= 5 and n >= 0.5 * seen[s]:
            out[s] = l
    return out


@lru_cache(maxsize=1)
def table() -> dict:
    """Learned on train folds 2-9 (dev = fold 1 and holdout = fold 0 stay unseen). Cached in data/work: it
    is derived from Amazon's rows, so it never goes into the repo."""
    if not TABLE.exists():
        from pairs import build
        src = TABLE.with_name("p2_pairs.parquet")
        if not src.exists():
            build(str(src))
        p = pd.read_parquet(src)
        t = learn(p[p.fold >= 2])
        pd.DataFrame({"indic": list(t), "latin": list(t.values())}).to_parquet(TABLE, index=False)
    t = pd.read_parquet(TABLE)
    return dict(zip(t.indic, t.latin))


def _span(p: str, tab: dict) -> str:
    return tab.get(p) or " ".join(tab.get(w) or roman(w) for w in p.split(" "))


def to_latin(s: pd.Series, tab: dict | None = None) -> pd.Series:
    """Romanize the Indic spans of each string: learned table first, rule romanizer for the rest.
    Latin text passes through untouched; rows without Indic text are returned unchanged. tab={} = rule only."""
    tab = table() if tab is None else tab
    s = s.astype("string")
    hit = s.str.contains(INDIC, regex=True).fillna(False).to_numpy(bool)
    out = s.copy()
    out[hit] = s[hit].map(lambda v: SPAN.sub(lambda m: _span(m.group(), tab), v))
    return out


def oov_flag(s: pd.Series, tab: dict | None = None) -> pd.Series:
    """True where some Indic word is not in the learned table (dev: 446 such records, 0 of them matched)."""
    tab = table() if tab is None else tab
    return s.astype("string").fillna("").map(
        lambda v: any(w not in tab for m in SPAN.findall(v) for w in m.split(" "))).astype(bool)


FOLD = [(r"[^a-z0-9 ]+", " "), (r"ph", "f"), (r"w", "v"), (r"sh|ch|c(?=[eiy])", "s"), (r"z", "j"),
        (r"ck|c|q", "k"), (r"x", "ks"), (r"([kgtdbj])h", r"\1"), (r"ee|y", "i"), (r"oo", "u"),
        (r"([a-z])\1+", r"\1"), (r"\s+", " ")]
FOLD = [(re.compile(a), b) for a, b in FOLD]


@lru_cache(maxsize=1 << 20)
def _fold(s: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))
    for a, b in FOLD:
        s = a.sub(b, s)
    return s.strip()


def fold(s: pd.Series) -> pd.Series:
    """Phonetic key, same for both scripts once romanized: 'Shiv Builders' and 'shiv bildars' converge."""
    return s.astype("string").fillna("").map(_fold).astype("string")


if __name__ == "__main__":
    t = pd.Series(["लाइफ प्रोजेक्ट्स प्राइवेट लिमिटेड", "క్లాసిక్ సాఫ్ట్‌వేర్", "ഗുഡ് ടെക്നോളജീസ്", "মডার্ন কেয়ার",
                   "શિવમ ટ્રેડિંગ", "தமிழ்நாடு", "भारत सूर्य", "Surya હોસ્પિટાલિટી Pvt", "plain latin", ""])
    r = to_latin(t, {})
    print(list(r)); print(list(fold(r)))
    assert r[0] == "laiph projekts praivet limited", r[0]
    assert r[6] == "bharat surya" and r[7].startswith("Surya ") and r[8] == "plain latin"
    r = to_latin(pd.Series(["लाइफ प्रोजेक्ट्स प्राइवेट लिमिटेड", "HN 742, PUNE, महाराष्ट्र", "D-19, दिल्ली", "Noida, उत्तर प्रदेश,", "प्रा. लि."]))
    print(list(r))
    assert list(r) == ["life projects private limited", "HN 742, PUNE, maharashtra", "D-19, delhi", "Noida, uttar pradesh,", "pvt. ltd."]
    assert list(fold(pd.Series(["Shree Photo Works", "shri foto varks", "Café"]))) == ["sri foto vorks", "sri foto varks", "kafe"]
    assert list(oov_flag(pd.Series(["मोटर्स", "लिमिटेड प्रा. लि.", "Latin", ""]))) == [True, False, False, False]
