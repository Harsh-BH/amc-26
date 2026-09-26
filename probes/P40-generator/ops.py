"""P40: tag the generator operations that turn an S1 record into an S2/S3 record (one pair at a time).

tag(n1, a1, nx, ax) -> int bitmask over OPS (name ops N_*, address ops A_*). Works on raw strings; it is a
descriptive detector for the catalogue / error join, not a model feature. Token logic mirrors normalize.py
(same LEGAL / HONORIFIC / STREET / FILLER tables, same digit-for-letter map).
    .venv/bin/python probes/P40-generator/ops.py        # self-check
"""
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P36-iu-residual")]
import normalize as NZ  # noqa: E402
from gib import n_syll  # noqa: E402

SYLL = set(__import__("json").loads((ROOT / "data/work/p36/morph_train.json").read_text()))
OPS = [
    # name
    "N_EXACT", "N_CASE", "N_SPACE", "N_ACCENT", "N_SCRIPT", "N_WEB", "N_HANDLE", "N_PIPE", "N_PHONE", "N_IDTAIL",
    "N_ALIAS", "N_HONOR", "N_BRACKET", "N_HYPHEN", "N_PAREN_DROP", "N_LEGAL_ADD", "N_LEGAL_DROP", "N_LEGAL_SWAP",
    "N_GENERIC_ADD", "N_TOKDROP", "N_REORDER", "N_DUP", "N_GIB_NAME", "N_INSERT", "N_T_DIGIT", "N_T_DEL", "N_T_INS",
    "N_T_SUB", "N_T_TRANSP", "N_T_MULTI", "N_ACRONYM", "N_UNRELATED",
    # address
    "A_EXACT", "A_EMPTY", "A_PLACEHOLDER", "A_NULLSEG", "A_NASEG", "A_UPPER", "A_HN_SAME", "A_HN_DROP", "A_HN_ADD",
    "A_HN_LEADDROP", "A_HN_PREFIXNUM", "A_HN_SMALL", "A_HN_SIB", "A_HN_OTHER", "A_HN_LABEL", "A_HN_SUFFIX",
    "A_STREET_ABBR", "A_SEG_DROP", "A_SEG_REORDER", "A_SEG_TYPO", "A_STATE_FORM", "A_STATE_SCRIPT", "A_FILLER",
    "A_SEG_ADD",
    # decoy-style edits (distractor generator candidates)
    "N_SYLL_APPEND", "N_SHORT_EDIT", "A_NUM_SUB",
]
BIT = {o: 1 << i for i, o in enumerate(OPS)}
SIB3 = {3, 4, 5, 7, 9, 11, 13, 21}
DIG = str.maketrans("0134568", "oleasgb")
GENERIC = {"center", "centre", "services", "service", "group", "holdings", "partners", "ventures", "industries",
           "overseas", "public", "east", "west", "north", "south", "midtown", "international", "global", "solutions",
           "enterprises", "associates", "hub", "labs", "systems", "co"}   # P12/P15 branch words (for N_GENERIC_ADD)
HON = NZ.HONORIFIC | {"sri", "shri", "smt", "m s"}
WEBRE = re.compile(r"www\.|https?://|\.(com|net|org|in|co|biz|info|io|us|fr)\b", re.I)
PHONE = re.compile(r"\d{7,}")
NA = {"", "null", "<null>", "n/a", "na", "none", "-"}
HN_LABEL = re.compile(r"(^|,\s*)(#|no\.?\s|h\.?\s?no|hn\b|door|plot|flat|shop|house|[a-z]-\d)", re.I)


def _strip(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch)).lower()


def _toks(s: str) -> list:
    return re.findall(r"\w+", _strip(s).replace("&", " and "))


def _script(s: str) -> bool:
    return any(ord(ch) > 0x24F and ch.isalpha() for ch in s)


def _is_legal(t: str, nxt: str) -> bool:
    return NZ._is_legal(t, nxt)


def _name_ops(n1: str, nx: str) -> int:
    b = 0
    if n1 == nx:
        return BIT["N_EXACT"]
    if n1.lower() == nx.lower():
        b |= BIT["N_CASE"]
    if "  " in nx and "  " not in n1:
        b |= BIT["N_SPACE"]
    if _script(nx):
        return b | BIT["N_SCRIPT"]
    if any(unicodedata.combining(ch) for ch in unicodedata.normalize("NFKD", nx)) and \
            _strip(nx) != nx.lower() and not any(unicodedata.combining(ch) for ch in unicodedata.normalize("NFKD", n1)):
        b |= BIT["N_ACCENT"]
    if "|" in nx:
        b |= BIT["N_PIPE"]
        nx = nx.split("|")[0]
    elif nx.lstrip().startswith("@"):
        return b | BIT["N_HANDLE"]
    elif WEBRE.search(nx) and len(nx.split()) == 1:
        return b | BIT["N_WEB"]
    if PHONE.search(nx) and not PHONE.search(n1):
        b |= BIT["N_PHONE"]
    if NZ._ID_TAIL.search(nx.lower()) and not NZ._ID_TAIL.search(n1.lower()):
        b |= BIT["N_IDTAIL"]
    if NZ._ALIAS.search(nx.lower()) and not NZ._ALIAS.search(n1.lower()):
        b |= BIT["N_ALIAS"]
    if "[" in nx and "[" not in n1:
        b |= BIT["N_BRACKET"]
    if "-" in nx and "-" not in n1:
        b |= BIT["N_HYPHEN"]
    if "(" in n1 and "(" not in nx:
        b |= BIT["N_PAREN_DROP"]
    t1 = [t for t in _toks(re.sub(r"\bm/s\b", "ms", n1, flags=re.I))]
    tx = [t for t in _toks(re.sub(r"\bm/s\b", "ms", PHONE.sub(" ", nx), flags=re.I))]
    if tx and tx[0] in HON and (not t1 or t1[0] != tx[0]):
        b |= BIT["N_HONOR"]
        tx = tx[1:]
    if len(tx) != len(set(tx)) and len(t1) == len(set(t1)):
        b |= BIT["N_DUP"]
    leg1 = {t for i, t in enumerate(t1) if _is_legal(t, t1[i + 1] if i + 1 < len(t1) else "")}
    legx = {t for i, t in enumerate(tx) if _is_legal(t, tx[i + 1] if i + 1 < len(tx) else "")}
    s1set = set(t1)
    used, order, extra = set(), [], []
    for i, t in enumerate(tx):
        if t in legx and t not in s1set:
            continue
        if t in s1set:
            j = t1.index(t)
            used.add(j)
            order.append(j)
            continue
        if not t.isdigit() and t.translate(DIG) in s1set and t.translate(DIG) != t:
            b |= BIT["N_T_DIGIT"]
            j = t1.index(t.translate(DIG))
            used.add(j)
            order.append(j)
            continue
        best, bj = 0.0, -1
        for j, u in enumerate(t1):
            if len(u) >= 3 and len(t) >= 3:
                r = Levenshtein.normalized_similarity(t, u)
                if r > best:
                    best, bj = r, j
        if best >= 0.6:
            u = t1[bj]
            used.add(bj)
            order.append(bj)
            if sorted(u) == sorted(t):
                b |= BIT["N_T_TRANSP"]
            else:
                kinds = {op.tag for op in Levenshtein.editops(u, t)}
                b |= BIT["N_T_DEL"] if kinds == {"delete"} else BIT["N_T_INS"] if kinds == {"insert"} else \
                    BIT["N_T_SUB"] if kinds == {"replace"} else BIT["N_T_MULTI"]
            continue
        extra.append(t)
    newx = [t for t in tx if t not in s1set and t.translate(DIG) not in s1set]
    for t in newx:
        tt = t.translate(DIG)
        for u in t1:
            if 2 <= len(u) <= 3 and not u.isdigit() and tt != u and (tt[:-1] == u or u[:-1] == tt or
                                                                     (len(tt) == len(u) and tt[:-1] == u[:-1])):
                if not _is_legal(tt, ""):
                    b |= BIT["N_SHORT_EDIT"]
            if len(u) >= 4 and tt != u:
                for j in (0, 1, 2):
                    stem = u[:len(u) - j]
                    if tt.startswith(stem) and len(tt) - len(stem) >= 2 and n_syll(tt[len(stem):], SYLL) >= 1:
                        b |= BIT["N_SYLL_APPEND"]
                        break
    if legx - leg1 and leg1 - legx:
        b |= BIT["N_LEGAL_SWAP"]
    elif legx - leg1:
        b |= BIT["N_LEGAL_ADD"]
    elif leg1 - legx:
        b |= BIT["N_LEGAL_DROP"]
    core1 = [j for j, t in enumerate(t1) if t not in leg1 and t not in NZ.STOP and t not in HON]
    if any(j not in used for j in core1) and used:
        b |= BIT["N_TOKDROP"]
    if order != sorted(order):
        b |= BIT["N_REORDER"]
    for t in extra:
        if t in GENERIC:
            b |= BIT["N_GENERIC_ADD"]
        elif t in NZ.STOP or t in HON:
            pass
        elif not used and n_syll(t.translate(DIG), SYLL) >= 2:
            b |= BIT["N_GIB_NAME"]
        else:
            b |= BIT["N_INSERT"]
    if not used:
        core = [t1[j] for j in core1]
        if len(core) >= 2 and "".join(tx) in ("".join(t[0] for t in core), "".join(t[0] for t in t1)):
            b |= BIT["N_ACRONYM"]
        elif not b & BIT["N_GIB_NAME"]:
            b |= BIT["N_UNRELATED"]
    return b


def _segs(a: str) -> list:
    return [s.strip() for s in a.split(",") if s.strip()]


def _nseg(s: str) -> str:
    return NZ._seg(NZ._basic(_strip(s)))


def _first_num(a: str) -> str:
    m = re.search(r"\d+", a)
    return m.group(0) if m else ""


def _addr_ops(a1: str, ax: str, h1: str, hx: str, country: str) -> int:
    if a1 == ax:
        return BIT["A_EXACT"] | (BIT["A_HN_SAME"] if h1 else 0)
    axs = ax.strip()
    if not axs:
        return BIT["A_EMPTY"]
    if axs.lower() in NA:
        return BIT["A_PLACEHOLDER"]
    b = 0
    sx = _segs(ax)
    s1 = _segs(a1)
    if any(s.upper() in ("NULL", "<NULL>") for s in sx):
        b |= BIT["A_NULLSEG"]
    if any(s.upper() == "N/A" for s in sx):
        b |= BIT["A_NASEG"]
    if ax.upper() == ax and a1.upper() != a1 and re.search("[A-Z]", ax):
        b |= BIT["A_UPPER"]
    # house number
    if h1 and hx:
        if h1 == hx:
            b |= BIT["A_HN_SAME"]
        else:
            nums_x = re.findall(r"\d+", ax)
            r1 = _first_num(a1)
            if r1 and len(r1) > 1 and _first_num(ax) == r1[1:]:
                b |= BIT["A_HN_LEADDROP"]
            elif h1 in [n.lstrip("0") for n in nums_x[1:]]:
                b |= BIT["A_HN_PREFIXNUM"]
            else:
                d = int(hx) - int(h1)
                b |= BIT["A_HN_SIB"] if d in SIB3 else BIT["A_HN_SMALL"] if abs(d) <= 2 else BIT["A_HN_OTHER"]
    elif h1:
        b |= BIT["A_HN_DROP"]
    elif hx:
        b |= BIT["A_HN_ADD"]
    if not h1 or h1 == hx:
        n1s = re.findall(r"\d+", a1)
        nxs = re.findall(r"\d+", ax)
        a_, b_ = set(n1s) - set(nxs), set(nxs) - set(n1s)
        if a_ and b_ and not any(y == x[1:] or y.lstrip("0") == x for x in a_ for y in b_):
            b |= BIT["A_NUM_SUB"]
    if HN_LABEL.search(ax) and not HN_LABEL.search(a1):
        b |= BIT["A_HN_LABEL"]
    if re.search(r"\d+-[A-Za-z]\b", ax) and not re.search(r"\d+-[A-Za-z]\b", a1):
        b |= BIT["A_HN_SUFFIX"]
    # segments (placeholders removed)
    sx = [s for s in sx if s.lower() not in NA]
    if any(_script(s) for s in sx):
        b |= BIT["A_STATE_SCRIPT"]
    raw1 = {t for s in s1 for t in _toks(s)}
    rawx = {t for s in sx for t in _toks(s)}
    ab = set(NZ.STREET)
    if (raw1 ^ rawx) & (ab | set(NZ.STREET.values())) and \
            {NZ.STREET.get(t, t) for t in raw1 & (ab | set(NZ.STREET.values()))} & \
            {NZ.STREET.get(t, t) for t in rawx & (ab | set(NZ.STREET.values()))}:
        b |= BIT["A_STREET_ABBR"]
    if (rawx - raw1) & NZ.FILLER:
        b |= BIT["A_FILLER"]
    st = NZ.STATES.get(country.lower(), {})
    n1 = [_nseg(s) for s in s1]
    nx = [_nseg(s) for s in sx]
    st1 = [st.get(NZ._basic(_strip(s))) for s in s1]
    stx = [st.get(NZ._basic(NZ.anyascii(s).lower())) for s in sx]
    if any(st1) and any(stx):
        a = [s for s, k in zip(s1, st1) if k]
        c = [s for s, k in zip(sx, stx) if k]
        if a and c and a[-1].lower() != c[-1].lower() and not _script(c[-1]):
            b |= BIT["A_STATE_FORM"]
    order, used, typo = [], set(), False
    for i, s in enumerate(nx):
        if stx[i]:
            k = [j for j, v in enumerate(st1) if v == stx[i]]
            if k:
                order.append(k[0])
                used.add(k[0])
            continue
        if s in n1:
            j = n1.index(s)
        else:
            sc = [(Levenshtein.normalized_similarity(s, u), j) for j, u in enumerate(n1)]
            r, j = max(sc) if sc else (0, -1)
            if r < 0.75:
                if s and not any(s in u or u in s for u in n1 if u):
                    b |= BIT["A_SEG_ADD"]
                continue
            typo = True
        order.append(j)
        used.add(j)
    if typo:
        b |= BIT["A_SEG_TYPO"]
    if any(j not in used for j in range(len(n1)) if n1[j]) and used:
        b |= BIT["A_SEG_DROP"]
    if order != sorted(order):
        b |= BIT["A_SEG_REORDER"]
    return b


def tag(n1: str, a1: str, nx: str, ax: str, h1: str, hx: str, country: str) -> int:
    """h1 / hx: normalize.house_number of a1 / ax (pass them precomputed, it is vectorised)."""
    return _name_ops(n1 or "", nx or "") | _addr_ops(a1 or "", ax or "", h1 or "", hx or "", country)


def tag_frame(n1, a1, nx, ax, country) -> np.ndarray:
    """Vectorised driver: arrays of raw strings -> uint64 bitmask per pair."""
    import pandas as pd
    h1 = NZ.house_number(pd.Series(a1, dtype="str")).to_numpy()
    hx = NZ.house_number(pd.Series(ax, dtype="str")).to_numpy()
    return np.fromiter((tag(*r) for r in zip(n1, a1, nx, ax, h1, hx, country)), np.uint64, len(n1))


def names(mask: int) -> list:
    return [o for o in OPS if mask & BIT[o]]


if __name__ == "__main__":
    def t(n1, a1, nx, ax, c="US"):
        return set(names(int(tag_frame([n1], [a1], [nx], [ax], [c])[0])))
    assert t("Grace Israel", "329 Plain Road, Great Barrington, MA", "GRACE ISRAEL LTD", "PLAIN RD, GREAT BARRINGTON, MA") \
        >= {"N_LEGAL_ADD", "A_HN_DROP", "A_STREET_ABBR", "A_UPPER"}
    assert "A_HN_LEADDROP" in t("X", "1007 Fairway Drive, Princeton, MN", "X", "007 FAIRWAY DRIVE, PRINCETON, MN")
    assert "A_HN_SIB" in t("X", "100 Main St, A, TX", "X Group", "107 Main St, A, TX")
    assert "N_GENERIC_ADD" in t("X Co", "a", "X Co Group", "a")
    assert {"N_T_DIGIT"} <= t("Sabrisoft Integrated Fund", "a", "5abrisoft Integrated Fund", "a")
    assert "N_T_TRANSP" in t("Kritva Trade Ltd", "a", "Kritva Tarde [Ltd]", "a")
    assert "N_REORDER" in t("Sabrisoft Integrated Fund", "a", "Sabrisoft Fund Integrated", "a")
    assert "N_GIB_NAME" in t("Jaipur Auto Pvt Ltd", "a", "Onyxrizabelo", "a", "India")
    assert "N_WEB" in t("Grace Israel", "a", "GISRAEL.COM", "a")
    assert "N_HANDLE" in t("Collins, Stanley and Logan", "a", "@collinsstanley", "a")
    assert "A_SEG_REORDER" in t("X", "3765 213th Street, Fairview Park, OH", "X", "FAIRVIEW PARK, 3765 213th Street, OH")
    assert "A_STATE_FORM" in t("X", "329 Plain Rd, Great Barrington, MA", "X", "329 Plain Rd, Great Barrington, Massachusetts")
    assert "A_HN_PREFIXNUM" in t("X", "Door No. 373, Church Road, Thrissur, Kerala", "X",
                                 "NO. 0296 DOOR NO. 373, CHURCH ROAD, Kerala", "India")
    assert "A_EMPTY" in t("X", "a, b", "X", "")
    assert "N_HONOR" in t("Royal Builders", "a", "Smt Royal Builders Corp", "a", "India")
    assert "N_SHORT_EDIT" in t("A Cure 7 IT", "a", "A Cure 7 ITY", "a")
    assert "N_SYLL_APPEND" in t("Delta Foundation", "a", "DELTALUM F0UNDATION", "a")
    assert "N_SYLL_APPEND" not in t("Delta Foundation", "a", "Deltas Foundation", "a")
    assert "A_NUM_SUB" in t("X", "2648 Martinsburg Pike, Unit UNIT 4, VA", "X", "2648 Martinsburg Pike, # UNIT 48, VA")
    assert "A_NUM_SUB" not in t("X", "2648 Martinsburg Pike, Unit 4, VA", "X", "2648 Martinsburg Pike, VA")
    print("ops self-check ok")
