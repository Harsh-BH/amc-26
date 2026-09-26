"""normalize.py v1 (P1-taxonomy) — name/address normalisers, pandas Series in, Series out.

    .venv/bin/python probes/P1-taxonomy/normalize.py      # self-check

Everything folds through anyascii (ISC) first, so every regex after `fold` sees ASCII only (RE2's
ASCII-only \\w/\\b are then safe). Indic names get anyascii's crude romanisation; if P2's `to_latin`
is adopted, apply it before these functions — they are idempotent on Latin text.

Word lists are MINED from train true pairs (US/India) unless tagged `# domain:` — those (France,
states that never occur in train) come from domain knowledge and are unvalidated.
"""
import re
from functools import lru_cache

import numpy as np
import pandas as pd
from anyascii import anyascii
from rapidfuzz.distance import Indel
from rapidfuzz.fuzz import ratio

# ---------------------------------------------------------------- folding
_NONASCII = r"[^\x00-\x7f]"


def fold(s: pd.Series) -> pd.Series:
    """lower-case ASCII; accents stripped, Indic crudely romanised, ° / № dropped (anyascii: '°' -> 'deg')."""
    s = s.fillna("").astype(str)
    s = s.str.replace("[°º№]", " ", regex=True)
    m = s.str.contains(_NONASCII, regex=True)
    if m.any():
        s = s.copy()
        s[m] = s[m].map(anyascii)
    return s.str.lower()


# ---------------------------------------------------------------- names
# mined: alias constructs in S2/S3 are "<fake name> [Sys|One|Co] <marker>[:] <real name>"
_ALIAS = re.compile(r"\b(?:formerly known as|formerly|f/k/a|fka|a/k/a|aka|d/b/a|dba|doing business as"
                    r"|trading as|t/a|nee)\b\s*:?\s*")      # 'nee' comes from 'née'; needs text on both sides
_ID_TAIL = re.compile(r"\(\s*id\s*:?\s*\d+\s*\)|#\d{3,}\s*$")
_JOIN = re.compile(r"[.'`]")                     # l.l.c. -> llc, tober's -> tobers
_SEP = re.compile(r"[^a-z0-9]+")                 # input is ASCII after fold
_ORD = re.compile(r"^\d+(st|nd|rd|th)$")
_DIGIT2LETTER = str.maketrans("0134568", "oleasgb")   # mined: denta1, c0m, 5ervices, 8rothers, 6roup, ...

LEGAL = {  # mined from dropped/swapped tokens on US + India true pairs
    "inc", "incorporated", "llc", "corp", "corporation", "co", "company", "ltd", "limited", "lp", "llp",
    "pc", "pllc", "pa", "plc", "lllp", "pvt", "private", "opc",
    # domain: French legal forms (unvalidated — France has no labels)
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "ei", "eirl", "scop", "scp", "selarl", "gie", "sca",
    "scs", "ets", "etablissement", "etablissements",
}
HONORIFIC = {"ms", "m/s", "sri", "shri", "smt", "mr", "mrs", "dr", "the"}   # mined prefixes (M/s -> "ms")
GENERIC = {"center", "centre", "services", "service"}   # mined: generator appends these 5-15x more than it drops them;
# stripping them: dense exact-key F0.5 0.7748 -> 0.7831 at equal precision (ledger exact-keys-D_*-dense-generic)
STOP = {"and", "of", "www", "com", "net", "org", "et", "de", "des", "du", "la", "le", "les", "l", "d"}  # fr: domain


def _tok_fix(t: str) -> str:
    if t.startswith("ln"):                        # I->l confusion: lnc, lndia, lnternational
        t = "i" + t[1:]
    if not t.isdigit() and not t.isalpha() and not _ORD.match(t):
        t = t.translate(_DIGIT2LETTER)
    return t


def _merge_letters(toks: list) -> list:
    """Join runs of single letters: 'l l c' -> 'llc', 'h q' -> 'hq' (from L.L.C / H/Q)."""
    out, run = [], []
    for t in toks + [""]:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.append("".join(run))
            run = []
        if t:
            out.append(t)
    return out


def _name_norm(x: str) -> str:
    x = _ID_TAIL.sub(" ", x)
    x = x.split("|")[0]
    m = None
    for m in _ALIAS.finditer(x):
        pass
    if m and x[m.end():].strip() and x[:m.start()].strip():
        x = x[m.end():]
    x = _SEP.sub(" ", _JOIN.sub("", x.replace("&", " and ")))
    toks = [_tok_fix(t) for t in _merge_letters(x.split())]
    return " ".join(t for i, t in enumerate(toks) if i == 0 or t != toks[i - 1])   # 'Shana Shana' duplication


def name_norm(s: pd.Series) -> pd.Series:
    """Folded, alias-resolved ('X fka Y' -> Y), '| site' and '(ID: n)' stripped, punctuation-free tokens."""
    return fold(s).map(_name_norm)


_LONG_LEGAL = ("private", "limited", "incorporated", "corporation", "company")


def _is_legal(t: str, nxt: str) -> bool:
    """Legal form, or a typo of one (mined: 'pirvate', 'privae', 'prcivtae Limited', 'Limted')."""
    if t in LEGAL:
        return True
    if len(t) < 5 or t[0] not in "plic" or t.isdigit():     # every _LONG_LEGAL word starts p/l/i/c
        return False
    if max(ratio(t, w) for w in _LONG_LEGAL) >= 85:
        return True
    return nxt in ("limited", "ltd") and t[0] == "p" and max(ratio(t, "private"),
                                                              ratio("".join(sorted(t)), "aeiprtv")) >= 75


def _core(x: str, country: str) -> str:
    toks = x.split()
    keep = [t for i, t in enumerate(toks) if t not in HONORIFIC and t not in STOP and t not in GENERIC and t != country
            and not _is_legal(t, toks[i + 1] if i + 1 < len(toks) else "")]
    return " ".join(keep) if keep else x


def name_core(s: pd.Series, country: pd.Series | None = None) -> pd.Series:
    """name_norm minus legal forms (and their typos), honorifics, stopwords, generator filler words and the record's
    own country word ('(India)'). Falls back to name_norm when nothing is left."""
    n = name_norm(s)
    c = fold(country) if country is not None else pd.Series("", index=s.index)
    return pd.Series([_core(a, b) for a, b in zip(n, c)], index=s.index, dtype="str")


def acronym(core: pd.Series) -> pd.Series:
    """First letters of a name_core with >= 2 tokens, else ''. Pass name_core output."""
    return core.map(lambda x: "".join(t[0] for t in x.split()) if len(x.split()) >= 2 else "")


# ---------------------------------------------------------------- legal forms, canonical (P14-legal)
# "Private Limited" = "Pvt. Ltd." = "Pirvate Limted" -> {pvt, ltd}: the tokens name_core strips (_is_legal), every
# spelling of a form mapped to one code, so two records can agree or conflict on their legal form.
_CANON = {"private": "pvt", "pvt": "pvt", "prvt": "pvt", "limited": "ltd", "ltd": "ltd", "incorporated": "inc",
          "inc": "inc", "corporation": "corp", "corp": "corp", "company": "co", "co": "co",
          "etablissement": "ets", "etablissements": "ets"}
_LONG = {"private": "pvt", "limited": "ltd", "incorporated": "inc", "corporation": "corp", "company": "co"}
LEGAL_CODES = sorted(set(_CANON.values()) | {_CANON.get(t, t) for t in LEGAL})
_BIT = {c: 1 << i for i, c in enumerate(LEGAL_CODES)}
assert len(LEGAL_CODES) < 63


def _legal_code(t: str, nxt: str) -> str:
    """Canonical code of token t (nxt = the token after it), '' if t is no legal form. _is_legal's cheap tests
    first, so only long p/l/i/c words reach the cached typo match."""
    if t in LEGAL:
        return _CANON.get(t, t)
    if t == "p":                                           # "P. Ltd."
        return "pvt" if nxt in ("ltd", "limited") else ""
    if len(t) < 5 or t[0] not in "plic" or t.isdigit():
        return ""
    return _legal_typo(t, nxt in ("ltd", "limited"))


@lru_cache(maxsize=1 << 20)
def _legal_typo(t: str, before_ltd: bool) -> str:
    if not _is_legal(t, "ltd" if before_ltd else ""):
        return ""
    return _LONG[max(_LONG, key=lambda w: Indel.normalized_similarity(t, w))]      # its closest long form


def legal_form(names) -> tuple:
    """Cleaned names (block.texts' `name`) -> (int64 bitmask over LEGAL_CODES, each name with its legal tokens
    replaced by their codes). ~320k names/s."""
    ms, cs = np.zeros(len(names), np.int64), [""] * len(names)
    for k, s in enumerate(names):
        toks = [_tok_fix(t) for t in _merge_letters((s or "").split())]
        m, out = 0, []
        for i, t in enumerate(toks):
            c = _legal_code(t, toks[i + 1] if i + 1 < len(toks) else "")
            m |= _BIT[c] if c else 0
            out.append(c or t)
        ms[k], cs[k] = m, " ".join(out)
    return ms, cs


_WEB = r"^\s*(?:https?://)?(?:www\.)?[a-z0-9][a-z0-9_.-]*\.(?:com|net|org|in|co|biz|info|io|us|fr|co\.in)\s*$" \
       r"|^\s*[@#][a-z0-9_.]+\s*$|^\s*www\.\S+\s*$|^[a-z0-9]{4,}com$"


def _web_clean(s: pd.Series) -> pd.Series:
    """Fold, drop '(ID: n)' tails, leading junk ('...', '--', '***') and a leading honorific ('Shri x.com')."""
    f = fold(s).str.replace(_ID_TAIL.pattern, " ", regex=True).str.strip()
    return f.str.replace(r"^[^a-z0-9@#]+|^(?:the|m/s|ms|sri|shri|smt|mr|mrs|dr)\s+", "", regex=True).str.strip()


def is_website(s: pd.Series) -> pd.Series:
    """The whole name is a URL / @handle / #hashtag (not a 'Name | www.x.com' join)."""
    return _web_clean(s).str.contains(_WEB, regex=True)


def website_core(s: pd.Series) -> pd.Series:
    """Squashed site label: 'www.Glandon-Div.com' -> 'glandondiv'; right side of 'Name | site'; '' if none."""
    f = _web_clean(s)
    pipe = f.str.contains("|", regex=False)
    site = f.where(~pipe, f.str.split("|").str[-1].str.strip())
    site = site.where(pipe | f.str.contains(_WEB, regex=True), "")
    site = site.str.replace(r"^(?:https?://)?(?:www\.)?|[@#]", "", regex=True)
    site = site.str.replace(r"\.(?:co\.in|com|net|org|in|co|biz|info|io|us|fr)$|com$", "", regex=True)
    return site.str.replace(r"[^a-z0-9]", "", regex=True)


# ---------------------------------------------------------------- addresses
US_STATES = {  # code -> full name (US S1 uses codes, S2/S3 often full names)
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california", "co": "colorado",
    "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas", "ky": "kentucky", "la": "louisiana",
    "me": "maine", "md": "maryland", "ma": "massachusetts", "mi": "michigan", "mn": "minnesota",
    "ms": "mississippi", "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee", "tx": "texas",
    "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia", "pr": "puerto rico",
}
IN_STATES = {  # mined codes (vehicle-registration style) for the 16 states in train; rest # domain:
    "mh": ["maharashtra", "महाराष्ट्र"], "dl": ["delhi", "दिल्ली"], "up": ["uttar pradesh", "उत्तर प्रदेश"],
    "ka": ["karnataka", "ಕರ್ನಾಟಕ"], "tn": ["tamil nadu", "தமிழ்நாடு"], "wb": ["west bengal", "পশ্চিমবঙ্গ"],
    "gj": ["gujarat", "ગુજરાત"], "hr": ["haryana", "हरियाणा"],
    "rj": ["rajasthan", "राजस्थान"], "kl": ["kerala", "കേരളം", "keralam"], "br": ["bihar", "बिहार"],
    "mp": ["madhya pradesh", "मध्य प्रदेश"], "pb": ["punjab", "ਪੰਜਾਬ"],
    # mined: S1 'Telangana' <-> S2/S3 'Andhra Pradesh' 370x (pre-2014 split) -> one canonical code
    "ap": ["andhra pradesh", "ఆంధ్రప్రదేశ్", "telangana", "తెలంగాణ", "tg", "ts"],
    "od": ["orissa", "odisha", "ଓଡ଼ିଶା", "or"],
    # domain: states/UTs never seen in train
    "ga": ["goa"], "as": ["assam"], "jh": ["jharkhand"], "cg": ["chhattisgarh", "chattisgarh", "ct"],
    "uk": ["uttarakhand", "uttaranchal", "ut"], "hp": ["himachal pradesh"], "jk": ["jammu and kashmir"],
    "ch": ["chandigarh"], "py": ["puducherry", "pondicherry"], "tr": ["tripura"], "ml": ["meghalaya"],
    "mn": ["manipur"], "nl": ["nagaland"], "ar": ["arunachal pradesh"], "mz": ["mizoram"], "sk": ["sikkim"],
    "an": ["andaman and nicobar islands"], "ld": ["lakshadweep"], "la": ["ladakh"],
    "dn": ["dadra and nagar haveli", "daman and diu"],
}
FR_REGIONS = {  # domain: 13 metropolitan régions -> départements (S1 France uses the région, S2/S3 mix in the département)
    "hauts de france": ["nord", "pas de calais", "somme", "oise", "aisne"],
    "pays de la loire": ["loire atlantique", "maine et loire", "mayenne", "sarthe", "vendee"],
    "nouvelle aquitaine": ["gironde", "landes", "pyrenees atlantiques", "lot et garonne", "dordogne",
                           "charente", "charente maritime", "deux sevres", "vienne", "haute vienne",
                           "creuse", "correze"],
    "ile de france": ["paris", "seine et marne", "yvelines", "essonne", "hauts de seine", "seine saint denis",
                      "val de marne", "val d oise"],
    "auvergne rhone alpes": ["ain", "allier", "ardeche", "cantal", "drome", "isere", "loire", "haute loire",
                             "puy de dome", "rhone", "savoie", "haute savoie"],
    "provence alpes cote d azur": ["alpes de haute provence", "hautes alpes", "alpes maritimes",
                                   "bouches du rhone", "var", "vaucluse", "paca"],
    "occitanie": ["ariege", "aude", "aveyron", "gard", "haute garonne", "gers", "herault", "lot", "lozere",
                  "hautes pyrenees", "pyrenees orientales", "tarn", "tarn et garonne"],
    "grand est": ["ardennes", "aube", "marne", "haute marne", "meurthe et moselle", "meuse", "moselle",
                  "bas rhin", "haut rhin", "vosges"],
    "bretagne": ["cotes d armor", "finistere", "ille et vilaine", "morbihan"],
    "normandie": ["calvados", "eure", "manche", "orne", "seine maritime"],
    "bourgogne franche comte": ["cote d or", "doubs", "jura", "nievre", "haute saone", "saone et loire",
                                "yonne", "territoire de belfort"],
    "centre val de loire": ["cher", "eure et loir", "indre", "indre et loire", "loir et cher", "loiret"],
    "corse": ["corse du sud", "haute corse"],
}


def _basic(x: str) -> str:
    """Segment-level cleaning shared by the state table and the parser (ASCII input)."""
    return " ".join(_SEP.sub(" ", x.replace("'", "").replace("&", " and ")).split())


def _state_table() -> dict:
    """country -> {normalised segment text -> canonical state}."""
    fr = {}
    for reg, deps in FR_REGIONS.items():
        fr[reg] = reg
        fr.update({d: reg for d in deps})
    us = {**{k: k for k in US_STATES}, **{v: k for k, v in US_STATES.items()}}
    ind = {k: k for k in IN_STATES}
    for k, vs in IN_STATES.items():
        ind.update({anyascii(v).lower(): k for v in vs})
    return {c: {_basic(k): v for k, v in t.items()} for c, t in {"us": us, "india": ind, "france": fr}.items()}


STATES = _state_table()
STREET = {  # mined (road<->rd, drive<->dr ...); canonical = the short form, so 'saint' and 'street' both -> st
    "street": "st", "saint": "st", "str": "st", "road": "rd", "drive": "dr", "avenue": "ave", "av": "ave",
    "lane": "ln", "court": "ct", "circle": "cir", "place": "pl", "boulevard": "blvd", "trail": "trl",
    "terrace": "ter", "highway": "hwy", "parkway": "pkwy", "cove": "cv", "fort": "ft", "mount": "mt",
    "ridge": "rdg", "point": "pt", "square": "sq", "turnpike": "tpke", "alley": "aly", "suite": "ste",
    "apartment": "apt", "floor": "fl", "building": "bldg", "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw", "centre": "center",
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6", "seventh": "7",
    "eighth": "8", "ninth": "9", "tenth": "10", "eleventh": "11", "twelfth": "12", "thirteenth": "13",
    "fourteenth": "14", "fifteenth": "15", "twentieth": "20",   # ordinal digits lose their suffix in _seg
    "crossing": "xing", "atl": "atlanta",
    "bombay": "mumbai", "calcutta": "kolkata", "madras": "chennai", "poona": "pune", "bengaluru": "bangalore",
    "gurugram": "gurgaon", "ahmadabad": "ahmedabad",   # mined alt city names (India)
    # domain: French street types / saints
    "rue": "r", "bd": "blvd", "impasse": "imp", "allee": "all", "chemin": "chem",
    "route": "rte", "faubourg": "fg", "quai": "qu", "cours": "crs", "sainte": "ste",
}
_STREET_TYPES = {"st", "rd", "dr", "ave", "ln", "ct", "cir", "pl", "blvd", "trl", "ter", "hwy", "pkwy", "cv", "aly",
                 "sq", "tpke", "way", "marg", "r", "imp", "chem", "rte", "qu", "crs"}   # a segment ending so is a street
FILLER = {"city", "cdp", "township", "twp", "town", "county", "region", "divreportingcircle", "null", "urban", "rural",
          "no", "door", "hno", "hn", "h", "unit"}   # mined: tokens the generator adds around city / house no.
_PLACEHOLDER = re.compile(r"<\s*null\s*>|\bn/a\b|\bnull\b")
_POBOX = re.compile(r"\b(?:p\s*o\s*box|pmb)\s*#?\s*\d+\b")
_LANDMARK = re.compile(r"^(?:near|nr|opp|opposite|behind|beside|next to|adjacent to|in front of|infront of)\b\s*\S"
                       r"|^c o\b")    # landmark phrase with words after it ('Opp, AL' is a town), or C/O person


def _seg(x: str) -> str:
    """_basic segment -> street types / ordinals / alt city names canonical, generator filler dropped."""
    toks = [STREET.get(t, t) for t in x.split() if t not in FILLER]
    toks = [t[:-2] if _ORD.match(t) else t for t in toks]              # mined '4th' <-> '4rd' / '5nd' noise
    return " ".join(toks[1:] if toks[:1] == ["of"] else toks)          # 'Town Of Rome' -> 'rome'


def _segments(addr: pd.Series) -> pd.Series:
    """Folded address -> list of _basic segments (placeholders, PO boxes, empty parts removed)."""
    f = fold(addr).str.replace(_PLACEHOLDER.pattern, " ", regex=True).str.replace(_POBOX.pattern, " ", regex=True)
    return f.str.split(",").map(lambda ss: [t for t in map(_basic, ss) if t])


def _ckey(country: pd.Series | None, n: int) -> list:
    return list(fold(country)) if country is not None else [""] * n


_DIGIT = re.compile(r"\d")


def address_fields(addr: pd.Series, country: pd.Series | None = None, freq: dict | None = None) -> pd.DataFrame:
    """One parse -> addr_norm, state, places, city_key, postcode, house_number (prefer this over the wrappers).

    addr_norm: unique cleaned segments, sorted (the generator shuffles segments, S1 too), state -> code.
    places:    digit-free, non-state, non-street, non-landmark segments (city/district/locality candidates).
    city_key:  with `freq` (segment -> count, from `place_freq`) the most frequent place (cities recur more than
               localities, shuffle-proof); without it, the last place segment."""
    norm, st_out, places = [], [], []
    for ss, c in zip(_segments(addr), _ckey(country, len(addr))):
        table = STATES.get(c, {})
        cand = [i for i, x in enumerate(ss) if x in table]      # state looked up BEFORE filler/abbreviation
        if len(cand) > 1:                                      # 'Washington, DC' / 'Delhi, ..., Delhi'
            codes = [i for i in cand if table[ss[i]] == ss[i]]
            cand = codes or cand
        pick = cand[-1] if cand else -1
        st = table[ss[pick]] if cand else ""
        rest = [y for i, x in enumerate(ss) if i != pick and not (x in table and table[x] == st)
                for y in [_seg(x)] if y]
        st_out.append(st)
        norm.append(", ".join(sorted(set(rest) | ({st} if st else set()))))
        places.append([x for x in rest if not _DIGIT.search(x) and not _LANDMARK.match(x)
                       and x.rsplit(" ", 1)[-1] not in _STREET_TYPES])
    if freq is None:
        ck = [r[-1] if r else "" for r in places]
    else:
        ck = [max(r, key=lambda x: (freq.get(x, 0), x)) if r else "" for r in places]
    return pd.DataFrame({"addr_norm": norm, "state": st_out, "places": places, "city_key": ck,
                         "postcode": postcode(addr, country).to_numpy(),
                         "house_number": house_number(addr, country).to_numpy()}, index=addr.index)


def addr_norm(addr: pd.Series, country: pd.Series | None = None) -> pd.Series:
    return address_fields(addr, country).addr_norm.astype("str")


def state(addr: pd.Series, country: pd.Series | None = None) -> pd.Series:
    """Canonical state/région code from the segment that is exactly a state name/code; '' if none."""
    return address_fields(addr, country).state.astype("str")


def place_segments(addr: pd.Series, country: pd.Series | None = None) -> pd.Series:
    return address_fields(addr, country).places


def city_key(addr: pd.Series, country: pd.Series | None = None, freq: dict | None = None) -> pd.Series:
    return address_fields(addr, country, freq).city_key.astype("str")


def place_freq(addr: pd.Series, country: pd.Series | None = None) -> dict:
    """segment -> count over `addr` (fit it on S1 addresses; see Risks in RESULT.md)."""
    return place_segments(addr, country).explode().dropna().value_counts().to_dict()


_POST = {"india": r"(?:^|[\s,])([1-9]\d{2}\s?\d{3})(?:$|[\s,])",   # 6-digit PIN, optional space
         "france": r"(?:^|[\s,])((?:0[1-9]|[1-8]\d|9[0-5])\d{3})\s+[a-z]",  # domain: '33000 Bordeaux'
         "us": r"\b[a-z]{2}\s+(\d{5})(?:-\d{4})?(?:$|[\s,])"}               # ZIP only when after a state code


def postcode(addr: pd.Series, country: pd.Series | None = None) -> pd.Series:
    """Postcode by country pattern ('' if none). Near-absent in train: 5-digit US numbers are house numbers."""
    f = fold(addr).str.replace(_POBOX.pattern, " ", regex=True)
    c = pd.Series(_ckey(country, len(f)), index=f.index)
    out = pd.Series("", index=f.index, dtype="str")
    for k, pat in _POST.items():
        m = c == k
        if m.any():
            out[m] = f[m].str.extract(pat, expand=False).fillna("").str.replace(" ", "", regex=False)
    return out


_HN_LABEL = re.compile(r"^(?:#|no\b|door\s*no\b|h\s*no\b|hn\b|house\s*no\b|plot\s*no\b|plot\b|flat\s*no\b"
                       r"|flat\b|shop\s*no\b|shop\b|unit\s*no\b|n)?[\s.:#-]*")


_UNIT = re.compile(r"(?:unit|apt|apartment|ste|suite|fl|floor)\b")


def house_number(addr: pd.Series, country: pd.Series | None = None) -> pd.Series:
    """Leading digit run of the first segment that starts (after a No/H.no/# label) with a number; '' if none.
    'N12815 Morgan Lake Dr' -> '12815', '##2823 Corbett' -> '2823', '1810-A Abbott' -> '1810'."""
    f = fold(addr).str.replace(_PLACEHOLDER.pattern, " ", regex=True).str.replace(_POBOX.pattern, " ", regex=True)

    def one(x: str) -> str:
        for seg in x.split(","):
            if _UNIT.match(seg.strip()):
                continue
            seg = _HN_LABEL.sub("", seg.strip().lstrip("#").strip())
            m = re.match(r"[a-z]?[-/]?(\d+)", seg)
            if m:
                return m.group(1).lstrip("0") or "0"
        return ""
    return f.map(one).astype("str")


# ---------------------------------------------------------------- self-check
def _check():
    S = lambda *v: pd.Series(list(v), dtype="str")
    got = name_norm(S("Hogue, Marquez & Ramirez (Incorporated)", "Deltahalo Co f/k/a Hogue, Marquez & Ramirez Inc",
                      "Tech Fimanie Private (Limited) | www.techfiman.com", "Golden Energy (ID: 85685)",
                      "Karrah Érwin, P.Á.", "H/Q Chir0n", "lnc Tober'S Marketing", "Metro Holding-Company"))
    assert list(got) == ["hogue marquez and ramirez incorporated", "hogue marquez and ramirez inc",
                         "tech fimanie private limited", "golden energy", "karrah erwin pa", "hq chiron",
                         "inc tobers marketing", "metro holding company"], list(got)
    core = name_core(S("M/s Lifewood Traders Pvt. Ltd.", "PRIVATE SHIVAM INVESTMENTS LIMITED",
                       "Seabird (India) Projects Corp", "The Lawlor Healthcare LLC", "Inc"),
                     S("India", "India", "India", "US", "US"))
    assert list(core) == ["lifewood traders", "shivam investments", "seabird projects", "lawlor healthcare",
                          "inc"], list(core)
    assert list(name_core(S("Shana Shana Mendoza", "Gainwell Network (India) Pirvate Limited",
                            "Blue Trading Parcvaet Limited", "Pioneer Products Limited", "Vionexecto nee Yevette Deli",
                            "Oncology LLC [Services]", "Services Inc"), S("US", "India", "India", "India", "US", "US", "US"))) == \
        ["shana mendoza", "gainwell network", "blue trading", "pioneer products", "yevette deli", "oncology", "services inc"]
    assert list(acronym(S("bureau parks recreation", "quex"))) == ["bpr", ""]
    web = S("GLANDONDIVIDEND.COM", "@irwincarrillo", "#balajiconsultants", "jaifoundationcom",
            "Ping Vacation LLP | www.pingvaca.com", "Twisted Deli", "Merchant Folhms [LLP]",
            "dermatologyclinic.com (ID: 55365)", "Shri Anjutrust.Com", "@ Cure 5 U, LLC")
    assert list(is_website(web)) == [True, True, True, True, False, False, False, True, True, False]
    assert list(website_core(web)) == ["glandondividend", "irwincarrillo", "balajiconsultants", "jaifoundation",
                                       "pingvaca", "", "", "dermatologyclinic", "anjutrust", ""], \
        list(website_core(web))
    a = S("1822 Perkins Road, Unit # 533, Stillwater, OK", "Stillwater, Unit # 533, Perkins Rd, Oklahoma",
          "14270 Main Road Circle, <NULL>, Mount Ppleasant CITY, Utah", "N12815 Morgan Lake Dr, PO BOX 7087, WI",
          "C/O Om Prakash, 12/B Rajab Ali Ln, Kolkata, Howrah, পশ্চিমবঙ্গ", "Shop No.5-6, Kharghar, महाराष्ट्र",
          "17794 Substation Road, Opp, AL", "Dannebrog, 119 1st Street, NE", "Kansas City, 4331 Wabash Ave, MO",
          "603 53nd St, Washington, District of Columbia", "9061 Little Osprey Dr, Calabash CITY, North Carolina",
          "Block D-741, Bhubaneswar, Khordha, ଓଡ଼ିଶା", "A-29. Lake View Park, KOLKATA, West Bengal")
    c = S("US", "US", "US", "US", "India", "India", "US", "US", "US", "US", "US", "India", "India")
    assert list(state(a, c)) == ["ok", "ok", "ut", "wi", "wb", "mh", "al", "ne", "mo", "dc", "nc", "od", "wb"], \
        list(state(a, c))
    assert list(house_number(a, c))[:8] == ["1822", "", "14270", "12815", "12", "5", "17794", "119"], \
        list(house_number(a, c))
    assert list(city_key(a, c)) == ["stillwater", "stillwater", "mt ppleasant", "", "howrah", "kharghar",
                                    "opp", "dannebrog", "kansas", "washington", "calabash", "khordha",
                                    "kolkata"], list(city_key(a, c))   # 'Opp, AL' is a town, kept
    an = addr_norm(a, c)
    assert an[0] == "1822 perkins rd, 533, ok, stillwater", an[0]
    assert an[7] == "119 1 st, dannebrog, ne", an[7]
    assert list(postcode(S("Jayanagar Bangalore 560082", "Raleigh, NC 27601", "12345 Oak St, Austin, TX",
                           "3 Rue X, 33000 Bordeaux"), S("India", "US", "US", "France"))) == \
        ["560082", "27601", "", "33000"]
    m, cn = legal_form(["shivam investments pvt ltd", "shivam investments private limited", "shivam pirvate limted",
                        "shivam investments llp", "twisted deli l l c", "lifewood p ltd", "sunrise 1td", "delhi public school"])
    assert [" ".join(c for c in LEGAL_CODES if v & _BIT[c]) for v in m] == \
        ["ltd pvt"] * 3 + ["llp", "llc", "ltd pvt", "ltd", ""], m
    assert cn[0] == cn[1] == "shivam investments pvt ltd" and cn[2] == "shivam pvt ltd", cn
    print("normalize self-check ok")


if __name__ == "__main__":
    _check()
