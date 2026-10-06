"""P67 side arm: wordfreq token commonness (allowed library) -> distinctive-name-token overlap features per frame pair,
for P55's combiner. Commonness of a token = max(Zipf en, Zipf fr) (language-, not country-keyed); rarity = 8 - that.
Names: casefold, split on non-word characters (Python re, Unicode-aware), legal-form tokens and 1-char tokens dropped.
  wf_jac        rarity-weighted Jaccard of the two name token sets
  wf_shared     rarity of the rarest shared token (0 if none)
  wf_sx_miss    rarity of the rarest SX token absent from S1 (0 if none)
  wf_s1_miss    rarity of the rarest S1 token absent from SX (0 if none)
    .venv/bin/python probes/P67-fr-synth/wf67.py feats      # -> data/work/p67/wf_<frame>.parquet (P65's frame rows)
"""
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P67-fr-synth")]
from gen67 import LEGAL_TOK  # noqa: E402

W = ROOT / "data/work/p67"
_SPLIT = re.compile(r"[\W_]+")


@lru_cache(maxsize=1)
def _freq() -> dict:
    """token -> max(frequency en, fr), wordfreq's own tables loaded once (zipf_frequency re-tokenises every call)."""
    from wordfreq import get_frequency_dict
    en, fr = get_frequency_dict("en"), get_frequency_dict("fr")
    return {**fr, **{k: max(v, fr.get(k, 0.0)) for k, v in en.items()}}


def rarity(t: str) -> float:
    f = _freq().get(t, 0.0)
    return 8.0 - max(0.0, np.log10(f * 1e9)) if f > 0 else 8.0


def toks(name: str) -> set:
    return {t for t in _SPLIT.split(name.casefold()) if len(t) > 1 and t not in LEGAL_TOK}


def pair(n1: str, nx: str):
    a, b = toks(n1), toks(nx)
    sh, un = a & b, a | b
    ru = sum(map(rarity, un))
    return (sum(map(rarity, sh)) / ru if ru else np.nan, max(map(rarity, sh), default=0.0),
            max(map(rarity, b - a), default=0.0), max(map(rarity, a - b), default=0.0))


def name_of(s: pd.Series) -> np.ndarray:
    """'name | addr' -> name, cut at the LAST ' | ' (P65's split)."""
    p = pd.Series(s.to_numpy(), dtype="str").str.rpartition(" | ")
    return p[0].where(p[1] != "", p[2]).to_numpy(dtype=object)


def feats():
    for fr in ("US_India", "India_US", "France"):
        t0 = time.time()
        f = pd.read_parquet(ROOT / f"data/work/p65/sm/frame_{fr}.parquet", columns=["row", "sx_txt", "s1_txt"])
        assert (f.row.to_numpy() == np.arange(len(f))).all()
        n1, nx = name_of(f.s1_txt), name_of(f.sx_txt)
        del f
        r = np.array([pair(a, b) for a, b in zip(n1, nx)], dtype=np.float32)
        d = pd.DataFrame(r, columns=["wf_jac", "wf_shared", "wf_sx_miss", "wf_s1_miss"])
        d.to_parquet(W / f"wf_{fr}.parquet", index=False)
        print(fr, len(d), d.describe().loc[["mean", "50%"]].round(3).to_dict(), f"{time.time() - t0:.0f}s", flush=True)


def _selfcheck():
    j, sh, sxm, s1m = pair("Boulangerie Lafayette SARL", "boulangerie lafayette")
    assert abs(j - 1) < 1e-9 and sxm == 0 and s1m == 0, (j, sxm, s1m)
    j, sh, sxm, s1m = pair("Boulangerie Lafayette", "Boulangerie du Centre")
    assert 0 < j < 1 and sh == rarity("boulangerie") and s1m == rarity("lafayette")
    print("selfcheck ok")


if __name__ == "__main__":
    {"feats": feats, "self": _selfcheck}[sys.argv[1]]()
