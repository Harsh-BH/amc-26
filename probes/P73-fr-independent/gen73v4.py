"""P73 v4 synthetic pairs (user, 27 Sep 19:25: 'more synthetic data from our data analysis'). v3 (gen73n: P67 op catalogue +
the target country's learned noise-word swap list) plus the generator patterns our analysis found that v3 lacks:
  decoy  sib     house number moved by the generator's SIBLING offsets {3,4,5,7,9,11,13,21} (P12/P15; France also +1/+2),
                 replacing P67's uniform 3-40 shift
  decoy  legal   same name with a DIFFERENT legal form of the country (SARL<->SAS ...; P64: conflicts 36.8% of France's rejected
                 confident pairs vs <0.4% of true pairs), address = the S1's (half with a sibling shift)
  decoy  twin    a copy of ANOTHER S1 with the same name core, address emptied (P40's address-less twin floor class)
  pos    trunc   house number truncated (4712 -> 471, or lead digit dropped) on true copies (P72: copies truncate, decoys step)
Everything is label-free / from the provided records; the LOCO arms use each country's own records and noise list.
    .venv/bin/python probes/P73-fr-independent/gen73v4.py India|US|France N   -> data/work/p73/syn_<C>_v4.parquet
"""
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P67-fr-synth"), str(Path(__file__).resolve().parent)]
import gen67 as G  # noqa: E402
from gen73n import swap_list, P_NOISE  # noqa: E402

SIB = [3, 4, 5, 7, 9, 11, 13, 21]
KIND_P4 = [0.12, 0.13, 0.55, 0.10, 0.10]          # sib, name-swap, neighbour, legal, twin
P_TRUNC = 0.03
_LEG = re.compile(r"\b(" + "|".join(sorted({t for t in G.LEGAL_TOK}, key=len, reverse=True)) + r")\b\.?", re.I)


def core(name):
    return " ".join(_LEG.sub(" ", str(name).lower()).replace(",", " ").split())


def sib_addr(a, c, rng):
    segs = [s.strip() for s in a.split(",") if s.strip()]
    si = next((k for k, s in enumerate(segs) if G._HN.match(s)), None)
    if si is None:
        return a
    m = G._HN.match(segs[si])
    offs = SIB + ([1, 2] if c == "France" else [])
    segs[si] = f"{int(m.group(2)) + int(rng.choice(offs))}{segs[si][m.end(2):]}"
    return ", ".join(segs)


def trunc_addr(a, rng):
    segs = [s.strip() for s in a.split(",") if s.strip()]
    si = next((k for k, s in enumerate(segs) if G._HN.match(s)), None)
    if si is None:
        return a
    m = G._HN.match(segs[si]); h = m.group(2)
    if len(h) >= 2:
        h = h[:-1] if rng.random() < 0.7 else h[1:]
        segs[si] = f"{h}{segs[si][m.end(2):]}"
    return ", ".join(segs)


def legal_conflict(name, c, rng):
    toks = name.split()
    base = [t for t in toks if t.lower().strip("().,") not in G.LEGAL_TOK]
    have = {t.lower().strip("().,") for t in toks} & {x.lower() for x in G.LEGAL[c]}
    opts = [x for x in G.LEGAL[c] if x.lower() not in have]
    return " ".join(base + [opts[int(rng.integers(len(opts)))]]) if base and opts else None


def gen(c, n, seed=74):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    words, wp = swap_list(c)
    noise = {w.lower() for w in words}
    d = G.s1_records(c)
    R, dept = dict(G.RATES[G.OTHER[c]]), G.state_map(c)
    vocab = pd.Series(" ".join(d.business_name.sample(min(len(d), 50_000), random_state=1)).split())
    vocab = vocab[(vocab.str.len() >= 3) & ~vocab.str.lower().isin(G.LEGAL_TOK) & ~vocab.str.lower().isin(noise)].to_numpy()
    s1_txt = G.clean(d.business_name.to_numpy(), d.business_address.to_numpy(), c)
    cores = d.business_name.map(core).to_numpy()
    grp = pd.Series(np.arange(len(d))).groupby(cores).agg(list)
    twin_of = {}
    for members in grp:
        if len(members) >= 2:
            for i in members:
                twin_of[i] = members
    anc = rng.choice(len(d), size=min(int(n), len(d)), replace=False)
    pos = np.r_[anc, anc[rng.random(len(anc)) < 0.5]]
    nb = pd.Series(G.neighbours(s1_txt, anc, rng), index=anc)
    neg_a = np.r_[anc, anc[rng.random(len(anc)) < 0.5]]
    hashn = d.business_address.str.contains(r"(?:^|,)\s*\d+", regex=True).to_numpy()
    kind = rng.choice(5, size=len(neg_a), p=KIND_P4)       # 0 sib, 1 name-swap, 2 neighbour, 3 legal, 4 twin
    kind[(kind == 0) & ~hashn[neg_a]] = 2
    has_twin = np.array([i in twin_of for i in neg_a])
    kind[(kind == 4) & ~has_twin] = 2
    neg_b = neg_a.copy()
    for j in np.flatnonzero(kind == 2):
        neg_b[j] = nb.get(neg_a[j], -1)
    for j in np.flatnonzero(kind == 4):
        others = [x for x in twin_of[neg_a[j]] if x != neg_a[j]]
        neg_b[j] = others[int(rng.integers(len(others)))]
    ok = neg_b >= 0
    neg_a, neg_b, kind = neg_a[ok], neg_b[ok], kind[ok]
    src = np.r_[pos, neg_b]
    knd = np.r_[np.full(len(pos), -1), kind]
    nm = [G.gen_name(d.business_name.iat[i], c, R, rng, vocab) for i in src]
    is_core = lambda x: x.lower().strip("().,") not in G.LEGAL_TOK
    cnt = {"noise_swap": 0, "trunc": 0, "legal": 0}
    for j in np.flatnonzero(knd == -1):
        if len(words) and rng.random() < P_NOISE:
            t = nm[j].split()
            if sum(map(is_core, t)) >= 2 and (k := G._pick(t, rng, is_core)) is not None:
                t[k] = str(words[rng.choice(len(words), p=wp)]).capitalize(); nm[j] = " ".join(t); cnt["noise_swap"] += 1
    for j in np.flatnonzero(knd == 1):
        t = nm[j].split()
        if (k := G._pick(t, rng, is_core)) is not None:
            t[k] = vocab[int(rng.integers(len(vocab)))]
        nm[j] = " ".join(t)
    for j in np.flatnonzero(knd == 3):
        v = legal_conflict(d.business_name.iat[src[j]], c, rng)
        if v is not None:
            nm[j] = v; cnt["legal"] += 1
    ad = []
    for i, k in zip(src, knd):
        a = d.business_address.iat[i]
        if k == 0:
            ad.append(sib_addr(a, c, rng))
        elif k == 3:
            ad.append(sib_addr(a, c, rng) if rng.random() < 0.5 else G.gen_addr(a, c, R | {"empty": 0.0, "nullseg": 0.0}, rng, dept))
        elif k == 4:
            ad.append("")
        else:
            x = G.gen_addr(a, c, R, rng, dept)
            if k == -1 and rng.random() < P_TRUNC:
                x = trunc_addr(x, rng); cnt["trunc"] += 1
            ad.append(x)
    tb = G.clean(np.array(nm, dtype=object), np.array(ad, dtype=object), c)
    a1 = np.r_[pos, neg_a]
    y = np.r_[np.ones(len(pos), np.int8), np.zeros(len(neg_a), np.int8)]
    out = pd.DataFrame({"ta": s1_txt[a1], "tb": tb, "y": y, "p": np.float32(0.5),
                        "s1": d.entity_id.to_numpy()[a1], "sxsrc": d.entity_id.to_numpy()[src], "kind": knd.astype(np.int8)})
    out = out.sample(frac=1, random_state=seed).reset_index(drop=True)
    out.to_parquet(ROOT / f"data/work/p73/syn_{c}_v4.parquet", index=False)
    kc = pd.Series(knd[len(pos):]).value_counts().sort_index().to_dict()
    print(f"syn_{c}_v4: {len(out):,} pairs, pos {y.mean():.3f}, decoy kinds {kc}, {cnt}, twins {len(twin_of):,}; "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    gen(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 60000)
