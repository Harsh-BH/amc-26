"""P73 v3 synthetic pairs: P67's generator + the TARGET country's learned noise-word list (user idea, 27 Sep 16:50).
P67's v2 decoys "name-swap" replace a core word with another S1's word and label it 0, and its positives only INSERT the
hand-written GENERIC words — so a CE learns "a swapped word = decoy", which is wrong for a country's copy-noise words
(France: groupe / développement / associés ...). v3 fixes both, label-free:
  positive op noise_swap (p .12): one core token -> a word from the country's swap list (P71 lens_<C>.csv: sub:w with
           ratio >= 3, n >= 20, s <= .10; drawn in proportion to n)
  decoy name-swap: the replacement word is never a noise word
The lists come from each country's own S2/S3-vs-S1 token statistics (no labels), so the LOCO arms stay fair: India and
US use their own lists, France its own. Op rates: the OTHER country's (strict 'h' recipe; France keeps France's).
    .venv/bin/python probes/P73-fr-independent/gen73n.py India|US|France N   -> data/work/p73/syn_<C>_v3n.parquet
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "probes/P67-fr-synth"))
import gen67 as G  # noqa: E402

P_NOISE = 0.12


def swap_list(c):
    L = pd.read_csv(ROOT / f"probes/P71-fr-noiseword/work/lens_{c}.csv", index_col=0)
    L = L[(L.n >= 20) & (L.ratio >= 3) & L.index.str.startswith("sub:") & (L.s <= 0.10)]
    return L.word.to_numpy(object), (L.n / L.n.sum()).to_numpy()


def gen(c, n, seed=73):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    words, wp = swap_list(c)
    noise = {w.lower() for w in words}
    d = G.s1_records(c)
    R, dept = dict(G.RATES[G.OTHER[c]]), G.state_map(c)
    vocab = pd.Series(" ".join(d.business_name.sample(min(len(d), 50_000), random_state=1)).split())
    vocab = vocab[(vocab.str.len() >= 3) & ~vocab.str.lower().isin(G.LEGAL_TOK) & ~vocab.str.lower().isin(noise)].to_numpy()
    s1_txt = G.clean(d.business_name.to_numpy(), d.business_address.to_numpy(), c)
    anc = rng.choice(len(d), size=min(int(n), len(d)), replace=False)
    pos = np.r_[anc, anc[rng.random(len(anc)) < 0.5]]
    nb = G.neighbours(s1_txt, anc, rng)
    neg_a = np.r_[anc, anc[rng.random(len(anc)) < 0.5]]
    hashn = d.business_address.str.contains(r"(?:^|,)\s*\d+", regex=True).to_numpy()
    kind = rng.choice(3, size=len(neg_a), p=G.KIND_P["v2"])
    kind[(kind == 0) & ~hashn[neg_a]] = 2
    neg_b = np.where(kind == 2, pd.Series(nb, index=anc).reindex(neg_a).to_numpy(), neg_a)
    ok = neg_b >= 0
    neg_a, neg_b, kind = neg_a[ok], neg_b[ok], kind[ok]
    src = np.r_[pos, neg_b]
    knd = np.r_[np.full(len(pos), -1), kind]
    nm = [G.gen_name(d.business_name.iat[i], c, R, rng, vocab) for i in src]
    is_core = lambda x: x.lower().strip("().,") not in G.LEGAL_TOK
    n_sw = 0
    for j in np.flatnonzero(knd == -1):                                   # positives: noise-word swap
        if len(words) and rng.random() < P_NOISE:
            t = nm[j].split()
            if sum(map(is_core, t)) >= 2 and (k := G._pick(t, rng, is_core)) is not None:
                t[k] = str(words[rng.choice(len(words), p=wp)]).capitalize()
                nm[j], n_sw = " ".join(t), n_sw + 1
    for j in np.flatnonzero(knd == 1):                                    # decoys: swap in an ordinary word only
        t = nm[j].split()
        k = G._pick(t, rng, is_core)
        if k is not None:
            t[k] = vocab[int(rng.integers(len(vocab)))]
        nm[j] = " ".join(t)
    ad = [G.gen_addr(d.business_address.iat[i], c, R, rng, dept, shift=k == 0) for i, k in zip(src, knd)]
    tb = G.clean(np.array(nm, dtype=object), np.array(ad, dtype=object), c)
    a1 = np.r_[pos, neg_a]
    y = np.r_[np.ones(len(pos), np.int8), np.zeros(len(neg_a), np.int8)]
    out = pd.DataFrame({"ta": s1_txt[a1], "tb": tb, "y": y, "p": np.float32(0.5),
                        "s1": d.entity_id.to_numpy()[a1], "sxsrc": d.entity_id.to_numpy()[src], "kind": knd.astype(np.int8)})
    out = out.sample(frac=1, random_state=seed).reset_index(drop=True)
    path = ROOT / f"data/work/p73/syn_{c}_v3n.parquet"
    out.to_parquet(path, index=False)
    print(f"syn_{c}_v3n: {len(out):,} pairs, pos {y.mean():.3f}, noise swaps {n_sw:,} (list {list(words)}); "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    gen(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 60000)
