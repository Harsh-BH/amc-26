"""Mine substitution dictionaries, legal-suffix inventory and city aliases from train true pairs.

    .venv/bin/python probes/P1-taxonomy/mine.py        # -> probes/P1-taxonomy/mined.json (token-level aggregates only)

Pairs: the train world's true pairs minus the analysis sample's S1s (sample.py), so city aliases are
evaluated on pairs they were not mined from.
"""
import sys, json, time, resource
from collections import Counter
sys.path[:0] = ["code/business_entity_resolution/src", "probes/P1-taxonomy", "."]
import numpy as np, pandas as pd
from ledger import log
import normalize as N

INDIC = "[ऀ-෿]"
FREQ = json.load(open("/tmp/p1/place_freq.json"))
ck = lambda r: max(r, key=lambda s: (FREQ.get(s, 0), s)) if len(r) else ""


def pairs(n_s1=60_000):
    W = "data/work/worlds/train/"
    samp = set(pd.read_parquet("/tmp/p1/samp_s1.parquet", columns=["entity_id"]).entity_id)
    links = pd.read_parquet(W + "truth.parquet")
    keep = pd.Series(links.s1.unique())
    keep = keep[~keep.isin(samp)].sample(n_s1, random_state=0)
    links = links[links.s1.isin(set(keep))]
    s1 = pd.read_parquet(W + "s1.parquet").set_index("entity_id")
    sx = pd.read_parquet(W + "sx.parquet").set_index("entity_id")
    a = s1.loc[links.s1.to_numpy()].reset_index(drop=True).add_prefix("a_")
    b = sx.loc[links.sx.to_numpy()].reset_index(drop=True).add_prefix("b_")
    return pd.concat([a, b], axis=1)


def diff_counts(A, B):
    """token add/drop/one-for-one substitution counts between aligned token strings."""
    add, drop, sub, tot = Counter(), Counter(), Counter(), Counter()
    for a, b in zip(A, B):
        sa, sb = set(a.split()), set(b.split())
        tot.update(sa)
        ra, rb = sa - sb, sb - sa
        add.update(rb); drop.update(ra)
        ra = [t for t in ra if not t.isdigit()]; rb = [t for t in rb if not t.isdigit()]
        if len(ra) == 1 and len(rb) == 1:
            sub[(ra[0], rb[0])] += 1
    return add, drop, sub, tot


def main():
    t0 = time.time()
    p = pairs()
    out, metrics = {}, {}
    latin = ~p.b_business_name.str.contains(INDIC)
    p["a_n"] = N.name_norm(p.a_business_name); p["b_n"] = N.name_norm(p.b_business_name)
    for c in sorted(p.a_country.unique()):
        q = p[(p.a_country == c) & latin]
        add, drop, sub, tot = diff_counts(q.a_n, q.b_n)
        out[f"name_subs_{c}"] = [[a, b, n] for (a, b), n in sub.most_common(150)]
        # generator-added words: added far more often than dropped
        out[f"name_noise_added_{c}"] = [[t, n, drop[t]] for t, n in add.most_common(400)
                                        if n >= 100 and n > 3 * drop[t]][:40]
        # legal-suffix inventory: last token of S1 names, with how often the S2/S3 side loses it
        last = q.a_n.str.split().str[-1].value_counts()
        rows = []
        for t, n in last.head(60).items():
            if n < 200:
                continue
            rows.append([t, int(n), round(n / len(q), 4), round(drop[t] / max(tot[t], 1), 3)])
        out[f"s1_last_token_{c}"] = rows
        qa = p[p.a_country == c]
        fa = N.address_fields(qa.a_business_address, qa.a_country, FREQ)
        fb = N.address_fields(qa.b_business_address, qa.b_country, FREQ)
        A = [" ".join(r) for r in fa.addr_norm.str.split(", ")]; B = [" ".join(r) for r in fb.addr_norm.str.split(", ")]
        add, drop, sub, tot = diff_counts(A, B)
        out[f"addr_subs_residual_{c}"] = [[a, b, n] for (a, b), n in sub.most_common(60)]   # what v1 still misses
        # city aliases: S1 city vs S2/S3 place segments, when they share no place segment
        cnt, base = Counter(), Counter()
        for pa, pb in zip(fa.places, fb.places):
            ca = ck(pa)
            if not ca or not len(pb):
                continue
            base[ca] += 1
            if not set(pa) & set(pb):
                cnt.update((ca, x) for x in pb)
        alias = {f"{a}|{b}": n for (a, b), n in cnt.items() if n >= 5 and n / base[a] >= 0.02}
        out[f"city_alias_{c}"] = alias
        metrics[f"n_city_alias_{c}"] = len(alias)
        del fa, fb
    json.dump(out, open("probes/P1-taxonomy/mined.json", "w"), indent=0)
    # evaluate city aliases on the analysis sample (true vs random), which they were not mined from
    f1 = pd.read_parquet("/tmp/p1/feat_s1.parquet").set_index("id")
    fx = pd.read_parquet("/tmp/p1/feat_sx.parquet").set_index("id")
    alias = set()
    for c in ("US", "India"):
        alias |= {tuple(k.split("|")) for k in out[f"city_alias_{c}"]}
    for tag, f in (("true", "/tmp/p1/samp_true.parquet"), ("random", "/tmp/p1/samp_rand.parquet")):
        pr = pd.read_parquet(f)
        a, b = f1.loc[pr.s1.to_numpy()], fx.loc[pr.sx.to_numpy()]
        ov = np.array([bool(set(x) & set(y)) for x, y in zip(a.places, b.places)])
        al = np.array([any((ck(x), z) in alias for z in y) for x, y in zip(a.places, b.places)])
        metrics[f"{tag}_place_overlap"] = round(float(ov.mean()), 4)
        metrics[f"{tag}_overlap_or_alias"] = round(float((ov | al).mean()), 4)
    print(json.dumps(metrics, indent=1))
    for c in ("US", "India"):
        print(c, "top name subs", out[f"name_subs_{c}"][:25])
        print(c, "noise-added words", [t for t, _, _ in out[f"name_noise_added_{c}"]])
        print(c, "S1 last tokens [tok, n, share, P(drop)]", out[f"s1_last_token_{c}"][:25])
        print(c, "residual addr subs", out[f"addr_subs_residual_{c}"][:30])
    pk = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
    log("exp", probe="P1-taxonomy", subproblem="dictionaries", name="mine-subs-city-alias-v1", world="train-60kS1",
        metrics=metrics, config={"alias_rule": "count>=5 and >=2% of S1 city", "pairs_from": "train world minus sample"},
        hypothesis="a mined city-alias table recovers alternate municipality names", verdict="inconclusive",
        runtime_s=round(time.time() - t0, 1), peak_gb=pk)
    print("runtime_s", round(time.time() - t0, 1), "peak_gb", pk)


if __name__ == "__main__":
    main()
