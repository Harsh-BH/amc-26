"""Score the full-corpus hits (full.py run) on the holdout S1: recall@k curves, fwd/rev/derived combos, per country
and per P1 difficulty class.

    .venv/bin/python probes/P3-block/score_full.py classes          # P1 taxonomy for holdout true pairs (cached)
    .venv/bin/python probes/P3-block/score_full.py v3 India [US]    # -> data/work/p3/score_v3_<countries>.csv
    .venv/bin/python probes/P3-block/score_full.py export v4 India  # recommended rule -> cands_holdout_full_India
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "P1-taxonomy")]
import block as B
import evalb as E
from full import HITS, OUT, TXT, WORK, peak_gb
from worlds import load_world  # noqa: E402 (classes step)

KS_F = (5, 10, 20, 50, 100)
KS_R = (1, 2, 3, 5, 10)


def p1_classes():
    """P1's 12-class taxonomy (analyze.classify) for every holdout true pair -> data/work/p3/holdout_p1cls.parquet."""
    import analyze as P1
    from featurize import featurize
    s1, sx, truth = load_world("holdout")
    tp = E.truth_df(truth)
    fa, _ = featurize(s1)
    fa["raw_name"] = s1.business_name.to_numpy()
    fb, _ = featurize(sx[sx.entity_id.isin(tp.sx)])
    fb["raw_name"] = sx[sx.entity_id.isin(tp.sx)].business_name.to_numpy()
    p = P1.join(fa.set_index("id"), fb.set_index("id"), tp)
    tp["cls"] = P1.classify(p)
    tp.to_parquet(OUT / "holdout_p1cls.parquet", index=False)
    print(tp.cls.value_counts(normalize=True).round(4).to_dict(), f"peak {peak_gb():.2f} GB")


def load(tag, countries):
    """Hits and holdout truth on int64 keys per country (s1 = pool row, sx = global SX row, both offset by country),
    so 20M+ hit rows never become Python strings. Forward hits are folded to top-100 per S1 while streaming (numpy)."""
    s1w = pd.read_parquet(WORK / "worlds/holdout/s1.parquet", columns=["entity_id", "country"])
    cls = pd.read_parquet(OUT / "holdout_p1cls.parquet")
    TP, S1, R, F, meta = [], [], [], [], []
    for ci, c in enumerate(countries):
        o1, ox = ci * 10_000_000, ci * 100_000_000
        ids1 = pd.Index(pd.read_parquet(TXT / f"{c}_s1.parquet", columns=["entity_id"]).entity_id.to_numpy())
        hold = s1w.entity_id[s1w.country == c].to_numpy()
        S1.append(pd.DataFrame({"entity_id": ids1.get_indexer(hold).astype(np.int64) + o1, "country": c}))
        sx_ids, fw, off = [], None, 0
        for p in sorted(TXT.glob(f"{c}_sx_*.parquet")):
            d = pd.read_parquet(p, columns=["entity_id", "addr"])
            e, noaddr = d.entity_id.to_numpy(), (d.addr == "").to_numpy(bool)
            sx_ids.append(e)
            hp = HITS / f"{tag}_{p.stem}.parquet"
            if hp.exists():
                h = pd.read_parquet(hp, columns=["q1", "qx", "score", "rank", "dir"])
                f = (h["dir"] == "f").to_numpy()
                q1, qx, sc, rk = (h[k].to_numpy() for k in ("q1", "qx", "score", "rank"))
                R.append(B._hits(q1[~f].astype(np.int64) + o1, qx[~f].astype(np.int64) + off + ox, sc[~f], rk[~f],
                                 "na", "r").assign(noaddr=noaddr[qx[~f]]))
                new = (q1[f], (qx[f] + off).astype(np.int32), sc[f])
                fw = new if fw is None else B._best(*map(np.concatenate, zip(fw, new)), 100)
                del h, q1, qx, sc, rk, new
                meta.append(json.loads(hp.with_suffix(".json").read_text()) | {"country": c})
            off += len(e)
        if fw is not None:
            fw = B._best(*fw, 100)
            F.append(B._hits(fw[0].astype(np.int64) + o1, fw[1].astype(np.int64) + ox, fw[2], B.ranks(fw[0]), "na", "f"))
        t = cls[cls.s1.isin(set(hold))]
        sx_ix = pd.Index(np.concatenate(sx_ids))
        TP.append(pd.DataFrame({"s1": ids1.get_indexer(t.s1).astype(np.int64) + o1,
                                "sx": sx_ix.get_indexer(t.sx).astype(np.int64) + ox, "country": c, "cls": t.cls.to_numpy()}))
        assert (TP[-1].sx >= ox).all(), "every true SX is in the corpus"
        del sx_ix, sx_ids, ids1, fw
        print(f"loaded {c}: peak {peak_gb():.2f} GB", flush=True)
    h = pd.concat(R + F, ignore_index=True).rename(columns={"q1": "s1", "qx": "sx"})
    return pd.concat(S1, ignore_index=True), pd.concat(TP, ignore_index=True), h, pd.DataFrame(meta)


def key(s1, sx):
    return np.asarray(s1, np.int64) * (1 << 28) + np.asarray(sx, np.int64)


def rank_of(tp, g):
    """Rank of each true pair in hit set g (1e9 if absent)."""
    k, o = key(g.s1, g.sx), None
    o = np.argsort(k)
    k = k[o]
    t = key(tp.s1, tp.sx)
    if len(k) == 0:
        return np.full(len(t), 1e9)
    pos = np.clip(np.searchsorted(k, t), 0, len(k) - 1)
    found = k[pos] == t
    return np.where(found, g["rank"].to_numpy()[o][pos], 1e9)


def metrics(sel, tp, s1, per_class=True):
    """Same fields as evalb.metrics, on int64 pair keys (no string merges)."""
    sk = np.unique(key(sel.s1, sel.sx))
    hit = np.isin(key(tp.s1, tp.sx), sk)
    m = {"pair_recall": hit.mean(), "cands_per_s1": len(sk) / len(s1)}
    g = pd.Series(hit).groupby(tp.s1.to_numpy())
    m["s1_full_recall"], m["s1_any_recall"] = g.all().mean(), g.any().mean()
    m["s1_no_cands"] = 1 - len(np.unique(sk >> 28)) / len(s1)
    s1c = s1.set_index("entity_id").country
    per = pd.Series(sk >> 28).map(s1c).value_counts()
    for c, gg in tp.groupby("country"):
        m[f"recall_{c}"] = hit[gg.index].mean()
        m[f"cands_{c}"] = per.get(c, 0) / (s1.country == c).sum()
    if per_class:
        for c, gg in tp.groupby("cls"):
            m[f"recall_{c}"] = hit[gg.index].mean()
    return m


def export(tag, country, kr=2, kna=10, kf=10):
    """The recommended rule on full-corpus holdout hits -> data/work/p3/cands_holdout_full_<country>.parquet
    (s1, sx ids, block_score, ranks, via, is_true): a realistic-density candidate set for matcher training / tuning."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    s1, tp, h, _ = load(tag, [country])
    r = h[h["dir"] == "r"]
    rank, na = r["rank"].to_numpy(), r.noaddr.to_numpy(bool)
    q1, qx, sc = r.s1.to_numpy(np.int64), r.sx.to_numpy(np.int64), r.score.to_numpy(np.float32)
    del h, r
    m = rank < np.where(na, kna, kr)
    p = rank < 30                                    # the per-S1 pool sees every hit (no-state SX searched to 30)
    g = B.finalize([(q1[m] << 32) | qx[m]], [sc[m]], [B._best(q1[p], qx[p], sc[p], kf)])
    ids1 = pq.read_table(TXT / f"{country}_s1.parquet", columns=["entity_id"]).column(0).combine_chunks()
    idx = pa.concat_arrays([pq.read_table(f, columns=["entity_id"]).column(0).combine_chunks()
                            for f in sorted(TXT.glob(f"{country}_sx_*.parquet"))])
    true = np.isin((g.q1.to_numpy(np.int64) << 32) | g.qx.to_numpy(np.int64),
                   (tp.s1.to_numpy(np.int64) << 32) | tp.sx.to_numpy(np.int64))
    t = pa.table({"s1": ids1.take(pa.array(g.q1.to_numpy())), "sx": idx.take(pa.array(g.qx.to_numpy())),
                  "block_score": g.block_score.to_numpy(), "block_rank_s1": g.block_rank_s1.to_numpy(),
                  "block_rank_sx": g.block_rank_sx.to_numpy(), "via": pa.array(g.via.astype(str).to_numpy()),
                  "is_true": true})
    pq.write_table(t, OUT / f"cands_holdout_full_{country}.parquet")
    print(f"{country}: {len(g):,} pairs ({len(g) / len(s1):.2f}/S1), pair recall {true.sum() / len(tp):.4f}, "
          f"true share {true.mean():.3f}, peak {peak_gb():.2f} GB")


if __name__ == "__main__":
    if sys.argv[1] == "classes":
        p1_classes()
        sys.exit()
    if sys.argv[1] == "export":
        export(sys.argv[2], sys.argv[3])
        sys.exit()
    tag, countries = sys.argv[1], sys.argv[2:] or ["India", "US"]
    s1, tp, h, meta = load(tag, countries)
    fw, rv = h[h["dir"] == "f"], h[h["dir"] == "r"]
    if len(fw) == 0:                          # v4 is reverse-only
        fw = rv.iloc[:0]
    rf, rr = rank_of(tp, fw), rank_of(tp, rv)
    print("n S1", len(s1), "true pairs", len(tp), "classes", tp.cls.value_counts(normalize=True).round(3).to_dict())
    print("fwd recall@k", {k: round(float((rf < k).mean()), 4) for k in KS_F})
    print("rev recall@k", {k: round(float((rr < k).mean()), 4) for k in KS_R})
    per_cls = {k: {c: round(float((rr[g.index] < k).mean()), 4) for c, g in tp.groupby("cls")} for k in (1, 2)}
    print("rev recall@1/2 per P1 class", per_cls)
    combos = {f"f:na{k}": lambda k=k: fw[fw["rank"] < k] for k in (10, 20, 40)}
    combos |= {f"r:na{k}": lambda k=k: rv[rv["rank"] < k] for k in KS_R}
    combos |= {f"f:na{kf}+r:na{kr}": lambda kf=kf, kr=kr: pd.concat([fw[fw["rank"] < kf], rv[rv["rank"] < kr]])
               for kf in (5, 10) for kr in (1, 2)}
    combos |= {f"derived kr{kr} kf{kf}": lambda kr=kr, kf=kf: B.derive(rv[rv["rank"] < 10], kr, kf, "s1")
               for kr in (1, 2, 3) for kf in (5, 10, 20)}
    if "noaddr" in rv:
        na = rv.noaddr.to_numpy(bool)
        combos |= {f"derived kr{kr} kf{kf} knoaddr{ke}": lambda kr=kr, kf=kf, ke=ke: B.derive(
                       rv[(rv["rank"] < 10) | na], np.where(na[(rv["rank"] < 10).to_numpy() | na], ke, kr), kf, "s1")
                   for kr, kf in ((1, 5), (1, 10), (2, 10), (2, 20), (3, 10)) for ke in (10, 20, 30)}
    if len(fw) == 0:
        combos = {k: v for k, v in combos.items() if not k.startswith("f:")}
    rows = [{"combo": n, **metrics(sel(), tp, s1)} for n, sel in combos.items()]
    d = pd.DataFrame(rows).set_index("combo")
    pd.set_option("display.width", 400)
    pd.set_option("display.max_columns", 60)
    print(d.round(4).to_string())
    d.to_csv(OUT / f"score_{tag}_{'_'.join(countries)}.csv")
    print(meta.groupby("country")[["n_sx", "secs"]].sum().to_string(), f"\npeak {peak_gb():.2f} GB")
