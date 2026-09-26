"""Rebuild ONE test country (France) end to end under a variant, in a scratch ER_WORK, with #6's models (m2_big +
pruner tau 0.03): texts -> rules.feats -> P3 blocking -> X -> pruner + M2 p. src/ is untouched: the variant is a data
or vectorizer swap before the unchanged pipeline runs.

  --idf S   France-aware IDF: the train IDF (1M train S2/S3) refit with a seeded sample of the country's own S2/S3
            text added at share S of the pool (0.144 = France's share of test S2/S3). US / India never see it.
  --fill    state filled from the city: an S2/S3 with no parsed state, one of whose places is a city with a
            >= 95%-pure state among the same split's S1 of its country (>= 10 S1), gets ", <state as S1 writes it>"
            appended to its raw address -- so blocking, rules and features all see it as a stated record.
  --fr6     P6's France layer (probes/P6-france/france.py) as the rules.feats of France rows.

    .venv/bin/python probes/P14-france-shift/variant.py NAME [--idf 0.144] [--fill] [--fr6] [--keep-x]
    -> data/work/p14/prob_test_France_NAME.parquet (s1, sx, p) + data/work/p14/stats_NAME.json + a ledger row
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/work"
C = "France"
SHARED = ["test_source1.parquet", "train_source1.parquet", "place_freq_train.json", "p2_xlit_table.parquet",
          "gt_pairs.parquet", "s1_folds.parquet", "p4/m2_big.txt", "p10/pruner.txt", "rules_s1feat_train.parquet"]


def link(v: Path, rel: str):
    d = v / rel
    d.parent.mkdir(parents=True, exist_ok=True)
    if not d.exists() and (BASE / rel).exists():
        d.symlink_to(BASE / rel)


def setup(v: Path, a):
    for rel in SHARED:
        link(v, rel)
    if not a.fill:
        for s in (2, 3):
            link(v, f"test_source{s}.parquet")
        for f in (BASE / "p3/txt_test").glob(f"{C}_*.parquet"):
            link(v, f"p3/txt_test/{f.name}")
    if not a.fr6:
        link(v, f"rf_test_s1_{C}.parquet")
        if not a.fill:
            link(v, f"rf_test_sx_{C}.parquet")
    if not (a.idf or a.idf_dom):
        link(v, "p3/vecs_hash.pkl")
        if not a.fill:                          # same texts + same IDF -> #6's blocking, bit for bit
            link(v, f"p3/cands_test_{C}.parquet")


def city_states(r1: pd.DataFrame, min_n: int = 10, purity: float = 0.95) -> pd.DataFrame:
    """place -> (n, purity, state) over S1 rules.feats rows (state, places) that carry a state; kept if pure."""
    e = r1[r1.state != ""].assign(place=r1.places.str.split("|")).explode("place")
    e = e[e.place.notna() & (e.place != "")]
    g = e.groupby("place").state.agg(["size", lambda s: s.value_counts(normalize=True).iloc[0],
                                      lambda s: s.value_counts().index[0]])
    g.columns = ["n", "purity", "state"]
    return g[(g.n >= min_n) & (g.purity >= purity)]


def fill_state(rx: pd.DataFrame, city: pd.DataFrame) -> pd.Series:
    """Filled state per SX row ('' = not filled): only rows without a state, from their best-supported mapped place."""
    n = city.n.to_dict()
    def pick(pl):
        c = [p for p in pl.split("|") if p in n] if pl else []
        return city.state[max(c, key=n.get)] if c else ""
    out = pd.Series("", index=rx.index, dtype=object)
    m = rx.state == ""
    out[m] = rx.places[m].map(pick)
    return out


def state_display(s1_addr: pd.Series, country: str) -> dict:
    """state -> the raw segment S1 most often writes it as ('Hauts-de-France')."""
    import normalize as N
    segs = s1_addr.str.split(",").explode().str.strip().reset_index(drop=True)
    st = N.address_fields(segs, pd.Series(country, index=segs.index)).state
    return segs[st != ""].groupby(st[st != ""]).agg(lambda s: s.value_counts().index[0]).to_dict()


def append_state(d: pd.DataFrame, add: dict) -> pd.DataFrame:
    """Raw rows with ', <segment>' appended to the address of every entity_id in add."""
    m = d.entity_id.map(add)
    h = m.notna()
    d = d.copy()
    d.loc[h, "business_address"] = (d.business_address[h].str.rstrip(", ") + ", " + m[h]).str.lstrip(", ")
    return d


def fill_sources(v: Path):
    """Raw France S2/S3 with ', <state>' appended where the state is missing and a place names a pure city."""
    r1 = pd.read_parquet(BASE / f"rf_test_s1_{C}.parquet", columns=["state", "places"])
    s1 = pd.read_parquet(BASE / "test_source1.parquet", columns=["business_address", "country"])
    disp = state_display(s1.business_address[s1.country == C], C)
    city = city_states(r1)
    rx = pd.read_parquet(BASE / f"rf_test_sx_{C}.parquet", columns=["id", "state", "places"])
    fill = fill_state(rx, city)
    add = dict(zip(rx.id[fill != ""], fill[fill != ""].map(disp)))
    stats = {"sx_no_state": float((rx.state == "").mean()), "sx_filled": float((fill != "").mean()),
             "cities_mapped": len(city), "display": disp}
    for s in (2, 3):
        d = pd.read_parquet(BASE / f"test_source{s}.parquet")
        append_state(d[d.country == C].reset_index(drop=True), add).to_parquet(v / f"test_source{s}.parquet", index=False)
    print("fill:", stats, flush=True)
    return stats


def idf_vecs(v: Path, share: float, seed: int = 0):
    """P3's hashed TF-IDF with the country's own S2/S3 text added to the IDF pool at `share`."""
    import pickle
    import pyarrow.parquet as pq
    import block as B
    vec = pickle.loads((BASE / "p3/vecs_hash.pkl").read_bytes())
    files = sorted((v / "p3/txt_test").glob(f"{C}_sx_*.parquet"))
    t = pd.concat([pq.read_table(f, columns=["name", "addr"]).to_pandas() for f in files], ignore_index=True)
    out = {}
    for view in ("name", "addr"):
        hv, tf = vec[view].steps[0][1], vec[view].steps[1][1]
        idf = tf.idf_.astype(np.float64)
        n = round(np.exp(idf.max() - 1) - 1)                       # smooth idf: max = ln(1 + n) + 1 (df = 0)
        df = np.rint((1 + n) / np.exp(idf - 1) - 1)
        m = round(n * share / (1 - share))
        X = hv.transform(t[view].sample(n=min(m, len(t)), random_state=seed)).tocsc()
        dfc = np.diff(X.indptr)                                    # docs containing each hashed feature
        tf.idf_ = (np.log((1 + n + X.shape[0]) / (1 + df + dfc)) + 1).astype(tf.idf_.dtype)
        out[view] = {"n_train": n, "m_country": X.shape[0]}
    (v / "p3").mkdir(parents=True, exist_ok=True)
    (v / "p3/vecs_hash.pkl").write_bytes(pickle.dumps(vec))
    print("idf:", out, flush=True)
    return out


def dom_vecs(v: Path):
    """Train-only fallback of idf_vecs: the address IDF of French domain words already listed in normalize.py
    (street types, articles, régions + départements, house-number words) lowered to the median train IDF of the US
    street types. City names stay at max IDF: no list of them exists without external data."""
    import pickle
    import normalize as N
    vec = pickle.loads((BASE / "p3/vecs_hash.pkl").read_bytes())
    hv, tf = vec["addr"].steps[0][1], vec["addr"].steps[1][1]
    fr = {k: w for k, w in N.STREET.items() if k in {"rue", "bd", "impasse", "allee", "chemin", "route", "faubourg",
                                                     "quai", "cours", "sainte", "avenue", "av", "boulevard", "place",
                                                     "saint"}}
    words = set(fr) | set(fr.values()) | {"de", "des", "du", "la", "le", "les", "l", "d", "et", "en", "sur", "sous",
                                          "aux", "au", "ndeg", "no", "bis", "ter", "france"}
    for reg, deps in N.FR_REGIONS.items():
        words |= {w for x in [reg, *deps] for w in x.split()}
    ix = lambda ws: hv.transform([" ".join(ws)]).indices
    val = float(np.median(tf.idf_[ix(["st", "rd", "dr", "ave", "ln", "ct", "blvd"])]))
    idf = tf.idf_.copy()
    j = ix(sorted(words))
    idf[j] = np.minimum(idf[j], val)
    tf.idf_ = idf
    (v / "p3").mkdir(parents=True, exist_ok=True)
    (v / "p3/vecs_hash.pkl").write_bytes(pickle.dumps(vec))
    out = {"dom_words": len(words), "idf_value": round(val, 3)}
    print("idf-dom:", out, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--idf", type=float, default=0.0)
    ap.add_argument("--idf-dom", action="store_true", help="train-only fallback of --idf (dom_vecs)")
    ap.add_argument("--fill", action="store_true")
    ap.add_argument("--fr6", action="store_true")
    ap.add_argument("--keep-x", action="store_true")
    ap.add_argument("--from", dest="src", help="reuse w_SRC's raw S2/S3, texts (and rules.feats unless --fr6)")
    a = ap.parse_args()
    t0 = time.time()
    v = BASE / "p14" / f"w_{a.name}"
    v.mkdir(parents=True, exist_ok=True)
    setup(v, a)
    if a.src:
        w = BASE / "p14" / f"w_{a.src}"
        for f in [*w.glob("test_source[23].parquet"), *(w / "p3/txt_test").glob("*"),
                  *([] if a.fr6 else w.glob(f"rf_test_sx_{C}.parquet"))]:
            d = v / f.relative_to(w)
            d.parent.mkdir(parents=True, exist_ok=True)
            if not d.exists():
                d.symlink_to(f.resolve())
    os.environ["ER_WORK"] = str(v)
    sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT), str(ROOT / "probes/P6-france")]
    import block as B
    import features as F
    import model as M
    import rules
    import run as RUN
    B.CFG["threads"] = int(os.environ.get("P14_THREADS", 8))  # ponytail: 15 threads thrash on a shared, loaded box
    info = {}
    if a.fill and not (v / "test_source2.parquet").exists():
        info["fill"] = fill_sources(v)
    if a.fr6:
        import france
        rules.feats = lambda df, chunk=None, procs=None: france.feats(df)     # in-process (no forkserver workers)
    RUN.prepare("test", C)                                     # texts (unless linked) + rules.feats caches
    if a.idf and not (v / "p3/vecs_hash.pkl").exists():
        info["idf"] = idf_vecs(v, a.idf)
    if a.idf_dom and not (v / "p3/vecs_hash.pkl").exists():
        info["idf_dom"] = dom_vecs(v)
    d = M.split_probs("test", C)                               # P3 blocking -> X -> pruner + M2
    id1, idx = F.ids("test", C)
    q1, qx, p = d.q1.to_numpy(), d.qx.to_numpy(), d.p.to_numpy(np.float32)
    out = pd.DataFrame({"s1": id1.take(q1).to_numpy(zero_copy_only=False),
                        "sx": idx.take(qx).to_numpy(zero_copy_only=False), "p": p})
    out.to_parquet(BASE / "p14" / f"prob_test_{C}_{a.name}.parquet", index=False)
    mg = []
    for f in F.xfiles("test", C):
        X = pd.read_parquet(f, columns=["q1", "qx", "sx_margin", "addr_cos", "name_cos", "block_score"])
        mg.append(X)
    X = pd.concat(mg, ignore_index=True).merge(d, on=["q1", "qx"])
    am = M.argmax_sx(X.qx.to_numpy(), X.p.to_numpy())
    info["acc95_means"] = X[am & (X.p.to_numpy() >= 0.95)][["sx_margin", "addr_cos", "name_cos", "block_score"]] \
        .mean().round(4).to_dict()
    info["pairs_blocked_all"] = int(sum(len(x) for x in mg))
    del X, mg
    if not a.keep_x:
        for f in F.xfiles("test", C):
            f.unlink()
    info["runtime_s"] = round(time.time() - t0)
    info["peak_gb"] = round(B.peak_gb(), 2)
    (BASE / "p14" / f"stats_{a.name}.json").write_text(json.dumps(info, default=str, indent=1))
    print(json.dumps(info, default=str), flush=True)


if __name__ == "__main__":
    main()
