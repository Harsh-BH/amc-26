"""P1 analyses on the sample (sample.py -> featurize.py first). Prints aggregate tables only, logs ledger rows.

    .venv/bin/python probes/P1-taxonomy/analyze.py lift|taxonomy|lookalike|address
"""
import sys, time, json, resource
sys.path[:0] = ["code/business_entity_resolution/src", "probes/P1-taxonomy", "."]
import numpy as np, pandas as pd
from rapidfuzz.distance import Levenshtein
from metric import score
from ledger import log
import normalize as N

PUNCT = r"[\s,.;:'\"()\[\]{}\-&/|#!?*+_@%$~`^<>=]+"          # s0_exact's normaliser = the "before" baseline
INDIC = "[ऀ-෿]"
pd.set_option("display.width", 250)


def s0(s):
    return s.str.lower().str.replace(PUNCT, " ", regex=True).str.strip()


def peak():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)


def load():
    f1 = pd.read_parquet("/tmp/p1/feat_s1.parquet").set_index("id")
    fx = pd.read_parquet("/tmp/p1/feat_sx.parquet").set_index("id")
    return f1, fx, pd.read_parquet("/tmp/p1/samp_true.parquet"), pd.read_parquet("/tmp/p1/samp_rand.parquet")


def join(f1, fx, pr):
    a = f1.loc[pr.s1.to_numpy()].reset_index().add_prefix("a_")
    b = fx.loc[pr.sx.to_numpy()].reset_index().add_prefix("b_")
    p = pd.concat([a, b], axis=1)
    p["src"] = p.b_id.str[:2]
    return p


def jacc(x, y):
    X, Y = set(x.split()), set(y.split())
    return len(X & Y) / len(X | Y) if X | Y else 0.0


def jv(x, y):
    return np.array([jacc(u, v) for u, v in zip(x, y)])


def both_eq(x, y):
    """(share where both sides non-empty, P(equal | both non-empty))"""
    m = (x != "") & (y != "")
    return m.mean(), (x[m] == y[m]).mean() if m.any() else float("nan")


def place_overlap(x, y):
    return np.array([bool(set(u) & set(v)) for u, v in zip(x, y)])


def place_fuzzy(x, y, cut=85):
    from rapidfuzz.fuzz import ratio
    return np.array([any(ratio(u, v) >= cut for u in U for v in V) for U, V in zip(x, y)])


def raw_seg_overlap(x, y):   # the train_report 'shared comma-segment' signal, before normalisation
    f = lambda s: {t.strip().lower() for t in s.split(",") if t.strip()}
    return np.array([bool(f(u) & f(v)) for u, v in zip(x, y)])


def signals(p, freq=None):
    ra, rb = s0(p.a_raw_name), s0(p.b_raw_name)
    out = {"name_exact_s0": (ra == rb).mean(), "name_exact_norm": (p.a_name_norm == p.b_name_norm).mean(),
           "name_exact_core": (p.a_name_core == p.b_name_core).mean()}
    for tag, (x, y) in {"s0": (ra, rb), "core": (p.a_name_core, p.b_name_core)}.items():
        j = jv(x, y)
        out[f"name_jacc>=.5_{tag}"], out[f"name_anytok_{tag}"] = (j >= 0.5).mean(), (j > 0).mean()
    a0 = s0(p.a_raw_addr.str.split(",").str[0]); b0 = s0(p.b_raw_addr.str.split(",").str[0])
    out["addr_seg0_eq_s0"] = (a0 == b0).mean()                   # what s0_exact keys on
    out["addr_rawseg_overlap"] = raw_seg_overlap(p.a_raw_addr, p.b_raw_addr).mean()
    out["addr_norm_eq"] = (p.a_addr_norm == p.b_addr_norm).mean()
    out["place_overlap"] = place_overlap(p.a_places, p.b_places).mean()
    out["place_fuzzy85"] = place_fuzzy(p.a_places, p.b_places).mean()
    out["city_last_eq"] = ((p.a_city_key == p.b_city_key) & (p.a_city_key != "")).mean()
    if freq is not None:
        ca = [max(r, key=lambda s: (freq.get(s, 0), s)) if len(r) else "" for r in p.a_places]
        cb = [max(r, key=lambda s: (freq.get(s, 0), s)) if len(r) else "" for r in p.b_places]
        ca, cb = np.array(ca, dtype=object), np.array(cb, dtype=object)
        out["city_freq_eq"] = ((ca == cb) & (ca != "")).mean()
    for f in ("state", "house_number", "postcode"):
        out[f"{f}_both"], out[f"{f}_eq|both"] = both_eq(p[f"a_{f}"], p[f"b_{f}"])
    return out


def lift():
    f1, fx, T, R = load()
    freq = json.load(open("/tmp/p1/place_freq.json")) if len(sys.argv) > 2 else None
    T, R = join(f1, fx, T), join(f1, fx, R)
    rows = {}
    for tag, p in [("true", T), ("random", R)]:
        for c in ["all"] + sorted(p.a_country.unique()):
            rows[(tag, c)] = signals(p if c == "all" else p[p.a_country == c], freq)
    df = pd.DataFrame(rows).T
    print(df.T.round(4).to_string())
    return df


def placefreq():
    """place-segment counts over all train S1 addresses (chunked) -> /tmp/p1/place_freq.json"""
    import pyarrow.parquet as pq
    from collections import Counter
    cnt = Counter()
    for bt in pq.ParquetFile("data/work/train_source1.parquet").iter_batches(200_000, columns=["business_address", "country"]):
        d = bt.to_pandas()
        for r in N.place_segments(d.business_address, d.country):
            cnt.update(r)
    json.dump(dict(cnt), open("/tmp/p1/place_freq.json", "w"))
    print(len(cnt), "distinct place segments")


# ------------------------------------------------------------------ taxonomy
CLASSES = ["exact_norm", "legal_suffix_only", "typo_le2", "reorder", "website", "pipe_joined", "cross_script",
           "acronym", "token_subset", "partial_overlap", "no_token_addr_agrees", "different_dba"]


def classify(p):
    ac, bc = p.a_name_core.to_numpy(), p.b_name_core.to_numpy()
    srt = lambda s: " ".join(sorted(set(s.split())))
    ed = np.array([Levenshtein.distance(u, v) for u, v in zip(ac, bc)])
    ed_sorted = np.array([Levenshtein.distance(srt(u), srt(v)) for u, v in zip(ac, bc)])
    A = [set(u.split()) for u in ac]; B = [set(v.split()) for v in bc]
    shared = np.array([len(x & y) for x, y in zip(A, B)])
    subset = np.array([bool(x & y) and (x <= y or y <= x) for x, y in zip(A, B)])
    acro = ((p.b_name_core == p.a_acronym) & (p.a_acronym != "")) | ((p.a_name_core == p.b_acronym) & (p.b_acronym != ""))
    addr = (((p.a_house_number == p.b_house_number) & (p.a_house_number != "")).to_numpy()
            | place_overlap(p.a_places, p.b_places))
    conds = [p.a_name_norm.to_numpy() == p.b_name_norm.to_numpy(), ac == bc, ed <= 2, ed_sorted <= 2,
             p.b_is_website.to_numpy(), p.b_raw_name.str.contains("|", regex=False).to_numpy(),
             p.b_raw_name.str.contains(INDIC).to_numpy(), acro.to_numpy(), subset, shared > 0, addr]
    return np.select(conds, CLASSES[:-1], CLASSES[-1])


def matchers(p):
    """Naive name-only matchers (recall side): True = the matcher finds the pair."""
    ra, rb = s0(p.a_raw_name), s0(p.b_raw_name)
    return {"M0_exact_s0": (ra == rb).to_numpy(), "M1_jacc.5_s0": jv(ra, rb) >= 0.5,
            "M2_jacc.5_core": (jv(p.a_name_core, p.b_name_core) >= 0.5) | (p.a_name_core == p.b_name_core).to_numpy()}


def f05_without(truth, groups, p, drop_mask):
    pred = {s: set(t) for s, t in truth.items()}
    for a, b in zip(p.a_id[drop_mask], p.b_id[drop_mask]):
        pred[a].discard(b)
    return score(pred, truth, groups)


def taxonomy():
    t0 = time.time()
    f1, fx, T, _ = load()
    p = join(f1, fx, T)
    p["cls"] = classify(p)
    M = matchers(p)
    truth = {s: set() for s in f1.index}
    for a, b in zip(T.s1, T.sx):
        truth[a].add(b)
    groups = f1.country.to_dict()
    print("pairs", len(p), "S1", len(truth))
    cols = {"all": p.cls}
    for c in sorted(p.a_country.unique()):
        cols[c] = p.cls[p.a_country == c]
    for s_ in ("S2", "S3"):
        cols[s_] = p.cls[p.src == s_]
    for c in sorted(p.a_country.unique()):
        for s_ in ("S2", "S3"):
            cols[f"{c}/{s_}"] = p.cls[(p.a_country == c) & (p.src == s_)]
    share = pd.DataFrame({k: v.value_counts(normalize=True).reindex(CLASSES) for k, v in cols.items()})
    print("\nclass shares (first match wins)\n", (100 * share).round(2).fillna(0).to_string())
    rows = []
    for c in CLASSES:
        m = (p.cls == c).to_numpy()
        r = {"class": c, "share": m.mean()}
        for k, v in M.items():
            r[f"miss_{k}"] = 1 - v[m].mean() if m.any() else float("nan")
        r["f05_at_stake_ceiling"] = 1 - f05_without(truth, groups, p, m)["f05"]
        r["f05_at_stake_M1"] = 1 - f05_without(truth, groups, p, m & ~M["M1_jacc.5_s0"])["f05"]
        r["f05_at_stake_M2"] = 1 - f05_without(truth, groups, p, m & ~M["M2_jacc.5_core"])["f05"]
        rows.append(r)
    st = pd.DataFrame(rows).set_index("class").sort_values("f05_at_stake_M2", ascending=False)
    print("\nper class: miss rates of naive name-only matchers, recall-side F0.5 at stake\n", st.round(4).to_string())
    for k, v in M.items():
        s = f05_without(truth, groups, p, ~v)
        print(f"{k}: recall-only F0.5 if it were the whole matcher (no FPs) = {s['f05']:.4f}",
              {g: round(s[f'f05_{g}'], 4) for g in ('US', 'India')})
    # raw noise prevalence (not first-match-wins)
    b = p.b_raw_name.str.lower()
    flags = {"alias_marker": b.str.contains(N._ALIAS.pattern), "pipe": b.str.contains("|", regex=False),
             "website_or_handle": p.b_is_website, "indic_script": p.b_raw_name.str.contains(INDIC),
             "id_tail": b.str.contains(r"\(id:"), "accent": p.b_raw_name.str.contains("[À-ÿ]"),
             "digit_in_word": b.str.contains(r"[a-z][0-9]|[0-9][a-z]{2}") & ~b.str.contains(r"\d(st|nd|rd|th)\b"),
             "bracket": b.str.contains(r"[\[\(]"), "junk_prefix": b.str.contains(r"^(\.\.\.|--|\*\*\*)"),
             "empty_addr": p.b_raw_addr.str.strip() == "", "null_addr": p.b_raw_addr.str.contains("NULL|null")}
    fl = pd.DataFrame({k: v.groupby([p.a_country, p.src]).mean() for k, v in flags.items()}).T
    fl.insert(0, "all", [v.mean() for v in flags.values()])
    print("\nraw noise prevalence on the S2/S3 side of true pairs (%)\n", (100 * fl).round(2).to_string())
    # website recoverability
    w = p[p.cls == "website"]
    sq = w.a_name_core.str.replace(" ", ""); sqn = w.a_name_norm.str.replace(" ", "")
    hit = np.array([bool(c) and (c in x or c in y or x.startswith(c[:8])) for c, x, y in zip(w.b_website_core, sq, sqn)])
    print(f"\nwebsite class: website_core is a substring/8-prefix of the squashed S1 name in {hit.mean():.3f}")
    log("exp", probe="P1-taxonomy", subproblem="taxonomy", name="taxonomy-v1", world="train-sample30k",
        metrics={**{f"share_{c}": round(float(st.loc[c, 'share']), 4) for c in CLASSES},
                 **{f"stakeM2_{c}": round(float(st.loc[c, 'f05_at_stake_M2']), 4) for c in CLASSES},
                 "website_core_recoverable": round(float(hit.mean()), 4)},
        config={"pairs": len(p), "s1": len(truth), "order": CLASSES, "M2": "core jacc>=.5 or core exact"},
        hypothesis="where is the recall-side F0.5 after normalize v1", verdict="keep",
        runtime_s=round(time.time() - t0, 1), peak_gb=peak())
    p[["a_id", "b_id", "cls"]].to_parquet("/tmp/p1/classes.parquet", index=False)
    return st


# ------------------------------------------------------------------ look-alikes
def s1cores():
    """name_norm / name_core for every train and test S1 (chunked) -> /tmp/p1/s1core_{train,test}.parquet"""
    import pyarrow.parquet as pq
    for split in ("train", "test"):
        parts = []
        for bt in pq.ParquetFile(f"data/work/{split}_source1.parquet").iter_batches(
                200_000, columns=["entity_id", "business_name", "country"]):
            d = bt.to_pandas()
            n = N.name_norm(d.business_name)
            parts.append(pd.DataFrame({"id": d.entity_id, "country": d.country, "norm": n,
                                       "core": N.name_core(d.business_name, d.country)}))
        pd.concat(parts, ignore_index=True).to_parquet(f"/tmp/p1/s1core_{split}.parquet", index=False)
        print(split, sum(map(len, parts)), "peak", peak())


def lookalike():
    t0 = time.time()
    out = {}
    for split in ("train", "test"):
        d = pd.read_parquet(f"/tmp/p1/s1core_{split}.parquet")
        for key in ("norm", "core"):
            g = d.groupby(["country", key]).id.transform("size")
            for c in ["all"] + sorted(d.country.unique()):
                m = slice(None) if c == "all" else (d.country == c).to_numpy()
                gs = g[m]
                out[(split, key, c)] = {"n_s1": len(gs), "share_in_group>=2": (gs >= 2).mean(),
                                        ">=5": (gs >= 5).mean(), ">=20": (gs >= 20).mean(),
                                        "distinct_keys": d[m].groupby(key).ngroups}
    print("\nS1 sharing a normalised name with another S1 of the same country (chains)\n",
          pd.DataFrame(out).T.round(4).to_string())
    # true pairs: does a WRONG S1 (full train S1, same country) carry the S2/S3 record's core name?
    d = pd.read_parquet("/tmp/p1/s1core_train.parquet", columns=["id", "country", "core"])
    f1, fx, T, _ = load()
    p = join(f1, fx, T)
    idx = d.groupby(["country", "core"]).id.agg(list)
    wrong = [[w for w in idx.get((c, k), []) if w != a] for c, k, a in zip(p.a_country, p.b_name_core, p.a_id)]
    p["n_wrong"] = [len(w) for w in wrong]
    ex = (p.a_name_core == p.b_name_core).to_numpy()
    res = {"P(wrong same-core S1 exists)": (p.n_wrong > 0).mean(),
           "P(wrong exists | b core == a core)": (p.n_wrong[ex] > 0).mean(),
           "P(wrong exists | b core != a core)": (p.n_wrong[~ex] > 0).mean(),
           "P(b core matches only wrong S1s)": ((p.n_wrong > 0) & ~ex).mean()}
    for c in sorted(p.a_country.unique()):
        res[f"P(wrong exists)_{c}"] = (p.n_wrong[p.a_country == c] > 0).mean()
    print("\n", pd.Series(res).round(4).to_string())
    # address evidence: b vs its true S1, b vs up to 20 wrong same-core S1s
    rng = np.random.default_rng(0)
    sub = p[(p.n_wrong > 0) & ex].sample(min(20_000, int(((p.n_wrong > 0) & ex).sum())), random_state=0)
    wl = [list(rng.choice(w, min(20, len(w)), replace=False)) for w in
          (wrong[i] for i in sub.index)]
    need = set(w for ws in wl for w in ws)
    import pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa
    tb = pq.read_table("data/work/train_source1.parquet", columns=["entity_id", "business_address", "country"])
    tb = tb.filter(pc.is_in(tb["entity_id"], value_set=pa.array(list(need)))).to_pandas()
    freq = json.load(open("/tmp/p1/place_freq.json"))
    wf = N.address_fields(tb.business_address, tb.country, freq).set_index(tb.entity_id)
    ck = lambda r: max(r, key=lambda s: (freq.get(s, 0), s)) if len(r) else ""
    sub = sub.assign(a_ck=sub.a_places.map(ck), b_ck=sub.b_places.map(ck))
    def ev(bh, bst, bck, bpl, wh, wst, wck, wpl):
        return {"hn_eq": bh != "" and bh == wh, "state_eq": bst != "" and bst == wst,
                "city_eq": bck != "" and bck == wck, "place_overlap": bool(set(bpl) & set(wpl))}
    rows_t, rows_w, decide = [], [], []
    for (_, r), ws in zip(sub.iterrows(), wl):
        et = ev(r.b_house_number, r.b_state, r.b_ck, r.b_places, r.a_house_number, r.a_state, r.a_ck, r.a_places)
        rows_t.append(et)
        sc = lambda e: 4 * e["hn_eq"] + 2 * e["city_eq"] + e["place_overlap"] + e["state_eq"]
        best_wrong = 0
        for w in ws:
            x = wf.loc[w]
            e = ev(r.b_house_number, r.b_state, r.b_ck, r.b_places, x.house_number, x.state, x.city_key, x.places)
            rows_w.append(e)
            best_wrong = max(best_wrong, sc(e))
        decide.append((sc(et) > best_wrong, sc(et) == best_wrong, sc(et) < best_wrong))
    ev_t, ev_w = pd.DataFrame(rows_t).mean(), pd.DataFrame(rows_w).mean()
    dec = np.array(decide).mean(0)
    print("\naddress evidence, b vs true S1 / b vs wrong same-core S1\n",
          pd.DataFrame({"true_S1": ev_t, "wrong_lookalike": ev_w}).round(4).to_string())
    print(f"score 4*hn+2*city+place+state: true strictly best {dec[0]:.4f}, tie {dec[1]:.4f}, wrong better {dec[2]:.4f}"
          f" (n={len(sub)}, <=20 wrong each)")
    log("exp", probe="P1-taxonomy", subproblem="lookalike", name="lookalike-chains-v1", world="train-full-S1+sample",
        metrics={**{k: round(float(v), 4) for k, v in res.items()},
                 **{f"true_{k}": round(float(v), 4) for k, v in ev_t.items()},
                 **{f"wrong_{k}": round(float(v), 4) for k, v in ev_w.items()},
                 "addr_true_strictly_best": round(float(dec[0]), 4), "addr_tie": round(float(dec[1]), 4),
                 "addr_wrong_better": round(float(dec[2]), 4),
                 "train_core_chain_share": round(float(out[("train", "core", "all")]["share_in_group>=2"]), 4),
                 "test_core_chain_share": round(float(out[("test", "core", "all")]["share_in_group>=2"]), 4),
                 "test_core_chain_share_France": round(float(out[("test", "core", "France")]["share_in_group>=2"]), 4)},
        config={"wrong_pool": "all train S1 same country+core", "max_wrong": 20, "score": "4hn+2city+place+state"},
        hypothesis="exact-core name is ambiguous for chains; address separates them", verdict="keep",
        runtime_s=round(time.time() - t0, 1), peak_gb=peak())


# ------------------------------------------------------------------ address structure
def _struct(d):
    """Aggregate address-structure stats for a frame with business_address, country."""
    f = N.fold(d.business_address)
    segs = f.str.split(",").map(lambda ss: [N._basic(x) for x in ss if N._basic(x)])
    c = N.fold(d.country).to_numpy()
    pos, form = [], []
    for ss, cc in zip(segs, c):
        t = N.STATES.get(cc, {})
        idx = [i for i, x in enumerate(ss) if x in t]
        if not idx:
            pos.append("none"); form.append("none"); continue
        i = idx[-1]
        pos.append("last" if i == len(ss) - 1 else "first" if i == 0 else "middle")
        form.append("code" if len(ss[i]) <= 2 else "full")
    script = d.business_address.str.contains(INDIC).to_numpy()     # Indic in addresses = the state segment only
    form = np.where(script & (np.array(form) != "none"), "script", form)
    hn_first = segs.map(lambda ss: bool(ss) and bool(re.match(r"^(?:#|no |h ?no |door no |plot no )?[a-z]?[- ]?\d", ss[0])))
    raw = d.business_address
    lower = raw.str.lower()
    return {"n": len(d), "empty": (raw.str.strip() == "").mean(), "null_placeholder": lower.str.contains(r"<null>|\bnull\b|\bn/a\b").mean(),
            "segments_mean": segs.map(len).mean(),
            **{f"state_pos_{k}": (np.array(pos) == k).mean() for k in ("last", "first", "middle", "none")},
            **{f"state_form_{k}": (np.array(form) == k).mean() for k in ("code", "full", "script")},
            "number_in_first_seg": hn_first.mean(), "has_digit": lower.str.contains(r"\d").mean(),
            "postcode": (N.postcode(raw, d.country) != "").mean(),
            "landmark_phrase": lower.str.contains(r"\b(?:near|nr|opp|opposite|behind|beside|next to)\b[ .]*[a-z]").mean(),
            "opp_as_whole_segment": segs.map(lambda ss: "opp" in ss).mean(),
            "po_box_pmb": lower.str.contains(r"\bp\.? ?o\.? box|\bpmb\b").mean(),
            "unit_apt_ste": lower.str.contains(r"\b(?:unit|apt|ste|suite)\b").mean(),
            "city_filler(city/cdp/twp/township)": lower.str.contains(r"\b(?:cdp|twp|township)\b| city\b").mean(),
            "care_of(c/o)": lower.str.contains(r"\bc/o\b").mean()}


import re


def address():
    t0 = time.time()
    import pyarrow.parquet as pq
    rows = {}
    s1 = pd.read_parquet("/tmp/p1/samp_s1.parquet"); sx = pd.read_parquet("/tmp/p1/samp_sx.parquet")
    for c in sorted(s1.country.unique()):
        rows[("train", c, "S1")] = _struct(s1[s1.country == c])
        for src in ("S2", "S3"):
            q = sx[(sx.country == c) & sx.entity_id.str.startswith(src)]
            rows[("train", c, src)] = _struct(q)
    for i, src in ((1, "S1"), (2, "S2"), (3, "S3")):      # test: every 40th row, structure only (nothing fitted)
        parts = [bt.to_pandas().iloc[::40] for bt in pq.ParquetFile(f"data/work/test_source{i}.parquet").iter_batches(
            200_000, columns=["business_address", "country"])]
        d = pd.concat(parts, ignore_index=True)
        for c in sorted(d.country.unique()):
            rows[("test", c, src)] = _struct(d[d.country == c].reset_index(drop=True))
    df = pd.DataFrame(rows)
    print((100 * df.drop("n")).round(2).to_string()); print(df.loc["n"].astype(int).to_string())
    log("exp", probe="P1-taxonomy", subproblem="address", name="address-structure-v1", world="train-sample+test-1/40",
        metrics={f"{a}/{b}/{c}/{k}": round(float(v), 4) for (a, b, c), col in df.items() for k, v in col.items()
                 if k in ("empty", "null_placeholder", "postcode", "landmark_phrase", "state_pos_last", "state_pos_none",
                          "state_form_code", "state_form_script", "number_in_first_seg")},
        config={"sample": "train sample30k; test every 40th row"}, hypothesis="per-country address layout",
        verdict="keep", runtime_s=round(time.time() - t0, 1), peak_gb=peak())
    return df


if __name__ == "__main__":
    t0 = time.time()
    res = globals()[sys.argv[1]]()
    print("runtime_s", round(time.time() - t0, 1), "peak_gb", peak())
