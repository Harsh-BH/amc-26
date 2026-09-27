"""End-to-end pipeline: raw TSVs -> matching_results.tsv + candidate_pairs.tsv. Run from the repo root.

    python code/business_entity_resolution/src/run.py --split test --stage model --out output/   # submission #37_gc70 (LB 0.986304)
        --recipe sub37_small (the final zip: same matches, 4.42 candidates per S1) | sub37_hn90 | sub34 | sub31 | … | sub4
        --rescue T / --gc T / --hn-extra T / --veto TAU / --t-country France=T: the unseen country's bars (0 = off)
    python code/business_entity_resolution/src/run.py --split holdout --stage model   # dense holdout F0.5 -> pipe ledger
    python code/business_entity_resolution/src/run.py --split dev --stage model       # the small dev world
    python code/business_entity_resolution/src/run.py --split test --stage rules      # submission #2 (rules only)

One country at a time, labels taken from the S1 file (an open set: France and unseen labels included):
  1 feats   rules.feats: P2 to_latin -> P1 name_core / website_core / address_fields, cached in data/work
  2 cands   P3 blocking (block.py, cached in data/work/p3) ∪ the pairs the rules accept (rules.match by_state)
  3 score   rules: the accept set (v0) | model: M2 LightGBM on 65 features (features.py, model.py) -> p, then for
            the countries the cross-encoder was trained on (India, US) the band 0.02 < p < 0.98 is rescored by
            rerank.py in its own process under the GPU lock and blended into p; then P17's house-number stacker (US)
            and P34's a10 residual stacker (a10.py, India / US: #34). UNSEEN = the test S1 labels absent from train
            S1 (unseen(): France) get france.py's chain (#31, FRANCE = "fr31") -- the rebuilt idf144 world, c_hn,
            the bge blend, tier / bxneg / [veto] / cap. No country is named in the code: "France" in a name below is
            the chain's name (it was built for the one unseen test label)
  4 decide  each S2/S3 only at its argmax-p S1, kept if p >= t: T (0.71) for a country with training labels;
            UNSEEN T_FRANCE_OVERRIDE (#31: 0.9868), fixed from the leaderboard ("a stricter threshold for countries
            unseen in training" is allowed, organisers 27 Sep); P6's label-free rule when it is None -- the t where
            the predicted empty-S1 rate equals the labelled countries' rate at M2's own optimum T_M2 (0.69), clipped
            to [T_M2, T_M2 + 0.2]. Then, UNSEEN only (france.complete): empty-S1 rescue (#34v, p >= 0.90) and
            group completion (#37, p >= 0.70)
  5 write   one row per S1, matches ⊆ candidates, only S2-/S3- ids of the split; then the official validator
Splits:
  test     the submission (all 1,732,544 test S1)
  holdout  DENSE: the holdout S1 (fold 0) are scored while every train S1 and every train S2/S3 of the country
           compete -- the test-shaped proxy
  holdout-world  (rules only) every train S1 competes, but only the holdout world's S2/S3 -- the old 0.8215
  dev      the dev world (fold 1, 30k S1, plain): in-memory blocking + features, the recompute path at small scale
Every artifact is rebuilt from the training data when absent: parquet copies, folds and worlds (ensure), the P2
transliteration table, place_freq, the TF-IDF vectorizers, the P3 texts + candidates (block.split_cands), the
rules.feats caches (feats), the X feature files (features.build), M2 (train_matcher.py) and the cross-encoder
(rerank.py band + train), c_hn (train_matcher.py --hn). Nothing is downloaded except the e5-small base weights
(README); the bge CE is fine-tuned on AWS and its checkpoint downloaded (SETUP.md).
"""
import argparse
import functools
import gc
import hashlib
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SRC = Path(__file__).resolve().parent
ROOT = SRC.parents[2]
sys.path[:0] = [str(SRC), str(ROOT)]
import block as B
import features as F
import model as M
import rerank as RR
import rules as R
from metric import score
from worlds import WORK

PROCS = 2           # ponytail: rules.feats workers (~0.5 GB each) sized for a 2.6 GB cap; 4+ on a bigger box
PIECE = 500_000     # raw rows per rules.feats call when building a feature cache
N_TEST_S1 = 1_732_544
CCOLS = ["s1", "sx", "block_score", "block_rank_s1", "block_rank_sx", "via"]
T = 0.71            # threshold on the CE-blended p (#7: m2_big + 2-epoch CE, refit on holdout half 0; #5/#6 0.69)
T_M2 = 0.69         # M2's own holdout optimum (m2_big 0.69): the reference point of the unseen-country rule (P6)
T_FRANCE_OVERRIDE = 0.95   # France's t (#7, LB-validated: D4 0.981142 vs 0.980017 at P6's 0.78); None = P6's rule
T_COUNTRY = {}             # more per-country t overrides (--t-country C=t); any other unseen label keeps P6's rule
# submission -> booster (+ its train folds), blend w, T, T_M2, pruner tau, CE (+ family, epochs, cached-score seed),
# stacked countries, France's scorer (m2 | idf144 | idfdom | bge | sib) and t (None = P6 rule). --recipe switches
# them all; each recipe reproduces its staged files byte for byte. The default is sub8c (use() at the module end).
_E5 = dict(ce_base="e5", r_seed={}, stack={}, france="m2", w_france=None, w_stack_fit=None)
_P4B = dict(ce="p4b/ce.pt", ce_epochs=1, t_france=None, **_E5)
_BIG = dict(booster="m2_big.txt", folds=(2, 3, 4, 5), t_m2=0.69, tau=0.03)
# #8c base: P13's bge-reranker-v2-m3 CE for India/US (w fit on holdout half 0), P17's US-only house-number stacker.
# Its test / holdout scores are seeded from P13's files (the exact numerics of the staged submissions).
# w at full precision: P13's staged test blends used eval_bge_poolck_holdout_vs_p11s2P_plain.json's fit unrounded (the
# documented [0.5901, 0.6319, 0.2973] moves every band pair by up to 2.4e-5 and flips 4 S1 lines).
_BGE = dict(_BIG, ce="p13/ckpt/ce_bge_pool.pt", ce_base="bge", ce_epochs=2, t=0.71,
            w=[0.5901490903244407, 0.6319357246516834, 0.29732167912424273],
            r_seed={"test": ["p13/r/r_bge_pool_test_{c}.parquet", "p19/r/r_bge_pool_test_{c}.parquet"],
                    "holdout": ["p13/r/r_bge_poolck_holdout.parquet"]},   # P13's scores; P19's for idf144 France
            stack={"US": "p17/stack_sup-bge-l21.0_US.txt"}, w_france=[0.9245, 0.2595, 0.0515],
            w_stack_fit=[0.5901, 0.6319, 0.2973])   # P17 fit the stacker on holdout p blended with the rounded w
RECIPES = {"sub4": dict(booster="m2_ladder.txt", folds=(2,), w=[0.636, 0.757, 0.355], t=0.70, t_m2=0.71, tau=None, **_P4B),
           "sub5": dict(_BIG, w=[0.678, 0.715, 0.338], t=0.69, tau=None, **_P4B),
           "sub6": dict(_BIG, w=[0.678, 0.715, 0.338], t=0.69, **_P4B),
           "sub7": dict(_BIG, w=[0.617, 0.666, 0.261], t=0.71, ce="p11/ce_s2P.pt", ce_epochs=2, t_france=0.95, **_E5),
           "sub8c": dict(_BGE, france="m2", t_france=0.95),       # France = #6c's (M2 p, t 0.95)
           "sub9a": dict(_BGE, france="idf144", t_france=0.975),  # + P14: France's own S2/S3 text in the IDF (14.4%)
           "sub9b": dict(_BGE, france="bge", t_france=0.98),      # + the bge CE on France, out-of-domain weights
           "sub9s": dict(_BGE, france="sib", t_france=0.95),      # + P15's house-number sibling rule on France
           "sub9d": dict(_BGE, france="idfdom", t_france=0.98),   # + P14's train-only IDF fallback (domain words)
           "sub9ab": dict(_BGE, france="idf144_bge", t_france=0.985)}  # #9a's France candidates + #9b's bge blend
# #31: France = france.py's chain (P26 c_hn on P29's rebuild, bge blend, P29 tier, P28 bxneg, P29 cap); India / US = #8c.
# Its band pairs' bge r were scored by P13 / P19 / P26 / P29 / P31 (first file wins, as P31 blended).
_R31 = {"test": _BGE["r_seed"]["test"] + [f"{p}/r_bge_pool_test_{{c}}.parquet" for p in ("p26/bge/r", "p29/r", "p31/r")],
        "holdout": _BGE["r_seed"]["holdout"]}
RECIPES["sub31"] = dict(_BGE, france="fr31", t_france=0.9868, r_seed=_R31)
RECIPES["sub31_veto"] = dict(RECIPES["sub31"], veto=(0.1, "p30/r/r_pool_France.parquet"))  # + P30's bge veto, tau 0.1
RECIPES["sub31_small"] = dict(RECIPES["sub31"], drop_sib_cands=True)   # #31's matching, 4.42 candidates per S1
# #34: India / US + P34's a10 residual stacker (fit on the holdout half 0); #34_veto = #34 + #31_veto's France.
# #37_gc70 (LB 0.986304, best) = #34_veto + the unseen country's empty-S1 rescue (p >= 0.90, #34v_r90 LB 0.98596) +
# group completion (p >= 0.70). _small: its decoy stage (candidate_pairs.tsv 4.83 -> 4.42 per S1, same matches).
RECIPES["sub34"] = dict(RECIPES["sub31"], a10="p34/a10_h0.txt")
RECIPES["sub34_veto"] = dict(RECIPES["sub34"], veto=RECIPES["sub31_veto"]["veto"])
RECIPES["sub37"] = dict(RECIPES["sub34_veto"], rescue=0.90, gc=0.70)
RECIPES["sub37_small"] = dict(RECIPES["sub37"], drop_sib_cands=True)
RECIPES["sub37_hn90"] = dict(RECIPES["sub37"], hn_extra=0.90)       # staged #37_gc70_hn90 (P37): + two-sided hn extras
# #52_max's India / US: P50's full-holdout refit of the P44 combiner + P45 r_all / zoo columns in a10's place (iu_stack)
RECIPES["sub52_iu"] = dict(RECIPES["sub37"], iu="fullzoo")
# staged #42_both (P42's joint re-tune): decision t 0.99 on the post-rules of t 0.9868, rescue 0.97, hn extras 0.80
RECIPES["sub42_both"] = dict(RECIPES["sub37"], t_france=0.99, t_post=0.9868, rescue=0.97, hn_extra=0.80)
# staged #41 / #41_hn90 (P41): + synthetic-brand recovery at p >= 0.30 (france.brand_mask)
RECIPES["sub41"] = dict(RECIPES["sub37"], brand=0.30)
RECIPES["sub41_hn90"] = dict(RECIPES["sub37_hn90"], brand=0.30)
# staged #40x / #40x_t70 (P40): #41 + address-less exact-name variants (france.noaddr_mask) at p >= 0.90 / 0.70
RECIPES["sub40x"] = dict(RECIPES["sub41"], noaddr=0.90)
RECIPES["sub40x_t70"] = dict(RECIPES["sub41"], noaddr=0.70)
# #43_r90's levers in one sequential recipe (the staged file is a union of P42 / P41 / P40 pair sets)
RECIPES["sub43_r90"] = dict(RECIPES["sub42_both"], rescue=0.90, brand=0.30, noaddr=0.70)
# P46's noise-word swap tier on top (france.swap_mask; lens s <= 0.05, 'g' variant 0.09)
RECIPES["sub46_on43"] = dict(RECIPES["sub43_r90"], swap=0.05)
RECIPES["sub46g_on43"] = dict(RECIPES["sub43_r90"], swap=0.09)
# #clean (27 Sep): the unseen country's final chain (france.chain: P42 recipe t .99 on the post-rules of .9868, rescue .90,
# gc .70, hn .80 -> P41 brand .30 on the gc70-style lists -> P48 CE veto r < .2 -> P54 in-band CE rescue r >= .99 -> P47
# hn-change class p >= .80 -> P55 combiner veto c < .6 in the band / adds c > .95), then P54's duplicate step (all
# countries). India / US: iu (P44 combiner + P45 r_all + P50 zoo refit), or #34's a10 (_iu34).
_CHAIN = dict(anchor=0.9868, rescue=0.90, gc=0.70, hn=0.80, brand=0.30, veto=0.20, u=0.99, cls=0.80,
              p55=(0.6, 0.95, ("p55/fr_plain_s0.txt", "p55/fr_plain_s1.txt", "p55/fr_plain_s2.txt")))
RECIPES["sub_clean"] = dict(RECIPES["sub37"], t_france=0.99, t_post=0.9868, rescue=None, gc=None, chain=_CHAIN,
                            r_lo="p48/r_lo_{c}.parquet", dup=True, iu="fullzoo")
RECIPES["sub_clean_iu34"] = dict(RECIPES["sub_clean"], iu="a10")
# the default final (27 Sep, master): #37_gc70's unseen-country recipe + #54's India / US (the combiner) with P54's
# duplicate step on the labelled countries only, as #54 staged those lines (the unseen country's lines are #37's).
RECIPES["sub_iu54_frgc70"] = dict(RECIPES["sub37"], iu="fullzoo", dup="seen")
# #54 (uploaded 27 Sep 05:32, LB 0.986685): the chain in the order it was staged (build38.full + fr54): recipe -> brand and
# P40 address-less (both on the gc70-style lists, unioned) -> P46 tier -> P48 veto -> P47 class -> P54 rescue; no P55
RECIPES["sub54"] = dict(RECIPES["sub_clean"], chain=dict(_CHAIN, xf=0.70, swap=0.05, p55=None,
                                                         order=("brand", "xf", "swap", "veto54", "cls", "u54")))
RECIPE = "sub37"            # LB 0.986304 (best); later bars: --rescue / --gc / --hn-extra / --veto / --t-country
CHN = WORK / "p31" / "c_hn.txt"   # P26's hn-aware matcher (AWS: probes/P26-france-da, SETUP.md), for fr31
CAP = None                  # --cap 5,6: at most 5 S2 and 6 S3 matches per S1 (the train truth never exceeds them)
DROP_SIB_CANDS = False      # --drop-sibling-cands: unaccepted decoy-offset pairs leave candidate_pairs.tsv
VETO = None                 # (tau, r file): fr31's P30 veto (recipe sub31_veto)
A10 = None                  # P34's a10 booster (data/work-relative) for the CE countries (recipe sub34 / sub37)
IU = None                   # --iu: the CE countries' stage, overriding the recipe's iu (a10 default | full | fullzoo)
RESCUE = GC = HN_EXTRA = None   # the unseen country's decision-time bars (france.complete; recipe sub37: 0.90 / 0.70)
SWAP = None                 # --swap: P46's noise-word swap tier, its lens s_max (france.swap_mask), the last stage
NOADDR = None               # --noaddr: P40's address-less exact-name bar (france.noaddr_mask), after the brand stage
CHAIN = None                # the unseen country's final chain (france.chain bars), recipe sub_clean*
R_LO = None                 # P48's CE scores of the pairs outside the band (data/work-relative, {c} = the label): P55 input
DUP = False                 # P54's duplicate step: True = every country, "seen" = the labelled ones (recipe sub_iu54_frgc70)
BRAND = None                # --brand: P41's synthetic-brand recovery bar (france.brand_mask), after the others
T_POST = None               # --t-post: fr31's post-rules at this t, the decision at its own (P42's #42_both: 0.9868 / 0.99)
FALLBACK = ("p11/ce_s2P.pt", [0.617, 0.666, 0.261])   # --ce-missing fallback: #7's e5 CE + w for pairs without an r
CE_MISSING = "score"        # band pairs without a cached r: score them (GPU) | "fallback" (holdout proxy, P17's yardstick)
                            # | "keep" (their p stays: a CPU box, or a check of how many lack an r)
GPU_LOCK = "/tmp/amc-gpu.lock"


def use(recipe: str):
    """Switch booster, blend weights, thresholds, pruner, CE, stacker and France to one submission's recipe."""
    global T, T_M2, T_FRANCE_OVERRIDE, RECIPE, FRANCE, W_FRANCE, STACK, R_SEED, W_STACK_FIT, VETO, DROP_SIB_CANDS, \
        A10, RESCUE, GC, HN_EXTRA, T_POST, BRAND, NOADDR, SWAP, CHAIN, R_LO, DUP
    r = RECIPES[recipe]
    M.BOOSTER, M.FOLDS, M.TAU, M._M, RR.CFG["w"] = WORK / "p4" / r["booster"], r["folds"], r["tau"], None, list(r["w"])
    RR.CE, RR.EPOCHS, RR.FAMILY = WORK / r["ce"], r["ce_epochs"], r["ce_base"]
    T, T_M2, T_FRANCE_OVERRIDE, RECIPE = r["t"], r["t_m2"], r["t_france"], recipe
    FRANCE, W_FRANCE, STACK, R_SEED = r["france"], r["w_france"], dict(r["stack"]), dict(r["r_seed"])
    W_STACK_FIT, VETO, DROP_SIB_CANDS = r["w_stack_fit"], r.get("veto"), r.get("drop_sib_cands", False)
    A10, RESCUE, GC, HN_EXTRA, T_POST = r.get("a10"), r.get("rescue"), r.get("gc"), r.get("hn_extra"), r.get("t_post")
    BRAND, NOADDR, SWAP = r.get("brand"), r.get("noaddr"), r.get("swap")
    CHAIN, R_LO, DUP = r.get("chain"), r.get("r_lo"), r.get("dup", False)


@functools.cache
def unseen() -> frozenset:
    """UNSEEN: the test S1 country labels absent from train S1 (France). The organisers allow a stricter rule for
    "countries unseen in training" when it is defined that way, not by naming a country: every unseen-country
    branch keys on this set."""
    import train_matcher
    test = pq.read_table(WORK / "test_source1.parquet", columns=["country"]).column(0).unique().to_pylist()
    return frozenset(test) - frozenset(train_matcher.countries())


def ensure(split: str):
    """Rebuild data/work from the raw TSVs where absent: parquet copies, gt_pairs + folds, the split's world."""
    import io_tsv
    import worlds
    if not all((WORK / f"{s}_source{i}.parquet").exists() for s in ("train", "test") for i in (1, 2, 3)):
        io_tsv.convert()
    if not ((WORK / "gt_pairs.parquet").exists() and (WORK / "s1_folds.parquet").exists()):
        worlds.build_base()            # translit.table() learns from these, the worlds sample from them
    w = split.split("-")[0]
    if w in worlds.WORLDS and not (WORK / "worlds" / w / "truth.parquet").exists():
        worlds.build_world(w, *worlds.WORLDS[w])


def arr(x) -> pa.Array:
    """ids (pandas / arrow / list) -> one arrow string array."""
    a = x.combine_chunks() if isinstance(x, pa.ChunkedArray) else pa.array(x)
    return a.cast(pa.string())


def codes(ids: pa.Array, values) -> np.ndarray:
    """Row position of each value in ids, computed in arrow memory. Every value must be present."""
    c = pc.index_in(arr(values), value_set=ids)
    assert c.null_count == 0, f"{c.null_count} ids outside this split / country"
    return c.to_numpy().astype(np.int64)


def feats(split: str, side: str, country: str, load: bool = True) -> pd.DataFrame | None:
    """rules.feats of one split / side ("s1" | "sx") / country, cached at data/work/rf_<split>_<side>_<country>.parquet
    (the files rules.run_test writes). Built PIECE raw rows at a time, so a big country never sits in RAM raw."""
    path = WORK / f"rf_{split}_{side}_{country}.parquet"
    if not path.exists():
        tmp, w, buf = path.with_suffix(".tmp"), None, []

        def flush():
            nonlocal w
            t = pa.Table.from_pandas(R.feats(pd.concat(buf, ignore_index=True), procs=PROCS), preserve_index=False)
            w = w or pq.ParquetWriter(tmp, t.schema)
            w.write_table(t.cast(w.schema))
            buf.clear()

        for d in (d for i in ([1] if side == "s1" else [2, 3]) for d in B._batches(f"{split}_source{i}", country)):
            buf.append(d)
            if sum(map(len, buf)) >= PIECE:
                flush()
        if buf or w is None:
            flush()
        w.close()
        tmp.rename(path)
    return pd.read_parquet(path) if load else None


def prepare(split: str, country: str):
    """Full-scale inputs of the model stage for one country of a split (test | train): P3 texts, then the rules.feats
    caches; the P3 candidates and the X files follow in features.build."""
    if not (B.TXT[split] / f"{country}_s1.parquet").exists():
        B.prep(split, country)
    for side in ("s1", "sx"):
        feats(split, side, country, load=False)


def p3_cands(split: str, country: str, scored=None, cols=CCOLS) -> pa.Table:
    """P3 blocking pairs of one country (s1, sx ids + block features), from data/work/p3 or rebuilt by block.py.
    holdout = the holdout S1's pairs out of the all-train-S1 run: the same rule P3 exported as cands_holdout_full_*."""
    if split == "test":
        p = B.P3 / f"cands_test_{country}.parquet"
        if not p.exists():
            B.split_cands("test", country)
    else:
        p = B.P3 / f"cands_holdout_full_{country}.parquet"
        if not p.exists():
            t = pq.read_table(B.split_cands("train", country), columns=CCOLS)
            pq.write_table(t.filter(pc.is_in(t["s1"], value_set=arr(scored))), p)
    return pq.read_table(p, columns=cols)


def holdout_s1():
    s1 = pd.read_parquet(WORK / "worlds/holdout/s1.parquet", columns=["entity_id", "country"])
    t = pd.read_parquet(WORK / "worlds/holdout/truth.parquet")
    return s1, t


def truth_of(ids, t: pd.DataFrame) -> dict:
    truth = {s: set() for s in ids}
    for a, b in zip(t.s1, t.sx):
        if a in truth:
            truth[a].add(b)
    return truth


def inputs(split: str, world=None):
    """Yield (country, f1, fx, cands, scored S1 ids, truth | None) per S1 country label. f1 = the S1 pool that
    competes (holdout: every train S1 of the country), fx = every S2/S3 of the country, cands(cols) loads the P3 pairs.
    Full-scale splits give rules.feats frames; in-memory worlds give features.rec tables (a superset)."""
    if split == "test":
        for c in sorted(pq.read_table(WORK / "test_source1.parquet", columns=["country"]).column(0).unique().to_pylist()):
            f1 = feats("test", "s1", c)
            yield c, f1, feats("test", "sx", c), lambda cols, c=c: p3_cands("test", c, cols=cols), f1.id, None
        return
    if split.startswith("holdout"):
        s1, t = holdout_s1()
        wx = arr(pq.read_table(WORK / "worlds/holdout/sx.parquet", columns=["entity_id"]).column(0)) \
            if split == "holdout-world" else None
        for c in sorted(s1.country.unique()):
            ids = s1.entity_id[s1.country == c]
            fx = feats("train", "sx", c)
            if wx is not None:
                fx = fx[pc.is_in(arr(fx.id), value_set=wx).to_numpy(zero_copy_only=False)].reset_index(drop=True)
            load = lambda cols, c=c, ids=ids: p3_cands("holdout", c, ids, cols)
            if wx is not None:
                load = lambda cols, l=load: (lambda t: t.filter(pc.is_in(t["sx"], value_set=wx)))(l(cols))
            yield c, feats("train", "s1", c), fx, load, ids, truth_of(ids, t)
        return
    if world is None:                                   # dev
        from worlds import load_world
        s1, sx, truth = load_world(split)
        cp = WORK / "p8" / f"cands_{split}.parquet"
        if not cp.exists():
            cp.parent.mkdir(parents=True, exist_ok=True)
            B.candidates(s1, sx, B.CFG, B.vecs())[CCOLS].to_parquet(cp, index=False)
        cands = pd.read_parquet(cp)
    else:                                               # injected frames (test_pipeline)
        s1, sx, truth = world
        cands = B.candidates(s1, sx, B.CFG, B.vecs(), log=lambda *_: None)
    for c in sorted(s1.country.unique()):
        a, b = s1[s1.country == c], sx[sx.country == c]
        cc = cands[cands.s1.isin(set(a.entity_id))]
        yield (c, F.rec(a), F.rec(b) if len(b) else None,
               lambda cols, cc=cc: pa.Table.from_pandas(cc[cols], preserve_index=False),
               a.entity_id, {s: truth[s] for s in a.entity_id})


def assemble(stage: str, f1: pd.DataFrame, fx: pd.DataFrame, load, keep: np.ndarray):
    """-> (accepted keys, candidate keys, pairs | None). A key is int64 q1 << 32 | qx over f1 / fx rows; everything is
    cut to the scored S1 (keep). Candidates = P3 ∪ accepted pairs. model: `pairs` carries them with their block
    features (NaN for a pair only the rules proposed) and the accept flag, ready for model.score."""
    id1, idx = arr(f1.id), arr(fx.id)
    key = lambda s1, sx: (codes(id1, s1) << 32) | codes(idx, sx)
    cands = load(["s1", "sx"] if stage == "rules" else CCOLS)   # rules: ids only, freed before the join
    kp = key(cands["s1"], cands["sx"])                  # asserts: P3 ids are S1 / S2 / S3 of this split + country
    m = keep[kp >> 32]
    if not m.all():
        cands, kp = cands.filter(pa.array(m)), kp[m]
    if stage == "rules":
        del cands
    f1r, ka = f1[F.RF], []
    for i in range(0, len(fx), 400_000):              # exact: each S2/S3 is decided on its own key matches
        acc, _ = R.match(f1r, fx.iloc[i:i + 400_000][F.RF], by_state=True, keep_cands=False)
        k = key(acc.s1, acc.sx) if len(acc) else np.zeros(0, np.int64)
        ka.append(k[keep[k >> 32]])                     # cut to the scored S1 per chunk: bounded on the full corpus
    ka = np.concatenate(ka)
    del acc, f1r
    if stage == "rules":
        return ka, np.union1d(kp, ka), None
    K = np.unique(np.concatenate([kp, ka]))
    o = np.argsort(kp)
    pos = np.minimum(np.searchsorted(kp[o], K), max(len(kp) - 1, 0))
    hit = (kp[o][pos] == K) if len(kp) else np.zeros(len(K), bool)
    src = o[pos] if len(kp) else pos
    col = lambda n, fill: np.where(hit, cands[n].to_numpy()[src], fill) if len(kp) else np.full(len(K), fill)
    via = pd.Categorical(cands["via"].to_numpy(zero_copy_only=False), categories=["r", "f", "r+f"])
    pairs = pd.DataFrame({"q1": K >> 32, "qx": K & 0xFFFFFFFF,
                          "block_score": col("block_score", np.nan).astype(np.float32),
                          "block_rank_s1": col("block_rank_s1", -1).astype(np.int32),
                          "block_rank_sx": col("block_rank_sx", -1).astype(np.int32),
                          "via": pd.Categorical.from_codes(np.where(hit, via.codes[src], -1) if len(kp) else
                                                           np.full(len(K), -1), via.categories),
                          "acc": np.isin(K, ka)})
    return ka, K, pairs


def seed_keys(f: Path, id1: pa.Array, idx: pa.Array) -> pd.DataFrame:
    """A scored (s1, sx, r) file -> k (q1 << 32 | qx over this split's rows), r; pairs of other countries dropped."""
    t = pq.read_table(f, columns=["s1", "sx", "r"])
    a = pc.index_in(t.column("s1"), value_set=id1).to_numpy(zero_copy_only=False)
    b = pc.index_in(t.column("sx"), value_set=idx).to_numpy(zero_copy_only=False)
    ok = ~(pd.isna(a) | pd.isna(b))                     # a holdout file holds both countries
    return pd.DataFrame({"k": (a[ok].astype(np.int64) << 32) | b[ok].astype(np.int64),
                         "r": t.column("r").to_numpy()[ok].astype(np.float32)})


def r_cache(split: str, c: str, id1: pa.Array, idx: pa.Array, ce: Path = None) -> Path:
    """The CE-score cache of one split / country / checkpoint (int64 key q1 << 32 | qx, r). A recipe with an r_seed
    (the bge CE: P13's scores, then P19's / P26's / P29's / P31's top-ups) merges every seed file the cache has not
    taken yet (listed in <cache>.seeds) into it; a pair already in the cache (or in an earlier seed) keeps its r."""
    ce = ce or RR.CE
    path = M.CACHE / f"r_{split}_{c}_{M.tag(ce)}.parquet"
    log = path.with_suffix(".seeds")
    done = set(log.read_text().split()) if log.exists() and path.exists() else set()
    seeds = [WORK / f.format(c=c) for f in R_SEED.get(split, [])] if ce == RR.CE else []
    new = [f for f in seeds if f.exists() and str(f.relative_to(WORK)) not in done]
    if new:
        parts = [pd.read_parquet(path)] if path.exists() else []
        for f in new:
            parts.append(seed_keys(f, id1, idx))
            print(f"{c}: r cache seeded from {f.relative_to(WORK)}: {len(parts[-1]):,} pairs", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(parts, ignore_index=True).drop_duplicates("k", keep="first").to_parquet(path, index=False)
        log.write_text("".join(f"{x}\n" for x in sorted(done | {str(f.relative_to(WORK)) for f in new})))
    return path


def rerank(c: str, q1: np.ndarray, qx: np.ndarray, p: np.ndarray, text, cache: Path | None = None,
           fallback: Path | None = None) -> tuple:
    """(p', r): p with the band of a CE-trained country blended with the cross-encoder's r (rerank.CFG w; France
    with W_FRANCE when FRANCE == "bge"), any other country's p unchanged; r = the CE score of the band pairs (NaN
    elsewhere). The CE runs in its own process under the GPU lock; text(q1, qx) -> ("name | addr" of the S1, same of
    the SX). cache: the r cache (r_cache), so a rerun only scores new band pairs. fallback: with CE_MISSING =
    "fallback", band pairs the cache lacks take #7's e5 blend from that cache instead of being scored (the holdout
    proxy P17 used: P13 scored only the holdout S1's own band pairs)."""
    lo, hi = RR.CFG["band"]
    full = np.full(len(p), np.nan, np.float32)
    w = W_FRANCE if c in unseen() and FRANCE in ("bge", "idf144_bge", "fr31") else RR.CFG["w"] if c in RR.CFG["countries"] \
        else None
    if w is None:
        return p, full
    b = np.flatnonzero((p > lo) & (p < hi))
    k = (q1[b].astype(np.int64) << 32) | qx[b].astype(np.int64)
    old = pd.read_parquet(cache) if cache is not None and cache.exists() else pd.DataFrame({"k": [], "r": []})
    r = pd.Series(old.r.to_numpy(np.float32), index=old.k.to_numpy(np.int64)).reindex(k).to_numpy(np.float32).copy()
    todo = np.isnan(r)
    if todo.any() and CE_MISSING == "fallback" and fallback is not None:
        f = pd.read_parquet(fallback)
        rf = pd.Series(f.r.to_numpy(np.float32), index=f.k.to_numpy(np.int64)).reindex(k[todo]).to_numpy(np.float32)
        full[b] = r
        out = RR.blend(p, full, w, lo, hi)
        out[b[todo]] = RR.blend(p[b[todo]], rf, FALLBACK[1], lo, hi)
        print(f"{c}: {int(todo.sum()):,} of {len(b):,} band pairs without an r keep {Path(FALLBACK[0]).name}'s blend "
              f"(--ce-missing fallback)", flush=True)
        return out, full
    if todo.any() and CE_MISSING == "keep":
        print(f"{c}: {int(todo.sum()):,} of {len(b):,} band pairs without an r keep p (--ce-missing keep)", flush=True)
        todo[:] = False
    if todo.any():
        ensure_ce()
        t0 = time.time()
        ta, tb = text(q1[b[todo]], qx[b[todo]])
        with tempfile.TemporaryDirectory(dir=WORK) as d:
            pd.DataFrame({"ta": ta, "tb": tb}).to_parquet(f"{d}/in.parquet", index=False)
            del ta, tb
            gc.collect()
            subprocess.run(["flock", GPU_LOCK, sys.executable, str(SRC / "rerank.py"), "score", f"{d}/in.parquet",
                            f"{d}/out.parquet", str(RR.CE), RR.FAMILY], check=True)   # the recipe's CE, not the default
            r[todo] = pd.read_parquet(f"{d}/out.parquet").r.to_numpy(np.float32)
        print(f"{c}: {int(todo.sum()):,} of {len(b):,} band pairs rescored in {time.time() - t0:.0f}s", flush=True)
        if cache is not None:
            pd.DataFrame({"k": np.concatenate([old.k.to_numpy(np.int64), k[todo]]),
                          "r": np.concatenate([old.r.to_numpy(np.float32), r[todo]])}).to_parquet(cache, index=False)
    full[b] = r
    return RR.blend(p, full, w, lo, hi), full


def ensure_ce():
    """Train the recipe's cross-encoder from the train data when rerank.CE is absent (~30 min per epoch, GPU)."""
    if RR.CE.exists():
        return
    assert RR.FAMILY == "e5", (f"{RR.CE} is P13's bge CE, fine-tuned on AWS (probes/P13-aws-rerank, SETUP.md): download "
                               "it to that path; this pipeline trains only the e5-small CE")
    import train_matcher
    train_matcher.prepare()
    for c in RR.CFG["countries"]:
        if not (RR.OUT / f"trainband_{c}.parquet").exists():
            subprocess.run([sys.executable, str(SRC / "rerank.py"), "band", c], check=True)
    subprocess.run(["flock", GPU_LOCK, sys.executable, str(SRC / "rerank.py"), "train", f"epochs={RR.EPOCHS}",
                    f"out={RR.CE}"], check=True)


def empty_curve(q1: np.ndarray, qx: np.ndarray, p: np.ndarray, n_s1: int, ts) -> pd.Series:
    """Predicted empty-S1 rate by threshold under the decision (argmax S1 per SX, p >= t); P6 fr_diag."""
    am = M.argmax_sx(qx, p)
    top = np.full(n_s1, -1.0)
    np.maximum.at(top, q1[am], p[am])            # an S1 is non-empty at t iff its best argmax-kept pair has p >= t
    return pd.Series([(top < t).mean() for t in ts], index=ts)


def thresholds(parts: list) -> dict:
    """Per-country t. A country with training labels gets T on the final p. Any other label (France) gets P6's
    rule on M2's p -- the model it is scored with: the t where its empty-S1 rate matches the labelled countries'
    (S1-weighted) at T_M2, clipped to [T_M2, T_M2 + 0.2]; T_M2 + 0.1 if the run holds no labelled country.
    T_FRANCE_OVERRIDE (every UNSEEN label) / T_COUNTRY replace the result for their countries (the rule's value is
    still printed)."""
    import train_matcher
    seen = set(train_matcher.countries())
    ts = np.round(np.arange(0.5, 0.99, 0.01), 2)
    ref = [(len(d["id1"]), empty_curve(d["q1"], d["qx"], d["p_m2"], len(d["id1"]), [T_M2]).iloc[0])
           for d in parts if d["c"] in seen]
    out = {}
    for d in parts:
        if d["c"] in seen:
            out[d["c"]] = T
        elif not ref:
            out[d["c"]] = round(T_M2 + 0.1, 2)
        else:
            e = empty_curve(d["q1"], d["qx"], d["p_m2"], len(d["id1"]), ts)
            r = sum(n * v for n, v in ref) / sum(n for n, _ in ref)
            out[d["c"]] = float(min(max(float((e - r).abs().idxmin()), T_M2), round(T_M2 + 0.2, 2)))
            print(f"{d['c']}: unseen label, t = {out[d['c']]:.2f} (empty-S1 rate {e.loc[out[d['c']]]:.4f} vs labelled "
                  f"{r:.4f} at T_M2 {T_M2})", flush=True)
    fixed = ({c: T_FRANCE_OVERRIDE for c in unseen()} if T_FRANCE_OVERRIDE is not None else {}) | T_COUNTRY
    for c, t in fixed.items():
        if c in out:
            print(f"{c}: t overridden {out[c]:.2f} -> {t:g}", flush=True)
            out[c] = float(t)
    return out


def cap(d: dict, acc: np.ndarray) -> np.ndarray:
    """At most CAP = (n S2, n S3) accepted matches per S1, the highest p kept: the train truth never has more
    (max 5 S2 / 6 S3 per S1), so a longer list holds at least one wrong pair."""
    q1, p = d["K"] >> 32, d["p"]
    j = np.flatnonzero(acc)
    s3 = pc.starts_with(d["idx"].take(pa.array(d["K"][j] & 0xFFFFFFFF)), "S3-").to_numpy(zero_copy_only=False)
    o = np.lexsort((-p[j], s3, q1[j]))                   # per (S1, source), best p first
    g = np.r_[True, (q1[j][o][1:] != q1[j][o][:-1]) | (s3[o][1:] != s3[o][:-1])]
    start = np.maximum.accumulate(np.where(g, np.arange(len(o)), 0))
    rank = np.empty(len(o), np.int64)
    rank[o] = np.arange(len(o)) - start
    over = rank >= np.where(s3, CAP[1], CAP[0])
    out = acc.copy()
    out[j[over]] = False
    print(f"{d['c']}: cap {CAP[0]} S2 / {CAP[1]} S3 per S1 drops {int(over.sum()):,} matches of "
          f"{len(np.unique(q1[j[over]])):,} S1", flush=True)
    return out


def drop_sib_cands(d: dict, acc: np.ndarray):
    """A decoy stage before the final decision: a pair at a generator sibling offset (hn_x - hn_1 in P15's
    {3,4,5,7,9,11,13,21}) survives only if it is its SX's best pair above t (accepted). The others leave the
    candidate list; the decision is unchanged by construction (they were neither an accepted pair nor, being below
    t or not the SX's best, able to change another SX's decision)."""
    import stack as S
    sib = S.sib_mask(d["hn"]("s1"), d["hn"]("sx"))
    keep = ~sib | acc
    n_ids = max(len(d["ids"]), 1)
    ids_rows = np.zeros(len(d["id1"]), bool)
    ids_rows[codes(d["id1"], d["ids"])] = True
    scored = ids_rows[d["K"] >> 32]                     # the scored S1's pairs (holdout: fold 0)
    print(f"{d['c']}: sibling-decoy candidates dropped {int((~keep & scored).sum()):,}: "
          f"{scored.sum() / n_ids:.2f} -> {(keep & scored).sum() / n_ids:.2f} candidates per S1", flush=True)
    d["K"], d["p"], acc_kept = d["K"][keep], d["p"][keep], acc[keep]
    return acc_kept


def half(ids) -> np.ndarray:
    """P5's tune (0) / report (1) halves of the S1: a hash of the id, stable across runs and probes."""
    return (pd.util.hash_pandas_object(pd.Series(list(ids)), index=False).to_numpy() >> 7) % 2


def refit_w(parts: list) -> list | None:
    """rerank.CFG["w"] refitted on this run: P4b's logistic stack on the band pairs of the scored half-0 S1
    (M2 p, CE r, truth). The adopted w came out of the same fit on P4b's holdout files."""
    P, Rr, Y = [], [], []
    for d in parts:
        b = np.flatnonzero(~np.isnan(d["r"]))
        s1 = np.array(d["id1"].take(pa.array(d["q1"][b])).to_pylist(), object)
        sx = d["idx"].take(pa.array(d["qx"][b])).to_pylist()
        m = np.fromiter((s in d["truth"] for s in s1), bool, len(s1))    # np.isin on objects is quadratic
        m[m] &= half(s1[m]) == 0
        P.append(d["p_m2"][b][m]), Rr.append(d["r"][b][m])
        Y.append(np.array([x in d["truth"][s] for s, x in zip(s1[m], np.array(sx, object)[m])], bool))
    if not sum(map(len, Y)):
        return None
    w = RR.fit_blend(*(np.concatenate(z) for z in (P, Rr, Y)))
    print(f"blend w refit on holdout half 0: {np.round(w, 3).tolist()} (CFG {RR.CFG['w']}), "
          f"{sum(map(len, Y)):,} band pairs", flush=True)
    return [round(x, 4) for x in w]


def france_variant(name: str, c: str) -> pd.DataFrame:
    """Unseen country c's M2 p under P14's IDF variant (#9a / #9ab idf144, #9d idfdom): a world (data/work/p9/fr_<name>)
    whose inputs are links to this one, except the vectorizers (block.idf_country / idf_domain). A subprocess (ER_WORK =
    that dir) reblocks c, rebuilds its X, prunes and runs M2: q1 / qx are rows of the same P3 texts, the candidates
    change."""
    import pickle
    import prune
    v = M.CACHE / f"fr_{name}"
    for rel in ["test_source1.parquet", "test_source2.parquet", "test_source3.parquet", "train_source1.parquet",
                "place_freq_train.json", "p2_xlit_table.parquet", "gt_pairs.parquet", "s1_folds.parquet",
                "rules_s1feat_train.parquet", f"rf_test_s1_{c}.parquet", f"rf_test_sx_{c}.parquet",
                str(M.BOOSTER.relative_to(WORK)), str(prune.MODEL.relative_to(WORK)),
                *[str(f.relative_to(WORK)) for f in B.TXT["test"].glob(f"{c}_*.parquet")]]:
        d = v / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        if not d.exists():
            d.symlink_to((WORK / rel).resolve())
    vp = v / "p3" / "vecs_hash.pkl"
    if not vp.exists():
        vec = B.vecs()
        if name == "idf144":
            files = sorted(B.TXT["test"].glob(f"{c}_sx_*.parquet"))
            tx = pd.concat([pq.read_table(f, columns=["name", "addr"]).to_pandas() for f in files], ignore_index=True)
            B.idf_country(vec, tx, 0.144)
        else:
            B.idf_domain(vec)
        vp.write_bytes(pickle.dumps(vec))
    cache = v / "p9" / M.probs_path("test", c).name
    if not cache.exists():
        env = {**__import__("os").environ, "ER_WORK": str(v)}
        subprocess.run([sys.executable, str(SRC / "run.py"), "--split", "test", "--probs-only", c,
                        "--recipe", RECIPE, "--tau-country", *[f"{k}={v}" for k, v in M.TAU_COUNTRY.items()]],
                       env=env, check=True)
    return pd.read_parquet(cache)


def france31(c: str) -> dict:
    """#31's chain (france.py) for unseen country c, one model_parts dict: c_hn's p on P29's rebuild of the idf144
    world (its own process, cached in data/work/p9/fr31/p9), #9ab's bge blend (W_FRANCE), then tier / bxneg / [VETO]
    / cap. The rules decide at c's t, so it must be fixed (T_COUNTRY / T_FRANCE_OVERRIDE). p_m2 = c_hn's p."""
    import os
    import france as FR
    import prune
    import stack as S
    FR.C = c
    t = T_POST or T_COUNTRY.get(c, T_FRANCE_OVERRIDE)
    assert t is not None, "fr31's post-rules need the unseen country's t"
    france_variant("idf144", c)                         # the base world (its P3 texts, blocking and X)
    v = FR.world(M.CACHE / "fr_idf144")
    path = v / "p9" / f"chn_test_{c}_{M.tag(CHN)}_{M.tag(prune.MODEL)}_{M.tau(c)}.parquet"
    if not path.exists():
        if not CHN.exists():                            # P26's recipe from the train data (~1 h, > 3 GB): SETUP.md
            import train_matcher
            train_matcher.main(CHN, M.FOLDS, hn=True)
        assert M.tau(c) == prune.TAU, "france.chn prunes at prune.TAU"
        subprocess.run([sys.executable, str(SRC / "france.py"), "chn", str(CHN), str(path), c],
                       env={**os.environ, "ER_WORK": str(v)}, check=True)
    d = pd.read_parquet(path)
    q1, qx, pm = d.q1.to_numpy(), d.qx.to_numpy(), d.p.to_numpy(np.float32)
    del d
    id1, idx = F.ids("test", c)
    cache = r_cache("test", c, id1, idx)
    p, r = rerank(c, q1, qx, pm, lambda a, b: RR.texts("test", c, a, b), cache)
    veto = None
    if VETO:                                            # P30's pool scores first, then the cache (P31's order)
        k = (q1.astype(np.int64) << 32) | qx.astype(np.int64)
        rk = pd.concat([seed_keys(WORK / VETO[1], id1, idx), pd.read_parquet(cache)]).drop_duplicates("k")
        veto = (VETO[0], pd.Series(rk.r.to_numpy(np.float32), index=rk.k.to_numpy()).reindex(k).to_numpy(np.float32))
    learn = pd.read_parquet(B.P3 / f"cands_test_{c}.parquet", columns=["s1", "sx"])   # main P3 list, as P31
    pbl = p
    p = FR.post(q1, qx, id1, idx, pbl, pm, t, v, learn, veto)
    extra = {}
    if CHAIN is not None:                               # france.chain's inputs: the blend, CE r, no-veto p, P48's r
        assert veto is not None, "the final chain reads the veto's CE r (P30's pool first, then the cache)"
        extra = {"pbl": pbl, "r_all": veto[1], "v": v}
        if CHAIN.get("p55") is not None or {"u54", "veto54"} & set(CHAIN.get("order", ())):
            print(f"{c}: post-rules again without the veto (P42's frame for P55 / fr54)", flush=True)
            extra["p0"] = FR.post(q1, qx, id1, idx, pbl, pm, t, v, learn, None)
            lo = seed_keys(WORK / R_LO.format(c=c), id1, idx)
            k = (q1.astype(np.int64) << 32) | qx.astype(np.int64)
            extra["r_lo"] = pd.Series(lo.r.to_numpy(np.float64), index=lo.k.to_numpy()).reindex(k).to_numpy(np.float64)
    del learn
    hn = lambda side: S.hn_of("test", side, c, id1 if side == "s1" else idx, q1 if side == "s1" else qx,
                              WORK if side == "s1" else v)      # the fixed '(n)' house numbers on the SX side
    return {"c": c, "id1": id1, "idx": idx, "ids": id1, "q1": q1, "qx": qx, "p_m2": pm, "p": p, "r": r,
            "hn": hn, "truth": None, **extra}


def ensure_stack(c: str):
    """P17's stacker booster for country c (the recipe's STACK path); fit from the holdout when absent: every
    competing row of the country's holdout list with the recipe's blended p (W_STACK_FIT's w: what the shipped
    booster was fit on, so the refit is bit-identical), trained on the holdout S1 of half 0."""
    import lightgbm as lgb
    import stack as S
    path = WORK / STACK[c]
    if not path.exists():
        w, RR.CFG["w"] = RR.CFG["w"], list(W_STACK_FIT or RR.CFG["w"])
        try:
            d = model_parts("holdout", only=[c], stack=False)[0]
        finally:
            RR.CFG["w"] = w
        X = S.X_of(d["q1"], d["p"], S.hn_of("train", "s1", c, d["id1"], d["q1"]),
                   S.hn_of("train", "sx", c, d["idx"], d["qx"]))
        pos, y = rows_truth(d)
        hv = half(d["ids"].to_pylist())
        S.train(X, y, (pos >= 0) & (hv[np.maximum(pos, 0)] == 0), path)
        print(f"{c}: stacker trained on holdout half 0 -> {path}", flush=True)
    return lgb.Booster(model_file=str(path))


def rows_truth(d: dict) -> tuple:
    """pos (row -> index of its S1 in d['ids'], -1 = a competing S1 that is not scored) and y (true pair) per row."""
    inv = np.full(len(d["id1"]), -1, np.int64)
    inv[codes(d["id1"], d["ids"])] = np.arange(len(d["ids"]))
    pos = inv[d["q1"]]
    a = [s for s, v in d["truth"].items() for _ in v]
    b = [x for v in d["truth"].values() for x in v]
    ka = pc.index_in(pa.array(a, pa.string()), value_set=d["id1"]).to_numpy(zero_copy_only=False)
    kb = pc.index_in(pa.array(b, pa.string()), value_set=d["idx"]).to_numpy(zero_copy_only=False)
    ok = ~(pd.isna(ka) | pd.isna(kb))
    y = np.isin((np.asarray(d["q1"], np.int64) << 32) | np.asarray(d["qx"], np.int64),
                (ka[ok].astype(np.int64) << 32) | kb[ok].astype(np.int64))
    return pos, y


def after_ce(c: str, split: str, q1, qx, p: np.ndarray, id1, idx, hn=None, stack: bool = True) -> np.ndarray:
    """The house-number stage after the CE: P17's stacker for a country in STACK (#8c: US), P15's sibling rule for
    an UNSEEN label when FRANCE == "sib" (#9s). hn(side) -> house numbers of the rows ("s1" -> of q1, "sx" -> of qx)."""
    import stack as S
    xs = "train" if split == "holdout" else "test"
    hn = hn or (lambda side: S.hn_of(xs, side, c, id1 if side == "s1" else idx, q1 if side == "s1" else qx))
    if stack and c in STACK:
        p = S.stacked(ensure_stack(c), S.X_of(q1, p, hn("s1"), hn("sx")))
        print(f"{c}: stacked ({Path(STACK[c]).name})", flush=True)
    if c in unseen() and FRANCE == "sib":
        sib = S.sib_mask(hn("s1"), hn("sx"))
        p = np.where(sib, 0.0, p).astype(p.dtype)
        print(f"{c}: sibling rule, {int(sib.sum()):,} pairs zeroed", flush=True)
    return p


def a10_stage(split: str, c: str, q1, qx, p: np.ndarray, pm, r, id1: pa.Array, idx: pa.Array, ids: pa.Array) -> tuple:
    """P34's a10 residual stacker (a10.py) on the country's #8c p -> (p' float32, am). am = #8c's argmax, which the
    decision keeps (re-deriving it from p' moves ~600 test decisions per country, P34). Cached per booster and input p
    (data/work/p9/a10_*): the features take 1.5-3 min and ~2 GB per country."""
    import a10 as A
    booster = ensure_a10()
    path = M.CACHE / f"a10_{split}_{c}_{M.tag(WORK / A10)}_{hashlib.sha1(np.ascontiguousarray(p).tobytes()).hexdigest()[:8]}.parquet"
    if not path.exists():
        t0 = time.time()
        scored = None if split == "test" else np.isin(np.arange(len(id1)), codes(id1, ids))[q1]
        pn, am, _ = A.apply(booster, "train" if split == "holdout" else split, c, q1, qx, p, pm, r, id1, idx, scored)
        pd.DataFrame({"p": pn.astype(np.float32), "am": am}).to_parquet(path, index=False)
        print(f"{c}: a10 stacked {int(am.sum()):,} argmax pairs in {time.time() - t0:.0f}s, peak {B.peak_gb():.2f} GB",
              flush=True)
    d = pd.read_parquet(path)
    return d.p.to_numpy(np.float32), d.am.to_numpy(bool)


def iu_stage(tag: str, split: str, c: str, q1, qx, p: np.ndarray, pm, r, id1: pa.Array, idx: pa.Array) -> tuple:
    """a10_stage's slot for iu = full / fullzoo (iu_stack.py: #50e / #52_max's combiner) -> (p'' float32, am). Cached
    per booster set and input p (data/work/p9/iu_*): the features take ~5 min and ~1.3 GB per country."""
    import iu_stack as I
    path = M.CACHE / f"iu_{tag}_{split}_{c}_{I.booster_tag(tag)}_{hashlib.sha1(np.ascontiguousarray(p).tobytes()).hexdigest()[:8]}.parquet"
    if not path.exists():
        t0 = time.time()
        pn, am = I.apply(tag, split, c, q1, qx, p, pm, r, id1, idx)
        pd.DataFrame({"p": pn, "am": am}).to_parquet(path, index=False)
        print(f"{c}: iu {tag} combined {int(am.sum()):,} argmax pairs in {time.time() - t0:.0f}s, peak "
              f"{B.peak_gb():.2f} GB", flush=True)
    d = pd.read_parquet(path)
    return d.p.to_numpy(np.float32), d.am.to_numpy(bool)


def ensure_a10():
    """P34's a10 booster (the recipe's A10 path); fit from the holdout when absent, as probes/P34-a10-stack/ho.py did:
    #8c's holdout rows of each CE country (P17's stack on, a10 off; band pairs without a cached bge r take #7's e5
    blend, CE_MISSING "fallback" = P23's dump), a10.features on the scored S1's argmax pairs, one fit on the S1 of
    holdout half 0 with the countries pooled (~2 GB, minutes with the holdout caches)."""
    import a10 as A
    import lightgbm as lgb
    global A10, CE_MISSING
    path = WORK / A10
    if not path.exists():
        keep, X, Y = (A10, CE_MISSING), [], []
        A10, CE_MISSING = None, "fallback"
        try:
            for c in RR.CFG["countries"]:
                d = model_parts("holdout", only=[c])[0]
                pos, y = rows_truth(d)
                k1, u1 = pd.factorize(d["q1"])
                kx, ux = pd.factorize(d["qx"])
                s1t, sxt = A.texts("train", c, d["id1"].take(pa.array(u1)).to_pylist(),
                                   d["idx"].take(pa.array(ux)).to_pylist())
                a = A.features(k1, kx, d["p"], d["p_m2"], d["r"], M.argmax_sx(d["qx"], d["p"]), pos >= 0, s1t, sxt)
                m = half(d["ids"].to_pylist())[pos[a.row.to_numpy()]] == 0
                X.append(a[A.COLS][m])
                Y.append(y[a.row.to_numpy()][m])
                del d, s1t, sxt, a
                gc.collect()
        finally:
            A10, CE_MISSING = keep
        path.parent.mkdir(parents=True, exist_ok=True)
        A.fit(pd.concat(X), np.concatenate(Y)).save_model(str(path))
        print(f"a10 fit on holdout half 0 ({sum(map(len, Y)):,} argmax pairs) -> {path}", flush=True)
    return lgb.Booster(model_file=str(path))


def model_parts(split: str, world=None, only=None, stack: bool = True) -> list:
    """One dict per country: id1 / idx = S1 pool / SX ids (row order of q1 / qx), ids = the scored S1, q1, qx, p_m2
    (M2), p (after the reranker and the house-number stage), truth. Full-scale splits read the X files; worlds are
    featurized in memory. only: those countries; stack=False: stop after the CE (the stacker's training input)."""
    parts = []
    if split in ("test", "holdout"):
        xs = "train" if split == "holdout" else "test"
        if split == "holdout":
            s1, t = holdout_s1()
            labels = sorted(s1.country.unique())
        else:
            labels = sorted(pq.read_table(WORK / "test_source1.parquet", columns=["country"]).column(0).unique().to_pylist())
        for c in (x for x in labels if only is None or x in only):
            tc = time.time()
            prepare(xs, c)
            if c in unseen() and FRANCE == "fr31":
                parts.append(france31(c))
                continue
            fv = {"idf144": "idf144", "idf144_bge": "idf144", "idfdom": "idfdom"}.get(FRANCE) if c in unseen() else None
            d = france_variant(fv, c) if fv else M.split_probs(split, c)
            q1, qx, pm = d.q1.to_numpy(), d.qx.to_numpy(), d.p.to_numpy(np.float32)
            del d
            id1, idx = F.ids(xs, c)
            cache = r_cache(split, c, id1, idx) if RR.CE.exists() or Path(f"{RR.CE}.tag").exists() else None
            fb = M.CACHE / f"r_{split}_{c}_{M.tag(WORK / FALLBACK[0])}.parquet" if (WORK / FALLBACK[0]).exists() else None
            p, r = rerank(c, q1, qx, pm, lambda a, b, c=c: RR.texts(xs, c, a, b), cache, fb)
            p = after_ce(c, split, q1, qx, p, id1, idx, stack=stack)
            hn = lambda side, c=c, id1=id1, idx=idx, q1=q1, qx=qx: __import__("stack").hn_of(
                xs, side, c, id1 if side == "s1" else idx, q1 if side == "s1" else qx)
            ids = id1 if split == "test" else arr(s1.entity_id[s1.country == c])
            am = None
            iu = IU or RECIPES[RECIPE].get("iu", "a10")
            if iu != "a10" and stack and c in RR.CFG["countries"]:
                p, am = iu_stage(iu, split, c, q1, qx, p, pm, r, id1, idx)
            elif A10 and stack and c in RR.CFG["countries"]:
                p, am = a10_stage(split, c, q1, qx, p, pm, r, id1, idx, ids)
            parts.append({"c": c, "id1": id1, "idx": idx, "ids": ids, "q1": q1, "qx": qx, "p_m2": pm, "p": p, "r": r,
                          "am": am, "hn": hn, "truth": None if split == "test" else truth_of(ids.to_pylist(), t)})
            print(f"{c}: {len(q1):,} pairs scored, {time.time() - tc:.0f}s, peak {B.peak_gb():.2f} GB", flush=True)
            gc.collect()
        return parts
    for c, f1, fx, cands, ids, tr in inputs(split, world):
        id1, ids = arr(f1.id), arr(ids)
        if fx is None or len(fx) == 0:                  # a country with no S2/S3 at all: every row empty
            q1 = qx = np.zeros(0, np.int64)
            pm, idx = np.zeros(0, np.float32), pa.array([], pa.string())
        else:
            _, K, pairs = assemble("model", f1, fx, cands, np.ones(len(f1), bool))
            keep, pm = M.score(pairs, f1, fx)
            K, idx = K[keep], arr(fx.id)
            q1, qx = K >> 32, K & 0xFFFFFFFF
            del pairs
        tx = lambda a, b, f1=f1, fx=fx: tuple((f.name.astype(str) + " | " + f.addr.astype(str)).to_numpy(object)[i]
                                              for f, i in ((f1, a), (fx, b)))
        p, r = rerank(c, q1, qx, pm, tx)
        hn = lambda side, f1=f1, fx=fx, q1=q1, qx=qx: (f1 if side == "s1" else fx).hn.fillna("").to_numpy(
            object)[q1 if side == "s1" else qx] if len(q1) else np.zeros(0, object)
        p = after_ce(c, split, q1, qx, p, id1, idx, hn=hn)
        parts.append({"c": c, "id1": id1, "idx": idx, "ids": ids, "q1": q1, "qx": qx, "p_m2": pm, "p": p, "r": r,
                      "hn": hn, "truth": tr})
    return parts


def write_lists(f, s1_ids: pa.Array, q1: np.ndarray, sx_ids: pa.Array, qx: np.ndarray, size: int = 200_000,
                tail: dict | None = None):
    """Append one row per S1 in s1_ids order (empty list allowed): 's1<TAB>sx,sx,...', ids sorted and de-duplicated.
    Pairs come as row numbers (q1 into s1_ids, qx into sx_ids), so the pair ids are Python strings only one chunk
    of S1 at a time (from probes/P7-scale/chunks.py write_lists). tail: {row: [sx id, ...]} appended after the sorted
    ids of that row, in order (the duplicate step's copies, as P54 wrote them)."""
    k = np.unique((q1.astype(np.int64) << 32) | qx.astype(np.int64))
    q1, qx = k >> 32, k & 0xFFFFFFFF
    bounds = np.searchsorted(q1, np.arange(len(s1_ids) + 1))
    s1 = s1_ids.to_numpy(zero_copy_only=False)
    for c0 in range(0, len(s1), size):
        c1 = min(c0 + size, len(s1))
        ids = sx_ids.take(pa.array(qx[bounds[c0]:bounds[c1]])).to_numpy(zero_copy_only=False)
        cut = bounds[c0:c1 + 1] - bounds[c0]
        f.write("".join(f"{s}\t{','.join(sorted(ids[cut[i]:cut[i + 1]]) + (tail or {}).get(c0 + i, []))}\n"
                        for i, s in enumerate(s1[c0:c1])))


def sets(keys: np.ndarray, pos: np.ndarray, s1_ids: pa.Array, sx_ids: pa.Array) -> dict:
    """{scored s1 id: set(sx ids)} from keys (for scoring holdout / dev only)."""
    q1, qx = pos[keys >> 32], keys & 0xFFFFFFFF
    s1, sx = s1_ids.take(pa.array(q1)).to_pylist(), sx_ids.take(pa.array(qx)).to_pylist()
    out = {}
    for a, b in zip(s1, sx):
        out.setdefault(a, set()).add(b)
    return out


def dup_keys(split: str) -> pd.DataFrame:
    """P54: (id as int64 = source * 1e10 + number, key = hash of the lower-cased, stripped name + address + country,
    country) of every S2 / S3 record whose key occurs >= 2 times: content-identical records."""
    ids, ks, cs = [], [], []
    for i in (2, 3):
        for b in pq.ParquetFile(WORK / f"{split}_source{i}.parquet").iter_batches(
                500_000, columns=["entity_id", "business_name", "business_address", "country"]):
            d = b.to_pandas()
            x = d[["business_name", "business_address", "country"]].apply(lambda s: s.str.lower().str.strip())
            ks.append(pd.util.hash_pandas_object(x, index=False).to_numpy())
            ids.append((d.entity_id.str.slice(1, 2).astype(np.int64) * 10**10 + d.entity_id.str.slice(3).astype(np.int64)).to_numpy())
            cs.append(d.country.astype("category"))
            del d, x
    k = pd.DataFrame({"id": np.concatenate(ids), "key": np.concatenate(ks)})
    k["country"] = pd.api.types.union_categoricals(cs)
    del ids, ks, cs
    return k[k.key.duplicated(keep=False)].reset_index(drop=True)


_DUPK = {}


def dup_add(d: dict, km: np.ndarray, K: np.ndarray, split: str) -> tuple:
    """P54's duplicate consistency, the last stage (keys = q1 << 32 | qx of one country): every unassigned
    content-identical copy of an accepted S2 / S3 goes to the same S1, past the 5 / 6 cap (train truth: 60,319 such
    groups, 100% one owner), and joins that S1's candidates when missing there. Groups whose accepted copies sit at > 1
    S1 are left alone. The key holds the country, so a group never spans two countries. -> (km, K)."""
    if split not in _DUPK:
        _DUPK[split] = dup_keys("test" if split == "test" else "train")
    k = _DUPK[split]
    eid = lambda a: (pc.utf8_slice_codeunits(a, 1, 2).cast(pa.int64()).to_numpy() * 10**10
                     + pc.utf8_slice_codeunits(a, 3).cast(pa.int64()).to_numpy())
    ux = eid(d["idx"])                                   # every SX of the country, as P54's int ids
    k = k[np.isin(k.id.to_numpy(), ux)]
    if not len(k) or not len(km):
        return km, K
    pos = pd.Series(np.arange(len(ux)), index=ux)
    acc = pd.DataFrame({"s1": km >> 32, "sx": ux[km & 0xFFFFFFFF]})
    k = k.merge(acc[acc.sx.isin(k.id.to_numpy())], left_on="id", right_on="sx", how="left").drop(columns="sx")
    g = k.groupby("key").s1.agg(["count", "nunique", "size"])
    one = g[(g["count"] > 0) & (g["count"] < g["size"]) & (g["nunique"] == 1)].index
    fix = k[k.key.isin(one) & k.s1.isna()].drop(columns="s1").merge(
        k[k.key.isin(one) & k.s1.notna()][["key", "s1"]].drop_duplicates(), on="key")
    add = pd.unique((fix.s1.to_numpy(np.int64) << 32) | pos.reindex(fix.id).to_numpy(np.int64))   # P54's order
    new_c = add[~np.isin(add, K)]
    print(f"{d['c']}: duplicate step (P54): {len(add):,} copies to {fix.s1.nunique():,} S1, {len(new_c):,} new candidates; "
          f"groups over > 1 S1 skipped {int((g['nunique'] > 1).sum()):,}", flush=True)
    d["tail"] = (add, new_c)                         # written after each line's sorted ids, as P54 appended them
    return np.concatenate([km, add]), np.concatenate([K, new_c])


def validate(out: Path):
    """The official validator (student_resource/utils, next to the dataset). It holds every candidate id as a Python
    string (~4 GB at test scale): if the OOM killer stops it, it reruns on matching_results alone -- run() has
    already asserted the candidate rules while writing."""
    from io_tsv import DATA
    cwd = DATA.resolve().parent
    if not (cwd / "utils/validate_submission.py").exists():
        print(f"official validator not found under {cwd}/utils -- skipped (in-process checks passed)", flush=True)
        return
    cmd = [sys.executable, "utils/validate_submission.py", "--matching", str((out / "matching_results.tsv").resolve()),
           "--test-dir", str(DATA.resolve() / "test")]
    victim = lambda: Path("/proc/self/oom_score_adj").write_text("1000")   # the OOM killer takes it, never run.py
    r = subprocess.run(cmd + ["--candidate", str((out / "candidate_pairs.tsv").resolve())], cwd=cwd, preexec_fn=victim)
    if r.returncode in (-9, 137):
        print("VALIDATOR OOM-KILLED on candidate_pairs.tsv -- rerunning on matching_results.tsv only", flush=True)
        r = subprocess.run(cmd, cwd=cwd, preexec_fn=victim)
    assert r.returncode == 0, f"official validator failed ({r.returncode})"


def rules_parts(split: str, world=None):
    """The rules stage per country, in model_parts' shape (km = accepted keys instead of p)."""
    for c, f1, fx, cands, ids, tr in inputs(split, world):
        id1, ids = arr(f1.id), arr(ids)
        keep = np.zeros(len(f1), bool)
        keep[codes(id1, ids)] = True
        if fx is None or len(fx) == 0:
            idx, km, K = pa.array([], pa.string()), np.zeros(0, np.int64), np.zeros(0, np.int64)
        else:
            idx = arr(fx.id)
            km, K, _ = assemble("rules", f1, fx, cands, keep)
        yield {"c": c, "id1": id1, "idx": idx, "ids": ids, "K": K, "km": km, "truth": tr}
        del f1, fx, cands
        gc.collect()


def run(split: str, stage: str, out: Path, world=None, log=True, check=True) -> dict:
    """Run the pipeline on one split (or on injected (s1, sx, truth) frames) and write both files to out
    (out/<split>/ for anything but test). Returns the metrics when the split has truth."""
    t0 = time.time()
    if world is None:
        ensure(split)
    out = out if split == "test" else out / split
    out.mkdir(parents=True, exist_ok=True)
    w = None
    if stage == "model":
        parts = model_parts(split, world)
        ts = thresholds(parts)
        if split == "holdout":
            w = refit_w(parts)
        for d in parts:                                 # the decision; candidates = every scored pair
            d["K"] = (d.pop("q1").astype(np.int64) << 32) | d.pop("qx").astype(np.int64)
            am = d.pop("am", None)                      # a10's countries keep #8c's argmax
            am = M.argmax_sx(d["K"] & 0xFFFFFFFF, d["p"]) if am is None else am
            acc = am & (d["p"] >= ts[d["c"]])
            if CAP:
                acc = cap(d, acc)
            if d["c"] in unseen() and len(acc) and CHAIN is not None:     # the final chain replaces acc
                import france as FR
                FR.C, qx = d["c"], d["K"] & 0xFFFFFFFF
                ids = lambda a, q: a.take(pa.array(q)).to_numpy(zero_copy_only=False)
                f = {"q1": d["K"] >> 32, "qx": qx, "p": d["p"], "p0": d.get("p0"), "g": d["p_m2"], "pbl": d["pbl"],
                     "r": d["r_all"], "r_lo": d.get("r_lo"), "h1": d["hn"]("s1"), "hx": d["hn"]("sx"),
                     "is2": pc.starts_with(d["idx"].take(pa.array(qx)), "S2-").to_numpy(zero_copy_only=False),
                     "s1": ids(d["id1"], d["K"] >> 32), "sx": ids(d["idx"], qx), "n": len(d["id1"]), "v": d["v"],
                     "learn": pd.read_parquet(B.P3 / f"cands_test_{d['c']}.parquet", columns=["s1", "sx"])
                     if "swap" in CHAIN.get("order", ()) else None}
                cfg = dict(CHAIN, t=ts[d["c"]])
                if cfg.get("p55") is not None:
                    cfg["p55"] = (*cfg["p55"][:2], [WORK / x for x in cfg["p55"][2]])
                acc = FR.chain(f, cfg)
                del f
            elif d["c"] in unseen() and len(acc) and any(x is not None for x in (RESCUE, GC, HN_EXTRA, BRAND, NOADDR, SWAP)):
                import france as FR
                FR.C, qx = d["c"], d["K"] & 0xFFFFFFFF
                is2 = pc.starts_with(d["idx"].take(pa.array(qx)), "S2-").to_numpy(zero_copy_only=False)
                brand = noaddr = swap = None
                if (BRAND, NOADDR, SWAP) != (None, None, None):   # full scale: the test sources + the fr31 world
                    assert FRANCE == "fr31", "--brand / --noaddr / --swap run on the fr31 chain"
                    ids = lambda a, q: a.take(pa.array(q)).to_numpy(zero_copy_only=False)
                    s1i, sxi = ids(d["id1"], d["K"] >> 32), ids(d["idx"], qx)
                    brand = (BRAND, FR.brand_mask(s1i, sxi, M.CACHE / "fr31")) if BRAND is not None else None
                    noaddr = (NOADDR, FR.noaddr_mask(s1i, sxi)) if NOADDR is not None else None
                    if SWAP is not None:                # P46 learns its sibling words on P3's blocked list
                        learn = pd.read_parquet(B.P3 / f"cands_test_{d['c']}.parquet", columns=["s1", "sx"])
                        t_rule = T_POST or T_COUNTRY.get(d["c"], T_FRANCE_OVERRIDE)
                        swap = (SWAP, FR.swap_mask(s1i, sxi, np.asarray(d["p"], np.float64), t_rule, learn, SWAP))
                        del learn
                    del s1i, sxi
                acc = FR.complete(d["K"] >> 32, qx, is2, d["p"], am, acc, d["hn"]("s1"), d["hn"]("sx"), RESCUE, GC,
                                  HN_EXTRA, brand, noaddr, swap)
            if DROP_SIB_CANDS:
                acc = drop_sib_cands(d, acc)              # d["K"] loses the unaccepted decoys; acc realigned
            d["km"] = d["K"][acc]
            if DUP and world is None and (DUP is True or d["c"] not in unseen()):   # P54's duplicate step, last
                d["km"], d["K"] = dup_add(d, d["km"], d["K"], split)
            for k in ("p", "p_m2", "r", "hn", "pbl", "r_all", "p0", "r_lo", "v"):
                d.pop(k, None)
    else:
        parts, ts = rules_parts(split, world), {}
    fm, fc = open(out / "matching_results.tsv", "w", encoding="utf-8"), open(out / "candidate_pairs.tsv", "w", encoding="utf-8")
    fm.write("source1_entity_id\tmatched_entity_ids\n")
    fc.write("source1_entity_id\tcandidate_entity_ids\n")
    rows, seen, pred, cset, truth, groups = 0, set(), {}, {}, {}, {}
    for d in parts:
        c, id1, idx, ids, K, km, tr = (d[k] for k in ("c", "id1", "idx", "ids", "K", "km", "truth"))
        assert len(pc.unique(ids)) == len(ids) and not seen & {c}, "one row per S1"
        seen.add(c)
        pos = np.full(len(id1), -1)
        pos[codes(id1, ids)] = np.arange(len(ids))
        K, km = K[pos[K >> 32] >= 0], km[pos[km >> 32] >= 0]     # the scored S1's pairs (holdout: fold 0)
        assert np.isin(km, K).all(), "matches ⊆ candidates"
        ux = pc.unique(idx.take(pa.array(np.unique(K & 0xFFFFFFFF))))
        assert pc.all(pc.or_(pc.starts_with(ux, "S2-"), pc.starts_with(ux, "S3-"))).as_py() in (True, None), "S2-/S3- only"
        assert len(np.unique(km & 0xFFFFFFFF)) == len(km) or stage == "rules", "each S2/S3 matched to one S1 at most"
        tm, tc = d.get("tail", (np.zeros(0, np.int64), np.zeros(0, np.int64)))
        tails = []
        for t in (tm, tc):                               # {row: [sx ids in order]} of the duplicate step's appends
            tl = {}
            for a, b in zip(pos[t >> 32].tolist(), idx.take(pa.array(t & 0xFFFFFFFF)).to_pylist()):
                tl.setdefault(a, []).append(b)
            tails.append(tl)
        write_lists(fm, ids, pos[km[~np.isin(km, tm)] >> 32], idx, km[~np.isin(km, tm)] & 0xFFFFFFFF, tail=tails[0])
        write_lists(fc, ids, pos[K[~np.isin(K, tc)] >> 32], idx, K[~np.isin(K, tc)] & 0xFFFFFFFF, tail=tails[1])
        rows += len(ids)
        if tr is not None:
            truth |= tr
            groups |= dict.fromkeys(tr, c)
            pred |= sets(km, pos, ids, idx)
            cset |= sets(K, pos, ids, idx)
        print(f"{c}: {len(ids):,} S1, {len(K):,} candidates ({len(K) / max(len(ids), 1):.1f}/S1), {len(km):,} matches, "
              f"{len(np.unique(km >> 32)) / max(len(ids), 1):.3f} of S1 matched" + (f", t {ts[c]:g}" if ts else "")
              + f", peak {B.peak_gb():.2f} GB", flush=True)
        del d, K, km, pos
        gc.collect()
    fm.close()
    fc.close()
    if split == "test" and world is None:
        assert rows == N_TEST_S1 == pq.ParquetFile(WORK / "test_source1.parquet").metadata.num_rows, rows
        ids = pq.read_table(WORK / "test_source1.parquet", columns=["entity_id"]).column(0)
        assert pc.count_distinct(ids).as_py() == len(ids), "S1 ids unique across countries: no row written twice"
        if check:
            validate(out)
    if not truth:
        return {}
    tp = sum(len(pred.get(s, set()) & truth[s]) for s in truth)
    n_true = sum(map(len, truth.values()))
    hs = pd.Series(half(truth), index=list(truth))
    m = {**score(pred, truth, groups), "pair_precision": tp / max(sum(map(len, pred.values())), 1),
         "pair_recall": tp / max(n_true, 1), "cand_recall": sum(len(cset.get(s, set()) & truth[s]) for s in truth) / max(n_true, 1),
         "cands_per_s1": sum(map(len, cset.values())) / len(truth), "runtime_s": round(time.time() - t0, 1),
         "peak_gb": round(B.peak_gb(), 2)}
    for h in (0, 1):
        m[f"f05_half{h}"] = score(pred, {s: v for s, v in truth.items() if hs[s] == h})["f05"]
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}, flush=True)
    if log and world is None:
        try:
            from ledger import log as ledger_log       # the team ledger lives at the repo root, not in the package
        except ImportError:
            return m
        cfg = {"feats": "rules.feats (P2 to_latin + P1)", "cands": "P3 block.CFG ∪ rule-accepted (by_state)",
               "score": "rules.match accept set", "decide": "none",
               "sx": {"holdout": "full train corpus", "holdout-world": "holdout world sx"}.get(split, split)}
        if stage == "model":
            cfg |= {"score": f"M2 {M.BOOSTER.name} ({M.tag(M.BOOSTER)}) + CE {RR.CE.name} ({M.tag(RR.CE)}) "
                             f"band {RR.CFG['band']} w {RR.CFG['w']} on {list(RR.CFG['countries'])}",
                    "decide": f"argmax S1 per SX + t {ts}", "unseen_rule": f"P6 empty-rate match on M2 p at {T_M2}",
                    "recipe": RECIPE, "pruner_tau": M.TAU, "france": FRANCE, "stack": STACK, "ce_missing": CE_MISSING,
                    "blend_w_refit_half0": w}
        ledger_log("pipe", version=f"p9-{stage}", world=split + ("-dense" if split.startswith("holdout") else ""),
                   metrics={k: round(float(v), 4) for k, v in m.items()}, config=cfg)
    return m


use(RECIPE)                       # module defaults = the default recipe


if __name__ == "__main__":        # forkserver workers (rules.feats) re-import this file: keep the run under main
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "holdout", "holdout-world", "dev"], required=True)
    ap.add_argument("--stage", choices=["rules", "model"], default="rules")
    ap.add_argument("--out", type=Path, default=Path("output"))
    ap.add_argument("--recipe", choices=sorted(RECIPES), default=RECIPE)
    ap.add_argument("--iu", choices=["a10", "full", "fullzoo"],
                    help="India / US stage after the CE: a10 (P34) | full / fullzoo (P50's combiner; default: recipe's)")
    ap.add_argument("--france", choices=["m2", "idf144", "idfdom", "bge", "sib", "idf144_bge", "fr31"],
                    help="the unseen country's scorer (default: the recipe's); its t: --t-country <label>=<t>")
    ap.add_argument("--ce-missing", choices=["score", "fallback", "keep"], default="score",
                    help="band pairs without a cached CE score: score them (GPU), take #7's e5 blend (holdout proxy) "
                         "or keep their p (no GPU)")
    ap.add_argument("--probs-only", metavar="COUNTRY", help=argparse.SUPPRESS)   # france_variant's subprocess
    ap.add_argument("--no-validate", action="store_true",
                    help="skip the official validator (~4 GB RAM), e.g. when diffing against a validated staged file")
    ap.add_argument("--cap", help="S2,S3: at most that many matches per S1 (e.g. 5,6), the highest p kept")
    ap.add_argument("--drop-sibling-cands", action="store_true",
                    help="drop unaccepted pairs at a generator sibling offset from candidate_pairs.tsv")
    ap.add_argument("--tau-country", nargs="*", default=[], metavar="C=tau", help="per-country pruner tau")
    for flag, what in (("--rescue", "empty-S1 rescue"), ("--gc", "group completion"), ("--hn-extra", "hn-equal extras"),
                       ("--brand", "synthetic-brand recovery"), ("--noaddr", "address-less exact-name"),
                       ("--swap", "noise-word swap (its lens s_max)")):
        ap.add_argument(flag, type=float, help=f"unseen country's {what} bar on p (recipe's by default; 0 = off)")
    ap.add_argument("--veto", type=float, metavar="TAU", help="fr31's bge veto tau (recipe's file; 0 = off)")
    ap.add_argument("--t-post", type=float, metavar="T", help="fr31's post-rules (tier / bxneg / veto / cap) at T, the "
                    "decision at the country's t (default: both at that t)")
    ap.add_argument("--t-country", nargs="*", default=[], metavar="C=t",
                    help="fixed thresholds per country, e.g. France=0.88 (default: France and unseen labels by P6's rule)")
    a = ap.parse_args()
    use(a.recipe)
    IU = a.iu
    FRANCE, CE_MISSING, DROP_SIB_CANDS = a.france or FRANCE, a.ce_missing, DROP_SIB_CANDS or a.drop_sibling_cands
    CAP = tuple(int(x) for x in a.cap.split(",")) if a.cap else None
    bar = lambda x, dflt: dflt if x is None else x or None      # 0 turns a bar off
    RESCUE, GC, HN_EXTRA, T_POST = bar(a.rescue, RESCUE), bar(a.gc, GC), bar(a.hn_extra, HN_EXTRA), a.t_post or T_POST
    BRAND, NOADDR, SWAP = bar(a.brand, BRAND), bar(a.noaddr, NOADDR), bar(a.swap, SWAP)
    if a.veto is not None:
        VETO = (a.veto, (VETO or RECIPES["sub31_veto"]["veto"])[1]) if a.veto else None
    M.TAU_COUNTRY.update({k: float(v) for k, v in (x.split("=") for x in a.tau_country)})
    T_COUNTRY.update({k: float(v) for k, v in (x.split("=") for x in a.t_country)})
    if a.probs_only:                                  # inside france_variant's world: blocking -> X -> prune -> M2
        prepare("test", a.probs_only)
        M.split_probs("test", a.probs_only)
        sys.exit()
    assert not (a.stage == "model" and a.split == "holdout-world"), "holdout-world is the rules-only regression link"
    run(a.split, a.stage, a.out, check=not a.no_validate)
