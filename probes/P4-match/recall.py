"""Candidate recall on the holdout S1 (fold 0) at full density: P3 alone, P3 + rule-accepted pairs (what run.py
featurizes), and P3 + every rule-KEY pair (same (country, name_core), or website core = squashed S1 core; state
compatible as in rules._key_join), with how many pairs each union adds.

    .venv/bin/python probes/P4-match/recall.py      # after featrun.py feat train <c> for both countries
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "probes/P3-block"), str(HERE.parents[1])]
from featrun import OUT, TXT, WORK
from train import ids, xfiles


def main():
    from ledger import log
    res = {}
    for c in ("India", "US"):
        s1ids, sxids = ids("train", c)
        gt = pq.read_table(WORK / "gt_pairs.parquet")
        a = pc.index_in(gt.column("s1"), value_set=s1ids).to_numpy(zero_copy_only=False)
        b = pc.index_in(gt.column("sx"), value_set=sxids).to_numpy(zero_copy_only=False)
        fo = pq.read_table(WORK / "s1_folds.parquet")
        fold = fo.column("fold").to_numpy()[pc.index_in(s1ids, value_set=fo.column("entity_id").combine_chunks())
                                            .to_numpy()]
        ok = ~np.isnan(a)
        a, b = a[ok].astype(np.int64), b[ok].astype(np.int64)
        h = fold[a] == 0
        a, b = a[h], b[h]
        tk = (a << 32) | b
        cand = pd.read_parquet(OUT / f"cands_train_full_{c}.parquet", columns=["q1", "qx", "fold"])
        cand = cand[cand.fold == 0]
        ck = (cand.q1.to_numpy(np.int64) << 32) | cand.qx.to_numpy(np.int64)
        X = pd.concat([pq.read_table(p, columns=["q1", "qx", "fold", "blocked"], filters=[("fold", "==", 0)]).to_pandas()
                       for p in xfiles("train", c)])
        xk = (X.q1.to_numpy(np.int64) << 32) | X.qx.to_numpy(np.int64)
        # rule keys of the missed true pairs
        r1 = pd.read_parquet(WORK / "rules_s1feat_train.parquet", columns=["id", "country", "core", "state"])
        r1 = r1[r1.country == c].set_index("id").reindex(s1ids.to_pandas())
        rx = pq.read_table(WORK / f"rf_train_sx_{c}.parquet", columns=["core", "web", "is_web", "state"]).to_pandas()
        miss = ~np.isin(tk, ck)
        ma, mb = a[miss], b[miss]
        c1, cx = r1.core.to_numpy()[ma], rx.core.to_numpy()[mb]
        s1st, sxst = r1.state.to_numpy()[ma], rx.state.to_numpy()[mb]
        sq1 = np.char.replace(c1.astype(str), " ", "")
        web = rx.web.to_numpy()[mb]
        key = ((c1 == cx) & (c1 != "")) | (rx.is_web.to_numpy()[mb] & (np.char.str_len(web.astype(str)) >= 4) & (web == sq1))
        stok = (s1st == "") | (sxst == "") | (s1st == sxst)
        # pairs every key would add for the holdout S1: SX sharing the S1's core, minus those already candidates
        cnt = pd.Series(rx.core.to_numpy()).value_counts()
        hold = np.flatnonzero(fold == 0)
        core_h = r1.core.to_numpy()[hold]
        add_key = int(pd.Series(core_h[core_h != ""]).map(cnt).fillna(0).sum())
        m = {"n_true": len(tk), "recall_p3": float(1 - miss.mean()), "recall_p3_rule_acc": float(np.isin(tk, xk).mean()),
             "recall_p3_rule_key": float(((~miss).sum() + (key & stok).sum()) / len(tk)),
             "p3_per_s1": len(ck) / len(hold), "rule_acc_added_per_s1": (len(xk) - len(ck)) / len(hold),
             "rule_key_added_per_s1_upper": add_key / len(hold), "missed_key_share": float((key & stok).mean())}
        res[c] = m
        print(c, {k: round(v, 4) for k, v in m.items()}, flush=True)
    tot = {k: sum(res[c][k] * (res[c]["n_true"] if k.startswith("recall") else 1) for c in res)
           / (sum(res[c]["n_true"] for c in res) if k.startswith("recall") else 1) for k in res["India"]}
    log("exp", probe="P4-match", subproblem="candidates", name="recall-p3-union-rules", world="holdout-dense-full",
        metrics={f"{k}_{c}": round(float(v), 4) for c in res for k, v in res[c].items()} |
                {k: round(float(v), 4) for k, v in tot.items() if k.startswith("recall")},
        config={"p3": "block.CFG over all train S1", "rule": "rules.match by_state accepted / key = core or web"},
        hypothesis="rule-accepted pairs add the exact-key true pairs P3 misses",
        verdict="keep", runtime_s=0, peak_gb=0)


if __name__ == "__main__":
    main()
