"""#7 holdout with a retrained cross-encoder in place of P11's ce_s2P.pt: does `rerank.py train epochs=2` (src, P4b's
in-loop recipe) reproduce P11's checkpoint? Same pipeline, same pruner, blend w and t (sub7 recipe); only rerank.CE
changes. Its r cache is keyed by the new checkpoint's hash, so P11's scores are untouched.

    .venv/bin/python probes/P9-integrate/ce_retrain_eval.py data/work/p9/ce_retrain.pt
"""
import sys
import time
from pathlib import Path

sys.path[:0] = ["code/business_entity_resolution/src", "."]
import rerank as RR
import run
from metric import score

if __name__ == "__main__":
    ce = Path(sys.argv[1])
    t0 = time.time()
    run.use("sub7")
    RR.CE = ce
    import csv
    import pandas as pd
    m = run.run("holdout", "model", Path("output/p9ce"), log=False)
    sub = pd.read_csv("output/p9ce/holdout/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False,
                      quoting=csv.QUOTE_NONE)
    pred = {a: set(filter(None, b.split(","))) for a, b in zip(sub.source1_entity_id, sub.matched_entity_ids)}
    s1, t = run.holdout_s1()
    truth = run.truth_of(list(s1.entity_id), t)
    h = pd.Series(run.half(list(truth)), index=list(truth))
    f = {f"f05_half{k}": round(score(pred, {s: v for s, v in truth.items() if h[s] == k})["f05"], 5) for k in (0, 1)}
    f["f05"] = round(score(pred, truth)["f05"], 5)
    ref = {"f05": 0.98881, "f05_half0": 0.98894, "f05_half1": 0.98869}          # P11's ce_s2P.pt, same pipeline
    print("retrained CE:", f, "P11 ce_s2P:", ref, flush=True)
    from ledger import log
    log("exp", probe="P9-integrate", subproblem="integration", name="ce-retrain-2ep-holdout", world="holdout-dense",
        metrics=f | {f"{k}_p11": v for k, v in ref.items()} | {"diff_half1": round(f["f05_half1"] - ref["f05_half1"], 5),
                                                             "pair_precision": round(m["pair_precision"], 4),
                                                             "pair_recall": round(m["pair_recall"], 4)},
        config={"ce": str(ce), "recipe": "sub7 (pruner, m2_big, w [0.617, 0.666, 0.261], t 0.71)",
                "train": "src/rerank.py train epochs=2 (P4b recipe, in-loop tokenisation, char-length buckets)"},
        hypothesis="rerank.train(epochs=2) reproduces P11's 2-epoch CE within noise (+-0.0001 on the report half)",
        verdict="keep" if abs(f["f05_half1"] - ref["f05_half1"]) <= 0.0002 else "inconclusive",
        runtime_s=round(time.time() - t0), peak_gb=0)
