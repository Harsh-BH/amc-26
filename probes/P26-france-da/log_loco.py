"""One ledger row per LOCO result JSON.   .venv/bin/python probes/P26-france-da/log_loco.py loco_ui_a loco_ui_b ..."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ledger import log

for tag in sys.argv[1:]:
    d = json.loads((ROOT / f"data/work/p26/res/{tag}.json").read_text())
    m = {}
    for a, v in d["arms"].items():
        m |= {f"{a}_src": round(v["src_f05"], 5), f"{a}_tgt_rule": round(v["tgt_f05_rule"], 5), f"{a}_t_rule": v["tgt_t_rule"],
              f"{a}_tgt_oracle": round(v["tgt_f05_oracle"], 5)}
        for k in ("vs_a0_tgt_rule", "vs_a0_src"):
            if k in v:
                m[f"{a}_{k}"], m[f"{a}_{k}_se"] = round(v[k]["delta"], 5), round(v[k]["se"], 5)
        if "w_ess" in v:
            m[f"{a}_w_ess"] = round(v["w_ess"], 3)
    for k, x in (d.get("adv") or {}).items():
        m[f"adv_{k}_auc"] = round(x["auc"], 4)
    arms = list(d["arms"])
    good = any(v.get("vs_a0_tgt_rule", {}).get("delta", 0) >= 0.01 for v in d["arms"].values())
    log("exp", probe="P26-france-da", subproblem="france-da", name=f"loco-{d['src']}>{d['tgt']}-{'+'.join(arms)}",
        world="holdout-full-corpus-fold0-pruned", metrics=m,
        config={"arms": d["hp"]["arms"], "train": "src fold 2 (all pairs unless train=kept), early stop src fold 1 x0.25",
                "lgbm": "m2_big params, lr 0.1", "eval": "fold 0 of both countries on the #6-pruned list, argmax over all train S1; "
                "src t* = best F0.5, tgt t = P6 rule (tgt empty = src empty at t*)", "job": tag},
        hypothesis="reweighting / hn features / pruned-list context transfer to an unseen country (France proxy)",
        verdict="keep" if good else "drop", runtime_s=d["elapsed_s"], peak_gb=d["peak_gb"])
    print("logged", tag)
