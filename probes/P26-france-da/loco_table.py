"""Markdown table of the LOCO result JSONs (data/work/p26/res/loco_*.json).   .venv/bin/python probes/P26-france-da/loco_table.py"""
import json
from pathlib import Path
R = Path(__file__).resolve().parents[2] / "data/work/p26/res"
print("| job | arm | trees | ESS | src F0.5 @t* | Δsrc vs a0 | tgt F0.5 rule @t | Δtgt rule vs a0 (±SE) | tgt oracle @t | Δtgt oracle | tgt @src t* | hn gain share |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for f in sorted(R.glob("loco*.json")):
    d = json.loads(f.read_text())
    for a, v in d["arms"].items():
        dr, ds, do = v.get("vs_a0_tgt_rule", {}), v.get("vs_a0_src", {}), v.get("vs_a0_tgt_oracle", {})
        fm = lambda x: f"{x['delta']:+.4f} ± {x['se']:.4f}" if x else "—"
        print(f"| {d['src']}→{d['tgt']} {f.stem.split("_", 1)[1]} | {a} | {v['trees']} | {v.get('w_ess', 1):.2f} | {v['src_f05']:.4f} @{v['src_t']:.2f} | {fm(ds)} | "
              f"{v['tgt_f05_rule']:.4f} @{v['tgt_t_rule']:.2f} | {fm(dr)} | {v['tgt_f05_oracle']:.4f} @{v['tgt_t_oracle']:.2f} | {fm(do)} | "
              f"{v['tgt_f05_at_src_t']:.4f} | {v['hn_gain_share']:.2f} |")
    if d.get("adv"):
        print(f"| {d['src']}→{d['tgt']} {f.stem.split("_", 1)[1]} | adversarial | " + "; ".join(f"{k}: AUC {x['auc']:.4f}" for k, x in d["adv"].items()) + " |||||||||| |")
