# P60-qwen-bigdata: fine-tune Qwen3-Reranker-4B on folds 1–5 (~2.5× data) and rebuild France with it
Why: Qwen beats bge as France's out-of-domain CE in both LOCO directions (P38: +0.0014 / +0.0011 France-F0.5 at 8 SE), but it
was trained on the same limited folds as bge. P59 is retraining bge on folds 1–5 (fold 0 = holdout stays out). Do the same
for Qwen. Read probes/P38-qwen4b-fr (solution_p38.py LoRA DDP job, S3 weight cache amc26-p38/weights/, gate script, build38.py
clean() chain, RESULT.md), probes/P59-ce-bigdata (fold 1–5 channel builder: reuse its channel), probes/P50-idea-batch
(integrator; sub_clean / sub38_clean_p55). Own worktree; code in probes/P60-qwen-bigdata/.
Plan (AWS --profile harsh; quota 1 per instance type; P59 uses some g5/g6/g6e types, so check what's free):
1. LOCO pair first, if time allows: Qwen LoRA on US folds 1–5 → score India holdout band; India folds 1–5 → US. Gate with
   P38's gate vs P38's LOCO Qwen (same pairs): both directions ≥ +0.0005 France-F0.5. In parallel, start the pooled folds 1–5
   job (checkpoints to S3), so it's ready if the gate passes.
2. If it passes: score France band pairs + out-of-band pairs, rebuild France with build38 clean() on the current best rule set
   (currently sub_clean's chain: recipe t .99 / r .90 / gc .70 / hn .80 → gib → P48 veto → P54 rescue → cls80, no P46/P40)
   → hand to P50 (it runs P55 + dup and splices). Stage output/sub60 if you build it yourself.
Budget $45. Smoke first; in-job deadline so scoring always fits; MaxRuntime ≤ 4 h. Memory 1.5 GB local. DISK ~2.4 GB.
Never end your turn to wait. Deadlines: LOCO verdict by 10:30, France rebuild by 12:30 (hard stop 13:30).
