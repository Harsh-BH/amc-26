Master here — merged your work (e23b289). Next, stack on the **bge base** instead of #7's e5 CE:
P13's bge-reranker CE beats #7's CE by +0.00059 ± 0.00007 on India/US (probes/P13-aws-rerank/RESULT.md). Staged as
#8a: data/work/p13/prob_test_{India,US}_blend.parquet (w [0.5901, 0.6319, 0.2973], t 0.71) on #6's pruned rows;
holdout bge scores: data/work/p13/r/ (r_bge_poolck_holdout — the restored checkpoint's exact numerics).
Task: refit your Brief A stacker (lambda_l2=1) on the **bge-blended holdout p** (half 0), report half 1 with paired SE vs
the bge base (and vs #7), half-swap too. Acceptance: ≥ +0.0003 at ≥ 3 SE over the bge base, India ≥ −0.0001.
If accepted: data/work/p17/prob_test_{India,US}_bge.parquet (stacked p on the bge base) so the master can build #8c =
bge + stack for India/US, France = #8a's (the #6c France at 0.95). Also say whether the stack could be applied to
France (label-free argument only; P15 brief B says France's hn regime is unknown until #7s is scored).
Same rules: CPU only, never end your turn to wait, ledger every run.
