# SDD ledger — plan: docs/superpowers/plans/2026-09-25-amc26-master-probe-plan.md
Spec: docs/PROBLEM.md
Setup: Ruling: work on main, no worktree — user explicitly asked to commit+push to main as we go (teammates pull main) — cost if wrong: history on main instead of a branch.
Setup: Ruling: graphify-out/ gitignored, not committed — generated local tool cache — cost if wrong: user commits it later.
Setup: Ruling: TDD adapted to repo convention (assert self-checks in __main__): write _check() first, run it RED (NameError: function not defined), then implement GREEN — cost if wrong: none beyond style.
Pre-flight: T0.1 produces data/work/*.parquet -> T0.3 consumes train_source{1,2,3}.parquet + train_ground_truth.parquet: names match (f.stem). OK
Pre-flight: T0.2 produces score(pred,true,groups)->{f05,...} -> T0.6/T2.1 consume score(): match. OK
Pre-flight: T0.3 produces load_world(name)->(s1,sx,truth) -> _COMMON.md, T0.6, T2.1 consume same triple. OK
Pre-flight: T0.4 produces log(kind, **row) with REQUIRED exp={probe,subproblem,name,world,metrics,config} -> _COMMON.md example passes all 6. OK; T0.6 pipe row passes {version,world,metrics,config}. OK
Pre-flight: T0.5 run_probe.sh reads probes/<id>/session_id written by launch.sh. OK
Task 0.0: complete (deps installed, import check ok; Ruling: ai4bharat-transliteration left out of base install — heavy deps, P2 evaluates it in its own probe — cost if wrong: P2 installs it)
Task 0.2: complete (commit 899357e, tests: metric.py self-check → ok; RED NameError first)
Task 0.4: complete (commit 899357e, tests: ledger.py selftest → ok; RED NameError first)
Task 0.1: complete (commit 899357e, tests: io_tsv.py --check → ok; convert → 7 files, all row counts == wc -l - 1)
Task 0.3: complete (commit 098b3e3, tests: worlds.py self-check → ok; RED NameError first; peak 4.78GB 34s)
Task 0.5: complete (commit 5396c40, tests: P0-smoke probe → exit=0, RESULT.md 7 headings, 1 ledger row; RED 'No such file' first)
Task 0.6: complete up to upload (commit ba3487b, tests: holdout f05=0.1775, validator → PASS); upload pending user
Task 0.7: drafted Google Form questions for the user (in chat); submission pending user
Wave A: Ruling: runner OOMPolicy=continue + resume mode after P2 died (exit 143, scope oom-kill) — reproduced RED 143 / GREEN parent survives — cost if wrong: none; P1/P3 still on old policy until they finish
Wave A: Ruling: DONE now records result:<subtype> vs CRASHED from the stream's last event — exit codes lie on scope teardown (P1 showed exit=0) — cost if wrong: none
Wave A: Ruling: probe tool shells get oom_score_adj=1000 via SHELL=bash + BASH_ENV — kernel OOM killed P2's CLI — RED parent 137 / GREEN child -9 — cost if wrong: none
Wave A: P2-script complete (result:success, 13 ledger rows). Master holdout check: exact folded-name hit 0.9522 on 54,843 cross-script pairs (dev 0.949). Promote translit.to_latin/fold/oov_flag at MERGE-1.
Wave A: Ruling: short generic fixture strings in probe self-checks may coincide with dataset names (P2 translit.py) — allowed; bulk/raw rows never — cost if wrong: 5 generic names in a private repo, same code goes to Amazon in the zip.
Wave A: Ruling: freed 0.4 GB (uv cache prune, partial LaBSE blobs, my scratch venvs); disk still 96% (8.1 GB free) — left user's Downloads untouched — cost if wrong: full-scale runs may need space later; ask user about the 1.1 GB dataset zip.
Wave A: Ruling: run_probe.sh body in main() + atomic mv; watch.sh detects dead-without-DONE — P3 lost DONE to in-place edit — P3 resumed at 3.5G (RAM gate: 5.4G free) — cost if wrong: P3 full-corpus eval must chunk tighter
NIGHT (user asleep 04:51 IST, back in the morning; wants the pipeline and everything done): autonomous run.
  Sleep inhibitor active 12 h (systemd-inhibit sleep:idle:lid). Heartbeat every ~25 min + probes/watch.sh.
  Order: (1) finish Wave A (P1, P3, P7) → collect, verify, commit
         (2) submission #2 candidate = P1 exact-key rules (holdout 0.785) → test run → validator → output/sub2/
         (3) MERGE-1 → promote translit/normalize/block into src/
         (4) Wave B P4-match + P6-france → MERGE-2; Wave C P5-decide (+P4b-rerank if RAM/GPU allow) → MERGE-3
         (5) Phase 2: src/run.py v1 end to end, test_pipeline, holdout → test → validator → output/subN/
         (6) Phase 3 improver probes until morning; keep the best validated submission staged
         (7) Morning report: SUBMISSIONS.md (upload queue with holdout scores) + summary. Every step committed+pushed.
  No leaderboard uploads possible overnight (user uploads); day-1 slots are open until 23:59 IST 25 Sep.
Wave A: P1-taxonomy complete (result:success, 26 rows). Promoted normalize.py; P2 translit.py/pairs.py promoted.
Wave A: Ruling: all pipeline scoring is DENSE (holdout S1 scored while all 2.2M train S1 compete) — P1 showed the plain world hides chain collisions (name-only 0.684 plain vs 0.431 dense) and absent parents inflate FPs — cost if wrong: none, dense is strictly closer to test.
Wave A: Ruling: test_rules guard moved to dense (f05>=0.7831, precision>=0.98) — plain-holdout precision 0.974 was an artifact of absent parents; dense 0.9937 — cost if wrong: none.
Wave A: Ruling: place_freq fitted on train S1 only (not test S1) pending Google Form Q4 — cost if wrong: some France city_key agreement.
Wave A: s2-rules dense holdout F0.5 0.8201, precision 0.9937 → submission #2 candidate (test run in progress).
05:22 Claude Code's memory-pressure reaper stopped ALL master background tasks: sub2 test run (no output),
      systemd-inhibit (laptop may suspend), probes/watch.sh x2, heartbeat. Harness instruction: do not restart
      them on my own; report and restart only when the user asks. Complied. tmux probes P3 + P7 unaffected (alive).
      Ruling: no new probes (Wave B) launched tonight — memory just hit critical and P3's full-corpus eval is the
      likeliest casualty; Wave B briefs to be refreshed and launched at the user's go — cost if wrong: ~4-6 h of
      overnight probe time lost.
      Cause of the spike: rules.py test with 5 forked workers (~2.5 GB RSS each, much shared) + P3 full-corpus + P7.
MERGE-1 written 11:20. Wave A complete: P1, P2, P3 success; P7 crashed (partial result adopted). rules.feats forkserver fix: test_rules GREEN 0.8201/0.9937.
11:40 User: keep the original plan — v1 after P4 + P5 + P6 (MERGE-2/3), P4b stays in Wave C; the 'ship v1 early / drop P4b' proposal was declined.
12:18 P8 finding: rules full-corpus F0.5 0.7708 / precision 0.912 vs dense-world 0.8215/0.994 — dense-world under-samples distractors. Ruling: full-corpus (all train S1 + all train S2/S3, holdout S1 scored) is the yardstick from now on — cost if wrong: none, it is the faithful mirror of test.
P8 merged at f133388 ; suite green
13:40 User approved: if P4b passes the gate (+0.003 full-corpus F0.5 over P4's final model), resume P4b automatically with probes/P4b-rerank/RESUME_expand.md (folds 2-9, 2 epochs).
13:41 User cancelled the P4b expansion queue: RESUME_expand.md deleted; no automatic expansion training. P4b's current (original-brief) run continues.
13:42 User: let P4b's current run finish (no expansion).
14:20 P4 LightGBM verified CLEAN by 2 independent Opus audits (workflow wf_3c3e0791-7a8): recomputed full-corpus holdout
      F0.5 0.981413 (exact match); no label leakage (65 label-free features); train fold 2 / early-stop fold 1 / score fold 0;
      threshold tuned on holdout but optimism <= 0.00003 (fold-3-picked t gives 0.98139). Risk (not leakage): 87% of holdout
      FPs are orphan SX; test SX/S1 5.8 vs train 4.7 -> test precision may drop. MERGE-2 to-do: orphan-density stress test +
      P6 profile threshold. Improvement idea: train on folds 2-9 (P4 used fold 2 only).
14:22 User away: 'take the best decisions and complete the work'. Autonomous. Milestone updates instead of 5-min; heartbeat 10 min. Plan: collect P4/P5/P6/P4b -> MERGE-2 (careful, suite after each) -> v1 (P4 LGBM + P4b CE + P5 decide + P6 France t) -> full-corpus holdout + orphan stress -> test -> validator -> stage #3 -> improve (folds 2-9) -> draft doc.
14:58 Submission #3 staged: P4 LGBM (0.9814 full-corpus, audited), t=0.71, France t=0.83 (P6), validator PASS. Ruling: France t=0.83 from P6 profile rule (LOCO: within 0.003 of oracle; unseen country needs stricter t) — cost if wrong: small France F0.5 loss, curve flat.
15:40 #4 staged (M2 + CE India/US t0.70, France M2 t0.82): holdout 0.9881, validator PASS; France profile = P6 prediction.
      P5 blend refit: stack +0.0001 < bar -> dropped; #4 keeps the plain threshold. Remaining loss = matcher recall (0.0089 partial).
      Next: (1) when P4 exits, merge its code; (2) integration probe: run.py --stage model reproduces #4 end to end (final package);
      (3) improvements day 2: P4 on folds 2-9 + CE refit, P3 recall-max variant (0.9931@22.6), France normalisation rules.
15:56 #5 candidate test (P5 bench, tune half0 / report half1): #4 M2+CE 0.9881 (reproduced), stage2 alone 0.9821,
      stage2+CE refit 0.9886 (+0.0005). Ruling: keep #4 (single-stage M2 + CE) — +0.0005 < 0.002 simpler-wins bar, and
      P9 integrates M2 — cost if wrong: ~0.0005 F0.5.
      Improvement round: resume P4 to retrain single-stage M2 on more train folds; gate >= +0.001 and > 3 SE over 0.9881.
17:12 P4 improvement: M2 on folds 2-5 + CE = 0.9885 vs #4 0.9881 (+0.00040 +- 0.00006, 6.7 SE); no-CE 0.9814 -> 0.9834.
      Ruling: ADOPT M2-big despite the 0.001 bar — the bar guards against complexity for noisy gains; this is the same
      recipe (only more training folds), zero added complexity, gain is certain (6.7 SE) — cost if wrong: ~80 min compute.
      Resume P4: big.py test + blendtest India/US -> #5. P9 to switch train_matcher to folds 2-5 after it finishes.
18:15 P9 merged (4213515): run.py --stage model reproduces #4 byte for byte, holdout 0.9884, M2 retrain 0.9813; suite green.
      #5 staged (b271c80): m2_big (folds 2-5) + CE refit, t 0.69, France 0.78 (P6 rule) — 0.9885 report half, validator PASS.
      Doc final (a6d08dd): only team name/members left for the user. P9 resumed: make run.py reproduce #5 byte for byte.
