# Experiment scripts behind the recorded change sets

`run.py` rebuilds `sub66_final` from the raw data. Two post-model steps then apply recorded change sets:
`final_build.py` applies `final_changes.parquet` (sub66_final → sub77), and `final_patch.py` applies
`final_patch.parquet` (sub77 → sub90u, the final submission). Every recorded row comes from the output of an experiment
script in this folder, and the parquets are those cached outputs. A few merges between script outputs ran as one-off
commands with the same one-owner / caps rule and no saved script. They are listed under "Merges without a saved
script".

- **Inputs.** The scripts read intermediates under `data/work/`. Some were written by `run.py` stages: the rules
  features `rf_*`, P3 blocking, the France chain frame `p31/fr31v0.1_t9868.parquet`, and M2 / CE scores. Others were
  written by the GPU scoring jobs whose entry scripts are included here (`solution_*.py`). The rest are staged
  submission folders under `output/`.
- **Layout.** Each folder holds the files of `probes/<same name>/` that are on the chain, copied verbatim. The only
  edits are the redactions listed at the end. Paths inside the scripts assume the original location `probes/<dir>/<file>`, run from the repo root. To
  re-run one, copy its folder back to `probes/`.
  - `P80-crossfit/solution_p80.py` and `solution_p80score.py` came from the repo root.
  - `P79-p73-takeover/final_build.py` has the same code as `src/final_build.py` (comments aside). It is here because
    `build79.py` imports it from its own folder.
- **P81 (third-party rules).** The rules come from the public MIT repository
  https://github.com/PundarikakshNTripathi/present-day-ber-amlc-2026 (`scripts/fr_posthoc{,2,3}.py`), credited here.
  `chainA.sh` / `chain.sh` run those scripts unmodified from a checkout of that repo. None of their code is copied
  into `src/`. Only their ADD rules were kept, and P88's filter removed part of those adds (below).

## Step → script

Rows are the pairs each step changed in the final line set. `+` = added, `−` = removed, as logged by the builder.

### final_build.py steps (`final_changes.parquet`; sub66_final → sub77)

| step | experiment | script(s) | what it does | rows |
|---|---|---|---|---|
| 67q | P67-fr-synth | `gen67.py gen` → `prep67.py` → `launch67.py` + `solution_p67.py` (GPU: bge CE tuned on synthetic pairs) → `ev67.py frscore` → `build67.py … 0.95 noswap` → `build67.py qgate`; the c > .95 cell and the Qwen r ≥ .5 gate were picked by `cell67.py` / `qgate67.py`; Qwen r from `P38-qwen4b-fr/solution_p38.py` | France adds from P55's combiner plus a column from a CE tuned on synthetic France pairs; adds with France Qwen r < .5 dropped | +2,793 |
| 70a | P70-fr-roles | `p70.py scores_fr words_nn` → `p70.py build … u=0.98` → `p70.py changes` | France word-role combiner adds at c > .98, P64 legal-form veto last | 1,991 in the file, +424 new after 67q |
| t975ce | P69-fr-loosen | `p69.py 0.975` (run.py's France stage at t .975) → `build69ce.py 0.99` → `splice.py` | the t .975 chain's extra pairs with bge r ≥ .99, SX free, caps 5/6 | 5,270 in the file, +4,904 (1 rejected) |
| (#75a) | P75-combine | `combine.py` on sub66_final with sub67q, sub70a, sub66_t975ce | union of the three add sets: one owner per SX, caps 5/6 | — |
| n54 | P74-n54-adapt | `p74.py` (run.py `sub66_final_n54`, writes the lens frame) → `build74.py output/sub75a output/sub75a_n54` | decoy-word pairs leave (`france.decoy_mask`, s > .25) | −2,120 |
| 71a | P71-fr-noiseword | `fr71.py` (France lens and pool) → `build71.py a` (`p71.py` holds the logic); `ana71.py` writes the India/US lens used as the analogue | noise-word insert/swap copies at the identical house number | 4,554 in the file, +4,523 |
| 41 | P70-fr-roles | `p41_on.py` (P41's `p41.py` brand rule on sub66_final) | bare synthetic brand joining the S1 that holds the unique hn+street+city key | 5,834 in the file, +5,824 |
| 73v / 73q / 73x | P73-fr-independent, P79-p73-takeover | `gen73n.py` (v3 synthetic, P71 noise lists) → `prep73.py` + `prep73n.py` → `launch73.py` + `solution_p73.py` (GPU: electra CE fine-tuned without any India/US label) → `ev73.py frscore p67+ele4n` → `build79.py p67+ele4n` (build67 swap @ .95, rows split by Qwen r); `inc79.py` picked the Qwen gate | France-independent CE combiner: vetoes (73v) and adds (73q Qwen-gated, 73x the rest) | 73v −159, 73q +167; reach the final through #81 below (sub77 itself is `--upto 41`) |
| (rows) | P78-final-pkg2 | `extract.py` | re-derives the 67q…41 rows from the staged folders → `final_changes.parquet`; `build79.py` appends the 73v/q/x rows | 22,562 + 3,072 rows |

### final_patch.py change sets (`final_patch.parquet`; sub77 → sub90u)

`final_patch.parquet` holds every line of sub90u that differs from sub77: 28,969 matching lines and 7,039 candidate
lines. The intermediate submissions sub78g … sub89 were each built from sub77 by the scripts below, in this order.

| change set | experiment | script(s) | what it does | rows |
|---|---|---|---|---|
| xlf | P70-fr-roles | `xlf_build.py sub77 sub77_xlf` (imports P66 `loco66.py` / `prof.py`, P68 `fr_key.py`, P64 `lf.py`) | address-less SX whose name core equals a unique France S1 core, and whose full name differs only by the legal form | +2,019 |
| groupe | P71-fr-noiseword | `build71.py b` → sub71b; its groupe swaps grafted on #77 (sub78g), then on sub77_xlf (sub80) | one S1 word swapped for "groupe" at the identical house number | +4,062 |
| P73 #79q | P73 / P79 | the 73v / 73q rows above, grafted on sub80 (sub81) | France-independent CE vetoes plus Qwen-gated adds | −159 / +166 (of 167; 1 was already present) |
| P76 R3 | P76-segment-hill | `build_r3.py output/sub81 output/sub81_r3` (reads the `p69.py 0.9868` chain frame) | S1 with exactly 2 SX-argmax pairs lose base-stage pairs with p in [.9868, .9925), unless the pair is its source's last member | −263 |
| P80 (legal form kept) | P80-twin-research | `frcount.py output/sub81_r3` → `build_lf.py … p80_lfkept` → `P75-combine/combine.py sub81_r3 sub82 +p80_lfkept` | address-less SX that keeps its legal form and whose full name matches exactly one France S1 | +1,473 |
| broad type-word veto | P70-fr-roles | `tw_audit.py sub81` → `data/work/p70/tw_sub81.parquet` (words compagnie / institut / pharmacie taken) | SX adds a type word unrelated to the S1's words | −132 |
| sub68fr | P68-iu-retrieval | `fr_key.py build output/sub66_final output/sub68fr` | free SX whose (name core, house number, city) key belongs to exactly one France S1, pair never a candidate; P64 veto | +814 of 835 (21 owner/cap rejections) |
| never-candidate noise swaps | P70-fr-roles | `tw_audit.py sub81` → `data/work/p70/noise_nc_sub81.parquet` | noise-suffix swaps (développement / groupe / fils / associés) at the same house number, never in the candidate list | +661 |
| narrow P73 veto | P73-fr-independent | `prep_fc73.py` (confident-row frames) → seed jobs via `launch73.py` / `solution_p73.py` → `vt73.py build -4.5 <jobs> cls=23`; tau from `vt73.py gate`, class split from `report73.py` | confident France pairs at a sibling-offset house number (hn_rel 2/3) that the independent CE scores below logit −4.5 | −65 of 69 listed (4 kept so that no line empties) → sub84 |
| #86 India/US | P70-fr-roles | `traces.py` + `anchors.py` (features) → `ta_ship.py fit` / `probs` / `build sub84 sub86` (P44 / P50 combiner modules) | India/US lines re-decided by the fullzoo combiner plus 7 generator-trace and 5 anchor columns | India +654 / −605, US +548 / −365 |
| P88 (P81 passes 1+2) | P81-rival-posthoc, P88-p81-graft | `chainA.sh` (the third-party ADD rules only, on our file) → `apply_cs.py make` → `cs81A12.parquet`; variant A chosen by the holdout test `mkworld.py` + `evalph.py` (`chain.sh` = the full chain it was compared against); `profile.py` (P31 p, P70 `street_fr.py` street classes) → `cs88.parquet` → `apply_cs.py apply output/sub86 cs88.parquet output/sub88` | France adds (France rows only) from the third-party post-hoc rules; veto re-adds and different- / typo-street pairs dropped | +10,689 (203 + 405 dropped) |
| #89 (pass 3) | P81, P88 | the same chain, pass-3 adds (`cs81A` minus `cs81A12`), same filter → `cs89_p3.parquet` → `apply_cs.py apply` | the same, for the third pass | +5,429 |
| P80 US owner re-decision | P80-crossfit | `solution_p80.py` (stage 1: M2 cross-fit) → `solution_p80score.py` / `score_test.py` (test p) → `build_ce.py` + `launch_ce.py` + `solution_p80ce.py` (CE cross-fit) → `stagec.py frame / fit / test / ce_r / ce_cs / fit --ce / blend --ce / w1_test` → `build_sub.py sub89 sub90u … US`; `check_sub.py` verifies | stage C on out-of-fold M2 + CE scores re-decides each US SX's owner, t .72, caps 5/6 | US 2,706 S1: +2,326 / −424 |

## Imported helper modules (copied because the scripts above import them)

| folder | module(s) | used for |
|---|---|---|
| P3-block | `block.py`, `full.py` | text builder in `gen67.clean` |
| P4b-rerank | `rerank.py` | CE text / frame helpers (`prep67`, `prep_fc73`, `build_ce`, `p13`) |
| P13-aws-rerank | `p13.py`, `solution_p13.py` | pinned job requirements for the launchers; the bge CE train/score job, packaged as `p13sol.py` into the P67 / P73 / P80 GPU jobs |
| P15-research | `sib_rule.py`, `hn_stack.py`, `hn_check.py` | sibling house-number mask |
| P20-france-patterns | `patterns.py` | name relation (`p70`) |
| P34-a10-stack, P36-iu-residual | `a10.py`, `base.py`, `gib.py` | India/US base features; synthetic-brand detector |
| P37-fr-complete, P40-generator, P41-fr-brand | `p37.py`, `rule40.py`, `p41.py` | France group completion, generator rule, brand rule |
| P42-fr-joint-tune, P48-fr-ce-all | `p42.py`, `solution_p42.py`, `loco48.py`, `build48.py` | France LOCO frames and decide; P48 veto mask |
| P44-meta-stack, P50-idea-batch | `ev.py`, `meta.py`, `te44.py`, `best.py`, `ctx.py`, `arm_c.py`, `solution_p44.py`; `e_ship.py`, `e_zoo.py`, `zoo50.py` | India/US combiner (`ta_ship`) |
| P55-fr-combiner, P61-fr-cotrain | `p55.py`, `p61.py` | France combiner; LOCO frames with agreement pseudo-labels |
| P64-legal-form-feat, P66-fr-noaddr | `lf.py`, `frtest.py`, `attach.py`; `loco66.py`, `prof.py` | legal-form conflict veto; address-less keys |
| P76-segment-hill | `dive.py` | per-S1 F0.5 (`stagec`) |

**Not included:**
- Repo-root tooling: `ledger.py` (writes log rows), `sm_launch.py` / `sm_train.py` (SageMaker submit and entry
  wrapper).
- P44's job-side S3 helper `p25.py`.
- Modules imported only inside functions the chain does not call: P4-match, P5-decide, P6-france, P28, and P15
  `hn_test_mix`.
- The third-party repository's own scripts. Get them from the URL above.
- Modules a few scripts import that are not in the zip: `sm_launch` (repo-root SageMaker launcher), `ber` (the
  third-party repository's package, needed by P81 `mkworld.py`), and P37 `ho`, P13 `big`, P4b `bench` (benchmark
  helpers the chain does not call). The third-party scripts also need `polars`, which is not in `requirements.txt`.

**Merges without a saved script.** Four steps ran as one-off merge commands with the same rule as `combine.py` /
`apply_cs.py`:
one owner per SX, caps 5/6, candidates extended. Their inputs are the script outputs named in the tables; no separate
script was saved for them.
- the groupe graft (sub78g), and the sub80 / sub81 unions;
- the three sub83 merges (the three type words are the ones `tw_audit.py`'s lens marks at s .20–.24 with n ≥ 400);
- the application of the 69-pair narrow veto list (sub84);
- the final P88 row selection from `profile.py`'s columns (`cs88.parquet` / `cs89_p3.parquet`). The rule was: drop
  pairs with P31 p ≥ .9868 or owned by the same S1 in sub77 / sub66_final, and drop the `street_fr` classes
  different-street and typo-street.

`final_patch.parquet` holds the sub77 → sub90u difference as attributed rows (file, s1, pos, sx, step, op). They
were derived from the staged chain by `P91-patch-provenance/provenance.py`, and `final_patch.py --summary` prints
the per-step counts.

## Redactions in the copies

- The S3 bucket name is now `os.environ.get("BUCKET", …)`, and the AWS profile is now `$AWS_PROFILE` (default
  chain).
- Absolute home and scratch paths are now repo-relative (`data/work/...`), `Path(__file__).parents[2]`, or
  `${SCRATCH:-/tmp/work}`.
- One comment prefix now reads `Note:`.

No logic was changed. Every `.py` compiles (`python -m py_compile`), and both `.sh` files pass `bash -n`.
