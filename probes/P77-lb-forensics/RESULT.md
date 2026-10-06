# P77-lb-forensics — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
- 12 LB deltas priced pair by pair (sub5 → sub77; sub76 itself was never scored, the 0.988292 upload is **sub77**). The LB pays for France structural adds far above what their own Qwen/g scores predict. The 66→77 set (18,014 adds, −1,666 N54) is **~0.90–0.95 precise jointly** (scores say 0.65). The P71a noise-word swaps at Qwen 0.17, P41 brands at 0.62 and N54 removals at 0.80 are all **rejected by the LB**.
- The only punished France set is **sub54's 28,238 adds that sub77 does not carry**: 0.38–0.52 precise vs a 0.69 break-even, in two independent LB equations (gc70→54 and 54→66). They are org-word swaps (comité / club / amis / cie / compagnie / société …), same-hn multi-word names, and missing-hn same names. Do not re-add them.
- Coarse label-free classes (hn × name relation × word role) are **not identified**: good and bad sets share the same classes. The signal is at the word level (P71's lens s). No new cheap rule has direct LB support, so **I built none**. Test patterns on sub77: France matches India/US on every S1 segment (empty 6.05% vs 5.78%) and on addressed S2/S3 (assigned 0.58–0.60 vs 0.57–0.60). The one outlier is **address-less S2/S3: France assigns 0.35 of 43,078, India/US 0.54–0.62**. That is ≈ 10k plausible missed pairs, but P66's LOCO puts the leftover pool at 27–42% precision, so there is no cheap rule there.

## Recommendation
1. Keep **sub77** as the base. Every component of 66→77 is either LB-supported as a group or too small to matter. Joint implied add precision is 0.954 with the N54 removals at 0.35 true (0.904 if they were all false).
2. **Do not re-add sub54-era France adds** outside sub77. The LB puts them at 0.38–0.52 (below). That covers org-word swaps at the same hn (the N54 decoy class), the P46-style tier outside the développement word, and hn-miss same-name adds.
3. The next France lever should be **word-level**, not class-level. `probes/P71-fr-noiseword/work/lens_France.csv` has, per swap/insert word, e (same-hn share) and s (sibling share). The LB now validates s ≤ 0.05 swaps (sub:developpement s .043) as a group, and the decoy side (N54, s > .25) as removals. Untested middle: sub:groupe (s .091, e .878, 4,063 free pairs = `output/sub71b` − `output/sub71a`), sub:compagnie (s .20), sub:cie (s .10). Upload sub77 + groupe swaps only as a measured test. By the sibling-contamination estimate (a pure decoy word lands at D=0 about 1/32 as often as at a sibling offset: ins:participations 176 vs 5,634), groupe's D=0 decoy share is ≈ 593 / 32 / 5,747 ≈ 0.3%. So the prior is favourable. The LB has not seen it.
4. Stop trusting Qwen / bge vetoes on France same-hn noise-word pairs. The LB rejects Qwen's 0.17 for P71a outright (best case 470e-6 vs 901e-6 observed). Any existing CE veto whose removed France pairs are identical-hn noise-word swaps is suspect. P48/P55 vetoes are in the chain; their removed sets were not re-audited here (open question 2).
5. For **diag_frblank** (sub66_final with France blanked): with India/US byte-identical, LB(sub66_final) − LB(diag) = w_fr · (F_fr − s_fr), where s_fr is France's empty-and-singleton share. D1 on sub5 gave 0.130854 ⇒ F_fr(sub5) ≈ 0.93 at w 0.1498. From the transitions since, France should have gained ≈ +0.005 LB (≈ +0.035 France F) ⇒ expect diag_frblank ≈ 0.987391 − 0.1497 · (0.968 − 0.057) ≈ **0.851** (0.849–0.853 across the India/US priors). A diag well below 0.851 means France is better than that and India/US worse, and vice versa. Forward me the score and I'll add the equation.

## Evidence
**A. Per-transition implied France precision** (`fit.py mc`). Each pair is priced at its S1's state: gain a if true, loss b if false. The simulated state has kept pairs 0.95 true, 0.35 unfound members per S1, and a filled S1 is a singleton w.p. 0.86. India/US parts come from SUBMISSIONS' holdout LB-eq (6c −0.0002, 9ab +0.0010, 34_veto +0.0002, 54 +0.00072). The minor action is at a fixed prior (adds .85 / removals .35). SE = Bernoulli noise on a public share of 0.5.

| transition | ΔLB | India/US prior | France change solved for | implied π | break-even π |
|---|---|---|---|---|---|
| sub5 → d2 | +0.001007 | 0 | remove 29,034 | 0.487 ± 0.006 | 0.654 |
| d2 → d4 | +0.000118 | 0 | remove 35,003 | 0.635 ± 0.005 | 0.650 |
| d4 → sub6c | −0.000188 | −0.0002 | remove 202 | 0.39 ± 0.07 | 0.652 |
| sub6c → sub9ab | +0.003118 | +0.0010 | add 32,859 (+26,008 removes) | 0.702 ± 0.005 | 0.646 |
| sub9ab → sub31 | +0.001528 | 0 | add 36,547 (+10,860) | 0.769 ± 0.004 | 0.663 |
| sub31 → sub34_veto | +0.000047 | +0.0002 | remove 6,140 | 0.808 ± 0.009 (veto removed good pairs) | 0.680 |
| sub34_veto → rescue90 | +0.000313 | 0 | fill 1,205 empty S1 | 0.723 ± 0.018 | 0.475 |
| rescue90 → gc70 | +0.000344 | 0 | add 10,084 (missing source) | 0.823 ± 0.006 (scores: 0.67) | 0.683 |
| gc70 → sub54 | +0.000381 | +0.00072 | add 41,854 (+2,454) | **0.617 ± 0.004** (punished) | 0.689 |
| sub54 → sub66_final | +0.000706 | 0 (India/US byte-identical) | remove 40,888 (+6,245 adds) | 0.618 (removal gained) | 0.681 |
| sub66_final → sub77 | +0.000901 | 0 (France only) | add 18,014 (−1,666 N54) | **0.954 ± 0.003** (scores: 0.65) | 0.699 |

The ±SE is LB noise only. Model error (pricing state, India/US priors) is larger; see Risks. Precision levels move ±0.05–0.1 with the pricing state, but the ordering does not.

**B. 66→77 by component** (`bounds76.py`, `c76.py`). The ceiling (all adds true, all N54 removals false) is +1,196e-6 (+1,093 to +1,309 across 4 pricing states). Observed +901e-6, SE 9e-6.

| component | pairs | Qwen/g prior π0 | hn eq | LB lower bound on π (others perfect, 2 SE, 4 states) | component at its own prior |
|---|---|---|---|---|---|
| P41 brand | 5,824 | 0.624 | 1.00 | 0.50–0.78 | **rejected** (851 < 901) |
| t975ce | 4,674 | 0.982 | 0.81 | 0.38–0.72 | possible |
| P71a dev swap | 4,523 | 0.174 | 1.00 | 0.48–0.77 | **rejected** (470 ≪ 901) |
| P67q | 2,593 | 0.932 | 0.35 | 0.01–0.55 | possible |
| P70a | 400 | 0.505 | 0.50 | 0 | possible |
| N54 removal | 1,666 | 0.796 | 0.99 | ≤ 0.47–1.0 true | **rejected** (855 < 901) |

Joint precision of the adds: 0.904 / 0.954 / 1.018 (infeasible) for N54's removed-pair precision 0 / 0.35 / 0.80.

**C. sub54's France adds** (`mix.py`, `rest54.py`, `words54.py`). Of sub54's 41,854 France adds, 13,616 are in sub77; 9,543 of those were reverted at 54→66 and re-added at 66→77. The other 28,238 have implied π **0.38–0.50** from gc70→54 (India/US prior 0.0005–0.0009, P48/P55 veto removals at 0.2–0.5). The 31,345 removed at 54→66 and never re-added come out at **0.52** from the second equation. Break-even is 0.687. Top classes of the rest: eq-hn multi-word 8,719 (π0 .68), eq-hn one-word 8,619 (.55), missing-hn same name 2,990 (.50), eq-hn same name 2,935 (.76), ±1/2-hn same name 1,241 (.28). P71 lens words among the rest: sub:comite 635, club 445, amis 442, cie 356, compagnie 320, societe 280, amicale 265, sante 253, lycee 226, section 193, federation 155. Among the kept 13,616: sub:developpement 3,082, compagnie 208, cie 92, associes 43.

**D. Class-level logit fit** (`lfit.py`): failed. π = σ(logit π0 + d_hn + d_name + d_action + d_role), ridge τ 1.5, 40 bootstraps. It cannot reproduce gc70→54 (predicts +277e-6, observed −339e-6) or 66→77 (+171e-6 against +901e-6). Every shift's 10–90% bootstrap band crosses 0 except act=add (+1.14, 0.1–1.6) and act=rem (−0.72). So adds are more precise than their scores say and removals less, but no coarse class is identified.

**E. Test-set patterns, label-free, on sub77** (`patterns.py`, `noaddr.py`; raw examples only in `data/work/p77/examples_noaddr.txt`, which is gitignored).

| | France | India | US | train law |
|---|---|---|---|---|
| S1 | 259,452 | 809,986 | 663,106 | |
| predicted empty | 6.05% | 5.78% | 5.79% | 5.58% singletons |
| matches per S1 | 3.22 | 3.38 | 3.39 | 3.46 true |
| one-sided groups (only S2 or only S3) | 17.0% | 15.2% | 15.1% | |
| S2 / S3 per S1 | 2.71 / 2.82 | 2.86 / 2.97 | 2.82 / 2.93 | 4.7 total |
| S2 assigned (addressed / address-less) | 0.581 / **0.349** | 0.574 / 0.608 | 0.580 / 0.620 | address-less SX 97.7% owned (P66) |
| S3 assigned (addressed / address-less) | 0.597 / **0.350** | 0.587 / 0.545 | 0.596 / 0.621 | |
| content-duplicate SX assigned | 0.98 / 0.99 | 1.00 / 0.99 | 0.99 / 0.99 | |

- France's S1 segments (house number present, name twins, token count, legal form) all sit within 0.5 pt of India/US on empty rate. None deviates.
- France's lower matches per S1 is explained by fewer SX per S1 (5.53 vs 5.8). Addressed SX are assigned at India/US's rate.
- **Address-less France SX**: 43,078 rows, assigned 0.349. Of the unassigned ones, 25,774 have a France candidate. Their argmax S1 name relation: exact core 21,203, sub1 1,757, add1 1,428, drop1 657, typo 609. g: < .3 12,783, .3–.7 5,532, ≥ .7 7,459. With g ≥ .3: exact core + unique S1 core 2,905, exact core + twin S1 cores 8,294, sub1 + unique core 603. Only 5.2% of their argmax S1 are empty.
- The twin-core share is what makes this hard: which S1? P66's LOCO puts unaccepted address-less argmax pairs at 27–42% precision, and only `xfc` (≈ 0.98 / 0.89) held. The 2,905 exact-core unique-core pairs with g ≥ .3 are the one unexplored slice (legal form differs, or the full name is not unique). See open question 4.

Ledger rows: `P77-lb-forensics/*` (6 rows).

## What failed
- The class-level regression, both ridge-linear (`ana.py`: shifts ±40, meaningless) and logit (`lfit.py`, above). 11 LB points cannot separate ~15 coarse classes, and the classes do not carry the signal.
- The first pricing model (kept pairs all true, S1 otherwise complete) could not produce 66→77's gain at all (implied π > 1 before P41 was attributed). The simulated state replaced it.
- My word-role tag (`w_s` from `p70.role_frame` run on the changed pairs only) is computed on the wrong population. Use P71's `lens_France.csv` (all France argmax pairs), as `words54.py` does.

## Interfaces
- `collect.py`: streams every scored matching file (S3 via rclone, or local), producing `work/union.parquet` (key = s1 << 31 | is_S3 << 30 | sx, 5.9M pairs, 13-bit membership mask) and `work/files.json`. Incremental: new files append a bit. About 40 s per file, 1.3 GB peak.
- `tag.py`: label-free tags per changed pair (216k): hn relation, P20 kind, P64 lf_k, noaddr, gib, France g / bge r / Qwen r. `work/tagged.parquet`, 4 min, 1.75 GB.
- `fit.py [mc [pi_kept lam]]`: per-transition pricing and implied π, producing `work/trans_mc.json` and `work/trans_pairs.parquet`.
- `comp76.py` / `c76.py` / `bounds76.py`: 66→77 component flags, mix and bounds. `rest54.py`, `words54.py`, `mix.py`: set contrasts. `patterns.py <sub>`: part B.
- All CPU. Nothing touches `code/…/src`. `work/` is a symlink to `data/work/p77/` (gitignored; union + tags ≈ 60 MB, raw examples in `examples_noaddr.txt`).

## Risks
- **Pricing model.** The a/b per pair come from a simulated S1 state, not labels. Levels shift ±0.05–0.1 across the 4 states tried; ordering and the rejections hold in all 4.
- **India/US priors** for 6c/9ab/34_veto/54 come from holdout LB-eq, ±30%. They matter for 9ab and 54 (the sub54-rest estimate moves 0.38 → 0.50 across 0.0005–0.0009).
- sub31's LB is known to 4 decimals only (0.9856), so ±50e-6 on the 9ab→31 and 31→34 deltas.
- The public share is unknown (assumed 0.5). The SE scales with 1/√f, and it is tiny next to model error anyway.
- No labels, no external data. France is handled only as a country label in analysis; nothing goes into pipeline code.

## Open questions
1. diag_frblank's score: send it. It pins France's current F and the India/US share of the gains. My guess is 0.851–0.853.
2. Do the P48 / P55 CE vetoes still in the chain remove identical-hn noise-word pairs, the class Qwen mis-scores? The 31→34_veto step says a veto removed pairs at 0.81 precision. My guess: some yes. Audit with the P71 lens before trusting any France CE veto.
3. Is groupe (s .091) noise or decoy? The lens and contamination estimate say noise (≈ 0.3% D=0 decoys). The LB has not seen it. My guess: +0.0002–0.0003 if uploaded on sub77.
4. France address-less exact-core pairs with a unique S1 core and g ≥ .3 (2,905, outside `xfc` because the full name incl. legal form differs): what is their LOCO precision? P66's `rule66` frame can measure it in minutes. My guess: 0.8–0.9 when the legal forms differ only by presence (lf_k 2/3), and low when they conflict (lf_k 4, already vetoed by P64).
