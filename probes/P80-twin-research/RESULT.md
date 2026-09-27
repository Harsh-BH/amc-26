# P80-twin-research — RESULT (27 Sep ~19:20 IST)
## TL;DR
- **Exact twins (address-less copy, full name incl. legal form shared by ≥ 2 S1) are a Bayes floor.** Every exchangeability
  breaker the literature lists was tested on train truth and is closed. No paper, tool or rival repo breaks such ties (all abstain).
- **One viable slice, next door:** an address-less copy that KEEPS its legal form and whose full name matches exactly one S1 is
  that S1's 0.94–0.98 (core shared by 2) / 0.91–0.96 (by 3), even though other S1 share the core. Bare copies: 0.50–0.77 → leave out.
- France has **1,540** such copies unassigned in sub76_brand (E already a candidate for all). Expected **LB +0.00004 to +0.00007**. Needs P66's LOCO gate.

## Recommendation
- Twins: keep abstaining. Spend nothing more on the identical-name bucket.
- Optional France add (extension of P66 `xfc`): address-less SX, `fkey` full name == E's and unique in France, **SX name carries a legal
  form**, E's rf core count **2–3** (xfc needed 1), France p ≥ 0.3, caps 5/6, free non-sibling SX. Gate with `loco66.py` (India→US
  precision ≥ 0.85) before staging; `frcount.py` sizes it on any base.

## Evidence (train truth unless noted; India + US)
| test | result | verdict |
|---|---|---|
| het.py — address-drop iid per copy? | var ratio vs binomial 1.000 (k 2–8); P(na \| other na) 0.0446 vs 0.0442 | closed |
| roworder.py — row / ID leak on twins | owner later in S1 0.494 ± 0.009 India, 0.501 ± 0.006 US; ID closer 0.501; S1 row vs size corr 0.001 | closed |
| derive.py — copies of copies | noised x raw-equal to an addressed copy at one twin: owner 0.215 vs uniform 0.218 (n 5,020) | closed |
| counts.py — size prior by (fewer, more) addressed copies | best cells 0.70–0.85 *given owner is a twin*; 15.8% of these x belong to a different-key S1 | below break-even |
| budget.py — per-record modification budget | exists: address-less copies are websites 0.10×, drop legal form 0.61×, keep key 1.24×; copy never = S1 (P(addr kept \| name kept) 0.0001) | real, cannot split identical twins |
| exact.py — SX identical to an S1 | train 18 (all owned by it); test 1,710 (France 1,696), all already assigned to it | no change |
| lfdir.py — legal-form direction, unique exact name, shared core | x keeps LF: India .981/.956/.870, US .938/.909/.729 (core count 2/3/≥4); x bare: India .767/.744/.479, US .662/.555/.501 | **keep (LF kept, core 2–3)** |
| testsize.py — test bucket, sub76_brand | twin-key address-less unassigned: France 7,710, India 10,482, US 10,323 (≈43% in groups ≥ 5) | descriptive |
| frcount.py — France pool | LF kept, core 2–3, unassigned: 1,540 (assigned now 21–24% vs ~all in India/US); E a candidate 100% | stage if LOCO passes |

Literature (subagent review): identical likelihoods ⇒ posterior = partition prior (label-switching symmetry; Skinner & Elliot
2002's 1/F_k); fastLink/Splink give a missing field weight 0 under MAR; Sadinle 2017 / BRL, Bai–Binette–Reiter 2023 abstain;
F-beta add rule p > 0.8·F_j (p > 0.5 for an empty S1 whose only possible copy it is). Febrl/GeCo: per-copy noise from the original,
per-record budget (matches budget.py). Repos: 653 rival repos after 15:05 — all abstain or hard-reject name-only matches; best public LB 0.9822.

## Interfaces
Run from repo root under a memory cap: `systemd-run --user --scope -q -p MemoryMax=5G .venv/bin/python probes/P80-twin-research/<x>.py`.
`frame.py OUT` writes amb.parquet / s1tw.parquet / s1all.parquet (17 s); `counts.py`, `roworder.py`, `derive.py` read OUT. Others are standalone (5–20 s each, < 3 GB).

## Risks
- lfdir precision is over all such pairs; the pool a strict France chain leaves may be harder (xfc: cell 0.99 → LOCO pool 0.89–0.98).
- `LF` regex is a hand list of legal forms (allowed: small hand-written dictionary).

## Addendum (19:45) — built R1; traps found on the way
- **Built `output/p80_lfkept`** (portable add-set, 1,479 pairs on 1,432 France S1) = the R1 pool (1,540) minus 61 pairs where a
  same-core sibling's legal form turns into E's by one dropped letter (SAS→SA, SASU→SAS: the generator drops letters).
  Grafted on sub76_brand → `output/sub80_lf_on76b`: validator `--check-ids` PASS; 1,432 France lines changed, additions only,
  India/US byte-identical; 0 cap/owner rejections; 23 targets were empty. Graft on the real base: `combine.py sub77 <out> +p80_lfkept`.
  Expected LB **+0.00002 to +0.00005** (precision 0.85–0.95 after a haircut for test-only orphans; ~0.075 gain / 0.25 loss per pair).
  No p / argmax gate (France chain scores are not on this laptop): apply `am & p >= 0.3` at graft time if available.
- **Trap: "exact unique name + house-number typo" (digit drop/add 0.92/0.99, far 1-digit substitution 0.92/0.93 on train).** In France
  these 1,252 free pairs have a *different street and city* (another branch of the brand, its S1 absent from test). Dropped.
- **Test France free SX at hn ±1/±2 with a tweaked name** are the "house-number branch" decoys (test has 1.9× distractors). The
  chain is right to leave them. Exact name + unique S1 + hn≠ is 11% assigned in France vs 45% US, but US-train precision is 0.45.

## Addendum (19:40) — pool gate passed; staged
- **pool.py** (P57 holdout frames, in-country M2 p, holdout-S1 rows): R1 rows (address-less, exact name to one candidate S1, SX
  legal form = S1's, 2–3 candidate S1 share the core) are 0.919 (India, n 124) / 0.899 (US, n 474) true; on the pool a strict
  bar leaves (.3 ≤ p < .9868) **0.919 / 0.914** — the model is never confident on them, so the pool is the class. xfc-like
  (core 1, LB-proven) reads 0.965 / 0.910 on the same test; core ≥ 4 reads 0.71 / 0.81 (excluded). p < .3 rows are ~2% (0.11–0.16).
  In-country M2, not the P66 LOCO harness (its frames are not on this laptop).
- **Staged `output/sub_p80lf_on76b`** (md5 cae97b8c…) = sub76_brand + p80_lfkept (+1,479). sub76_brand = sub76 + P41 brands, the same
  recipe as #77 (≈100 brand pairs apart). Expected LB ≈ #77 + 0.00003 to 0.00005.
