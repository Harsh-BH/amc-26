# P50-idea-batch — RESULT (DRAFT, 04:35; updated as ideas land)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- A (hn suffixes / ranges) and B (French legal forms): **drop**. The parser and LEGAL already do what the ideas proposed.
- C (France exclusivity veto), D (CE order TTA), E (full-holdout refit), F (20 seeds): running.
- sub50 = sub44r + passing ideas (pending).

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- Pending.

## Evidence         (table: ledger name | world | key metrics | verdict)
| idea | ledger name | world | key metrics | verdict |
|---|---|---|---|---|
| A | A-hn-suffix-range | test France (label-free) + holdout US/India | parser already maps 12 bis/12bis/12b/12-14/12 à 14/12/14 → 12. France suffix share: 6.3% S1 / 5.7% SX. hn-equal pairs, same suffix both sides: 0.927 accepted (no-suffix 0.895); one-sided bis: 0.04 accepted, ~90% at p<.5. Range whose other end = the other hn: 27 chain pairs (10 accepted, all p≥.99). Suffix-veto label check: one-sided / conflicting letter suffix on hn-equal pairs, #34 accepted precision India 0.9991, US 0.9997 | drop |
| B | B-legal-form-coverage | test France (label-free) | generator forms = SARL/SAS/EURL/SA/SASU/SCI/EI/SNC (+ dotted, bracketed, SÀRL): all in LEGAL after fold + _JOIN + _SEP. Legal tokens left in R1 cores: 69 S1 / 476 SX (core fallback, e.g. "SARL France"). Typo-like last tokens 0.11%, mostly real words | drop |
| C | C-fr-exclusivity-veto | LOCO (P47 frames, P42 recipe, both analogues / directions) | best both-positive cell, chain m ≥ .3, bar .99: +0.000029 ± 0.000019 US→India / +0.000062 ± 0.000019 India→US (m2 +0.000006 / +0.000049). Best India→US cell +0.000114 has US→India −0.000012. Removed precision 0.35–0.55. Other-mass pairs are 0.34% of accepted France pairs vs 0.02–0.03% in LOCO | drop (gate +0.0003) |
| D | D-ce-order-tta | holdout, 20k band argmax pairs | swapped (SX, S1) AUC 0.659 vs 0.983 (India 0.75, US 0.61); logit corr 0.34; average 0.875; stack p + r + r_swap vs p + r: +0.00002 AUC. Re-score jitter vs r_all up to 0.05 | drop |
| F | (ef.py) | holdout h0→h1 | 20 vs 5 seeds −0.000014 ± 0.000018; 10 vs 5 +0.000008 ± 0.000011 | drop |

## What failed      (approaches tried and dropped, with the number that killed them)
- A: a suffix veto (remove hn-equal pairs whose bis/ter/letter suffix is one-sided or differs). On US/India these pairs are
  true at 0.999 among accepted. No label support; the France model already rejects one-sided "bis" pairs (4% accepted).
- B: nothing to add to LEGAL.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `b_legal.py`, `b_typo.py`: legal-form counts (40 s, 0.6 GB). `a_hn.py`, `a_loco.py`: hn suffix counts + US/India label check (5 min, 0.9 GB).
- `d_build.py`: the CE TTA channel (20k band argmax pairs, both orders) for P13's bge_ckpt job.
- `ef.py`: E learning curve + F seed bag on P44's A frames.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- Pending.

## Open questions   (for the master; each with your best guess)
- Pending.
