# P2-script — RESULT
_DRAFT, 2026-09-25 ~04:00. Numbers below are dev-world only; being updated as experiments land._

## TL;DR            (3 lines: the answer, the number, the recommendation)
- Cross-script India names are **token-by-token transliterations of the S1 name**: 100% equal token count, 96.8% of
  Indic tokens map to one Latin token by position, only ~1.3k Indic token types in the train world.
- Rule romanization + phonetic fold (no training) already lifts cross-script name AUC 0.50 → 0.90, r@1 0.01 → 0.40.
- Lead (in progress): a token dictionary learned from train pairs, with the rule romanizer as OOV fallback.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
Provisional: `probes/P2-script/translit.py` — `to_latin(s: pd.Series) -> pd.Series` on every S2/S3 name and
address before normalization; `fold(s)` on both sides for char-n-gram features. Final version will add the
learned dictionary.

## Evidence         (table: ledger name | world | key metrics | verdict)
Name similarity = char_wb 2-3gram TF-IDF cosine; 7,457 dev cross-script queries vs 11,994 India S1; AUC vs a
hard negative (random S1 sharing a name token with the true S1); ties count against the method.

| method | world | AUC_hard | r@1 | r@5 | r@20 | verdict |
|---|---|---|---|---|---|---|
| raw (no translit) | dev | 0.504 | 0.009 | 0.019 | 0.032 | baseline |
| rule (translit.py) | dev | 0.866 | 0.308 | 0.597 | 0.747 | superseded |
| rule + fold | dev | 0.898 | 0.401 | 0.758 | 0.872 | keep (fallback) |

## What failed      (approaches tried and dropped, with the number that killed them)
- `sanscript` (indic-transliteration) as the romanizer: per-script scheme calls, leaves ZWNJ, Tamil த→"dh",
  Bengali nukta unhandled. Replaced by a 60-line offset table in translit.py (all 9 scripts, one table).

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
- `translit.to_latin(s)`, `translit.fold(s)` — token-cached; full-scale timing pending.
- `eval_names.py` — harness (cross_script, hard_negatives, evaluate).

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- No third-party dependency in translit.py (stdlib + pandas). France/US contain no Indic text, so it is a no-op there.

## Open questions   (for the master; each with your best guess)
- pending
