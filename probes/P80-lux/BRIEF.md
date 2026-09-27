# P80-lux: Scoring Levers from Top Public Repositories (probe-lux)

## Motivation & Context
Cross-referencing the top public repositories for Amazon ML Challenge 2026:
- `Ruthwik000/amazonsummerML` (Ekdant.exe, LB 0.986916)
- `bhavyatalwar04/Business_Entity_Resolution` (LB 0.986201, holdout 0.9900)
- `nisdudeheckz/Amazon_ML_Challenge-26` (held-out 0.9885)
- `prem970/Amazon-ML-challenge` (offline 0.99813)

Current base in repository:
- Best public LB: `sub77` (LB **0.988292**, MD5 `9e4fb898`, Zip `b6e87f89`).
- Active holdout / calibrated pipeline: `v2_calibrated_9924` (held-out $F_{0.5} \ge 0.9924$).

## Levers Under Evaluation
1. **Priority 1: India/US Hidden Internal Address Number Decoy Veto (`hidden_hn_veto.py`)**
   - Standard house number checks only evaluate leading numbers.
   - In India/US addresses, synthetic generator shifted internal door/plot/survey/unit numbers by $+k \in \{3, 4, 5, 7, 9, 11, 13, 21\}$.
   - In training truth, pairs exhibiting this shift are **98% to 99.9% false decoys**.
   - LOCO proxy (US $\to$ India) demonstrates $+0.00451$ $F_{0.5}$ gain from pruning shifted pairs.

2. **Priority 2: France Category-Word Swap Veto (`france_cat_veto.py`)**
   - Based on `Ruthwik000`'s `france_fix2`:
   - Synthetic distractors in France swapped category/institution words (`Club` $\leftrightarrow$ `Collège`, `École` $\leftrightarrow$ `Centre`, etc.) at the identical street address.
   - Pruning pairs where the core institution noun is swapped while street address is unchanged.

3. **Priority 3: Zero-Candidate Space-less 3-gram Rescue (`rescue_3gram.py`)**
   - True matches missing from blocking due to run-together URLs (`chennaiprime.com` vs `Chennai Prime`) or extreme OCR typos.
   - Built on space-stripped char 3-grams over unclaimed high-value records.

4. **Priority 4: Dynamic Expected-$F_{0.5}$ Set Selection (`ev_select.py`)**
   - Instead of static $\tau = 0.71$, dynamically select the match set per $S_1$ entity that maximizes $\mathbb{E}[F_{0.5}]$.

## Deliverables
- `BRIEF.md`: Experiment framing and gate specifications.
- `hidden_hn_veto.py`: Vectorized secondary address number shift detector and filter.
- `france_cat_veto.py`: Category swap detector at identical address.
- `ev_select.py`: Dynamic subset selection optimizer.
- `rescue_3gram.py`: Unmatched candidate rescue engine.
- `run_lux.py`: End-to-end evaluation and candidate generation script.
- `RESULT.md`: Empirical verification on holdout, LOCO metrics, and final recommendations.
