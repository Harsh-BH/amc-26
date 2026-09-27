# P33-orgword-tier — RESULT
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **R05 as Brief 4 specifies it fails its LOCO gate.** Half-1 ΔF0.5 at t_profile is India→US **+0.00010 ± 0.00011** and US→India **+0.00002 ± 0.00009**; the gate is ≥ +0.0005 in both directions. The flipped pairs are only 71% / 67% true, below the ~0.73–0.79 break-even. In-domain it is harmless: US −0.00006 ± 0.00002, India −0.00000.
- **The s(w) ≤ 0.4 gate does not separate true from false swaps.** In LOCO the whole gain is one corruptor noise word, "partners" (1,118 flips, 100% true). The rest (sibling brand words, surnames) flip at 0–40%, at every s. The two lenses I pre-registered to isolate noise words both failed: e(w) ≥ 0.8 selects 0 pairs, and a(w) ≥ 2 scores −0.00032 / −0.00002.
- **Do not build sub33, and do not upload R05.** On France it would add 6,920 pairs on 6,765 S1 (85% developpement / groupe, 12% compagnie). That population has no labelled analog besides "partners", so no LOCO-validated version exists. A label-free spare is noted under Open questions.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
- **Adopt nothing into src. #31's France file stays as it is.** R05 misses the brief's gate (LOCO ≥ +0.0005 in both directions) by 5× in India→US and 25× in US→India.
  - R05 would also go into the same cell as R14 / R32. Brief 4 was ranked above them as the rule-level fix, so the cell is still open for R14 if the master wants a model-level attempt.
- **No `output/sub33` was built**, as the brief says: build only if the gate passes.
- The France file that R05 would produce exists for the record only: `data/work/p33/fr33_r05_t9868.parquet` (#31 + R05, cap re-applied, t 0.9868).
- One gate-free idea for a spare slot, R05a France-only, is under Open questions. It is not recommended by the evidence here.

## Evidence         (table: ledger name | world | key metrics | verdict)
ΔF0.5 is paired per S1: half 1 ± SE, with half 0 in brackets. "Flip" = a pair the rule turns from rejected to accepted. That is the only thing the rule changes, since it drops nothing.
Worlds:
- **LOCO** = P6's dense fold-0 holdout scored by an M2 trained on the other country (P27's copies, re-fetched from S3). t = t_profile (P6's empty-rate rule, label-free), then the oracle t.
- **In-domain** = #8c's dense holdout (P23 dump), t 0.71.

**R05 exactly as briefed:**
- argmax pair; sub1; not an R1 filler; hn equal and present; city (`po`) and street exact (P28 `st_x`);
- not a P28 sibling word (1,197 words on France's blocked list, 188 US / 139 India);
- remaining core unique among the SX's other candidate S1s;
- the SX core is not another S1's (core, city);
- s(w) ≤ 0.4 with support ≥ 20, s learned on the country's blocked list; p ≥ 0.02;
- then p := max(p, 0.99).

| ledger name | world | key metrics | verdict |
|---|---|---|---|
| r05-loco-India>US (18:38) | LOCO, France-like target, t 0.81 / 0.87 | base 0.92850. **r05 +0.00010 ± 0.00011 (h0 −0.00004)**: 3,236 rule pairs, 0.804 true; 1,735 flips, **0.710**. At oracle t: +0.00013. r05_s3 +0.00017, **r05_s2 +0.00041** (1,574 flips, 0.783), r05_nos −0.00041, r05_city (Brief 4's city-only address) +0.00009, r05_nouniq +0.00006, r05_nonocore +0.00011, r05_p5 +0.00010 (541 flips, 0.799). P28 bxsg on argmax pairs −0.00092 | **drop** |
| r05-loco-US>India (18:39) | LOCO, India hn regime, t 0.97 / 0.98 | base 0.90213. **r05 +0.00002 ± 0.00009**: 1,203 rule pairs, 0.806 true; 314 flips, **0.672**. r05_s2 +0.00002, r05_city **−0.00090** (769 flips, 0.41), r05_nonocore +0.00042 (508 flips, 0.77), r05_p5 −0.00007, bxsg_am +0.00040 | **drop** |
| r05-holdout-US (18:41) | #8c in-domain | base 0.98924. **r05 −0.00006 ± 0.00002**: 2,819 rule pairs, 0.987 true; only 32 flips, 0.094 true. bxsg_am −0.00339 (1,630 flips, 0.006), as in P28 | passes the −0.0001 bar |
| r05-holdout-India (18:42) | #8c in-domain | base 0.99004. **r05 −0.00000**: 1,020 rule pairs, 0.996; 5 flips. bxsg_am −0.00128 | passes the −0.0001 bar |
| r05-France-fr33_r05_t9868 (18:44) | test France, on `p31/fr31_t9868` (post tier + bxneg + cap), t 0.9868 | 85,661 sub1 / hn-eq / city / argmax pairs. Filtered out: street 13.6k, sibling guard 36.3k, fillers 20.5k, nocore 11.5k, floor 11.3k, uniq 8.7k. 9,697 R05 pairs. **+6,920 added / 0 dropped, 6,765 S1, 238 empties filled.** Profile 6.458% → 6.366% empty, 3.117 → 3.144 per S1 (in P6's bands). Added words: developpement 3,437, groupe 2,366, compagnie 855, the rest ≤ 19 each | not built (gate) |
| r05-loco-* / r05-holdout-* (R05e rows) | LOCO + in-domain | **R05e** = R05 + e(w) ≥ 0.8 (pre-registered). e(w) = the hn-equal share of w's hn-present sub1 pairs on the blocked list. **0 rule pairs in every world.** On a blocked list, random candidates with other house numbers keep every word below 0.7 | drop |
| r05-loco-* (R05a rows, 18:38–18:39) | LOCO | **R05a** = R05 + a(w) ≥ 2 (pre-registered). a(w) = the share of SX cores holding w over the share of S1 cores holding w. India→US **−0.00032 ± 0.00005** (196 flips, 0.077 true); US→India −0.00002. The a ≥ 1.5 sensitivity gives India→US +0.00060 but US→India +0.00003. It is post hoc anyway: US "partners" sits at a = 1.97 | drop |
| r05-holdout-* (R05a) | in-domain | US −0.00004 ± 0.00002, India 0.00000 | — |
| r05-France-fr33_r05a_t9868 (18:47) | test France | R05a: **+5,836 / 0 pairs, 5,730 S1, 182 empties filled**, 6.388% empty, 3.140 per S1. Almost all flips are developpement 3,437 and groupe 2,366 (compagnie drops out at a 0.90). The same ~5.8k at a ≥ 1.5 and a ≥ 4 | spare, not recommended |
| r05-gate-summary (18:48) | — | the numbers above | **drop** |

**Why s(w) does not work: LOCO flip precision by s bin.** This uses r05_nos, i.e. R05 with no s gate. Each cell is flips / share true; LOCO s is learned on the dense frame.

| s(w) bin (n ≥ 20) | India→US | US→India |
|---|---|---|
| ≤ 0.02 | 166 / 0.42 | 80 / 0.74 |
| 0.02–0.05 | 164 / 0.24 | 188 / 0.80 (jay / laxmi / jai / lakshmi: transliteration variants, 1.00) |
| 0.05–0.10 | 78 / 0.05 | 30 / 0.03 |
| 0.10–0.20 | **1,166 / 0.96 ("partners" 1,118 / 1.00)** | 14 / 0.00 |
| 0.20–0.40 | 161 / **0.00** (nexlum, aviquo, …: generated sibling brand words) | 2 / 0.00 |
| > 0.4 | 18 / 0.00 | 0 |
| support < 20 | 1,517 / 0.59 | 240 / 0.62 |

- Precision is not monotone in s. The true class is a handful of corruptor noise words, whatever their s.
- Surnames (adams s 0.025, walker 0.065) are 0% true at the same address.

**France, label-free: the scored-list hn profile of the words R05 would flip.** Columns: s = SIB3 share; e = hn-equal share of hn-present pairs.

| word | n | s | e | a(w) | R05 flips |
|---|---|---|---|---|---|
| developpement | 5,157 | 0.016 | **0.974** | 10.6 | 3,437 |
| groupe | 6,290 | 0.075 | **0.887** | 4.7 | 2,366 |
| compagnie | 2,188 | 0.194 | 0.742 | 0.90 | 855 |
| club / ecole / comite / amicale | 4.0–6.6k | 0.31–0.37 | 0.50–0.58 | 0.84–0.88 | ≈ 0 (sibling guard) |
| holding / participations | 68–105 | 0.57–0.60 | 0.21–0.23 | 341 / 3,099 | 0 (guard) |

- developpement and groupe have the true-pair hn profile (US true pairs keep the hn ~90%). They look like France's "partners", which is corruptor noise.
- compagnie sits in between. It is an S1 org word (a < 1), in the s band that flips at 0% in LOCO.

## What failed      (approaches tried and dropped, with the number that killed them)
- **R05 as briefed:** LOCO +0.00010 / +0.00002 against a +0.0005 gate. Flip precision 0.71 / 0.67.
- **Brief 4's literal address (city only, no street):** US→India −0.00090 to −0.00114 (flips 0.39–0.41 true). The street-exact clause is essential.
- **Its guards, one at a time:**
  - The uniq guard removes 611 / 549 candidates and is roughly neutral (+0.00004 when dropped in India→US).
  - The nocore guard is neutral in India→US and costs US→India 0.0004 (r05_nonocore +0.00042). India has many same-name S1s in one city, and the guard removes true pairs there. It still falls short of +0.0005 there.
  - The p ≥ 0.5 variant (Brief 4's conservative fallback): +0.00010 / −0.00007.
- **Tighter s gates:** s ≤ 0.2 gives +0.00041 / +0.00002. The India→US gain is "partners"; the US→India side has no noise-word mass.
- **No R05 variant clears +0.0005 in both directions.** The best mixed pair is r05_s2 in India→US (+0.00041) with r05_nonocore in US→India (+0.00042), and those are different rules.
- **R05e (e ≥ 0.8 on the blocked list)** selects 0 pairs anywhere.
- **R05a (a ≥ 2)** fails because generated sibling brand words appear only in S2/S3 (a ≈ 52), exactly like noise words. US "partners" sits at 1.97, just under the threshold.
- A run was paused by memguard (exit 147) once, when other probes held the RAM. Reran detached; no results were lost.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
`probes/P33-orgword-tier/r05.py`. It imports P28's `anchor` (rule_masks, boosted, decide, _rf, SIB3) and P20's `patterns.name_rel`, P29's `cap_mask`, and P28's `ho` / `loco` helpers. src is untouched.
- `s_table(pairs, xs, c) -> DataFrame[word: n, s, n_hn, n_eq, e]`: label-free s(w) / e(w) over a candidate list's sub1 pairs.
- `a_table(xs, c) -> Series[word: a]`: SX/S1 core document-frequency ratio. 2 s France, 10 s US train, < 0.7 GB.
- `r05_masks(d[s1,sx,p], split, c, t, am=None, best=None, learn=None, s_lists={tag: s_table}, a_tab=None) -> {variant: mask, 'info', 'w', 'a'}`. Variants: r05, r05_s3/s2, r05_nos, r05_city, r05_nouniq, r05_nonocore, r05_p5, r05_scored, bxsg_am, r05e*, r05a*.
- CLI:
  - `r05.py loco "India>US" "US>India"`: 90 s, 2.5 GB.
  - `r05.py ho US India`: 2.3 min, 2.5 GB.
  - `r05.py fr IN T OUT [variant]`: France file (post-rules) → + variant boost → P29 cap → OUT (s1, sx, p). 100 s, 2.6 GB.
  - `r05.py selftest`.
- Logs are in `data/work/p33/` (loco3.log / ho3.log are the full final runs). Files there:
  - `fr33_r05_t9868.parquet` and `fr33_r05a_t9868.parquet` (28 MB each);
  - the s tables.
- `data/work/p27/p6/S_holdout_{India,US}.parquet` was re-fetched from the S3 offload (102 MB). Delete it when disk is needed.

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Assumptions I made where the brief was open:**
  - s(w) = SIB3 share over **all** of w's sub1 pairs (P23's `orgswap_s.py` definition), learned on the **blocked** list (P28's lesson).
  - The scored-list s (`r05_scored`) gives the same verdict: India→US +0.00010, France 6,702 flips.
  - "No other kept candidate shares the remaining core" = no other candidate S1 of the SX contains the SX core minus w.
  - "SX core is not another S1's core in the same city" uses the S1 pool of the whole split.
- **France vs LOCO population mismatch.** LOCO cannot validate the words R05 flips on France (developpement / groupe / compagnie). The rule's LOCO failure comes from US / India words that the France guards mostly remove. So the failed gate does not prove R05 hurts France. It shows that s(w) is not a safe selector, and that nothing validates the France flips.
- R05 on France uses P28's original rf cores and hn (like #31's bxneg), not P29's '(n)' hn fix. That can only under-count eligible pairs.
- No licence, external-data or country-filter issue. Everything is test-side unlabelled statistics (ruling 2). The France-only application is combine.py's S1-country swap, as in every France probe.

## Open questions   (for the master; each with your best guess)
- **Spend a slot on R05a France-only (+5,836 developpement / groupe swap pairs, 5,730 S1)?** Best guess: **no**, unless a slot would otherwise go unused on 27 Sep.
  - For: the label-free profile is the true-pair one (e 0.97 / 0.89, SIB3 0.016 / 0.075, a 10.6 / 4.7). The closest labelled analog, corruptor noise word "partners" at the same address, flips at 100% in LOCO.
  - Against: R05a itself fails LOCO (−0.00032).
  - EV ≈ +0.0003 LB at 0.97 precision, 0 at the 0.73 break-even. Well below public/private noise on ~6k France S1.
  - Build if wanted (≈ 5 min + validator):
    ```
    .venv/bin/python probes/P14-france-shift/combine.py /home/harsh1/github-repos/amc-26/output/sub8c data/work/p33/fr33_r05a_t9868.parquet 0.9868 output/sub33a
    cd data/student_resource && python3 utils/validate_submission.py --matching <abs>/output/sub33a/matching_results.tsv --candidate <abs>/output/sub33a/candidate_pairs.tsv --test-dir dataset/test
    ```
  - On `sub31_veto` or another France base, re-run `r05.py fr <that post file> <t> <out> r05a` first.
- **Should R14 (re-computed sx margin over the pruned list) take this cell?** Best guess: yes, if anyone has time before the freeze. It fixes the depressed margin rather than choosing words by a lens, which is what failed here.
- **Is "groupe" safe when it is swapped in?** Best guess: mostly. P20 found it signals a sibling when **added** (add1, 51% SIB3), and P28's guard catches that form. As a swap it is SIB3 7.5%, hn-equal 0.89.
