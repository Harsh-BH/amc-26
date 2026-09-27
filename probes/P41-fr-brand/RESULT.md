# P41-fr-brand — RESULT (final, 27 Sep ~01:05)
## TL;DR            (3 lines: the answer, the number, the recommendation)
- **The rule passes, and `output/sub41` is staged (validator PASS).** A free France SX whose name is a *bare* generator brand joins the S1 that uniquely shares its exact hn + street + city key. "Bare" means one token built from P36's 44 syllables, keeping no word of any S1 name. The S1 must also be the SX's argmax, already have members, and have p ≥ 0.30 (caps 5/6). This adds **+5,742 pairs to 5,641 France S1** on #37_gc70. Among gib SX that have a pair, France's assignment rises from 0.60 to 0.79.
- **Gate: criterion 2 passes in the plain P37 LOCO.** India→US +0.00010 to +0.00014 France-F0.5 at precision 0.99; US→India +0.00008 to +0.00010 at precision 0.98–1.00. A France-matched LOCO, with gib acceptance lowered to France's ratio, gives +0.00037 / +0.00079. Train truth: a bare brand at a unique key is that S1's in 99.8% of US cases (34.8k) and 96.1% of India cases. **Expected LB +0.00016 to +0.00024 vs #37_gc70.**
- **Upload order: #41 first, then `output/sub41_hn90`.** #41 is the low-risk step. As pair sets, `sub41_hn90` = #41 ∪ #37_gc70_hn90, so the second upload tests hn90's 14,974 non-brand extras (the risky part) on top of #41. Of #41's adds, 3,206 (the p ≥ 0.9 part) are already in hn90; sub41_hn90 adds 2,536 new pairs on hn90.

## Recommendation   (exactly what the master should adopt: module, function signature, config)
**Queue, 27 Sep. Base = #37_gc70 (LB 0.986304). Written into SUBMISSIONS.md:**

| rank | folder | vs its base | France Δ | LOCO (plain P37; France-F0.5, precision) US→India / India→US | expected LB |
|---|---|---|---|---|---|
| **1** | **`output/sub41`** | #37_gc70 + gib rule (key1_bare, p ≥ 0.30) | +5,742 pairs, 5,641 S1; 3.137 → 3.159 per S1 | **+0.00008 (1.00) / +0.00010 (0.99)**; France-matched +0.00035 / +0.00077 | **+0.00016 to +0.00024** |
| 2 | `output/sub41_hn90` | #37_gc70_hn90 + the same rule | +2,536 pairs (p 0.30–0.90, plus same-source members of one-sided groups), 2,511 S1 | same rule | #37_gc70_hn90 + 0.00007 to 0.00011 |

- Both folders: validator PASS. `check.py` confirms that only France lines differ from the base (5,641 / 2,511), and that 0 matches fall outside candidate_pairs.tsv.
- `candidate_pairs.tsv` and `code.zip` are hard links to the base's: every added pair is a pair of the France file.
- **Alternative order**, if the master prefers P37's queue: #37_gc70_hn90 first; then #41_hn90 if it gains, or #41 if it loses.

**run.py integration.** France / unseen branch, right after P37's `gc_add` lines, in the same `acc` loop:
```python
if d["c"] in UNSEEN:                                  # P41 synthetic-brand recovery (after rescue, gc70, [hn90])
    gk = P41.gib_keys(k1, kx)                          # k1: S1 id/hn/street/city; kx: SX id/hn/street/city/gib
    gk["bare"] = P41.bare("test", d["c"], gk.index)    # one-token brand name
    g = P41.guards(s1_ids, sx_ids, d["p"], gk, k1.set_index("id"))["key1_bare"]
    acc |= FR.gc_add(s1c, sxc, is2, d["p"], am, acc, sib | ~g, 0.30, "all")   # non-empty groups, caps 5/6
```
- **Move into `france.py` (≈ 100 lines, stdlib + numpy + pandas):**
  - `gib.py` from P36: `toks`, `core`, `learn`, `n_syll`, `is_gib`;
  - from `p41.py`: `key_match`, `gib_keys`, `bare`, and only the `key1` / `key1_bare` part of `guards`.
- **Inputs:**
  - hn and city: the rules.feats files (S1 `rf_test_s1_France`, SX the fixed world's `rf_test_sx_France`);
  - street: `france.street_key` on the raw address;
  - gib syllables: `gib.learn` on the split's own names (label-free test-side statistics; the same 44 syllables on train and test).
- `d["p"]` is the post-rules France p (tier / bxneg / veto / cap), the p of `fr31v0.1_t9868.parquet`. Vetoed or capped pairs (p = 0) never pass the 0.30 floor.

## Evidence         (table: ledger name | world | key metrics | verdict)
| ledger name | world | key metrics | verdict |
|---|---|---|---|
| fr-gib-profile | test France, sub37_gc70 | gib 31,035, assigned 0.593 (non-gib 0.567); 12,637 unassigned: **95.8% in the pruned pair file**, median best p 0.73, argmax hn-eq 0.86 / street-eq 0.82 / city-eq 0.96 / sibling 0.01; 7,047 have a unique key S1 (92% of them its argmax). Assigned gib: key S1 = assigned S1 99.95% | keep |
| train-gib-key-truth | train truth | a gib SX whose hn+street+city key is unique among the country's S1 belongs to it: US 0.973 (134k), India 0.974 (39k). Decoys (gib SX matching nothing, US 11.6% / India 0.85%): 81% / 84% have no key S1 | keep |
| gib-decoy-signature | train truth + France | the bare-name split, below | keep |
| loco-plain-key1_bare_* | LOCO P27 M2 (P37 protocol) | FR 0.3: US→India +0.00008 / +0.00010 (dens / pool map; prec 1.00 / 0.98); India→US +0.00010 / +0.00014 (0.99) | **keep (gate criterion 2)** |
| loco-gm-key1_bare_* | LOCO, France-matched | FR 0.3: +0.00035 / +0.00037 (1.00); India→US +0.00077 / +0.00079 (1.00) | keep |
| loco-plain-key1_* (no bare guard) | LOCO P37 protocol | India→US −0.00001 to −0.00024 (prec 0.50–0.68); US→India +0.00006 to +0.00022 | drop |
| stage-sub41-key1_bare-0.3 / stage-sub41_hn90-key1_bare-0.3 | test France | +5,742 / +2,536 pairs; gib ratio to non-gib 0.87 → 1.15; 3,206 of sub41's adds are in hn90 | keep |
| stage-sub41 / stage-sub41_hn90 (key1 @ 0.7) | test France | first staging at 00:50 (+5,120 / +1,682); **superseded** by the key1_bare builds (same folder names) | superseded |

**1. France's missing brand SX are a decision miss, not a blocking miss.** Profile of the 12,637 unassigned gib SX on #37_gc70:

| best p in the France file | ≤ 0.01 | 0.01–0.1 | 0.1–0.3 | 0.3–0.5 | 0.5–0.7 | 0.7–0.9 | 0.9–0.9868 | not in the file |
|---|---|---|---|---|---|---|---|---|
| SX | 307 | 895 | 1,207 | 1,860 | 1,581 | 2,297 | 3,945 | 535 |

| S1 sharing the SX's exact hn + street + city key | 0 | 1 | 2 | 3 | 4 | 5+ |
|---|---|---|---|---|---|---|
| unassigned gib SX | 2,774 | **7,047** | 1,507 | 390 | 157 | 762 |
| assigned gib SX (18,398) | 325 | 14,646 | 1,992 | 483 | 176 | 776 |

- **The key S1 already has members:** its predicted group holds 2–5 SX in 83% of cases (0 members: 104).
- **Assigned gib SX follow the key:**
  - hn-equal to their S1 99.1%, street-equal 98.1%;
  - when the key is unique, the key S1 is the assigned S1 99.95% of the time.

**2. What ties a gib SX to its S1 (train truth), and why the bare guard exists:**

| key-unique gib SX | n | precision (true S1 = key S1) | decoys | exact address | S2 share | 3-syllable | all caps | keeps an S1 word |
|---|---|---|---|---|---|---|---|---|
| US, bare name | 34,777 | **0.998** | 50 | 0.15 (true) | 0.45 | 0.33 | 0.10 | 0 |
| US, not bare | 99,698 | 0.964 | **3,588** | true 0.69 / decoy 0.10 | 0.00 / 0.49 | 0.25 / 0.00 | 0.00 / 0.13 | 0.99 / 0.92 |
| India, bare | 15,083 | 0.961 (misses are true-elsewhere) | 37 | 0.43 | 0.23 | 0.30 | 0.06 | 0 |
| India, not bare | 23,597 | 0.982 | 4 | 0.72 | 0.02 | 0.25 | 0.00 | 0.90 |
| **France, unassigned key-unique argmax pool** | 6,505 | ? | ? | 0.26 | 0.38 | 0.32 | 0.08 | 0 (92% bare) |
| France, assigned key-unique | 14,638 | — | — | 0.64 | 0.06 | 0.26 | 0.01 | 0.71 (29% bare) |

- **US decoys are "brand + an S1 word"** at the S1's address: S2-heavy and caps-heavy.
- **True gib SX come in two kinds:**
  - brand + S1 word: S3 and not caps;
  - bare brand: S2 45%, caps 10%.
- **Why the France pool is 92% bare:** the France model accepts the brand + word kind and misses the bare kind (no name signal at its strict t 0.9868).
- **The France pool's signature matches US *true bare* SX** (S2 0.38 vs 0.45, caps 0.08 vs 0.10, 3-syllable 0.32 vs 0.33). It does not match US decoys (3-syllable 0.00, keeps an S1 word 0.92).

**3. LOCO by guard and strictness.**
- **Units:** dF × 1e-4 France-F0.5, with (added-pair precision, adds). LB ≈ 0.1497 × dF.
- **"plain":** P37's protocol. The base is t_profile + rescue90 + gc70, each at its density-mapped t.
- **"gm" (France-matched):** the same, except the target's gib argmax pairs need a t raised until the target's gib / non-gib assignment ratio equals France's 0.87. The targets' own ratios are 1.08–1.12.
- **Map:** the France p floor becomes a target t through P37's argmax density ("dens") or through the quantile of the rule's own pool ("pool").
- **Why the gm world:** in the plain LOCO the rule's pool is 8–17× smaller per S1 than France's (0.3–0.4% of S1 against 2.5%). The plain LOCO does not reproduce the France failure.

| guard | FR p ≥ | map | plain US→India | plain India→US | gm US→India | gm India→US | France adds |
|---|---|---|---|---|---|---|---|
| key1 | 0.9 | dens | +0.64 (1.00, 76) | −0.07 (0.68, 101) | +4.20 (1.00, 509) | +7.19 (0.94, 1692) | 3,438 |
| key1 | 0.7 | dens | +0.92 (1.00, 110) | −0.28 (0.63, 169) | +4.47 (1.00, 543) | +6.97 (0.92, 1760) | 5,120 |
| key1 | 0.7 | pool | +1.82 (1.00, 205) | −1.48 (0.55, 412) | +4.57 (1.00, 551) | +7.19 (0.94, 1689) | 5,120 |
| key1 | 0.3 | dens | +1.55 (1.00, 170) | −1.10 (0.56, 337) | +5.11 (1.00, 603) | +6.16 (0.89, 1928) | 6,223 |
| key1 | 0.0 | dens | +2.24 (0.99, 257) | −2.36 (0.50, 501) | +5.79 (1.00, 690) | +4.90 (0.85, 2092) | 6,414 |
| key1nc (no city) | 0.7 | dens | +0.88 (1.00, 102) | −0.04 (0.70, 227) | +4.17 (1.00, 502) | +7.98 (0.93, 1957) | 4,409 |
| key1nc | 0.3 | pool | +1.97 (0.99, 223) | −1.90 (0.58, 674) | +5.20 (1.00, 611) | +6.37 (0.86, 2358) | 5,379 |
| eq (key-equal argmax, not unique) | 0.7 | dens | +1.02 (0.97, 145) | −1.12 (0.51, 275) | +5.42 (0.99, 675) | +6.61 (0.89, 1963) | 5,645 |
| eq | 0.3 | dens | +1.74 (0.96, 226) | −3.39 (0.45, 621) | +6.15 (0.99, 756) | +4.34 (0.82, 2309) | 8,367 |
| key1_bare | 0.9 | dens | +0.34 (1.00, 38) | +0.34 (0.98, 66) | +3.09 (1.00, 378) | +7.05 (1.00, 1301) | 3,206 |
| key1_bare | 0.7 | dens | +0.48 (1.00, 55) | +0.55 (0.99, 100) | +3.24 (1.00, 395) | +7.26 (1.00, 1335) | 4,726 |
| key1_bare | 0.7 | pool | +0.89 (0.99, 102) | +1.10 (0.99, 191) | +3.07 (1.00, 374) | +6.41 (1.00, 1177) | 4,726 |
| **key1_bare (staged)** | **0.3** | dens | **+0.78 (1.00, 83)** | **+1.02 (0.99, 182)** | +3.54 (1.00, 423) | +7.74 (1.00, 1417) | **5,742** |
| **key1_bare (staged)** | **0.3** | pool | **+1.03 (0.98, 123)** | **+1.36 (0.99, 232)** | +3.74 (1.00, 453) | +7.88 (1.00, 1433) | 5,742 |
| key1_bare | 0.0 | dens | +1.06 (0.98, 127) | +1.40 (0.99, 239) | +3.81 (0.99, 467) | +8.13 (1.00, 1474) | 5,907 |
| eq_bare | 0.7 | dens | +0.53 (0.94, 82) | −0.11 (0.69, 192) | +4.00 (0.99, 502) | +6.94 (0.96, 1488) | 5,201 |
| eq_bare | 0.3 | dens | +0.80 (0.94, 123) | −0.92 (0.61, 439) | +4.28 (0.99, 543) | +6.15 (0.90, 1735) | 7,761 |

- **h1 half ± SE for the staged arm (dens, FR 0.3):**
  - plain: US→India +0.000066 ± 0.000012, India→US +0.000104 ± 0.000014;
  - gm: +0.000336 ± 0.000026 / +0.000761 ± 0.000033.
- **Empty-S1 arm (`_E`):** France +77 pairs at 0.3. LOCO only reaches 1–2 empty S1 (precision 1.0), so it is not staged.
- **Margin guard** (best − second p ≥ 0.10, `key1_m10`): it removes almost nothing from key1 (6,447 of 6,505 pool pairs remain), so it was dropped.

**Expected LB.**
- Every LOCO arm of key1_bare gains ≈ 0.072–0.083 France-F0.5 per added pair, which implies a precision of about 0.99.
- 5,742 × 0.073 / 259,452 = +0.0016 France-F0.5, i.e. **LB +0.00024**.
- At a France precision of 0.90 (per add 0.9 × 0.075 − 0.1 × 0.19 ≈ 0.049): **+0.00016**.
- sub41_hn90 vs hn90: 2,536 adds → **+0.00007 to +0.00011**.

## What failed      (approaches tried and dropped, with the number that killed them)
- **Key rule without the bare guard** (`key1`, first staged at 00:50 at p ≥ 0.7, now superseded). Plain LOCO India→US −0.00003 (dens) / −0.00015 (pool), added precision 0.55–0.63. The 515 non-bare France pool pairs are the US-decoy shape (brand + S1 word).
- **Non-unique key** (`eq`, `eq_bare`): the argmax among several S1 sharing the exact key. Plain India→US −0.0001 to −0.0006 (precision 0.42–0.69): buildings with several businesses.
- **No-city key** (`key1nc`): no better than key1 (India→US −0.00004 to −0.0002) and fewer France adds.
- **Margin guard, empty-S1 arm:** see Evidence (no effect, or no LOCO support).
- **Not attempted:**
  - the 2,774 unassigned gib SX with no key S1 (perturbed address or no hn);
  - the 2,816 with 2+ key S1.
  - The key cannot place them, and P36 found the address-cosine top-1 near-tied (≈ 8 S1 within 0.05) and 19% precise in the holdout.

## Interfaces       (functions in probes/<id>/*.py, inputs/outputs, runtime + RAM at full scale)
**How to run:** from the repo root, `probes/P41-fr-brand/py.sh probes/P41-fr-brand/p41.py <cmd>` (2.2 GB cap). A bare `p41.py` runs the toy self-checks for `key_match`, `guards` and `rule`.

**Caches** go in `data/work/p41/` (≈ 100 MB):
- `keys_*` (hn / street / city / gib per record);
- `bare_*`;
- `train_gib_{c}` (train gib with its truth and key S1);
- `fr_gib` (the France profile frame).

**Functions in `p41.py`:**
- `keys(split, c, sx_rf)` / `keys_train(c)` → S1 and SX key frames. France takes 50 s (street key for 1.43M SX); train (gib SX only) about 2 min.
- `key_match(k1, kx, city=True)` → (key S1 index, number of S1 sharing the key) per SX. An empty hn or street never matches.
- `gib_keys(k1, kx)` → per gib SX: its unique key S1, with and without city.
- `bare(split, c, ids)` → one-token brand flag.
- `guards(s1_ids, sx_ids, p, gk, k1i)` → per-pair masks `gib`, `key1`, `key1nc`, `eq`, `key1_m10`, `key1_bare`, `eq_bare`.
- `rule(P, s1, sx, is2, p, am, pred, sib, t, g, empty=False)` → the pairs to add. It is P37's `gc_add(..., sib | ~g, t, "all")`: free argmax pairs into non-empty groups, caps 5 / 6. `empty=True` adds P37's `rescue` on the same guard.

**Commands:**
- `prof`: France profile → `prof.json` (43 s, 1.1 GB).
- `train_prof` → `train_prof.json` (≈ 2 min, 1.5 GB).
- `loco [gm]` → `loco.json` / `loco_gm.json` (≈ 3 min, ≤ 2 GB).
- `build T OUT [guard]`: the base is `$P41_BASE` (default `output/sub37_gc70`) → OUT with `p41_info.json` (21 s, 1.3 GB).

**Helper scripts:**
- `decoy.py`: the signature table → `decoy.json` (4 min, 1.6 GB).
- `check.py NEW BASE`: only France lines changed, and matches ⊆ candidates.

**Rebuild the queue:**
```
probes/P41-fr-brand/py.sh probes/P41-fr-brand/p41.py build 0.3 output/sub41 key1_bare
P41_BASE=output/sub37_gc70_hn90 probes/P41-fr-brand/py.sh probes/P41-fr-brand/p41.py build 0.3 output/sub41_hn90 key1_bare
```

## Risks            (licence, external-data, France/unseen-country, scale, anything you assumed)
- **Compliance:**
  - no model and no external data;
  - the 44 syllables are learned label-free from the split's own names (test-side unsupervised use is allowed);
  - `street_key` is P28's hand-written normaliser;
  - India/US lines are byte-identical, and every S1 row is kept;
  - the rule is written for France; the final zip should branch on `c in UNSEEN`.
- **The plain LOCO passes, but only at +0.0001.** Its pool is tiny, so the France-size effect rests on two arguments:
  - (a) train truth, where a bare brand at a unique key is 99.8% (US) / 96.1% (India) true;
  - (b) France's pool signature matching US true-bare SX.
  - If France had a decoy kind unseen in train (bare brand names placed at a unique S1 address), precision would fall. Two bounds:
    - **Key-less share.** 10% of France gib have no key S1 (US true 14%, decoys 81%), so the France decoy share is probably ≈ 0.
    - **US decoy density.** At the US's density of key-unique decoys, *with or without* a bare name (3,638 per 1.32M S1 → 713 for France's 259k), all of them landing in the adds would still leave precision ≥ 1 − 713 / 5,742 = 0.88. That is above the ≈ 0.68 break-even.
- **The p floor 0.3 is a judgement call.** LOCO prefers lower floors (0.0 is best everywhere). The floor guards against France-specific low p from c_hn's hn features, which LOCO (M2, no hn features) cannot see. Only 165 France pairs lie below 0.3.
- **Assumptions:**
  - I worked in the main checkout's `probes/P41-fr-brand/`, as P36 and P37 did, not in a separate worktree;
  - LOCO dF is read in France-F0.5 units, with LB = 0.1497 × dF (France's S1 share of test).
- **SageMaker:** not used. The master offered it, but every step runs locally in ≤ 4 min at ≤ 2 GB.
- **Disk:** two output folders × 97 MB (the matching files; the rest are hard links), plus data/work/p41 at ≈ 100 MB.

## Open questions   (for the master; each with your best guess)
- **Does key1_bare also help India/US?** Plain LOCO +0.0001 (France units) in both directions, but that is on M2, not on a10's p'. Best guess: ≤ +0.00002 LB. Not worth a slot.
- **Empty-S1 arm (+77 France pairs)?** Best guess ±0.00002 LB. Skip unless a slot is spare.
- **If #41 loses on the LB:** France's bare brands then behave unlike train. Drop the rule, and treat hn90's gib subset (3,206 pairs) as suspect too.
- **If #41 gains:** upload #41_hn90 next, then consider floor 0.0 (+165 pairs, LOCO-best everywhere; ≈ +0.000007 LB).
