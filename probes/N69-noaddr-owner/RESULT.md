# N69-noaddr-owner — RESULT (laptop, 27 Sep ~18:00 IST) — France add-on, staged `output/sub76_n69`

## TL;DR
- **Address-less SX are almost always true matches, and France leaves most of them unassigned.** Train truth: SX with an
  empty/NULL address are 97.7% true (India 0.978 / US 0.977; decoys keep an address). Test carries the same volume per S1
  (US 0.1665, India 0.138, France 0.166). sub76 assigns US 62%, India 57%, **France 35%**.
- **Rule (France / unseen-country only):** a free empty-address SX joins its argmax S1 when the S1's name core equals the SX's
  (name_rel eq, or typo), **that core belongs to exactly one France S1**, the pair is not a legal-form conflict (P64 class 4),
  France p >= 0.3, caps 5/6. Train: a unique exact core names the owner of an address-less SX 96–97% of the time.
- **LOCO (France-like chain, N63 frames):** US→India **+0.00065 ± 0.00006 France-F on both halves** (852 adds, prec 0.96);
  India→US −0.00002 ± 0.00004 (122 adds, prec 0.61, not significant). France's shape matches US→India: 82.3% of
  address-less unique-core pairs already accepted (US→India 81.7%, India→US 97.9%), leftovers at p .6–.98 like US→India
  (where they are 94–98% true). Typo arm passes both ways (+0.00006 / +0.00002).
- **`output/sub76_n69` = sub76 + 2,753 France pairs on 2,693 S1 (40 empties filled)**, p median 0.92. Only France rows change,
  adds only, no SX owned twice; official validator PASS (plain). md5 293e2a89…. Expected **≈ +0.0001 LB** (France-F +0.0007).
  Disjoint from sub77's P41 brand adds (those SX carry an address) → stacks on sub77.

## Rebuild on any base (Harsh's box, needs data/work/p31/pairs_France.parquet)
```
.venv/bin/python probes/N69-noaddr-owner/fr2.py output/<base>/matching_results.tsv          # 20 s: data/work/n69/fr_noaddr.parquet
.venv/bin/python probes/N69-noaddr-owner/build.py output/<base>/matching_results.tsv output/<base>_n69 output/<base>/candidate_pairs.tsv
```
The third argument appends any added pair missing from the base candidate set (P68 did the same).

## What failed (same probe)
- Shared core (≥ 2 France S1 carry it; 5,918 free pairs): LOCO prec 0.54–0.66, −0.0002/−0.0005. Unique full name within a shared
  core: train precision 0.76–0.94 only. Not staged.
- drop1 variants: −0.0001/−0.0002 both ways. sub1/add1: pass LOCO but France's are the decoy vocabulary (groupe, club↔école), excluded.
- India/US version: the in-domain combiner already takes these; the #8c holdout leftovers at a unique core are 30–47% precise.

## Other probes this session (all negative / small)
- N64 France gap map: France lists are short (3.20 vs 3.39 per S1, 5+ lists 20% vs 24.5%); free p≥.5 pool is ~96k, mostly
  sibling-offset decoys and org-word swaps. Clean legal-form cells are ≤ 1.3k pairs (+0.00004).
- N65 India/US bare-brand unique key: test free 667 (India) / 352 (US); US 0.993 precise, India 0.53 → ≈ +0.00001, not staged.
- N66 India/US empty-S1 rescue: negative every τ (empties are 96–97% true singletons).
- N67 France '(n)' hn parse: `france.street_key` still returns '' for '(4) R. …' (23k France SX), but acceptance of these pairs
  equals plain ones (the matcher's rf was fixed by P29). Only 83–95 brand pairs hidden from P41's key. No action.
- N68 class-deficit map vs US train law: France accepts MORE hn-equal pairs than US truth but half the no-hn / empty-address SX
  (→ N69) and almost no hn digit-drop pairs.
