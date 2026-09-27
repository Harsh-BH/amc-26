# A-series common rules (27 Sep evening; read probes/NIGHT/AGY_COMMON.md FIRST — all its hard rules apply)

Deadline: the round closes 27 Sep 23:59 IST. Your RESULT.md must be FINAL by 21:45 IST. Draft RESULT.md by 20:00.
Current LB best: sub77 = 0.988292. Target 0.989. LB weights: India 0.468, US 0.383, France 0.149 (France-F +0.001 = LB +0.00015).

Bases on this laptop:
- France test base: output/sub76_n69/matching_results.tsv + candidate_pairs.tsv (= sub76 + N69; India/US lines = sub77's IU).
  sub77 itself (= sub76 + P41 brands) is NOT local: re-derive P41's adds with probes/P41-fr-brand/p41.py if you must
  dedupe against them. Your build must work on ANY base folder (the lead will apply it to sub81).
- France test pairs: data/work/p31/pairs_France.parquet (always pass columns=[...]; 1.33M rows).
- France-like LOCO frame (use this for ALL France gates): data/work/n63/chain_{US_India,India_US}.parquet + meta_*.npz,
  built by probes/N63-floor-probes/chain.py. Columns: s1 sx p y am src rel hrel n54 sib hn_eq dl c1 cx h1 hx acc pos noaddr
  name_1 name_x addr_1 addr_x. See probes/N76-hn-edit-ops/gate.py for exactly how to score a rule
  (fbeta per S1 via pos/ntrue, capped_add from probes/N63-floor-probes/cells.py). Report per-half (h0/h1) ± SE like N76.
- India/US holdout: data/work/p23/pairs_{US,India}.parquet (#8c; holdout==True rows only), data/work/p34 (a10 stacker),
  data/work/p42/ho_*.parquet.

France gate: LOCO France-F delta > 0 in BOTH directions (US→India and India→US), each half positive, added-pair precision
≥ .80 in both. Report expected LB = mean LOCO France-F delta × 0.149 × (France pool size / LOCO pool size per S1 ratio) —
state your scaling.
Dedupe against what the base already contains (only add SX not assigned to any S1 in the base; caps 5 S2 / 6 S3).
Memory: ONE python job at a time, < 2 GB. Another agent runs next to you.

## STRICTER GATE (added 19:10 by the master; this overrides the France gate above)
LB lesson (P70 addendum 4): sub77_hnst passed LOCO on the US side but scored −0.00045 on the LB; France behaved like
India's free pool. The lead now ships a France class only if, in BOTH LOCO targets, the pool of pairs the reference
leaves FREE (unaccepted) for that class has precision with Wilson 95% lower bound ≥ .75 and n ≥ 200. Also report the
free-share diagnostic: the share of the class left free in France vs in each LOCO target. Treat an arm as shippable only
if its added-pair precision is ≥ .95 in both directions (only ≥ .98-precise tiers have proven out on the LB).
