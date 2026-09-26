# P50-fr-moved-addr — exact-name SX at a DIFFERENT street in the same city: true "moved" variant or decoy?
Read probes/NIGHT/AGY_COMMON.md first, then probes/P40-generator/RESULT.md (the generator op catalogue) and
probes/P28-addr-anchor/RESULT.md.

Observation (master, unaccepted France argmax pairs p 0.3–0.99 on the P31 base): pairs where the SX name equals the S1
name (core equal, often raw-equal up to case/accents/legal form) and the city is equal, but the street AND house number
are completely different (hn_cls 2/3/4 in pairs_France.parquet), sit at p 0.3–0.6 and are rejected.

Task:
1. Train truth (full, US and India separately; gt_pairs + raw sources + rf features): for SX whose core equals its
   argmax/any S1's core, same city, different street and hn (not a sibling offset): what fraction are true for that S1?
   Split by: name raw-equal vs core-equal; core twin count k among the country's S1 (P40: twins are a coin flip);
   whether the S1 has other members at its own address; SX source (S2/S3).
2. Same on the holdout dumps (p23, holdout==True) restricted to pred==False, by p band — the gain lives there.
3. If a cell is >= 0.90 precise among pred==False in BOTH US and India, count the France analogue on the P31 proxy base
   (free argmax SX, hn_cls != 1), check label-free France stats agree (e.g. share of such pairs vs the US/India share
   among argmax pairs), and write p50.py build/selftest per AGY_COMMON. If no cell passes, say so clearly with the
   numbers — a clean negative is a fine result.
Deadline: RESULT.md final by 07:30 IST.
