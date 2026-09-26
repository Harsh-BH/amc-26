# P41-fr-brand: recover France's missing synthetic-brand SX (P36 hand-off)
Finding (P36, label-free, test #34v_rescue90): SX whose name is a generator brand (44 syllables, probes/P36-iu-residual/gib.py
`learn`/`is_gib`) are assigned at India 89.5% / US 73.3% but **France 54.8%** (31,035 gib SX); non-gib SX are 56–58% in
all three. At the US gib/non-gib ratio France would assign ≈ 70% → ≈ 4.7k missing France brand SX. P37 (train truth):
same-address SX with invented brand names are true variants (7,763 true vs 159 false at p > 0.6). P36: an address key
alone fails in France (≈ 8 near-tied S1 by address, top-1 address gap 0.023) → needs hn-aware scoring.
Base = output/sub37_gc70 (LB 0.986304, best); also report on top of output/sub37_gc70_hn90 (staged, overlaps: measure
how many of your adds hn90 already contains). Read probes/P36-iu-residual/RESULT.md (gib.py, recover.py),
probes/P37-fr-complete/RESULT.md (gc rule, hn90, LOCO protocol + the finding that the LB tracks India→US),
src/france.py (hn_feats, c_hn, sibling rule), data/work/p31/fr31v0.1_t9868.parquet. Own worktree; code in probes/P41-fr-brand/.
Task:
1. Profile France's unassigned gib SX: are they in the France candidate file? at what p? argmax to which S1? hn-equal /
   street-equal / sibling offset to their best S1? Contrast with India/US assigned gib SX (train truth: what makes a gib SX
   belong to its S1 — same hn + same street? the S1's group size/shape?).
2. Rule: assign a free gib SX to the S1 with exact hn + street(+city) equality, non-sibling, unique best (margin), caps
   5/6. Variants by strictness. LOCO both directions (quantile-matched p as P37 does), report added-pair precision
   (break-even ≈ 0.68 for groups already partly right, higher for empty S1).
3. If it passes (LOCO ≥ +0.0005 both ways, or India→US positive with precision ≥ 0.8 and US→India not below −0.0003 —
   flag which), build output/sub41 on sub37_gc70 and, separately, on sub37_gc70_hn90 (India/US byte-identical),
   validator PASS, rank in SUBMISSIONS.md queue, run.py notes.
RESULT.md with tables. Memory 3 GB. DISK ~3 GB free: small intermediates, never > 300 MB locally. Never end your turn to
wait. Deadline: first staged file by 27 Sep 05:00 IST, final by 08:30.
