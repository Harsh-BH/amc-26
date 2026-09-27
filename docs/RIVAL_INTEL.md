# Rival intel — 27 Sep 15:05 IST (public repos + live leaderboard)

Source: 333 public repos pushed since 25 Sep (README/log scan) + 1,942 commit-message hits, 4 deep reads
(AyanAhmedKhan, bhavyatalwar04, sumit-0507, HBX814, nisdudeheckz, rajesh08114, mightbeanshuu notes, Reddit
r/Btechtards). No rival code was copied; ideas only. Rival claims are theirs; nothing below is LB-verified for us.

## Where we stand (Unstop snapshot 11:53 IST, 6,938 teams scored)
| | score |
|---|---|
| #1 GG (LPU) | 0.991811 |
| #2 CDS_Team_iisc | 0.991483 |
| #10 | 0.990545 · 21 teams ≥ 0.990 · 48 ≥ 0.989 · 86 ≥ 0.988 |
| **top-50 cut** | **0.988905** |
| **top-100 cut** | **0.987804** |
| **us, sub66_final 0.987391** | **rank ~117** (0.986685 was 163) |

LB = 0.8502·India/US + 0.1498·France. #1 needs India/US ≥ 0.992 even with France 0.99, so the gap is NOT France
only (≥ 0.0018 of it is India/US). #1's only public hint (Reddit, 27 Sep 02:23): "think abt france tht is not in ur
training set"; a 0.988 team's hint: "synthetic". Best public repo: bhavyatalwar04 0.986201 (below us).

## Untested levers, ranked by expected LB per hour
1. **France noise-word tier on sub66_final** (~1 h, CPU). 14,636 unassigned France SX at the S1's house number whose
   name = S1 name + one noise word or one word swapped for it (groupe 5,203 swap / 1,432 insert; développement 4,754 /
   1,386; associés 835; 14,419 with a single such S1). S2/S3-vs-S1 over-representation: développement 10.6×, groupe
   4.7× (median word 0.9×) → copy noise, not decoys. LB evidence: AyanAhmedKhan filler fix +0.000138; nisdudeheckz
   per-country noise-ratio features +0.0056 (from 0.9765). P46 tier was dropped from sub_clean on a Qwen-judge
   precision of 0.08 — but a US/India-trained CE learned "group/sons/holdings = decoy", so it is biased on exactly
   these pairs. Stage: développement + fils (~6k) first, groupe second. Expected +0.0002–0.0005. Risk: #54's France
   rule-adds bundle netted ≈ −0.0001 (but that bundle also held P54 rescue ≈45% precise and P40 xf_noaddr ≈0.19).
2. **France hn-shift drop rule** (~1.5–2 h). Drop a France link whose house number differs by 3–100 when the same S1
   already has an exact-number link: 4,540 links (0.0175/S1). Precision lever (the kind that paid for us). Gate on
   India→US LOCO. Expected up to +0.0003 (≈ −0.0001 if France copies shift numbers like US ones).
3. **Raw-text corruption marks as features** (2–3 h, CPU; the only lever that also hits India/US). Owned vs distractor
   on train: all-lowercase name 5.3–8.0% vs 1.8–3.9%; all-caps (S2) 16–23% vs 11–16%; accents injected 5–7.5% vs
   2.7–4.9%; name tokens 3.1–3.5 vs 3.9–4.2; empty address 3.8–4.9% vs 0.2–0.3%. Our normaliser lowercases/folds
   before features, so the combiners never see these. Add to P44 (India/US) + P55 (France) combiners; holdout + LOCO
   gates. Expected +0.0001–0.0004.
4. **Qwen on ALL France pairs / two-model veto** (3–4 h, AWS GPU). bhavyatalwar04's biggest France gain: Qwen3-4B LoRA
   scored all 1.68M France pairs as a stacker column → LB +0.00106 (France +0.0054). Ours scores the band only.
   Veto confident France pairs when bge AND Qwen are low; re-admit P30 vetoes Qwen strongly endorses. +0.0001–0.0004.
5. **France CE self-training on real France pseudo-labels** (4–5 h, GPU): P61 agreement labels (combiner & Qwen ≥ 0.7
   pos, ≤ 0.3 neg), 1 epoch LoRA/bge at low lr with 1:1 source replay (bhavya "France r1", pkd). Alternative to P67's
   synthetic pairs. +0.0002–0.0008, unmeasured anywhere on an LB.
6. **France-blank probe of sub66_final** (one slot, no rank cost): exact France vs India/US split of our score.

Smaller: India/US unique-key band recovery (shared address tokens → exactly one S1; shared name tokens → exactly one S1;
+0.00003–0.0001); P40 full-name address-less adds where the core is shared (+0.00002–0.0001); France parser gaps
"Appt/Apt" read as hn (~1,200 S1) and "12 B / 12 T" bis/ter (1,795 S1, 20.8k SX).

## Rival evidence that constrains us
- Confident-CE France adds are ≈45% precise on the LB (bhavya v8i −0.0005); our P54 rescue delivered ¼ of expected.
  sub66_final does not carry P54's adds — keep it that way.
- Precision/CE changes paid on our LB (#66_final +0.0007); rule adds did not (#54 France adds ≈ −0.0001).
- Addresses are noised per source (22–24% of same-source copies share the exact raw address) and decoys copy
  addresses verbatim (24% of exact-address groups span another S1): "same address ⇒ same owner" is unsafe.
- Test has ~1.9× distractors per S1 (house-number branch records); our accepted India/US links match the train-truth
  hn-class mix, so they are not leaking in.
- Group/sibling features gained on validation and lost on the LB for 4 teams.
