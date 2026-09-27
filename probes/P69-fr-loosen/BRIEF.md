# P69-fr-loosen: are France's strict threshold / rules dropping GENUINE pairs? (user hypothesis, 27 Sep ~15:05) — FAST
Context: final = output/sub66_final, LB 0.987391 (best). `run.py --recipe sub66_final` rebuilds it byte-for-byte from cached
inputs (code/business_entity_resolution/src/run.py, france.py). France is ~15% of LB, no labels; est. France F0.5 ≈ 0.963 vs
India/US ≈ 0.991. France's chain accepts at t .9868 (set label-free from India/US empty rate). Prior evidence to weigh:
LOCO says France-like band p .5–.9 is only 20–44% true (P52); P42 joint tune picked t .99; P52 rescue .80/gc .50 ≈ +0.00005.
BUT the LB just showed precise France rules transfer (sub66_final beat its estimate), so re-check strictness directly.
France's unassigned SX are 59% candidates (India/US 42–48%) and France's never-candidate rate is LOWER (17.6% vs 22–24%),
so retrieval is not the France bottleneck — the decision is. Verify that too (one line).
A. FUNNEL for France on sub66_final: per stage (P31 chain t .9868, P30 veto, rescue90, gc70, hn extras, gib, acronym, P48 low-CE
   veto, P54 rescue, cls80, P55 veto/add, P64 veto, P66 adds, caps 5/6, one-owner/argmax) → pairs added/removed, S1 affected.
   For every REMOVED / rejected set: pooled-Qwen r distribution (share r≥.9, .5–.9, <.5) and m2 p deciles.
B. Unassigned France SX that ARE candidates, by argmax p bin (<.3, .3–.5, .5–.7, .7–.9, .9–.95, .95–.9868): count, median Qwen r,
   share whose argmax S1 lacks a member from that source, and LOCO-analogue precision in the same bin (P42/P52 frames).
C. Build loosened files on sub66_final: France chain t → .975 (output/sub66_t975) and → .95 (output/sub66_t95); ALL vetoes (P30,
   P48, P64) and caps kept; India/US byte-identical. Also a "vetoes-off" diagnostic count (no file) — how many pairs each veto
   kills and their Qwen r. Validator --check-ids, pairs added, Qwen r of added pairs vs the LB-proven gc70/rescue90 adds, LOCO
   chain dF for the same t change (P42 machinery, both directions), SUBMISSIONS rows with expected LB.
Verdict: which (if any) loosened file deserves an LB slot (the user has ~4 today). A single upload of sub66_t975 is itself a
direct France measurement — say whether it is worth it.
RAM ≤ 3 GB local, disk ~1.9 GB (keep outputs lean; offload per rclone pattern if needed). Target: tables 15:45, files 16:15.
Never end your turn to wait. Message the lead (amc-26-a7) with the tables.
