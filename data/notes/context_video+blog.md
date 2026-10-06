## Business Entity Resolution — video + blog extraction, plus PS/guidelines PDFs, zip, and dataset

### PDFs, zip, dataset (the additional ask)
- **PS PDF** (`/home/harsh1/Downloads/6ab5628d5a817_amazon_ml_challenge_problem_statement.pdf`) and **guidelines PDF** (`/home/harsh1/Downloads/6ab56657b4f1a_guidelines_and_key_instructions_amazon_ml_challenge_2026.pdf`): opened directly (not just the `.txt` extracts). Both are plain text layouts, no diagrams/logos beyond what's already in `problem_statement_pdf.txt`/`guidelines_pdf.txt` — nothing visual was lost in the earlier text extraction.
- **student_resource.zip** (`/home/harsh1/Downloads/6ab10eb3b23ba_student_resource.zip`, 1.09 GB / 2.52 GB uncompressed): matches what's already extracted at `data/student_resource/` (2.4 GB) file-for-file — no need to re-extract.
- **Dataset** (`data/student_resource/dataset/`): `train_source1` 2,206,821 rows / `train_source2` 5,034,616 / `train_source3` 5,285,603 / `train_ground_truth` 2,206,821 (one row per S1, matching the DRAFT note); `test_source1` 1,732,544 / `test_source2` 4,887,273 / `test_source3` 5,082,316 — confirms the "~1.7M test entities" figure. Sample rows look like the PDF's description (noisy names/addresses, `S1-`/`S2-`/`S3-` prefixes, `country` column).

### (a) Timestamped video summary
6-slide deck (`1920x1080`, `355.87s`, mono AAC narration) — `video.mp4` and `transcript.txt` in `.../scratchpad/video/`.

- **0:00–0:29** — Slide 1/6, title "Business Entity Resolution": states the 3-source, no-shared-ID problem.
- **0:29–3:00** — Slide 2/6 "From sign-up to matched records" (THE PROBLEM), builds through 5 sub-panels: (1) an Amazon Business sign-up form capturing only name+address; (2) the same business as three noisy source records; (3) an entity-resolution diagram (match/no-match, incl. a "look-alike" example); (4) a blocking-table walkthrough naming `candidate_pairs.tsv`; (5) a matching-model diagram producing `matching_results.tsv`, labeled "SCORED ON LEADERBOARD."
- **3:00–3:49** — Slide 3/6 "Train to learn, test to prove it" (THE DATASET): train has all 3 sources + `train_ground_truth.tsv`; test has the same 3 sources, no labels; ground truth is one row per S1 entity; all files tab-separated, `sep="\t"`.
- **3:49–4:40** — Slide 4/6 "A file, then a package" (WHAT YOU SUBMIT): during-challenge upload is `matching_results.tsv` only; final zip needs `output/matching_results.tsv`, `output/candidate_pairs.tsv` (audited), `code/…` (runnable), `Documentation_template.md`; reminder to run `utils/validate_submission.py`.
- **4:40–5:46** — Slide 5/6 "Judging & tips" (JUDGING & TIPS): `F_0.5` formula, "a false merge costs about twice as much as a miss," singleton scoring (empty=1.0, any match=0.0), "blocking sets your recall ceiling," region-specific patterns, boxed warning "No external data lookup."
- **5:46–5:56** — Slide 6/6 "All the best," closing.

### (b) NEW or CONTRADICTING facts, tagged by source
**Video (slides + narration):** adds **no new rules, deadlines, or constraints**. It's a faithful narrated/visual recap of `problem_statement_pdf.txt` with one worked synthetic example (Acme Robotics across S1/S2/S3). Notably it says nothing at all about dates, submission caps, team size, the Google Form, login rules, or the top-100/50/10 stages — those live only in the guidelines PDF / Unstop page, not the video.

**Blog** (`https://builder.aws.com/content/3HiM6zDmFrF98fRzOUETnGFDoqz/...`, fetched via `r.jina.ai` since WebFetch/curl only got the JS-shell — `blog_jina.txt` in scratchpad) — this is where essentially everything new is:
- **[blog] Eligibility (not in any written source):** engineering students in India only (B.E./B.Tech/M.Tech/M.S./PhD), graduation year **2027 or 2028**, team size **2–4** (cross-college teams allowed), must have a valid **AWS Builder Center Profile ID** to register.
- **[blog] Prizes (not in any written source):** Winner ₹1,00,000; 1st runner-up ₹75,000; 2nd runner-up ₹50,000 (all + certificates/goodies) = ₹2.25 lakh total; **Pre-Placement Interviews at Amazon for the Top 50 teams**; certificates + Amazon swag for Top 10 and Top 10 women-only teams.
- **[blog] Registration window (not in written sources):** 7–22 Sep 2026 (separate from the 25–27 Sep hackathon window already known).
- **[blog] Scale:** "79,000+ students registered."
- **[blog] Prep session:** a live "Best Practices" session was hosted 21 Sep 2026, 5–6 PM IST, recording on Twitch (linked in post) by Jatin Mehrotra (AWS Developer Advocate) — this is what the guidelines PDF's "Prep before you start" link points to.
- No contradictions found between video/blog and the written sources on anything the written sources actually cover (metric, format, deadlines, no-external-data rule all match exactly).

### (c) AWS credits / SageMaker / compute — from the blog only (video says nothing about AWS)
- **$200 AWS credit for every participant**, no credit card required: $100 on signup + $100 more for completing 5 starter activities (e.g., launch an EC2 instance, an RDS database).
- **Extra $100 credits at the 48-hour mark for the Top 500 teams** — i.e. up to $300 total is possible mid-hackathon, not just $200.
- Separate, stackable **"Student Rewards"** program (launched 20 Aug 2026, up to **$579** value, no card ever required): verified student status → 12 months Skill Builder Premium ($449); 7 badges → $10 credit; 14 badges → $20 credit; 21 badges → $100 AWS cert voucher. Badges come from routine Builder Center activity (sign-ins, comments, articles).
- **SageMaker free tier** (2 months): Notebooks 250h on `ml.t3.medium`; Training 50h on `ml.m5.xlarge`; Inference 125h on `ml.m5.xlarge`; plus S3 5GB and Lambda 1M req/month always-free.
- **Caveats called out:** set billing alerts immediately; always delete endpoints (~$0.12/hr while idle, even a *stopped* notebook instance does not bill, only a live endpoint does); use `us-east-1` for best compatibility.
- **Recommended workflow for this challenge specifically:** a plain **Notebook Instance** (not Studio — "works on all accounts without quota issues"), **local training inside the notebook** (no S3 upload, no separate Training Job) with plain XGBoost/scikit-learn, **local prediction** via `model.predict()` rather than a hosted Endpoint — "you submit a CSV [file], not a running API." The blog explicitly frames Training Jobs / Endpoints as only needed for bigger data or GPUs, which the tiny synthetic demo dataset didn't require — worth weighing against `amc-26`'s existing `sm_launch.py` boto3-Training-Job framework once real dataset scale is known.
- The demo itself is a generic XGBoost churn-prediction walkthrough (synthetic AWS tutorial dataset) — not entity-resolution-specific, included only to teach the SageMaker mechanics.

### Notes on method
- Video frames: `ffmpeg` scene-detection (`select='gt(scene,0.25)'`) found **zero** cuts — this is a slide deck with soft cross-fades/builds, not hard cuts — so I switched to a fixed grid (`fps=1/18`, 20 frames) plus an `mpdecimate` change-frame pass (15 frames), viewed all of them.
- Whisper: GPU path failed (`RuntimeError: Library libcublas.so.12 is not found`, ctranslate2's CUDA build needs a system cuBLAS 12 the box doesn't have) — fell back to `device="cpu", compute_type="int8"` per the task's own fallback instruction; transcription completed fine, `language=en prob=0.998`.
- Did not touch the Google Form or any login page.

Files: `/tmp/claude-1001/-home-harsh1-github-repos-amc-26/b9a894ea-d7e4-441a-9f08-e7799e3d80b9/scratchpad/video/{video.mp4,transcript.txt,frames_grid/,frames/}`, `/tmp/claude-1001/-home-harsh1-github-repos-amc-26/b9a894ea-d7e4-441a-9f08-e7799e3d80b9/scratchpad/blog_jina.txt`.