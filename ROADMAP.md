# Roadmap: from prototype to a product people can rely on

Durations are rough estimates for a small team (1 to 2 engineers plus access to a domain
expert) and will move with your data access. Each phase has an **exit test**: do not start the
next phase until it passes.

## Where you are now (v2.0, built; status as of October 2026)

| Area | Status |
|---|---|
| Product detection + visible defects (crack, scratch, spot) + learned anomaly model | Works on synthetic data. On 61 real phone photos defect detection was at chance, so **a FAIL now needs a trained bank for the product and a photo close to the bank's; otherwise REVIEW** (`REQUIRE_BANK_FOR_FAIL`, `ANOMALY_TRUST_RATIO`). Needs your own rig photos |
| Auto-detect of the product (YOLO + whole-photo classifier) | Built and measured: 86.6% accuracy, 73% of products named, 9% wrong on unseen photographers; 12 of 15 brand-new photos right. Weak on the back/edge of a phone. Commons snapshots, not your rig |
| PASS / REVIEW / FAIL verdicts, image-quality gate, heatmaps, repair advice | Working |
| Live camera (WebSocket), batch API, edge runner, ONNX export | Working (ONNX for the anomaly backbone tested; YOLO export untested here) |
| Phone internal health: battery, storage, memory, symptoms -> score and advice | Working from supplied numbers. Android `adb` collector **not tested on a real phone** |
| Camera pulse (rPPG) | Working on synthetic signals. **Not validated on people** |
| Health screening framework (train, evaluate with CIs, Grad-CAM, triage, disclaimers) | Working. **Four research-grade models installed** (malaria_smear AUC 0.991, chest_xray 0.997, retina_dr 0.901, skin_lesion 0.959) after passing only a better-than-chance gate; **none is clinically validated** (see README). eye_disease was rejected (0.709); glaucoma and six more modules have no model |
| 11 camera-feasible modules with devices, evidence levels, referrals, red flags; dataset adapters (APTOS, HAM10000, PAD-UFES-20, malaria, chest X-ray, ODIR, CSV) | Adapters tested on **small fixtures that mimic each layout**, not on the real downloads |
| Temperature calibration, sensitivity-targeted thresholds, out-of-distribution guard, flip TTA, per-subgroup fairness report | Tested on synthetic and toy data only |
| Patient records (consent, owner-only, erasure, audit log) and the patient PDF report | Tested with the memory store; **Supabase tables not tested against a live project** |
| Supabase auth, RLS, storage, dashboard, Docker | Written. Supabase and Docker **not tested against real services** |

## What is possible and what is not

| You asked for | Reality | How this project does it |
|---|---|---|
| Phone **external** faults from a photo | Yes, for visible damage | Detector + OpenCV + anomaly model |
| Phone **internal** faults from a photo | No, a photo cannot see inside | Device-reported data (battery, storage, temperature), symptoms, and photos of *opened* devices (needs your data) |
| Human **external** disease from the camera | Only some skin/eye conditions, and only with a validated model | Screening modules you train and validate (skin lesion first) |
| Human **internal** disease from the camera | No. A camera cannot see organs | Needs the right sensor: ECG, X-ray, ultrasound, blood tests, a doctor |
| Pulse from a face video | Yes, under good conditions | `rPPG` module; no number when confidence is low |

## v2 next steps for the health track (in order)
1. Download **one** dataset you are allowed to use (APTOS is the easiest) and run the full chain in `MODULES.md`: prepare, train, evaluate on the locked test set, read the CI lower bounds.
2. Check the adapter against the real files and fix any column-name differences (they stop with a clear message).
3. For phone-only checks (anaemia, jaundice, oral, skin), plan **your own data collection**: image plus the reference test (haemoglobin, bilirubin, biopsy), with consent and skin-tone recording. Public data for these is scarce.
4. Run the subgroup report (`--subgroups`) and do not ship to a group where sensitivity is clearly lower.
5. Only then involve clinicians in a prospective study (Phase 6 below).

## Phase 0: Decide scope and success (week 0 to 1)
- Pick **one** product line (suggestion: used-phone grading) and decide what "defective" means.
- Write targets with numbers, e.g. "catch at least 95% of FAIL-grade defects with at most 3% of good units flagged", on a held-out set.
- Decide the human-health stance now: **"wellness / research use only"** until Phase 6 is done.
- **Exit test:** a one-page spec signed off by the people who will use the system.

## Phase 1: Real data and a baseline (weeks 1 to 4)
- Build the capture rig and follow the protocol in `DATASETS.md`; collect good units and defect examples.
- Run the three baselines on a **locked test set**: OpenCV only, anomaly bank, both.
  `python scripts/eval_anomaly.py ...`, `python scripts/benchmark.py ...`
- Check public sets (MVTec AD for method sanity, SSGD for glass defects if your use is non-commercial).
- **Exit test:** you have measured AUROC, defect catch rate and false-alarm rate with confidence intervals on your own data.

## Phase 2: Better detection (weeks 4 to 8)
- Label with Roboflow / CVAT; train a custom YOLO for product types and defect classes
  (`scripts/train_detector.py`): crack, scratch, dent, missing_screw, bulge, liquid_damage, gap.
- Add a **feedback loop**: reviewers confirm or correct each REVIEW/FAIL in the dashboard; corrections
  go to a retraining set. *(Built: History tab "Reviewer" column, `inspection_feedback` table, `POST /api/v1/inspections/{id}/feedback`, `GET /api/v1/feedback`. `scripts/export_feedback.py` exports the corrected photos into good/defective/borderline folders. Still to do: split by physical unit for a locked test set, and a scheduled retrain.)*
- Per-model banks and per-camera thresholds; re-calibrate when the lighting or camera changes.
- **Exit test:** Phase 0 targets met on the locked test set, with lower confidence bound above target.

## Phase 3: Phone internals (weeks 6 to 10, parallel with Phase 2)
- Test `collect_android_diagnostics.py` on at least 10 real phones across makers; fix parsing differences.
- Build a small **companion app** (Android first): BatteryManager and sysfs values, sensor presence
  and calibration, display dead-pixel colour test, touch-grid test, speaker/mic loopback, camera focus.
  Apps can read far more than adb; iPhone access is limited, so use manual entry plus Apple's battery data.
- Partner with a repair shop or refurbisher to collect **opened-device photos** and **symptom -> confirmed fault** logs.
  Use them to train an internal-photo model (swelling, corrosion, liquid indicator) and to tune the symptom rules.
- **Exit test:** on 100 phones with a confirmed fault list, the diagnostics flag at least 90% of battery-related faults without crying wolf on healthy phones.

## Phase 4: Production hardening (weeks 8 to 14)
- Real Supabase project, row-level security audit, secret management, backups, audit log of who inspected what.
- TLS reverse proxy, rate limiting, request size limits, dependency scanning, CI running the tests.
- Edge: export to ONNX/TensorRT, measure latency on the target device, watchdog and auto-restart, offline queue
  that syncs to Supabase when the network returns.
- Monitoring: dashboard warnings exist; add alerts (email/Slack), **drift detection** (score distribution shifts) and a
  weekly accuracy report from reviewer feedback.
- Privacy: privacy policy, retention periods, deletion on request, DPDP Act compliance review.
- **Exit test:** load test at 2x expected volume, a restore-from-backup drill, and a security review with no open high findings.

## Phase 5: Pilot with real users (weeks 12 to 20)
- Run beside the human process first (shadow mode): compare every verdict with the human grade.
- Pilot with 1 to 2 customers (refurbishers, repair chains, e-waste buyers, insurers, marketplaces).
- Track: time per inspection, disagreement rate, defects found after sale (returns), user trust.
- **Exit test:** disagreement with expert graders no higher than between two human graders.

## Phase 6: Human health track (start research now, ship only after validation)
Gate every step; none of it ships as a diagnosis until step 4 is complete.
1. **Pulse (rPPG):** validate against a pulse oximeter or ECG on at least 30 volunteers across skin tones, lights,
   and movement; report MAE and a Bland-Altman plot. Add face landmark tracking (YuNet/MediaPipe) and motion
   rejection. Do **not** claim blood pressure or SpO2 without calibration studies.
2. **Skin screening:** train on a dataset that matches your camera (dermoscopy sets will not transfer to phone photos),
   split by patient, report sensitivity and specificity with CIs per skin tone (`evaluate_classifier.py`).
3. **Other modalities, with the right sensor:** ECG patch/watch for rhythm, uploaded chest X-rays through a
   clinic workflow, retinal camera for diabetic retinopathy. A phone camera is the wrong sensor for these.
4. **Clinical validation and regulation:** prospective study with ethics-committee approval and a clinician-defined
   ground truth; seek regulatory advice early. In India, software that diagnoses or screens is likely a medical device
   under the Medical Devices Rules 2017 (CDSCO); FDA and CE/MDR apply abroad; expect ISO 13485 / IEC 62304 quality
   processes. **Confirm the exact classification with a regulatory consultant.**
5. Until step 4: market as **wellness / educational screening**, show disclaimers everywhere (already built in), and
   always route concerning results to "see a clinician".

## Suggestions: what I would do first

1. **Focus on used-phone grading.** It is the most achievable and sellable: external grade A/B/C from photos,
   plus the battery/storage health report, plus IMEI/blacklist and activation-lock checks (not built).
2. **Collect your own images from day one.** Everything else depends on them. Even 100 well-labelled phones beats any public set.
3. **Measure with confidence intervals on a locked test set**, and report the lower bound, not the average.
4. **Keep humans in the loop** for REVIEW and for every safety item (swollen battery, liquid damage).
5. **Treat the health features as a separate, later product** with its own consent, ethics, validation and legal path.
6. **Cheap wins to add next:** reviewer feedback buttons, an IMEI check, a printable inspection report (PDF) per unit,
   multi-view grading (front/back/edges combined into one grade), and e-mail/Slack alerts.
