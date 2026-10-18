# VisionInspect

Computer-vision quality inspection for products (phones, laptops, watches, tablets, ...) plus
an image-based health-screening module. Python, PyTorch, OpenCV, FastAPI, Supabase, Docker.

- **Detects the product** (YOLO, or a classical fallback) and **checks it for visible defects**
  with two complementary methods: OpenCV checks (cracks, scratches) and a learned anomaly model
  (PatchCore-style) trained on photos of *good* units only.
- Returns **PASS / REVIEW / FAIL** with reasons, defect boxes, an annotated image and a heatmap.
- **Real-time** via WebSocket (live camera in the browser, or `scripts/run_camera.py` on an edge box).
- **Supabase** for login (JWT), Postgres tables with row-level security, and a private image bucket.
- **Dashboard**: inspection metrics, daily trend, defect types, system warnings, edge stability.
- **Repair advice**: every defect comes with likely causes, urgency and what to do. Safety items
  (swollen battery, liquid damage) always rank first and trigger a QUARANTINE disposition.
- **Phone internals**: a health check from what the phone itself reports (battery capacity, cycles,
  temperature, storage, memory) plus symptoms, with an Android `adb` collector.
- **Live vitals**: camera pulse estimate (rPPG). Only colour numbers leave the browser, nothing is stored.
- **Health screening**: 11 camera-feasible modules (retina / diabetic eye disease, glaucoma, multi-disease eye, external
  eye, skin lesion, skin conditions, anaemia from the inner eyelid, jaundice, mouth lesion, malaria smear, chest X-ray), each
  labelled with the device it really needs and its evidence level. Calibrated probabilities, sensitivity-targeted triage,
  an out-of-distribution guard, Grad-CAM and mandatory disclaimers. **No clinical model is included**: four
  research-grade models are installed (malaria smear, chest X-ray in children, diabetic retinopathy, skin lesion on
  dermatoscope photos) after passing only a "clearly better than chance on a locked test set" gate; none is validated
  for clinical use (see [Project status](#project-status-october-2026) and [MODULES.md](MODULES.md)).
- **Patient report (PDF)**: per body area ("EYE (RETINA): needs professional evaluation"), plain-language English plus
  Roman-Hindi lines, heat-maps, referral and red flags, model details and limits, and a clinician sign-off block.
  Never states a diagnosis. Patient records are saved only with consent and a signed-in user, with an audit log.
- **Real-data tooling**: leakage-safe dataset splitting, honest evaluation with confidence intervals.
  See [DATASETS.md](DATASETS.md) (verified public datasets and their licences) and
  [ROADMAP.md](ROADMAP.md) (phases, exit tests, and what is and isn't possible).

## Project status (October 2026)

**Works and is tested** (89 automated tests pass): detection and verdicts, image-quality gate, repair advice, live camera
(WebSocket), batch API, dashboard, phone-health scoring from supplied numbers, camera pulse on synthetic signals (low/high
bpm zones drive the dashboard siren), patient records with consent / ownership / erasure / audit log, the patient PDF.

**Auto-detect of the product.** YOLO (COCO) cannot name headphones, tablets or wrist watches, and it mislabels parts of a
product. A whole-photo classifier (frozen ResNet50 + a small linear head, `scripts/train_product_classifier.py`) names the
product, with rules for the cases YOLO gets wrong (a laptop seen as "keyboard", worn watches/headphones). Measured on
photographers it never saw: 86.6% accuracy, 73% of real products named correctly, 9% of its answers wrong, and 0 of 135
non-product photos called a product. On 15 completely new photos 12 were right. Weak spots: the back or edge of a phone
(named "tablet"/"laptop"), tablet and headphone precision, unusual devices. Photos are Wikimedia Commons snapshots labelled
by search text (attributions in `data/product_types/*/attribution.csv`); this is **not** accuracy on your own rig.

**A FAIL needs evidence we trust.** On real, uncontrolled photos the OpenCV crack finder and the phone anomaly bank (built
from 40 synthetic drawings) have no skill (intact and cracked phones scored alike), so the pipeline returns REVIEW, never
FAIL, when (a) the product has no trained bank, (b) the photo is nothing like the bank's reference photos
(`ANOMALY_TRUST_RATIO`, default 6.5), or (c) the product was never located. Set `REQUIRE_BANK_FOR_FAIL=false` only for a
controlled rig. **To get real FAILs: photograph 50 to 200 good units per model on your capture rig and run
`scripts/build_memory_bank.py`.** Everything else in the quality story depends on that data.

**Health screening models installed** (locked test set, 95% interval; "gate" = lower AUC bound at least 0.70):

| Module | AUC (n) | Read this first |
|---|---|---|
| `malaria_smear` | 0.991 (1,259) | cell crops from thin smears, split by slide |
| `chest_xray` | 0.997 (416) | children only, one hospital's mirror; the near-perfect score suggests it may be reading image style, not disease |
| `retina_dr` | 0.901 (590) | refuses 55% of photos and finds about 49% of cases; no patient id, so optimistic |
| `skin_lesion` | 0.959 (514) | dermatoscope images, non-commercial licence, does not transfer to phone photos; weaker for age 30 to 49, women, legs |

`eye_disease` was trained and **rejected** (AUC 0.709, lower bound 0.651). `glaucoma` and six other modules have no model.
The glaucoma mirror is stored as PNG folders and is dominated by different sources per class; `scripts/prepare_hf.py glaucoma`
now samples equally per source, but it was not trained. None of this is a diagnosis: the app shows it as wellness screening.

**Not yet done (needs real data, services or people):** real capture-rig photos and per-model banks; testing on real phones
(`adb` collector) and real people (pulse); Supabase tables, RLS and Docker against live services; load, backup and security
drills; clinical validation and regulatory review. See [ROADMAP.md](ROADMAP.md).

## What it can and cannot do

| It can | It cannot |
|---|---|
| Find visible cracks, scratches, stains, and anything that looks different from known-good units | See inside a device from a photo. Internals come from the phone's own data (`/api/v1/diagnostics/phone`), not the camera |
| Say "this photo is too blurry/dark to judge" instead of guessing | Prove a product is genuine or new |
| Learn your own products from good-unit photos (30 to 200 per type) | Catch defect types it has never been shown, unless they look visibly unusual |
| Run at roughly 10 FPS on a 2-core CPU (measured on synthetic images) | See inside a person: a camera cannot detect internal disease; it can estimate pulse and screen some visible skin conditions after you train and validate a model |

A **PASS** only means "no visible problem found in this photo".

## Quick start (no Supabase needed)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-extra.txt        # YOLO + ONNX (recommended)
cp .env.example .env                         # then set AUTH_REQUIRED=false for a local demo

# try it with synthetic data
python scripts/make_demo_data.py
python scripts/build_memory_bank.py --images data/demo/good --product phone --no-crop
python -m uvicorn app.main:app --port 8000
# open http://localhost:8000
```

Without Supabase credentials the app uses an in-memory store (data is lost on restart) and the
dashboard says so in its warnings.

## Supabase setup

1. Create a project at supabase.com.
2. **SQL Editor** → paste `supabase/schema.sql` → Run. This creates `inspections`, `screenings`, `patients`, `audit_log`,
   `system_events`, the daily view, row-level-security policies and a private `inspections` bucket.
3. **Project Settings → API**: copy the project URL, `anon` key, `service_role` key and the
   **JWT secret** into `.env`.
4. **Authentication → Providers**: Email is on by default, so users can create an account from the dashboard's sign-in
   screen. Set the **URL Configuration** and, if you want "Continue with Google / GitHub", enable those providers:
   step by step in [AUTH_SETUP.md](AUTH_SETUP.md).
5. Set `AUTH_REQUIRED=true` in production.

Security model: the browser only ever gets the `anon` key. The backend verifies each user's
token, then writes with the `service_role` key and always sets `user_id` from the verified
token. RLS means that even direct database access only shows users their own rows.
**Never expose the service-role key.** Medical images are sensitive personal data: keep the
bucket private, and check GDPR / HIPAA / India DPDP obligations before storing them.

## Docker

```bash
docker compose up --build            # CPU, port 8000
# GPU: TORCH_INDEX=https://download.pytorch.org/whl/cu124 docker compose up --build
#      and uncomment the deploy.resources block in docker-compose.yml
```
`./models` is mounted into the container so trained weights survive rebuilds.
Camera access in the browser (`getUserMedia`) needs HTTPS (or localhost): put the container behind
a reverse proxy such as Caddy or nginx, which should also add rate limiting.

## Training for your own products

Three levels, use as many as you need:

1. **Nothing to train**: product detection with COCO YOLO (phone, laptop, keyboard, tv, mouse,
   remote, clock) and OpenCV crack/scratch checks work immediately. COCO has no "watch" class
   (`clock` is mapped to `watch`, which also matches wall clocks), so fine-tune for real watch work.
2. **Anomaly bank, from good photos only** (recommended first step):
   ```bash
   python scripts/build_memory_bank.py --images data/good_phones --product phone
   ```
   Use the same camera, lighting and distance as production, include normal variation, and use a
   different bank per product type (`default` is the catch-all). The script holds out 20% of the
   photos to calibrate the FAIL threshold and prints the held-out scores. It needs internet once to
   fetch ImageNet weights; offline it falls back to a random backbone, which works but is weaker.
3. **Custom YOLO**, when you need named defect classes (dent, missing screw...) or reliable
   product types: label images (Roboflow / CVAT), then
   ```bash
   python scripts/train_detector.py --data data/products.yaml --epochs 80
   # then set YOLO_WEIGHTS=models/yolo/products.pt
   ```

**Learn from reviewer corrections.** Once reviewers have confirmed or corrected verdicts in the History tab,
export those photos, sorted by the reviewer's verdict (needs Supabase, since photos are kept there):
```bash
python scripts/export_feedback.py --out data/feedback                 # add --only-overrides for just the disagreements
python scripts/build_memory_bank.py --images data/feedback/phone/good --product phone    # false alarms become good examples
```
It writes `<product>/good|defective|borderline/`, `labels.csv` (model vs reviewer, note) and `conflicts.csv`
(inspections where two reviewers disagree; not exported). Re-running only downloads new photos and moves a photo if its
reviewer changed their verdict. Keep a locked test set that the bank is never built from.

Check results on **images the bank never saw** (`scripts/benchmark.py`), and tune
`ANOMALY_FAIL_RATIO` / `ANOMALY_REVIEW_RATIO` to your tolerance for false alarms vs misses.

## Phone internal health (battery, storage, symptoms)

A photo cannot show a swollen battery inside a sealed phone or a worn cell, but the phone knows.

```bash
# Android: enable USB debugging, plug in, then
python scripts/collect_android_diagnostics.py --symptoms overheating fast_drain
python scripts/collect_android_diagnostics.py --post http://localhost:8000 --token $TOKEN
```
iPhone values (Settings > Battery) can be typed into the dashboard's **Phone health** tab. The result is
a 0 to 100 score, a verdict (GOOD / FAIR / POOR / UNSAFE), findings with possible causes, and next steps.
Which fields Android exposes depends on the maker and version; missing ones are skipped and the
`data_completeness` field says how much the score is based on. It scores supplied data and lists
*possible* causes: it cannot see board, connector or water damage.

## Live vitals (pulse from the camera)

Open the **Live vitals** tab, fit your face in the oval in even light and keep still for ~10 s. The browser
averages skin colour per frame and sends only numbers over `/ws/vitals`; no video leaves your device and
nothing is stored. A low-confidence signal returns **no number** instead of a guess. Tested on synthetic
signals only (55 to 150 bpm recovered within 3 bpm); real-world accuracy depends on lighting, motion and
skin tone and must be validated on people (see ROADMAP.md, Phase 6). It estimates pulse only.

## Preparing real datasets and evaluating honestly

```bash
# skin lesions: split BY LESION so the same lesion never appears in train and test
python scripts/prepare_ham10000.py --src ~/Downloads/ham10000 --out data/skin_lesion
# any folder-per-class dataset, grouped by patient/unit in the file name
python scripts/prepare_folder_dataset.py --src data_raw --out data/chest_xray --group-regex '^(P\d+)_'

python scripts/train_classifier.py --data data/skin_lesion --name skin_lesion --concern suspicious
python scripts/evaluate_classifier.py --name skin_lesion --data data/skin_lesion/test   # AUC, sens/spec with 95% CIs

# one tool for the other public sets (aptos, ham10000, pad_ufes, malaria, chest_xray, odir, csv); splits by patient/slide
python scripts/prepare_dataset.py --list
python scripts/prepare_dataset.py aptos --src ~/data/aptos2019 --out data/retina_dr
python scripts/evaluate_classifier.py --name retina_dr --data data/retina_dr/test --subgroups data/retina_dr/subgroups.csv   # fairness per sex / age / skin type
python scripts/eval_anomaly.py --mvtec path/to/mvtec_ad/bottle                          # or --train-good/--test-good/--test-bad
```
The trainer calibrates the image-quality gate from your training images, so dark X-rays or small cell crops
are not wrongly rejected. Most public sets are research-only or non-commercial: read DATASETS.md first.

## Edge deployment

- `scripts/run_camera.py --source 0|rtsp://... --max-fps 5` runs without the web server; frames are
  grabbed on a background thread (no backlog), verdicts are smoothed over 5 frames, rejects are saved
  locally and optionally to Supabase (`--supabase`).
- Inference uses `torch.inference_mode`, FP16 on CUDA, and a thread lock around YOLO.
- Lower `ANOMALY_IMAGE_SIZE` (e.g. 224) or use `yolov8n` for speed.
- `scripts/export_onnx.py` exports YOLO (ONNX / TensorRT / OpenVINO) and the anomaly backbone (ONNX).
- `/ready` and the dashboard's "Edge deployment health" show CPU, memory, error rate, p50/p95
  latency and a stability score; thresholds for warnings live in `app/metrics.py`.

## API

Interactive docs at `/docs`. Send `Authorization: Bearer <supabase access token>`.

| Method | Path | |
|---|---|---|
| POST | `/api/v1/inspect` | multipart `file`, optional `product_type`, `save`, `annotate` |
| POST | `/api/v1/inspect/batch` | up to 20 `files` |
| GET | `/api/v1/inspections` | your history (with the reviewer's verdict where given) |
| POST | `/api/v1/inspections/{id}/feedback` | reviewer confirms or overrides the model: `{"verdict": "PASS\|REVIEW\|FAIL", "note": "..."}` (one per reviewer, resending replaces) |
| GET | `/api/v1/feedback` | reviews plus agreement rate, false alarms and misses |
| WS | `/ws/stream?token=&product=` | send JPEG frames, receive JSON verdicts (send the next frame after each reply) |
| GET | `/api/v1/analytics/summary` | dashboard data |
| POST | `/api/v1/diagnostics/phone` | JSON battery / storage / memory / symptoms -> health score and advice |
| GET | `/api/v1/diagnostics/phone` | your diagnostics history |
| WS | `/ws/vitals?token=` | send `{"samples": [[t_ms, r, g, b, skin_frac], ...]}`, receive pulse estimates |
| POST | `/api/v1/vitals/estimate` | same samples in one request |
| GET | `/api/v1/screen/modules` | screening modules and install status |
| POST | `/api/v1/screen/{module}` | multipart `file`, optional `explain=true`, `patient_id` |
| POST | `/api/v1/patients` | create a patient (consent required); GET list / GET, DELETE `/{id}` (erases screenings), GET `/{id}/screenings`, GET `/audit/log` |
| POST | `/api/v1/report` | screening results (+ patient or `patient_id`) -> PDF report |
| GET | `/api/v1/report/sample` | clearly labelled SYNTHETIC example report |
| GET | `/health`, `/ready` | liveness / readiness |

```bash
curl -H "Authorization: Bearer $TOKEN" -F file=@phone.jpg -F product_type=phone \
     http://localhost:8000/api/v1/inspect
```

## Health screening: read this first

**Can code detect human disease from images?** Research systems can detect some conditions at
expert level on specific data (diabetic retinopathy in retinal photos, pneumonia on chest X-rays,
skin lesions), and a few are cleared medical devices. But:

- This project ships **no trained clinical model** and makes **no diagnostic claims**. A module
  is enabled only when you train and add one.
- Models fail quietly on images from different cameras, hospitals or populations than their
  training data. Accuracy on a public dataset is not accuracy on real patients.
- Using output to make health decisions needs clinical validation on an independent dataset and
  regulatory clearance (CDSCO in India, FDA, CE/MDR). Treat this as a research / education tool.
- Every response carries `not_a_diagnosis: true` and a disclaimer; poor-quality or low-confidence
  images return `INCONCLUSIVE` rather than a guess.

What a camera can look at, per module, with datasets and their licences: **[MODULES.md](MODULES.md)**. Internal organs
(heart, lungs, liver ...) are not visible to a camera; those modules (chest X-ray, malaria smear, retina) need the
matching device and are labelled so in the UI and the report.

Train a module (suggested public datasets are listed by `GET /api/v1/screen/modules`; check each licence):

```bash
# data/chest_xray/{train,val}/{normal,pneumonia}/*.jpg   <- split by PATIENT, not by image
python scripts/train_classifier.py --data data/chest_xray --name chest_xray \
       --arch densenet121 --concern pneumonia --dataset-note "RSNA pneumonia"
```
It writes `models/screening/chest_xray.pt` + `.json` (+ `.ood.npz`) and prints sensitivity, specificity and AUC per
class. It also fits temperature scaling, triage thresholds for a target sensitivity (`--target-sensitivity`, default 0.90) and
an out-of-distribution guard, all on the validation split (so confirm on a locked test set). Judge a screening model by those, not by accuracy: for rare diseases accuracy is misleading.

## Tests

```bash
pip install -r requirements-dev.txt && python -m pytest tests -q
```
61 tests cover the defect finder, quality gate, anomaly bank (build / save / load), smoothing,
every API route, validation limits, JWT auth (expired, wrong audience, per-user isolation), both
WebSockets, the screening flow, repair advice, Android output parsing, phone diagnostics rules, rPPG pulse
recovery and its refusal to guess, leakage-safe splitting, the 7 dataset adapters (on small fixtures that mimic each
published layout), calibration / threshold / OOD maths, a real train-load-predict-evaluate round trip (single and multi-label),
patient consent / isolation / erasure / audit, and the PDF report (content, hostile and non-Latin input, "incomplete never
reads as all-clear").

## Layout

```
app/
  main.py, config.py, auth.py, db.py, metrics.py, deps.py
  inspection/  detector.py cv_defects.py anomaly.py pipeline.py temporal.py types.py
  screening/   classifier.py catalog.py calibration.py     report/  summary.py builder.py sample.py
  repair/      advice.py            diagnostics/  phone.py android.py     vitals/  rppg.py
  routes/      inspect.py screening.py stream.py analytics.py diagnostics.py vitals.py patients.py report.py
  static/dashboard.html
scripts/       build_memory_bank  train_detector  train_classifier  export_onnx  run_camera  benchmark
               make_demo_data  collect_android_diagnostics  prepare_ham10000  prepare_folder_dataset
               prepare_dataset  make_sample_report  export_feedback
               evaluate_classifier  eval_anomaly
DATASETS.md    MODULES.md    ROADMAP.md
supabase/schema.sql    tests/    Dockerfile    docker-compose.yml
```

## Tuning notes

- **The OpenCV crack/scratch checks are for controlled capture only** (fixed camera, plain background, screen off). On
  uncontrolled snapshots (hands, tables, shop displays, screens on) they fire on bezels, speakers, logos, text and
  icons and do not separate cracked from intact phones: in a test on 61 public photos they were close to chance
  (AUROC 0.45; see [DATASETS.md](DATASETS.md)). Treat a verdict on such photos as a prompt for a human look, not a result.

- Too many REVIEWs on good units: add more varied good photos to the bank, or raise `ANOMALY_REVIEW_RATIO`.
- Defects slipping through: lower `ANOMALY_FAIL_RATIO`, add a custom YOLO for those defect types.
- The classical "spot" finder is informational only; it never changes a verdict by itself because
  logos, camera lenses and UI elements look like spots.
