# Datasets: what is real, what each is good for, and the catches

Facts below were checked against the dataset pages / papers while building this project
(October 2026). Anything marked **not verified** is a dataset I know of but did not re-check:
read its licence page before use. **Licences change and many research sets forbid commercial
use, so a product you sell needs data you own or data with a commercial licence.**

## 1. Phone / product surface defects

| Dataset | What it is | Licence / access | Good for | Catch |
|---|---|---|---|---|
| **SSGD** (Smartphone Screen Glass Dataset) | 2,504 annotated images, 3,914 defects, 7 classes: crack, broken, spot, scratch, light-leakage, blot, broken-membrane. GitHub: `VincentHancoder/SSGD`, paper arXiv 2303.06673 | "Academic purposes only". Download via Baidu / TsinghuaCloud links in the README | Training and benchmarking a defect **detector** (YOLO) on glass defects | Factory inspection images under controlled light, not photos a phone owner takes. Not for commercial use |
| **Roboflow Universe: "smartphone-crack"** | Community-labelled cracked-phone images | Per-project licence, **not verified** | Quick prototype for crack detection | Quality and labels vary; check licence and label consistency |
| **MVTec AD** | 15 object/texture categories, 5,000+ images, defect-free train sets, pixel-level defect masks | **CC BY-NC-SA 4.0: commercial use not allowed** | Validating that the anomaly method works: `scripts/eval_anomaly.py --mvtec <category>` | No phones or laptops. A benchmark, not training data for your product |
| **VisA** | Industrial anomaly benchmark | **Not verified** | Same as MVTec | Check licence |

**For a real product you need your own data.** Public sets will not match your camera,
lighting, phone models or your definition of "defective". See the capture protocol below.

## 2. Phone internals (battery, board, sensors)

There is no public dataset of "phone photos labelled with internal faults", and a photo of the
outside cannot show them. What exists in practice:

- **Device-reported data** (battery capacity vs design, cycle count, temperature, storage):
  collected live from the phone, no dataset needed. See `scripts/collect_android_diagnostics.py`
  and `POST /api/v1/diagnostics/phone`.
- **Opened-device photos** (swollen battery, corrosion, liquid-damage indicators, burnt
  components): you have to collect these yourself, ideally with repair shops or refurbishers.
  They can feed the same anomaly bank (`--product phone_internal`) and a custom YOLO.
- **Service/repair records** (symptom -> confirmed fault) are the best source for improving
  the symptom rules in `app/diagnostics/phone.py`. Ask pilot partners for anonymised logs.

## 3. Human health: images

| Dataset | What it is | Licence / access | Notes |
|---|---|---|---|
| **HAM10000** | 10,015 dermatoscopic images of 7 pigmented-lesion types (akiec, bcc, bkl, df, mel, nv, vasc); over half confirmed by histopathology | **CC BY-NC 4.0 (non-commercial)** via Harvard Dataverse / ISIC archive | Several photos per lesion: **split by lesion** (`scripts/prepare_ham10000.py` does). Dermatoscope images, so a model will not transfer to ordinary phone photos without more data. Heavily imbalanced (nevi dominate) |
| **RSNA Pneumonia Detection (2018)** | Chest X-rays from the NIH public chest X-ray set, with pneumonia annotations | RSNA's page says research, education and commercial use are allowed with attribution; read its Terms of Use PDFs | Needs an actual X-ray image: not something a phone camera produces |
| **APTOS 2019** (diabetic retinopathy) | 3,662 fundus images graded 0 to 4 (1,805 / 370 / 999 / 193 / 295) | Kaggle competition rules; check before commercial use | Needs a fundus camera. The public file has **no patient id**, so a per-image split can leak both eyes of one patient |
| **REFUGE** (glaucoma) | 1,200 fundus images, 120 glaucoma (10%); disc/cup masks by 7 specialists; Zeiss (train) vs Canon (test) cameras | Training set public; test labels not public; licence terms not stated in the paper | Camera shift between train and test is part of the benchmark |
| **ODIR-5K** | 5,000 patients, left and right fundus, 8 labels (normal, diabetes, glaucoma, cataract, AMD, hypertension, myopia, other), labels **per patient** | "Ethical and privacy rules of China"; download-page licence not read | Multi-label; keep both eyes in one split |
| **PAD-UFES-20** | Smartphone skin-lesion photos with patient data, 6 lesion types (about 2,300 samples from about 1,400 patients, **recalled, page not readable here**) | Mendeley Data; **not verified** | The right kind of data for phone photos. Split by patient |
| **SCIN** (Google Research) | 5,000+ contributions / 10,000+ phone images of common skin conditions; dermatologist labels; self-reported and estimated skin tone | SCIN Data Use License (read the repo LICENSE) | Real volunteer phone photos with skin-tone labels, ideal for a fairness check. Contains 15 duplicate images, 1 missing file, 48 gradable cases without a label |
| **Fitzpatrick 17k** | 16,577 atlas images, 114 conditions; skin types 1-2: 7,755, 3-4: 6,089, 5-6: 2,168 | Web-sourced atlas images; check terms | Authors' own VGG-16 reached only 20.2% top-1 on 114 classes, and a dermatologist judged 3.4% of sampled labels wrong. Light skin dominates |
| **NIH malaria cell images** (TFDS `malaria`) | 27,558 cell crops, balanced parasitized / uninfected | Not stated on the TFDS page; check the NIH homepage | Crops come from slides: split by slide |
| **ISIC archive**, **CheXpert**, **NIH ChestX-ray14**, **EyePACS**, **Messidor-2** | Standard public sets | **Not verified**: rules and licences differ | Retinal and X-ray images need the matching device |
| **Smartphone anaemia (inner eyelid), jaundice (sclera), oral-cancer studies** | Published research (PMC11854623, PMC8588081, PMC8397787) | Datasets are mostly **not public**; I could not open these papers in this build | You will have to collect paired data yourself: image plus a blood haemoglobin / bilirubin test or a biopsy |
| **PTB-XL** (12-lead ECG) | Large public ECG set | **Not verified** | For heart-rhythm models. Needs ECG signal from a sensor, not a camera |

## 4. Camera pulse (rPPG)

Public sets used in the literature: **UBFC-rPPG, PURE, COHFACE, LGI-PPGI** (licences/access
**not verified**: several require a request to the authors).
A 2025 real-time rPPG system paper reports heart-rate error around 1 bpm on UBFC (controlled
lighting) but roughly 8 to 11 bpm on COHFACE (uncontrolled lighting) and does not evaluate skin
tone. Read that as: **pulse from a camera works well when the room and the person cooperate and
poorly otherwise**, which is why this project returns no number at all when confidence is low.

## 5. Your own data: capture protocol that actually works

**Products**
1. Fixed rig: tripod, a diffuse light box or two soft lights, plain matte background, same camera
   and distance every time. Glare is the main enemy: use cross-polarised light for glass.
2. Photograph each unit from the same fixed set of views: front screen **off**, front screen
   **on** (white, black), back, 4 edges, 4 corners, ports.
3. Collect **50 to 200 good units per product model** (anomaly bank) and **at least 100 examples
   per defect class** (detector). Log unit ID, model, grade, and who graded it.
4. Two graders on a sample of images; measure agreement. Unclear labels cap your accuracy.
5. **Split by physical unit**, never by image (`scripts/prepare_folder_dataset.py --group-regex`).
   Lock a test set and do not look at it while tuning.
6. Keep collecting from production: every REVIEW that a human resolves is a new labelled example
   (the Supabase `inspections` table and image bucket already store them).

**People (any health image or video)**
1. Written informed consent that states purpose, storage, retention and who can see it. Ethics
   committee approval for anything beyond internal testing.
2. Record skin tone (e.g. Fitzpatrick scale), age, sex, device, lighting; **test performance per
   group**. Many models are worse on darker skin and you will not find out unless you measure.
3. De-identify; keep faces out of storage where you can; encrypt; set a retention limit.
   India's DPDP Act 2023, GDPR and HIPAA treat health data as sensitive.
4. Ground truth comes from a clinician or a lab test, not from your own labelling.

## Honest evaluation numbers from this project

On **synthetic drawn phone images only** (no real products): none of 12 defective examples (crack,
scratch, missing part) passed (8 FAIL, 4 REVIEW), 29 of 30 unseen good units passed and 1 went to
REVIEW, and the anomaly
model alone (random backbone, no ImageNet weights available in the build environment) reached
image-level AUROC 0.86 (95% CI 0.72 to 0.97, 18 defective images) and caught 50% of defects at
its FAIL threshold. **First real-world photos (October 2026):** 41 intact and 20 cracked phones from Wikimedia Commons
(`scripts/fetch_commons.py`; CC BY / CC BY-SA, attribution kept in `attribution.csv`; labels set by looking at the photos).
These are uncontrolled snapshots (hands, tables, lace, shop displays, screens on and off), not a capture rig. The phone
was found in 61 of 61 photos, but defect detection was close to chance: out of the box (YOLO + OpenCV) it flagged 73% of
intact phones and 70% of cracked ones, AUROC 0.45 (95% CI 0.29 to 0.59). With an anomaly bank built from 26 of the intact
phones, AUROC on 15 unseen intact + 20 cracked was 0.42, 0.63 and 0.62 over three splits (CIs all include 0.5).
The OpenCV scratch/crack/spot checks fire on busy backgrounds and on-screen content, and one bank cannot describe phones
that vary this much. This is the expected result for uncontrolled data and the reason the capture protocol above exists:
fixed rig, plain background, screen off, one bank per model. Do not read it as the accuracy you would get on a rig.

**Later results (October 2026).**
*Product naming:* a ResNet50 head trained on 1,742 Commons snapshots (labels from search text, with the worst noise removed
by eye) reached 86.6% accuracy on photographers it never saw (454 photos), named 73% of real products at its 0.55 cut-off,
was wrong in 9% of its answers and called 0 of 135 non-product photos a product. On 15 brand-new photos, 12 were right.
*Phone anomaly bank:* built from 40 synthetic drawings, it scored every real phone photo 6 to 9 times its threshold
(intact 5.9 to 9.0, cracked 6.9 to 9.4), so a ratio that high means "unlike the reference photos", not damage. The OpenCV
crack finder flagged a "high" crack in 34% of intact and 20% of cracked real phones; none of length, contrast, straightness
or angle separated them. Hence the FAIL policy in the README.
*Screening models (locked test, per-patient or per-slide splits where an id exists):* malaria_smear AUC 0.991 (n=1,259);
chest_xray 0.997 (416, children; near-perfect, possibly image-style shortcut); retina_dr 0.901 (590, no patient id, 55% of
photos refused, sensitivity about 0.49); skin_lesion 0.959 (514, dermatoscope, sensitivity 0.895 / specificity 0.842 at
MODERATE); eye_disease 0.709 (541, lower bound 0.651: rejected by the gate). All are research-grade only.

These show the code works. They say nothing about real phones on a rig: run
`scripts/eval_anomaly.py` and `scripts/evaluate_classifier.py` on your own held-out data.
