# Screening modules: what the camera can (and cannot) look at

Mirrors `app/screening/catalog.py` (a test fails if a module is missing here). **No clinical model ships with this project** (four research-grade ones are installed, see the note below). A module turns on when you put a trained `<name>.pt` + `<name>.json` in `models/screening/`.

Legend: **Device** is what produces the image. A phone selfie only works for the *phone* modules. **Evidence** says how strong the published support is. A dataset marked *verified* had its page or paper read while building this project (October 2026); *not verified* means I know of it but did not re-check size, labels or licence: read its page first.


> **Installed models (October 2026):** `malaria_smear`, `chest_xray`, `retina_dr`, `skin_lesion`. Each passed a gate (lower
> 95% AUC bound at least 0.70 on a locked test set) and nothing more; none is clinically validated. `eye_disease` was trained
> and rejected (AUC 0.709). `glaucoma` and the other modules have no model. Their locked-test reports are in `reports/`.

| Module | Body area | Device | Evidence |
|---|---|---|---|
| `anemia_conjunctiva` | eye (inner lower eyelid) | phone | Published smartphone research; not a proven clinical tool |
| `chest_xray` | lungs (X-ray image) | X-ray machine (a phone camera cannot see the lungs) | Well studied, but needs the specialist device |
| `eye_disease` | eye (retina) | fundus camera | Well studied, but needs the specialist device |
| `eye_external` | eye (front: lens, white of the eye) | phone | Early / small studies: treat as an experiment |
| `glaucoma` | eye (optic nerve) | fundus camera | Well studied, but needs the specialist device |
| `jaundice` | eye (white of the eye) and skin | phone | Published smartphone research; not a proven clinical tool |
| `malaria_smear` | blood (microscope slide) | microscope (not a phone selfie; a phone on a microscope eyepiece is research use) | Well studied, but needs the specialist device |
| `oral_lesion` | mouth | phone | Published smartphone research; not a proven clinical tool |
| `retina_dr` | eye (retina) | fundus camera (a phone selfie cannot see the retina) | Well studied, but needs the specialist device |
| `skin_condition` | skin (rashes, acne, eczema, fungal and more) | phone | Early / small studies: treat as an experiment |
| `skin_lesion` | skin (single lesion / mole) | dermatoscope for HAM10000-style data; phone close-ups need phone-photo training data | Published smartphone research; not a proven clinical tool |

## `anemia_conjunctiva`: Anaemia screen (inner eyelid colour)

- **Looks for:** pale inner eyelid, which can go with low haemoglobin (anaemia)
- **Input:** Phone photo of the inner lower eyelid pulled down, even light, no flash colour cast
- **Device:** phone
- **Evidence:** Published smartphone research; not a proven clinical tool
- **Labels:** not_anemic, anemic  (concern: anemic)
- **Refer to:** Family doctor for a haemoglobin blood test
- **Red flags:** Breathlessness at rest, chest pain or fainting; Heavy or long bleeding; Black or bloody stools

| Dataset | Size / content | Licence | Verified | Catch |
|---|---|---|---|---|
| Smartphone inner-eyelid anaemia studies (e.g. PMC11854623) | reported: thousands of smartphone eyelid photos | datasets are usually not public | **no** | Could not open the paper in this build. Ground truth is a blood haemoglobin test, so you must collect paired data. |

## `chest_xray`: Chest X-ray pneumonia screen

- **Looks for:** a pattern on the chest X-ray that can go with pneumonia
- **Input:** Frontal chest radiograph (PA/AP) file from a hospital or clinic
- **Device:** X-ray machine (a phone camera cannot see the lungs)
- **Evidence:** Well studied, but needs the specialist device
- **Labels:** normal, pneumonia  (concern: pneumonia)
- **Refer to:** Doctor / pulmonologist
- **Red flags:** Breathlessness at rest, blue lips or chest pain (emergency); Cough with high fever for more than 3 days; Coughing blood

| Dataset | Size / content | Licence | Verified | Catch |
|---|---|---|---|---|
| RSNA Pneumonia Detection Challenge | from the NIH ChestX-ray14 images | RSNA terms: research/education/commercial with attribution (read the PDFs) | yes | Group by patient. |
| NIH ChestX-ray14, CheXpert | large | not verified | **no** |  |

## `eye_disease`: Multi-disease eye screen

- **Looks for:** possible eye disease on the retina photo
- **Input:** Colour fundus photograph
- **Device:** fundus camera
- **Evidence:** Well studied, but needs the specialist device
- **Labels:** normal, diabetes, glaucoma, cataract, amd, hypertension, myopia, other  (concern: diabetes, glaucoma, cataract, amd, hypertension, myopia, other)  [multi-label]
- **Refer to:** Ophthalmologist
- **Red flags:** Sudden vision loss; Eye injury or chemical splash; Sudden double vision

| Dataset | Size / content | Licence | Verified | Catch |
|---|---|---|---|---|
| ODIR-5K (Peking University, ODIR-2019) | 5,000 patients, left+right fundus; 8 labels N/D/G/C/A/H/M/O; labels per PATIENT, a patient can have several | ethical/privacy rules of China apply; download page licence not read | yes | Because labels are per patient, both eyes must stay together in the split. |

Prepare: `python scripts/prepare_dataset.py odir --src <download> --out data/eye_disease`

## `eye_external`: External eye screen (redness, cloudy lens, yellow sclera)

- **Looks for:** visible changes on the front of the eye
- **Input:** Close, well-lit phone photo of one open eye, looking straight ahead
- **Device:** phone
- **Evidence:** Early / small studies: treat as an experiment
- **Labels:** normal, conjunctivitis, cataract, other_eye_condition  (concern: conjunctivitis, cataract, other_eye_condition)
- **Refer to:** Ophthalmologist or optometrist
- **Red flags:** Eye pain, light sensitivity or reduced vision with a red eye (do not wait); Contact-lens wearer with a painful red eye; Eye injury or chemical splash: rinse and go to emergency

| Dataset | Size / content | Licence | Verified | Catch |
|---|---|---|---|---|
| Public external-eye / cataract photo sets on Kaggle and Roboflow | varies | varies per set | **no** | I did not verify any specific set. Quality and labelling vary a lot; many are web images without clinical confirmation. |

## `glaucoma`: Glaucoma screen (optic disc)

- **Looks for:** signs of glaucoma (damage to the optic nerve)
- **Input:** Colour fundus photograph centred on the optic disc
- **Device:** fundus camera
- **Evidence:** Well studied, but needs the specialist device
- **Labels:** non_glaucoma, glaucoma  (concern: glaucoma)
- **Refer to:** Ophthalmologist (glaucoma needs eye-pressure and visual-field tests)
- **Red flags:** Severe eye pain with nausea or halos around lights (possible acute angle-closure: emergency); Gradual loss of side vision

| Dataset | Size / content | Licence | Verified | Catch |
|---|---|---|---|---|
| REFUGE (MICCAI 2018 challenge) | 1,200 fundus images: 120 glaucoma (10%), 1,080 not; disc/cup masks by 7 specialists | public training set; test labels not public; licence terms not stated on the paper | yes | Two cameras (Zeiss train, Canon test): expect a camera domain shift. Labels come from clinical records, not the image alone. |
| ODIR-5K (also has a glaucoma label) | 5,000 patients, both eyes | check download page | yes |  |

Prepare: `python scripts/prepare_dataset.py odir --src <download> --out data/glaucoma`

## `jaundice`: Jaundice screen (yellow eyes or skin)

- **Looks for:** yellow tint of the eyes or skin, which can go with high bilirubin (liver or blood problems)
- **Input:** Phone photo of the white of the eye (sclera) in even light, with a colour reference card if possible
- **Device:** phone
- **Evidence:** Published smartphone research; not a proven clinical tool
- **Labels:** normal, jaundice  (concern: jaundice)
- **Refer to:** Doctor for a bilirubin and liver blood test (urgent for newborns)
- **Red flags:** Yellow newborn: see a doctor the same day; Yellow eyes with fever, belly pain or dark urine; Confusion or drowsiness with yellow skin (emergency)

| Dataset | Size / content | Licence | Verified | Catch |
|---|---|---|---|---|
| Smartphone jaundice studies (e.g. PMC8588081) | small to medium | datasets usually not public | **no** | Could not open the paper in this build. Ground truth is a serum bilirubin test. Light colour changes the result a lot. |

## `malaria_smear`: Malaria blood-smear screen

- **Looks for:** possible malaria parasites on a blood smear
- **Input:** Microscope image of a thin blood-smear cell crop
- **Device:** microscope (not a phone selfie; a phone on a microscope eyepiece is research use)
- **Evidence:** Well studied, but needs the specialist device
- **Labels:** uninfected, parasitized  (concern: parasitized)
- **Refer to:** Lab for a rapid test / blood film and a doctor
- **Red flags:** Fever after travel to or living in a malaria area (get tested the same day); Confusion, seizures or very dark urine (emergency)

| Dataset | Size / content | Licence | Verified | Catch |
|---|---|---|---|---|
| NIH Malaria Cell Images (TFDS 'malaria') | 27,558 cell images, balanced parasitized / uninfected | licence not stated on the TFDS page; check NIH homepage | yes | Cells are cropped from slides: split by SLIDE/patient, not by cell, or results are inflated. |

Prepare: `python scripts/prepare_dataset.py malaria --src <download> --out data/malaria_smear`

## `oral_lesion`: Mouth lesion screen

- **Looks for:** a mouth patch or ulcer that deserves a dentist or doctor check
- **Input:** Phone photo inside the mouth, tongue or cheek lesion in focus, good light
- **Device:** phone
- **Evidence:** Published smartphone research; not a proven clinical tool
- **Labels:** normal_or_benign, suspicious  (concern: suspicious)
- **Refer to:** Dentist, ENT or oral oncologist
- **Red flags:** A mouth ulcer that has not healed in 3 weeks; A white or red patch that does not wipe off; A lump in the mouth or neck, trouble swallowing, or unexplained bleeding

| Dataset | Size / content | Licence | Verified | Catch |
|---|---|---|---|---|
| Smartphone oral-cancer image studies (e.g. PMC8397787) | published studies; sets mostly not public | usually need a request to the authors | **no** | Could not open the paper in this build. Ground truth is biopsy. High-risk groups (tobacco, betel) matter most in India. |

## `retina_dr`: Diabetic retinopathy screen

- **Looks for:** signs of diabetic retinopathy (diabetes damage at the back of the eye)
- **Input:** Colour fundus photograph (retina photo from a fundus camera)
- **Device:** fundus camera (a phone selfie cannot see the retina)
- **Evidence:** Well studied, but needs the specialist device
- **Labels:** no_dr, mild, moderate, severe, proliferative  (concern: moderate, severe, proliferative)
- **Refer to:** Ophthalmologist / retina specialist
- **Red flags:** Sudden loss of vision or a dark curtain over the vision; Many new floaters or flashes of light

| Dataset | Size / content | Licence | Verified | Catch |
|---|---|---|---|---|
| APTOS 2019 Blindness Detection | 3,662 fundus images, grades 0-4 (1,805 / 370 / 999 / 193 / 295) | Kaggle competition rules; check before commercial use | yes | Grades 0-4. Small, one hospital system (Aravind, India). |
| EyePACS / Kaggle DR 2015 | large (tens of thousands) | not verified | **no** |  |
| Messidor-2 | ~1,700 images | not verified | **no** |  |

Prepare: `python scripts/prepare_dataset.py aptos --src <download> --out data/retina_dr`

## `skin_condition`: Common skin conditions

- **Looks for:** a skin condition pattern that may need treatment
- **Input:** Clear phone photo of the affected skin area, good natural light
- **Device:** phone
- **Evidence:** Early / small studies: treat as an experiment
- **Labels:** no_condition, inflammatory_or_allergic, infection, other_skin_condition  (concern: inflammatory_or_allergic, infection, other_skin_condition)
- **Refer to:** Dermatologist or family doctor
- **Red flags:** Rash with fever, blisters in the mouth or eyes, or fast spread; Swelling of lips or face or trouble breathing (emergency); Skin that is hot, red, swelling and painful (possible infection)

| Dataset | Size / content | Licence | Verified | Catch |
|---|---|---|---|---|
| SCIN (Google Research) | 5,000+ contributions / 10,000+ phone images; dermatologist labels, self-reported and estimated skin tone | SCIN Data Use License (read LICENSE in the repo) | yes | Real phone photos by volunteers in the US; includes skin-tone labels, so you can test for fairness. |
| Fitzpatrick 17k | 16,577 atlas images, 114 conditions; skin types 1-2: 7,755, 3-4: 6,089, 5-6: 2,168 | web-sourced atlas images; check terms | yes | The authors' own VGG-16 got only 20.2% top-1 accuracy on 114 classes, so group the labels. Light skin dominates; 3.4% of sampled labels were wrong. |

## `skin_lesion`: Skin lesion triage

- **Looks for:** a skin spot with features that deserve a skin-specialist check
- **Input:** Close-up, in-focus photo of one skin lesion (dermoscopy images or a phone close-up, depending on training data)
- **Device:** dermatoscope for HAM10000-style data; phone close-ups need phone-photo training data
- **Evidence:** Published smartphone research; not a proven clinical tool
- **Labels:** benign, suspicious  (concern: suspicious)
- **Refer to:** Dermatologist
- **Red flags:** A mole that changed in size, shape or colour (ABCDE); A sore that bleeds or does not heal for 4+ weeks; A new dark spot with irregular border or several colours

| Dataset | Size / content | Licence | Verified | Catch |
|---|---|---|---|---|
| HAM10000 | 10,015 dermatoscopic images, 7 types; >half histology-confirmed | CC BY-NC 4.0 (non-commercial) | yes | Dermatoscope images only. Several photos per lesion: split by lesion. |
| PAD-UFES-20 | about 2,300 smartphone lesion samples from ~1,400 patients, 6 types, with patient data (recalled, page not readable here) | Mendeley Data; licence not verified | **no** | The right kind of data for PHONE photos. Has patient_id and lesion_id, so split by patient. |
| ISIC archive challenges | large | varies by collection | **no** |  |

Prepare: `python scripts/prepare_dataset.py ham10000 --src <download> --out data/skin_lesion`

## What a camera cannot detect

- Heart, lung, liver, kidney, brain or stomach disease (a camera cannot see inside the body; these need ECG, X-ray, ultrasound, scans or blood tests).
- Diabetes, cholesterol, thyroid, infections and cancers directly (they need blood or tissue tests). A few have indirect visible signs that a doctor can check.
- Blood pressure and oxygen level from an ordinary camera (not reliable without clinical calibration; this project does not report them).

## From dataset to a patient report

```bash
python scripts/prepare_dataset.py aptos --src ~/data/aptos2019 --out data/retina_dr   # patient-level split + verify
python scripts/train_classifier.py --data data/retina_dr --name retina_dr --arch efficientnet_b0 \
       --concern moderate severe proliferative --dataset-note "APTOS 2019"
python scripts/evaluate_classifier.py --name retina_dr --data data/retina_dr/test \
       --subgroups data/retina_dr/subgroups.csv --report retina_eval.json            # CIs + fairness per group
```
The trainer also fits **temperature scaling** (calibrated probabilities), picks the **MODERATE** threshold for a target
sensitivity (default 90%) and the **HIGH** threshold for 95% specificity, and fits an **out-of-distribution guard** so an
image unlike the training data returns INCONCLUSIVE. These are chosen on the validation split, so they are optimistic:
always confirm on a locked test set with `evaluate_classifier.py`.

Then open the dashboard, tab **Patient report**: enter the patient, tick consent, add one check per body area (photo or
camera), and download the PDF. `GET /api/v1/report/sample` shows a clearly labelled synthetic example.

