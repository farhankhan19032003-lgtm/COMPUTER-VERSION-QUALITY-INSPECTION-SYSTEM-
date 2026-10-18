"""Which screening modules the system knows about, what each can honestly do, and where to get data.

No clinical model ships with this project. A module becomes active when a trained
`<name>.pt` + `<name>.json` pair exists in SCREENING_MODEL_DIR (see scripts/train_classifier.py).

Honesty fields on every entry:
  device    - what camera/sensor the images really come from. "phone" = an ordinary phone camera;
              anything else needs specialist equipment, so a phone selfie will NOT work.
  evidence  - how strong the published support for image-based screening is:
                "device_established"  : well studied, but needs the specialist device (fundus camera, X-ray ...)
                "smartphone_research" : published smartphone research, not yet a proven clinical tool
                "early_research"      : small studies only; treat as an experiment
  verified  - per dataset: True only if the dataset page/paper was read while building this project.
              False = I know of it but did not re-check size, labels or licence. Read its page first.
Many research datasets forbid commercial use. A product you sell needs data you own or that is licensed for it.
"""
from __future__ import annotations

from typing import Any, Dict, List

DISCLAIMER = (
    "Screening aid only, NOT a medical diagnosis. Results can be wrong in both directions. "
    "A 'low' result does not rule out disease, and a 'high' result does not confirm it. "
    "Consult a qualified clinician about any health concern."
)

URGENCY_RANK = {"INCONCLUSIVE": 0, "LOW": 1, "MODERATE": 2, "HIGH": 3}


def _ds(name: str, size: str, licence: str, verified: bool, note: str = "", adapter: str = "") -> Dict[str, Any]:
    return {"name": name, "size": size, "licence": licence, "verified": verified, "note": note, "adapter": adapter}


CATALOG: Dict[str, Dict[str, Any]] = {
    # ------------------------------------------------------------------------------ EYES
    "retina_dr": {
        "title": "Diabetic retinopathy screen",
        "body_area": "eye (retina)",
        "finding": "signs of diabetic retinopathy (diabetes damage at the back of the eye)",
        "input": "Colour fundus photograph (retina photo from a fundus camera)",
        "device": "fundus camera (a phone selfie cannot see the retina)",
        "evidence": "device_established",
        "suggested_labels": ["no_dr", "mild", "moderate", "severe", "proliferative"],
        "suggested_concern": ["moderate", "severe", "proliferative"],
        "datasets": [
            _ds("APTOS 2019 Blindness Detection", "3,662 fundus images, grades 0-4 (1,805 / 370 / 999 / 193 / 295)",
                "Kaggle competition rules; check before commercial use", True,
                "Grades 0-4. Small, one hospital system (Aravind, India).", "aptos"),
            _ds("EyePACS / Kaggle DR 2015", "large (tens of thousands)", "not verified", False),
            _ds("Messidor-2", "~1,700 images", "not verified", False),
        ],
        "referral": "Ophthalmologist / retina specialist",
        "red_flags": ["Sudden loss of vision or a dark curtain over the vision", "Many new floaters or flashes of light"],
    },
    "glaucoma": {
        "title": "Glaucoma screen (optic disc)",
        "body_area": "eye (optic nerve)",
        "finding": "signs of glaucoma (damage to the optic nerve)",
        "input": "Colour fundus photograph centred on the optic disc",
        "device": "fundus camera",
        "evidence": "device_established",
        "suggested_labels": ["non_glaucoma", "glaucoma"],
        "suggested_concern": ["glaucoma"],
        "datasets": [
            _ds("REFUGE (MICCAI 2018 challenge)", "1,200 fundus images: 120 glaucoma (10%), 1,080 not; disc/cup masks by 7 specialists",
                "public training set; test labels not public; licence terms not stated on the paper", True,
                "Two cameras (Zeiss train, Canon test): expect a camera domain shift. Labels come from clinical records, not the image alone.", ""),
            _ds("ODIR-5K (also has a glaucoma label)", "5,000 patients, both eyes", "check download page", True, "", "odir"),
        ],
        "referral": "Ophthalmologist (glaucoma needs eye-pressure and visual-field tests)",
        "red_flags": ["Severe eye pain with nausea or halos around lights (possible acute angle-closure: emergency)",
                      "Gradual loss of side vision"],
    },
    "eye_disease": {
        "title": "Multi-disease eye screen",
        "body_area": "eye (retina)",
        "finding": "possible eye disease on the retina photo",
        "input": "Colour fundus photograph",
        "device": "fundus camera",
        "evidence": "device_established",
        "multilabel": True,
        "suggested_labels": ["normal", "diabetes", "glaucoma", "cataract", "amd", "hypertension", "myopia", "other"],
        "suggested_concern": ["diabetes", "glaucoma", "cataract", "amd", "hypertension", "myopia", "other"],
        "datasets": [
            _ds("ODIR-5K (Peking University, ODIR-2019)", "5,000 patients, left+right fundus; 8 labels N/D/G/C/A/H/M/O; labels per PATIENT, a patient can have several",
                "ethical/privacy rules of China apply; download page licence not read", True,
                "Because labels are per patient, both eyes must stay together in the split.", "odir"),
        ],
        "referral": "Ophthalmologist",
        "red_flags": ["Sudden vision loss", "Eye injury or chemical splash", "Sudden double vision"],
    },
    "eye_external": {
        "title": "External eye screen (redness, cloudy lens, yellow sclera)",
        "body_area": "eye (front: lens, white of the eye)",
        "finding": "visible changes on the front of the eye",
        "input": "Close, well-lit phone photo of one open eye, looking straight ahead",
        "device": "phone",
        "evidence": "early_research",
        "suggested_labels": ["normal", "conjunctivitis", "cataract", "other_eye_condition"],
        "suggested_concern": ["conjunctivitis", "cataract", "other_eye_condition"],
        "datasets": [
            _ds("Public external-eye / cataract photo sets on Kaggle and Roboflow", "varies", "varies per set", False,
                "I did not verify any specific set. Quality and labelling vary a lot; many are web images without clinical confirmation."),
        ],
        "referral": "Ophthalmologist or optometrist",
        "red_flags": ["Eye pain, light sensitivity or reduced vision with a red eye (do not wait)",
                      "Contact-lens wearer with a painful red eye", "Eye injury or chemical splash: rinse and go to emergency"],
    },
    # ------------------------------------------------------------------------------ SKIN
    "skin_lesion": {
        "title": "Skin lesion triage",
        "body_area": "skin (single lesion / mole)",
        "finding": "a skin spot with features that deserve a skin-specialist check",
        "input": "Close-up, in-focus photo of one skin lesion (dermoscopy images or a phone close-up, depending on training data)",
        "device": "dermatoscope for HAM10000-style data; phone close-ups need phone-photo training data",
        "evidence": "smartphone_research",
        "suggested_labels": ["benign", "suspicious"],
        "suggested_concern": ["suspicious"],
        "datasets": [
            _ds("HAM10000", "10,015 dermatoscopic images, 7 types; >half histology-confirmed", "CC BY-NC 4.0 (non-commercial)", True,
                "Dermatoscope images only. Several photos per lesion: split by lesion.", "ham10000"),
            _ds("PAD-UFES-20", "about 2,300 smartphone lesion samples from ~1,400 patients, 6 types, with patient data (recalled, page not readable here)",
                "Mendeley Data; licence not verified", False,
                "The right kind of data for PHONE photos. Has patient_id and lesion_id, so split by patient.", "pad_ufes"),
            _ds("ISIC archive challenges", "large", "varies by collection", False),
        ],
        "referral": "Dermatologist",
        "red_flags": ["A mole that changed in size, shape or colour (ABCDE)", "A sore that bleeds or does not heal for 4+ weeks",
                      "A new dark spot with irregular border or several colours"],
    },
    "skin_condition": {
        "title": "Common skin conditions",
        "body_area": "skin (rashes, acne, eczema, fungal and more)",
        "finding": "a skin condition pattern that may need treatment",
        "input": "Clear phone photo of the affected skin area, good natural light",
        "device": "phone",
        "evidence": "early_research",
        "suggested_labels": ["no_condition", "inflammatory_or_allergic", "infection", "other_skin_condition"],
        "suggested_concern": ["inflammatory_or_allergic", "infection", "other_skin_condition"],
        "datasets": [
            _ds("SCIN (Google Research)", "5,000+ contributions / 10,000+ phone images; dermatologist labels, self-reported and estimated skin tone",
                "SCIN Data Use License (read LICENSE in the repo)", True,
                "Real phone photos by volunteers in the US; includes skin-tone labels, so you can test for fairness.", ""),
            _ds("Fitzpatrick 17k", "16,577 atlas images, 114 conditions; skin types 1-2: 7,755, 3-4: 6,089, 5-6: 2,168",
                "web-sourced atlas images; check terms", True,
                "The authors' own VGG-16 got only 20.2% top-1 accuracy on 114 classes, so group the labels. Light skin dominates; 3.4% of sampled labels were wrong.", ""),
        ],
        "referral": "Dermatologist or family doctor",
        "red_flags": ["Rash with fever, blisters in the mouth or eyes, or fast spread", "Swelling of lips or face or trouble breathing (emergency)",
                      "Skin that is hot, red, swelling and painful (possible infection)"],
    },
    # ------------------------------------------------------------------------------ BLOOD / LIVER (indirect signs)
    "anemia_conjunctiva": {
        "title": "Anaemia screen (inner eyelid colour)",
        "body_area": "eye (inner lower eyelid)",
        "finding": "pale inner eyelid, which can go with low haemoglobin (anaemia)",
        "input": "Phone photo of the inner lower eyelid pulled down, even light, no flash colour cast",
        "device": "phone",
        "evidence": "smartphone_research",
        "suggested_labels": ["not_anemic", "anemic"],
        "suggested_concern": ["anemic"],
        "datasets": [
            _ds("Smartphone inner-eyelid anaemia studies (e.g. PMC11854623)", "reported: thousands of smartphone eyelid photos",
                "datasets are usually not public", False,
                "Could not open the paper in this build. Ground truth is a blood haemoglobin test, so you must collect paired data."),
        ],
        "referral": "Family doctor for a haemoglobin blood test",
        "red_flags": ["Breathlessness at rest, chest pain or fainting", "Heavy or long bleeding", "Black or bloody stools"],
    },
    "jaundice": {
        "title": "Jaundice screen (yellow eyes or skin)",
        "body_area": "eye (white of the eye) and skin",
        "finding": "yellow tint of the eyes or skin, which can go with high bilirubin (liver or blood problems)",
        "input": "Phone photo of the white of the eye (sclera) in even light, with a colour reference card if possible",
        "device": "phone",
        "evidence": "smartphone_research",
        "suggested_labels": ["normal", "jaundice"],
        "suggested_concern": ["jaundice"],
        "datasets": [
            _ds("Smartphone jaundice studies (e.g. PMC8588081)", "small to medium", "datasets usually not public", False,
                "Could not open the paper in this build. Ground truth is a serum bilirubin test. Light colour changes the result a lot."),
        ],
        "referral": "Doctor for a bilirubin and liver blood test (urgent for newborns)",
        "red_flags": ["Yellow newborn: see a doctor the same day", "Yellow eyes with fever, belly pain or dark urine",
                      "Confusion or drowsiness with yellow skin (emergency)"],
    },
    # ------------------------------------------------------------------------------ MOUTH
    "oral_lesion": {
        "title": "Mouth lesion screen",
        "body_area": "mouth",
        "finding": "a mouth patch or ulcer that deserves a dentist or doctor check",
        "input": "Phone photo inside the mouth, tongue or cheek lesion in focus, good light",
        "device": "phone",
        "evidence": "smartphone_research",
        "suggested_labels": ["normal_or_benign", "suspicious"],
        "suggested_concern": ["suspicious"],
        "datasets": [
            _ds("Smartphone oral-cancer image studies (e.g. PMC8397787)", "published studies; sets mostly not public",
                "usually need a request to the authors", False,
                "Could not open the paper in this build. Ground truth is biopsy. High-risk groups (tobacco, betel) matter most in India."),
        ],
        "referral": "Dentist, ENT or oral oncologist",
        "red_flags": ["A mouth ulcer that has not healed in 3 weeks", "A white or red patch that does not wipe off",
                      "A lump in the mouth or neck, trouble swallowing, or unexplained bleeding"],
    },
    # ------------------------------------------------------------------------------ LAB / RADIOLOGY (not a phone selfie)
    "malaria_smear": {
        "title": "Malaria blood-smear screen",
        "body_area": "blood (microscope slide)",
        "finding": "possible malaria parasites on a blood smear",
        "input": "Microscope image of a thin blood-smear cell crop",
        "device": "microscope (not a phone selfie; a phone on a microscope eyepiece is research use)",
        "evidence": "device_established",
        "suggested_labels": ["uninfected", "parasitized"],
        "suggested_concern": ["parasitized"],
        "datasets": [
            _ds("NIH Malaria Cell Images (TFDS 'malaria')", "27,558 cell images, balanced parasitized / uninfected", "licence not stated on the TFDS page; check NIH homepage", True,
                "Cells are cropped from slides: split by SLIDE/patient, not by cell, or results are inflated.", "malaria"),
        ],
        "referral": "Lab for a rapid test / blood film and a doctor",
        "red_flags": ["Fever after travel to or living in a malaria area (get tested the same day)", "Confusion, seizures or very dark urine (emergency)"],
    },
    "chest_xray": {
        "title": "Chest X-ray pneumonia screen",
        "body_area": "lungs (X-ray image)",
        "finding": "a pattern on the chest X-ray that can go with pneumonia",
        "input": "Frontal chest radiograph (PA/AP) file from a hospital or clinic",
        "device": "X-ray machine (a phone camera cannot see the lungs)",
        "evidence": "device_established",
        "suggested_labels": ["normal", "pneumonia"],
        "suggested_concern": ["pneumonia"],
        "datasets": [
            _ds("RSNA Pneumonia Detection Challenge", "from the NIH ChestX-ray14 images", "RSNA terms: research/education/commercial with attribution (read the PDFs)", True,
                "Group by patient.", ""),
            _ds("NIH ChestX-ray14, CheXpert", "large", "not verified", False),
        ],
        "referral": "Doctor / pulmonologist",
        "red_flags": ["Breathlessness at rest, blue lips or chest pain (emergency)", "Cough with high fever for more than 3 days", "Coughing blood"],
    },
}

# What the camera can and cannot do: shown in the UI and in the report.
NOT_DETECTABLE = [
    "Heart, lung, liver, kidney, brain or stomach disease (a camera cannot see inside the body; these need ECG, X-ray, ultrasound, scans or blood tests).",
    "Diabetes, cholesterol, thyroid, infections and cancers directly (they need blood or tissue tests). A few have indirect visible signs that a doctor can check.",
    "Blood pressure and oxygen level from an ordinary camera (not reliable without clinical calibration; this project does not report them).",
]


def catalog_names() -> List[str]:
    return sorted(CATALOG)


def module_info(name: str) -> Dict[str, Any]:
    """Catalog entry or a safe default for a module that has a model but no catalog entry."""
    return CATALOG.get(name, {
        "title": name.replace("_", " ").title(), "body_area": "body", "finding": "a possible finding",
        "device": "see model notes", "evidence": "early_research", "referral": "A qualified clinician", "red_flags": [],
    })
