#!/usr/bin/env python
"""Turn a public dataset mirror on Hugging Face (no login needed) into leakage-safe train/val/test folders.

    python scripts/prepare_hf.py --list
    python scripts/prepare_hf.py skin_lesion --out data/skin_lesion

Parquet shards are downloaded one at a time (and deleted after use) until each class has enough images, resized to
at most --max-side pixels, then split with the same group-aware splitter as prepare_dataset.py (by lesion / patient
where the data has an id).

READ THIS BEFORE TRUSTING A MODEL TRAINED ON THESE:
  * These are community mirrors of datasets published elsewhere. I could check the columns and look at sample images,
    not the labelling process. Where the original source is known it is named in each recipe.
  * Where a mirror has no patient / lesion id (diabetic retinopathy, glaucoma, chest X-ray if the file name is lost) the
    split is per image, so photos of the same patient can sit on both sides and test numbers are OPTIMISTIC.
    The script prints a warning for each such recipe.
  * Licences differ, and several are research / non-commercial only. Read DATASETS.md and each dataset page.
"""
import argparse
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from _common import ROOT

from prepare_dataset import _age_band, split_and_write, _need


def _skin(row):
    dx = str(row.get("dx", "")).strip().lower()
    label = "suspicious" if dx in {"actinic_keratoses", "basal_cell_carcinoma", "melanoma"} else "benign"
    return {"labels": [label], "group": str(row["lesion_id"]), "key": str(row["image_id"]),
            "attrs": {"sex": str(row.get("sex") or "unknown"), "age_band": _age_band(row.get("age")),
                      "site": str(row.get("localization") or "unknown")}}


DR_NAMES = ["no_dr", "mild", "moderate", "severe", "proliferative"]
ODIR_MAP = {"normal": "normal", "diabetes": "diabetes", "glaucoma": "glaucoma", "cataract": "cataract",
            "age_related_macular_degeneration": "amd", "age-related_macular_degeneration": "amd", "amd": "amd",
            "hypertension": "hypertension", "pathological_myopia": "myopia", "myopia": "myopia",
            "other_diseases_abnormalities": "other", "other": "other"}


def _xray(row):
    path = (row.get("image") or {}).get("path") or ""
    m = re.search(r"person(\d+)", path, re.I) or re.search(r"(IM-\d+)", path, re.I)
    return {"labels": ["pneumonia" if int(row["label"]) == 1 else "normal"], "group": m.group(0).lower() if m else None,
            "key": None, "attrs": {}}


def _odir(row):
    lab = ODIR_MAP.get(str(row["label"]).strip().lower().replace(" ", "_"))
    if lab is None:
        return None
    return {"labels": [lab], "group": f"p{row['patient_id']}", "key": None,
            "attrs": {"sex": str(row.get("sex") or "unknown"), "age_band": _age_band(row.get("age"))}}


RECIPES: Dict[str, Dict] = {
    "chest_xray": {
        "repo": "hf-vision/chest-xray-pneumonia", "module": "chest_xray", "arch": "resnet18", "concern": ["pneumonia"],
        "caps": {"normal": 1100, "pneumonia": 1100}, "row": _xray,
        "note": "Paediatric chest X-rays (Kermany et al., Guangzhou; CC BY 4.0 mirror). Do not apply to adults."},
    "skin_lesion": {
        "repo": "marmal88/skin_cancer", "module": "skin_lesion", "arch": "resnet18", "concern": ["suspicious"],
        "caps": {"suspicious": 1100, "benign": 1400}, "row": _skin,
        "note": "HAM10000 dermatoscope images (CC BY-NC 4.0, non-commercial), split by lesion id. Dermatoscope photos: "
                "this does NOT transfer to ordinary phone photos."},
    "retina_dr": {
        "repo": "youssefedweqd/Diabetic_Retinopathy_Detection", "module": "retina_dr", "arch": "resnet18",
        "concern": ["moderate", "severe", "proliferative"],
        "caps": {"no_dr": 900, "mild": 450, "moderate": 700, "severe": 450, "proliferative": 450},
        "row": lambda r: {"labels": [DR_NAMES[int(r["label"])]], "group": None, "key": None, "attrs": {}},
        "note": "Retinal photos graded 0-4 (a community mirror, probably of the EyePACS / Kaggle DR images; origin not "
                "stated on the page). No patient id: both eyes of one person can be on both sides of the split."},
    "glaucoma": {
        "repo": "Nj-1111/Glaucoma_Dataset", "module": "glaucoma", "arch": "resnet18", "concern": ["glaucoma"],
        "caps": {"non_glaucoma": 1100, "glaucoma": 1100}, "folder": True,
        "note": "Fundus photos labelled referable glaucoma (rg) vs not (nrg); CC BY 4.0 mirror, pooled from several public "
                "sets (EyePACS, OIA, G1020, ...). Stored as PNG folders, not parquet. The sources are very unevenly split between "
                "the classes (most glaucoma is EyePACS, most normal is OIA), so a model could learn the SOURCE instead of the "
                "disease: this recipe takes the SAME number of glaucoma and normal images from every source that has both. "
                "No patient id."},
    "eye_disease": {
        "repo": "bumbledeep/odir", "module": "eye_disease", "arch": "resnet18",
        "concern": ["diabetes", "glaucoma", "cataract", "amd", "hypertension", "myopia", "other"], "multilabel": True,
        "caps": {"normal": 500, "diabetes": 500, "glaucoma": 300, "cataract": 300, "amd": 300, "hypertension": 150,
                 "myopia": 300, "other": 500},
        "row": _odir,
        "note": "ODIR-2019 fundus images (Peking University), one label per image in this mirror, split by patient id."},
}


def image_from(row) -> Optional[np.ndarray]:
    im = row.get("image")
    data = im.get("bytes") if isinstance(im, dict) else None
    if not data:
        return None
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def _fetch_folder(name: str, raw: Path, max_side: int, seed: int) -> List[Dict]:
    """Datasets stored as image folders (<split>/<class>/<SOURCE>-....png). Takes equal numbers of each class from every
    source that has both, so the source cannot stand in for the label."""
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import HfApi, hf_hub_download

    rc = RECIPES[name]
    files = [f for f in HfApi().list_repo_files(rc["repo"], repo_type="dataset") if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    _need(files, f"no image files found in {rc['repo']}")
    label_of = {"rg": "glaucoma", "nrg": "non_glaucoma"}
    by: Dict[tuple, List[str]] = defaultdict(list)
    for f in files:
        parts = f.split("/")
        lab = label_of.get(parts[-2].lower())
        if lab:
            by[(parts[-1].split("-")[0], lab)].append(f)
    labels = list(rc["caps"])
    sources = sorted({s for s, _ in list(by) if all(by.get((s, l)) for l in labels)})
    _need(sources, "no source has both classes")
    avail = {s: min(len(by[(s, l)]) for l in labels) for s in sources}
    scale = min(1.0, min(rc["caps"][l] for l in labels) / float(sum(avail.values())))
    rng = random.Random(seed)
    picks = []
    for s in sources:
        n = int(avail[s] * scale)
        for l in labels:
            rng.shuffle(by[(s, l)])
            picks += [(f, l, s) for f in by[(s, l)][:n]]
    print(f"{rc['repo']}: sources used {dict((s, int(avail[s] * scale)) for s in sources)} per class")
    img_dir = raw / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    def one(i_item):
        i, (f, lab, src) = i_item
        try:
            local = Path(hf_hub_download(rc["repo"], f, repo_type="dataset", local_dir=str(raw / "hf")))
            img = cv2.imread(str(local))
            local.unlink(missing_ok=True)
            if img is None:
                return None
            h, w = img.shape[:2]
            k = max_side / max(h, w)
            if k < 1:
                img = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
            key = f"{src}_{i}"
            path = img_dir / f"{key}.jpg"
            cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            return {"labels": [lab], "group": key, "attrs": {"source": src}, "path": path, "no_id": True}
        except Exception as exc:
            print(f"  skip {f}: {type(exc).__name__}")
            return None

    recs = []
    with ThreadPoolExecutor(8) as ex:
        for n, r in enumerate(ex.map(one, enumerate(picks)), 1):
            if r:
                recs.append(r)
            if n % 200 == 0:
                print(f"  {n}/{len(picks)} images", flush=True)
    return recs


def fetch(name: str, raw: Path, max_side: int, seed: int) -> List[Dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    rc = RECIPES[name]
    if rc.get("folder"):
        return _fetch_folder(name, raw, max_side, seed)
    api = HfApi()
    shards = sorted(f for f in api.list_repo_files(rc["repo"], repo_type="dataset") if f.endswith(".parquet"))
    _need(shards, f"no parquet files found in {rc['repo']}")
    random.Random(seed).shuffle(shards)
    caps, counts = rc["caps"], Counter()
    img_dir = raw / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    recs: List[Dict] = []
    print(f"{rc['repo']}: {len(shards)} shard(s); target per class {caps}")
    for si, shard in enumerate(shards):
        if all(counts[c] >= n for c, n in caps.items()):
            break
        local = Path(hf_hub_download(rc["repo"], shard, repo_type="dataset", local_dir=str(raw / "hf")))
        pf = pq.ParquetFile(local)
        for batch in pf.iter_batches(batch_size=64):
            for ri, row in enumerate(batch.to_pylist()):
                info = rc["row"](row)
                if info is None or counts[info["labels"][0]] >= caps.get(info["labels"][0], 0):
                    continue
                img = image_from(row)
                if img is None:
                    continue
                h, w = img.shape[:2]
                s = max_side / max(h, w)
                if s < 1:
                    img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
                key = info["key"] or f"s{si}_{len(recs)}"
                path = img_dir / f"{key}.jpg"
                cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                info["path"] = path
                info["group"] = info["group"] or key           # no id in the data: each image is its own group
                info["no_id"] = info["group"] == key
                counts[info["labels"][0]] += 1
                recs.append(info)
        try:                                                   # keep the disk free: shards are 100s of MB
            pf.close()                                         # Windows will not delete a file that is still open
            local.unlink(missing_ok=True)
        except OSError as exc:
            print(f"  (could not delete {local.name}: {exc}; it will be removed with the scratch folder)")
        print(f"  shard {si + 1}/{len(shards)} done: {dict(counts)}")
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", nargs="?", choices=sorted(RECIPES))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--raw", help="scratch folder (default data/_raw/<dataset>)")
    ap.add_argument("--max-side", type=int, default=384)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-raw", action="store_true")
    args = ap.parse_args()
    if args.list or not args.dataset:
        for k, v in RECIPES.items():
            print(f"{k:12s} {v['repo']}\n{'':12s} {v['note']}")
        return
    _need(args.out, "--out is required")
    rc = RECIPES[args.dataset]
    raw = Path(args.raw or ROOT / "data" / "_raw" / args.dataset)
    recs = fetch(args.dataset, raw, args.max_side, args.seed)
    _need(recs, "no images were collected")
    no_id = sum(1 for r in recs if r.pop("no_id", False))
    for r in recs:
        r.pop("key", None)
    info = {"records": recs, "missing": 0, "multilabel": bool(rc.get("multilabel")), "concern": rc["concern"],
            "module": rc["module"], "arch": rc["arch"], "note": rc["note"]}
    res = split_and_write(info, Path(args.out), args.val, args.test, args.seed, link=False)
    print(f"\n{rc['note']}")
    if no_id:
        print(f"WARNING: {no_id} of {len(recs)} images have no patient/lesion id, so each is its own group. Photos of the "
              "same person may be on both sides of the split: test numbers are optimistic.")
    print(f"{res['n_groups']} groups split with no group shared between splits.")
    for split in ("train", "val", "test"):
        row = res["counts"].get(split, {})
        print(f"  {split:6s} {sum(row.values()):6d}  {dict(row) if len(row) <= 12 else ''}")
    if not args.keep_raw:
        shutil.rmtree(raw, ignore_errors=True)


if __name__ == "__main__":
    main()
