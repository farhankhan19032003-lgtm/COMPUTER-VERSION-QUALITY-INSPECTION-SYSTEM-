#!/usr/bin/env python
"""One tool to turn a downloaded public dataset into leakage-safe train/val/test folders.

    python scripts/prepare_dataset.py --list
    python scripts/prepare_dataset.py aptos      --src ~/data/aptos2019   --out data/retina_dr
    python scripts/prepare_dataset.py ham10000   --src ~/data/ham10000    --out data/skin_lesion
    python scripts/prepare_dataset.py pad_ufes   --src ~/data/pad-ufes-20 --out data/skin_lesion_phone
    python scripts/prepare_dataset.py malaria    --src ~/data/cell_images --out data/malaria_smear
    python scripts/prepare_dataset.py chest_xray --src ~/data/chest_xray  --out data/chest_xray
    python scripts/prepare_dataset.py odir       --src ~/data/odir5k      --out data/eye_disease
    python scripts/prepare_dataset.py csv        --src ~/data/mine --csv ~/data/mine/labels.csv --out data/anemia_conjunctiva

Every adapter splits by the unit that must never be shared between splits (patient, lesion, slide)
and then verifies it. Where the dataset has no patient id (APTOS) the script says so: that split can
leak and its metrics will be optimistic.

I wrote each adapter from the published file layout of the dataset (recalled, and checked only
against the small test fixtures in tests/test_datasets.py). Real downloads sometimes differ, so every
adapter checks the columns/folders it needs and stops with a clear message instead of guessing.
You must download the data yourself and accept each dataset's licence; nothing is fetched here.

Besides the images the script writes <out>/subgroups.csv (file, split, plus attributes such as sex,
age band or skin type where the dataset has them) so evaluate_classifier.py --subgroups can report
performance per group.
"""
import argparse
import csv
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List

from _common import list_images
from _split import assert_no_leakage, split_groups

Record = Dict  # {"path": Path, "labels": [str], "group": str, "attrs": {str: str}}


class DatasetError(SystemExit):
    pass


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise DatasetError(f"ERROR: {msg}")


def _age_band(v) -> str:
    try:
        a = float(v)
    except (TypeError, ValueError):
        return "unknown"
    return "<30" if a < 30 else "30-49" if a < 50 else "50-69" if a < 70 else "70+"


def _index_images(src: Path) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    for p in list_images(src):
        idx.setdefault(p.name, p)
        idx.setdefault(p.stem, p)
    return idx


def _read_table(path: Path) -> List[Dict[str, str]]:
    if path.suffix.lower() in (".xlsx", ".xls"):
        import pandas as pd

        df = pd.read_excel(path, dtype=str).fillna("")
        return df.to_dict("records")
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- adapters
def adapter_aptos(src: Path, **kw) -> Dict:
    csv_path = next(iter(src.rglob("train.csv")), None)
    _need(csv_path is not None, f"train.csv not found under {src} (APTOS 2019 'train.csv' with id_code,diagnosis)")
    rows = _read_table(csv_path)
    _need(rows and {"id_code", "diagnosis"} <= set(rows[0]), f"train.csv needs columns id_code, diagnosis; found {list(rows[0]) if rows else 'none'}")
    names = ["no_dr", "mild", "moderate", "severe", "proliferative"]
    imgs = _index_images(src)
    recs, missing = [], 0
    for r in rows:
        p = imgs.get(r["id_code"])
        if p is None:
            missing += 1
            continue
        recs.append({"path": p, "labels": [names[int(r["diagnosis"])]], "group": r["id_code"], "attrs": {}})
    return {"records": recs, "missing": missing, "multilabel": False,
            "concern": ["moderate", "severe", "proliferative"], "module": "retina_dr", "arch": "efficientnet_b0",
            "note": "APTOS 2019 (no patient id in the public file: the split is per image and CAN leak if both eyes of "
                    "a patient are present; treat validation numbers as optimistic)"}


def _adapter_lesion(src, meta_name, id_col, group_col, dx_col, suspicious, mode, attr_cols, module, note, concern="suspicious"):
    meta = next(iter(src.rglob(meta_name)), None)
    _need(meta is not None, f"{meta_name} not found under {src}")
    rows = _read_table(meta)
    need = {id_col, group_col, dx_col}
    _need(rows and need <= set(rows[0]), f"{meta_name} needs columns {sorted(need)}; found {list(rows[0]) if rows else 'none'}")
    imgs = _index_images(src)
    recs, missing = [], 0
    for r in rows:
        p = imgs.get(r[id_col]) or imgs.get(Path(r[id_col]).stem)
        if p is None:
            missing += 1
            continue
        dx = r[dx_col].strip().lower()
        label = ("suspicious" if dx in suspicious else "benign") if mode == "binary" else dx
        attrs = {}
        for out_name, col in attr_cols.items():
            v = r.get(col, "")
            attrs[out_name] = _age_band(v) if out_name == "age_band" else (v.strip() or "unknown")
        recs.append({"path": p, "labels": [label], "group": r[group_col], "attrs": attrs})
    return {"records": recs, "missing": missing, "multilabel": False,
            "concern": [concern] if mode == "binary" else sorted(suspicious), "module": module, "arch": "efficientnet_b0",
            "note": note}


def adapter_ham10000(src: Path, mode="binary", **kw) -> Dict:
    return _adapter_lesion(src, "HAM10000_metadata.csv", "image_id", "lesion_id", "dx", {"mel", "bcc", "akiec"}, mode,
                           {"sex": "sex", "age_band": "age", "site": "localization"}, "skin_lesion",
                           "HAM10000 dermatoscopic images, lesion-level split (CC BY-NC 4.0)")


def adapter_pad_ufes(src: Path, mode="binary", **kw) -> Dict:
    meta = next(iter(src.rglob("metadata.csv")), None)
    _need(meta is not None, f"metadata.csv not found under {src} (PAD-UFES-20)")
    rows = _read_table(meta)
    fz = "fitspatrick" if rows and "fitspatrick" in rows[0] else "fitzpatrick"
    out = _adapter_lesion(src, "metadata.csv", "img_id", "patient_id", "diagnostic", {"mel", "bcc", "scc", "ack"}, mode,
                          {"fitzpatrick": fz, "age_band": "age", "site": "region"}, "skin_lesion_phone",
                          "PAD-UFES-20 smartphone photos, PATIENT-level split. ACK (actinic keratosis, pre-cancer) is "
                          "counted as suspicious like HAM10000 akiec")
    return out


def adapter_malaria(src: Path, **kw) -> Dict:
    folders = {d.name.lower(): d for d in src.rglob("*") if d.is_dir() and d.name.lower() in ("parasitized", "uninfected")}
    _need(len(folders) == 2, f"need 'Parasitized' and 'Uninfected' folders under {src}")
    recs, no_slide = [], 0
    for key, label in (("parasitized", "parasitized"), ("uninfected", "uninfected")):
        for p in list_images(folders[key]):
            m = re.match(r"(.+?)_cell_\d+", p.stem)
            if not m:
                no_slide += 1
            recs.append({"path": p, "labels": [label], "group": m.group(1) if m else p.stem, "attrs": {}})
    return {"records": recs, "missing": 0, "multilabel": False, "concern": ["parasitized"], "module": "malaria_smear",
            "arch": "resnet18",
            "note": "NIH malaria cell crops, split by SLIDE (id taken from the filename)" +
                    (f"; {no_slide} files had no slide id and are their own group, so leakage is possible" if no_slide else "")}


def adapter_chest_xray(src: Path, **kw) -> Dict:
    classes = {"normal": "normal", "pneumonia": "pneumonia"}
    recs = []
    for p in list_images(src):
        parent = next((classes[a.name.lower()] for a in p.parents if a.name.lower() in classes), None)
        if parent is None:
            continue
        m = re.search(r"person(\d+)", p.stem, re.I) or re.search(r"(IM-\d+)", p.stem, re.I)
        recs.append({"path": p, "labels": [parent], "group": m.group(0).lower() if m else p.stem, "attrs": {}})
    _need(recs, f"no images found under NORMAL / PNEUMONIA folders in {src}")
    return {"records": recs, "missing": 0, "multilabel": False, "concern": ["pneumonia"], "module": "chest_xray",
            "arch": "densenet121",
            "note": "Kaggle chest X-ray (pediatric) re-split by patient id from the filename; the original val folder is only "
                    "16 images so all folders are pooled and re-split. Paediatric data: do not apply to adults."}


ODIR_FLAGS = {"N": "normal", "D": "diabetes", "G": "glaucoma", "C": "cataract", "A": "amd", "H": "hypertension",
              "M": "myopia", "O": "other"}


def adapter_odir(src: Path, **kw) -> Dict:
    table = next((p for p in src.rglob("*") if p.suffix.lower() in (".xlsx", ".xls", ".csv") and "annot" in p.name.lower()), None) \
        or next((p for p in src.rglob("*.csv")), None)
    _need(table is not None, f"annotation table (xlsx/csv) not found under {src}")
    rows = _read_table(table)
    need = {"ID", "Left-Fundus", "Right-Fundus"} | set(ODIR_FLAGS)
    _need(rows and need <= set(rows[0]), f"ODIR annotations need columns {sorted(need)}; found {list(rows[0]) if rows else 'none'}")
    imgs = _index_images(src)
    recs, missing = [], 0
    for r in rows:
        labels = [name for flag, name in ODIR_FLAGS.items() if str(r.get(flag, "0")).strip() in ("1", "1.0")]
        if not labels:
            continue
        for eye in ("Left-Fundus", "Right-Fundus"):
            p = imgs.get(r[eye].strip())
            if p is None:
                missing += 1
                continue
            recs.append({"path": p, "labels": labels, "group": str(r["ID"]),
                         "attrs": {"sex": (r.get("Patient Sex") or "unknown").strip() or "unknown",
                                   "age_band": _age_band(r.get("Patient Age"))}})
    return {"records": recs, "missing": missing, "multilabel": True,
            "concern": [n for n in ODIR_FLAGS.values() if n != "normal"], "module": "eye_disease", "arch": "efficientnet_b0",
            "note": "ODIR-5K: labels are per PATIENT (both eyes), so each eye image inherits the patient's labels. "
                    "Split by patient."}


def adapter_csv(src: Path, csv_path: str = None, **kw) -> Dict:
    _need(csv_path, "--csv is required for the generic adapter")
    rows = _read_table(Path(csv_path))
    _need(rows and {"file", "label"} <= set(rows[0]), "CSV needs columns file,label (optional: group, and any attribute columns)")
    imgs = _index_images(src)
    recs, missing = [], 0
    reserved = {"file", "label", "group"}
    for r in rows:
        p = (src / r["file"]) if (src / r["file"]).exists() else imgs.get(Path(r["file"]).name)
        if p is None:
            missing += 1
            continue
        labels = [l for l in r["label"].split("|") if l]
        recs.append({"path": Path(p), "labels": labels, "group": r.get("group") or Path(p).stem,
                     "attrs": {k: (v or "unknown") for k, v in r.items() if k not in reserved}})
    multilabel = any(len(r["labels"]) > 1 for r in recs)
    all_labels = sorted({l for r in recs for l in r["labels"]})
    return {"records": recs, "missing": missing, "multilabel": multilabel, "concern": all_labels[-1:], "module": "custom",
            "arch": "resnet50", "note": "generic CSV dataset. Set --concern yourself when training. Without a 'group' column "
                                         "every image is its own group, which can leak"}


ADAPTERS: Dict[str, Callable[..., Dict]] = {
    "aptos": adapter_aptos, "ham10000": adapter_ham10000, "pad_ufes": adapter_pad_ufes, "malaria": adapter_malaria,
    "chest_xray": adapter_chest_xray, "odir": adapter_odir, "csv": adapter_csv,
}


# ----------------------------------------------------------------------------- writing
def cap_records(recs: List[Record], per_class: int, seed: int = 0) -> List[Record]:
    """Keep about `per_class` images of each class, adding whole groups at a time so no patient / slide is cut in two.

    For quick CPU experiments and small disks. Single-label records only (multi-label sets are left alone).
    """
    import random

    rng = random.Random(seed)
    by_label: Dict[str, Dict[str, List[Record]]] = defaultdict(lambda: defaultdict(list))
    for r in recs:
        by_label[r["labels"][0]][r["group"]].append(r)
    kept: List[Record] = []
    for label, groups in sorted(by_label.items()):
        order = sorted(groups)
        rng.shuffle(order)
        n = 0
        for g in order:
            if n >= per_class:
                break
            kept += groups[g]
            n += len(groups[g])
    return kept


def split_and_write(info: Dict, out: Path, val: float, test: float, seed: int, link: bool) -> Dict:
    recs: List[Record] = info["records"]
    _need(recs, "no images matched the annotation table")
    names = [r["path"].name for r in recs]
    dup = [n for n, c in Counter(names).items() if c > 1]
    _need(not dup, f"duplicate image file names (first: {dup[:3]}); rename them or use the csv adapter")

    # One label per group for stratification (group's rarest label, so rare classes spread across splits).
    freq = Counter(l for r in recs for l in r["labels"])
    group_label: Dict[str, str] = {}
    for r in recs:
        lab = min(r["labels"], key=lambda l: freq[l])
        group_label.setdefault(r["group"], lab)
    assign = split_groups(group_label, val, test, seed)
    items = [(r["path"], assign[r["group"]], "|".join(r["labels"]), r["group"]) for r in recs]
    assert_no_leakage(items)

    counts: Dict[str, Counter] = defaultdict(Counter)
    attr_names = sorted({k for r in recs for k in r["attrs"]})
    out.mkdir(parents=True, exist_ok=True)
    label_rows: Dict[str, list] = defaultdict(list)
    sub_rows = []
    for r in recs:
        split = assign[r["group"]]
        lab = "|".join(r["labels"])
        dest_dir = out / split if info["multilabel"] else out / split / r["labels"][0]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / r["path"].name
        if not dest.exists():
            if link:
                dest.symlink_to(r["path"].resolve())
            else:
                shutil.copy2(r["path"], dest)
        counts[split][lab if not info["multilabel"] else "images"] += 1
        label_rows[split].append([r["path"].name, lab])
        sub_rows.append([r["path"].name, split, *[r["attrs"].get(a, "unknown") for a in attr_names]])
    if info["multilabel"]:
        for split, rows in label_rows.items():
            with open(out / split / "labels.csv", "w", newline="") as f:
                w = csv.writer(f); w.writerow(["file", "labels"]); w.writerows(rows)
    if attr_names:
        with open(out / "subgroups.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["file", "split", *attr_names]); w.writerows(sub_rows)
    return {"counts": counts, "n_groups": len(group_label), "attrs": attr_names}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", nargs="?", choices=sorted(ADAPTERS))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--src")
    ap.add_argument("--out")
    ap.add_argument("--csv", help="generic adapter: CSV with file,label[,group,...]")
    ap.add_argument("--mode", choices=["binary", "full"], default="binary", help="lesion sets: benign/suspicious or the original classes")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--link", action="store_true", help="symlink instead of copying")
    ap.add_argument("--max-per-class", type=int, default=0,
                    help="keep about this many images per class (whole patients/slides at a time); 0 = all")
    args = ap.parse_args(argv)

    if args.list or not args.dataset:
        print("Adapters:", ", ".join(sorted(ADAPTERS)), "\nSee MODULES.md for sizes, licences and caveats of each dataset.")
        return
    _need(args.src and args.out, "--src and --out are required")
    src = Path(args.src)
    _need(src.exists(), f"{src} does not exist")
    kw = {"mode": args.mode, "csv_path": args.csv}
    info = ADAPTERS[args.dataset](src, **kw)
    if info["missing"]:
        print(f"Warning: {info['missing']} annotation rows had no image file and were skipped")
    if args.max_per_class and not info["multilabel"]:
        before = len(info["records"])
        info["records"] = cap_records(info["records"], args.max_per_class, args.seed)
        print(f"--max-per-class {args.max_per_class}: using {len(info['records'])} of {before} images")
    res = split_and_write(info, Path(args.out), args.val, args.test, args.seed, args.link)

    print(f"\n{info['note']}")
    print(f"{res['n_groups']} groups split with no group shared between splits.")
    for split in ("train", "val", "test"):
        row = res["counts"].get(split, {})
        print(f"  {split:6s} {sum(row.values()):6d}  {dict(row) if len(row) <= 12 else ''}")
    multi = " --multilabel" if info["multilabel"] else ""
    print(f"\nNext:\n  python scripts/train_classifier.py --data {args.out} --name {info['module']} --arch {info['arch']}{multi} "
          f"--concern {' '.join(info['concern'])} --dataset-note \"{args.dataset}\"\n"
          f"  python scripts/evaluate_classifier.py --name {info['module']} --data {args.out}/test"
          + (f" --subgroups {args.out}/subgroups.csv" if res["attrs"] else ""))


if __name__ == "__main__":
    main()
