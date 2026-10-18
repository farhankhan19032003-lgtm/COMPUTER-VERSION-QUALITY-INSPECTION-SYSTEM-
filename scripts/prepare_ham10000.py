#!/usr/bin/env python
"""Turn the HAM10000 skin-lesion dataset into train/val/test folders, split BY LESION.

HAM10000 contains several photos of the same lesion (different angles/magnifications).
A naive per-image split puts the same lesion in train and test and inflates accuracy, so
this script splits by lesion_id and verifies nothing is shared.

Get the data (free, research use; licence is CC BY-NC 4.0, so NOT for commercial use):
Harvard Dataverse "HAM10000" or the ISIC archive. Unzip so that HAM10000_metadata.csv and the
image folders are somewhere under --src.

    python scripts/prepare_ham10000.py --src ~/Downloads/ham10000 --out data/skin_lesion
    python scripts/train_classifier.py --data data/skin_lesion --name skin_lesion \\
        --arch efficientnet_b0 --concern suspicious --dataset-note "HAM10000, lesion-level split"

--mode binary   benign vs suspicious (mel, bcc, akiec)    [default, simplest to validate]
--mode 7class   the seven original diagnoses (akiec bcc bkl df mel nv vasc)

HAM10000 is dermatoscopic images from specific clinics. A phone camera photo looks different,
so a model trained on it will NOT transfer reliably to phone photos without more data.
"""
import argparse
import csv
from pathlib import Path

from _common import list_images
from _split import assert_no_leakage, materialise, report, split_groups

SUSPICIOUS = {"mel", "bcc", "akiec"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["binary", "7class"], default="binary")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--link", action="store_true", help="symlink instead of copying (saves disk)")
    args = ap.parse_args()

    src = Path(args.src)
    meta = next(iter(src.rglob("HAM10000_metadata.csv")), None)
    if meta is None:
        raise SystemExit(f"HAM10000_metadata.csv not found under {src}")
    with open(meta, newline="") as f:
        rows = list(csv.DictReader(f))
    need = {"image_id", "dx", "lesion_id"}
    if not rows or not need <= set(rows[0]):
        raise SystemExit(f"Metadata must have columns {sorted(need)}; found {list(rows[0]) if rows else 'none'}")

    paths = {p.stem: p for p in list_images(src)}
    label = (lambda dx: "suspicious" if dx in SUSPICIOUS else "benign") if args.mode == "binary" else (lambda dx: dx)

    group_label, items_raw, missing = {}, [], 0
    for r in rows:
        p = paths.get(r["image_id"])
        if p is None:
            missing += 1
            continue
        group_label.setdefault(r["lesion_id"], label(r["dx"]))
        items_raw.append((p, r["lesion_id"], label(r["dx"])))
    if missing:
        print(f"Warning: {missing} metadata rows had no image file and were skipped")
    if not items_raw:
        raise SystemExit("No images matched the metadata")

    assign = split_groups(group_label, args.val, args.test, args.seed)
    items = [(p, assign[g], cls, g) for p, g, cls in items_raw]
    assert_no_leakage(items)
    counts = materialise(items, Path(args.out), link=args.link)
    report(counts, len(group_label))
    print(f"\nWrote {args.out}. Use train/ and val/ for training; keep test/ untouched until the final evaluation:\n"
          f"  python scripts/evaluate_classifier.py --name skin_lesion --data {args.out}/test")


if __name__ == "__main__":
    main()
