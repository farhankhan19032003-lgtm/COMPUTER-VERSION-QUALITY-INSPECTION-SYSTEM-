#!/usr/bin/env python
"""Split any folder-per-class image dataset into train/val/test without leakage.

    data_raw/<class>/<file>.jpg   ->   out/{train,val,test}/<class>/<file>.jpg

Works for medical sets (chest X-ray, retina, malaria...) and for your own product-defect photos.
Tell the script which photos belong together (same patient / same physical unit) with
--group-regex, applied to the file name without extension; the first capture group is the group id:

    # files like  P0012_left.jpg, P0012_right.jpg  -> group "P0012"
    python scripts/prepare_folder_dataset.py --src data_raw --out data/chest_xray --group-regex '^(P\\d+)_'

Without --group-regex every file is treated as its own group. That is only safe if each file is
a different patient or unit; otherwise your test results will be too optimistic.
"""
import argparse
import re
from pathlib import Path

from _common import list_images
from _split import assert_no_leakage, materialise, report, split_groups


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group-regex")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--link", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    rx = re.compile(args.group_regex) if args.group_regex else None
    if rx is None:
        print("WARNING: no --group-regex: assuming every image is a different patient/unit.")

    raw, group_label = [], {}
    for p in list_images(src):
        cls = p.relative_to(src).parts[0]
        if p.parent == src:
            continue                    # loose files without a class folder
        if rx:
            m = rx.search(p.stem)
            if not m:
                raise SystemExit(f"--group-regex did not match {p.name}")
            group = f"{cls}:{m.group(1)}"
        else:
            group = f"{cls}:{p.relative_to(src)}"
        if group_label.setdefault(group, cls) != cls:
            raise SystemExit(f"Group {group} appears under two different classes")
        raw.append((p, group, cls))
    if not raw:
        raise SystemExit("No images found in class sub-folders")

    assign = split_groups(group_label, args.val, args.test, args.seed)
    items = [(p, assign[g], cls, g) for p, g, cls in raw]
    assert_no_leakage(items)
    report(materialise(items, Path(args.out), link=args.link), len(group_label))


if __name__ == "__main__":
    main()
