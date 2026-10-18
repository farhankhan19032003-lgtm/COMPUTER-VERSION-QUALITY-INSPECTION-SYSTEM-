#!/usr/bin/env python
"""Train the small product-type classifier used by auto-detect (phone, laptop, headphones, ...).

    python scripts/train_product_classifier.py --stage fetch     # open-licence photos from Wikimedia Commons
    python scripts/train_product_classifier.py --stage train     # features -> linear head -> models/product/

YOLO's COCO classes cover phone, laptop, keyboard, mouse, TV and clock, but not headphones, tablets or wrist watches,
and it can miss a product that fills the frame. This head names the product from the whole photo, using the same
frozen backbone features as app/inspection/product_classifier.py (ResNet50 by default, --backbone to change).

Honesty notes
  * Photos are ordinary Commons snapshots, labelled only by the search text that found them. Search results are noisy,
    so a few labels are wrong. The numbers printed are measured on photographers the model never saw, but they say
    little about YOUR photos (rig, lighting, background). Check auto-detect on your own pictures.
  * "other" (landscapes, food, animals, buildings) teaches it to say "I don't know" instead of forcing a product.
  * Most photos are CC BY / CC BY-SA: keep attribution.csv with them and do not republish the photos without credit.
"""
import argparse
import csv
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from _common import ROOT, list_images

from app.inspection.product_classifier import DEFAULT_BACKBONE, LEGACY_BACKBONE, OTHER, embed, load_backbone

CLASSES = {
    "laptop": ["laptop computer", "notebook computer", "MacBook"],
    "tablet": ["tablet computer", "iPad", "Samsung Galaxy Tab"],
    "watch": ["wristwatch", "smartwatch", "Apple Watch"],
    "headphones": ["headphones", "over-ear headphones", "earphones", "wireless earbuds"],
    "monitor": ["computer monitor", "LCD monitor"],
    "keyboard": ["computer keyboard", "mechanical keyboard"],
    "mouse": ["computer mouse", "wireless mouse"],
    OTHER: ["mountain landscape", "city street", "food dish", "dog", "flower", "building", "beach", "tree"],
}
PHONE_SOURCES = [ROOT / "data" / "public" / "commons" / "good", ROOT / "data" / "public" / "commons" / "bad"]


def stage_fetch(data: Path, per_query: int) -> None:
    from fetch_commons import fetch

    phone = data / "phone"
    if not list_images(phone):
        phone.mkdir(parents=True, exist_ok=True)
        rows = []
        for src in PHONE_SOURCES:                                    # the intact + cracked phone photos fetched earlier
            att = {r["file"]: r for r in csv.DictReader(open(src / "attribution.csv", encoding="utf-8"))} if (src / "attribution.csv").exists() else {}
            for p in list_images(src):
                new = f"{src.name}_{p.name}"
                shutil.copy2(p, phone / new)
                rows.append({"file": new, **{k: att.get(p.name, {}).get(k, "") for k in ("title", "author", "license", "source")}})
        with open(phone / "attribution.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file", "title", "author", "license", "source"]); w.writeheader(); w.writerows(rows)
    print(f"phone: {len(list_images(phone))} photos (from earlier phone downloads)")
    for cls, queries in CLASSES.items():
        d = data / cls
        if len(list_images(d)) >= 40:
            print(f"{cls}: {len(list_images(d))} photos already there, skipping")
            continue
        n = fetch(queries, d, per_query=per_query, width=640, min_side=400)
        print(f"{cls}: saved {n} photos", flush=True)


def load_split(data: Path, test_frac: float, seed: int):
    """Images with labels and a train/test split by PHOTOGRAPHER within each class (series from one author stay together)."""
    rng = random.Random(seed)
    items = []                                                      # (path, class, author)
    for cdir in sorted(d for d in data.iterdir() if d.is_dir()):
        att = {}
        if (cdir / "attribution.csv").exists():
            att = {r["file"]: (r.get("author") or "").strip() for r in csv.DictReader(open(cdir / "attribution.csv", encoding="utf-8"))}
        for p in list_images(cdir):
            items.append((p, cdir.name, att.get(p.name) or f"anon-{p}"))
    by_cls = defaultdict(lambda: defaultdict(list))
    for p, c, a in items:
        by_cls[c][a].append(p)
    split = {}
    for c, authors in by_cls.items():
        names = sorted(authors)
        rng.shuffle(names)
        total, n_test, acc = sum(len(v) for v in authors.values()), 0, 0
        for a in names:
            to_test = acc < test_frac * total
            for p in authors[a]:
                split[p] = "test" if to_test else "train"
            acc += len(authors[a]) if to_test else 0
    return [(p, c, split[p]) for p, c, _ in items]


def stage_train(data: Path, out: Path, test_frac: float, seed: int, min_conf: float, backbone: str) -> None:
    from sklearn.linear_model import LogisticRegression

    rows = load_split(data, test_frac, seed)
    classes = sorted({c for _, c, _ in rows})
    print(f"{len(rows)} photos, classes: {classes}")
    net = load_backbone(backbone)
    feats, labels, splits = [], [], []
    for i in range(0, len(rows), 64):
        chunk, ims, keep = rows[i:i + 64], [], []
        for p, c, s in chunk:
            im = cv2.imread(str(p))
            if im is not None:
                ims.append(im); keep.append((c, s))
        if ims:
            feats.append(embed(net, ims)); labels += [classes.index(c) for c, _ in keep]; splits += [s for _, s in keep]
        print(f"  features {min(i + 64, len(rows))}/{len(rows)}", end="\r", flush=True)
    X, y, sp = np.concatenate(feats), np.array(labels), np.array(splits)
    tr, te = sp == "train", sp == "test"
    print(f"\ntrain {tr.sum()}  test {te.sum()} (test photographers never seen in training)")

    def fit(Xa, ya):
        return LogisticRegression(C=5.0, max_iter=3000, class_weight="balanced").fit(Xa, ya)

    m = fit(X[tr], y[tr])
    P = m.predict_proba(X[te]); pred = P.argmax(1); yt = y[te]
    print(f"\nheld-out accuracy: {(pred == yt).mean():.1%}  (n={te.sum()})")
    print(f"{'class':12s}{'n':>5s}{'recall':>9s}{'precision':>11s}")
    for k, c in enumerate(classes):
        n = int((yt == k).sum()); tp = int(((pred == k) & (yt == k)).sum()); pp = int((pred == k).sum())
        print(f"{c:12s}{n:5d}{(tp / n if n else float('nan')):9.0%}{(tp / pp if pp else float('nan')):11.0%}")
    oth = classes.index(OTHER) if OTHER in classes else -1
    conf = (P.max(1) >= min_conf) & (pred != oth)
    wrong = conf & (pred != yt)
    prod_true = yt != oth
    print(f"\nat the {min_conf:.2f} confidence cut-off the app uses:")
    print(f"  real products named correctly: {(conf & (pred == yt) & prod_true).sum()}/{prod_true.sum()} = {(conf & (pred == yt) & prod_true).sum() / max(1, prod_true.sum()):.0%}")
    print(f"  real products left as 'unknown' (user picks the type): {(~conf & prod_true).sum()}/{prod_true.sum()}")
    print(f"  a product named WRONGLY (incl. 'other' photos called a product): {wrong.sum()} of {conf.sum()} answers = {wrong.sum() / max(1, conf.sum()):.0%}")
    print(f"  'other' photos wrongly called a product: {(conf & ~prod_true).sum()}/{(~prod_true).sum()}")

    final = fit(X, y)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "product_head.npz", classes=np.array(classes), backbone=np.array(backbone), W=final.coef_.astype(np.float32), b=final.intercept_.astype(np.float32))
    (out / "product_head.json").write_text(json.dumps({
        "classes": classes, "backbone": backbone, "photos": int(len(y)), "held_out_photos": int(te.sum()), "min_conf": min_conf,
        "held_out_accuracy": float((pred == yt).mean()),
        "named_correctly": float((conf & (pred == yt) & prod_true).sum() / max(1, prod_true.sum())),
        "named_wrongly_of_answers": float(wrong.sum() / max(1, conf.sum())),
        "note": "Commons snapshots labelled by search text; held-out = photographers not seen in training. Not a measure of accuracy on your own photos."}, indent=2))
    print(f"\nSaved {out / 'product_head.npz'} (final head is refit on ALL photos; the numbers above are from the held-out run)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["fetch", "train"], required=True)
    ap.add_argument("--data", default=str(ROOT / "data" / "product_types"))
    ap.add_argument("--out", default=str(ROOT / "models" / "product"))
    ap.add_argument("--per-query", type=int, default=30)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--min-conf", type=float, default=0.55)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE, choices=[DEFAULT_BACKBONE, LEGACY_BACKBONE])
    a = ap.parse_args()
    data = Path(a.data)
    if a.stage == "fetch":
        stage_fetch(data, a.per_query)
    else:
        stage_train(data, Path(a.out), a.test_frac, a.seed, a.min_conf, a.backbone)


if __name__ == "__main__":
    main()
