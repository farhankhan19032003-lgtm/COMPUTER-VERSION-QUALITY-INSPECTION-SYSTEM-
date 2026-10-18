#!/usr/bin/env python
"""Train an image-screening classifier and write <name>.pt, <name>.json (+ <name>.ood.npz) for the API.

Single-label dataset layout (torchvision ImageFolder), made by scripts/prepare_dataset.py:
    data/<name>/train/<class>/*.jpg
    data/<name>/val/<class>/*.jpg        <- patients kept separate from train!

Multi-label layout (e.g. ODIR, one patient can have several findings):
    data/<name>/train/labels.csv  (columns: file,labels   labels like "diabetes|myopia")  + the images
    data/<name>/val/labels.csv

    python scripts/train_classifier.py --data data/retina_dr --name retina_dr --arch efficientnet_b0 \
        --concern moderate severe proliferative --dataset-note "APTOS 2019"

After training it also fits, on the VALIDATION split only:
  * temperature scaling (calibrated probabilities)
  * triage thresholds for a target sensitivity (--target-sensitivity, default 0.90)
  * a Mahalanobis out-of-distribution guard (images unlike the training data become INCONCLUSIVE)

IMPORTANT: a model trained here is a research prototype. Using it on real patients needs clinical
validation on an independent, representative dataset and regulatory clearance.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from _common import ROOT

from _metrics import auc_binary
from app.inspection.cv_defects import assess_quality
from app.screening.calibration import (MahalanobisOOD, choose_triage_thresholds, collect_outputs,
                                       expected_calibration_error, fit_temperature)
from app.screening.classifier import SUPPORTED_ARCHS, build_arch

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class LabelCsvDataset(Dataset):
    """Multi-label images listed in <root>/labels.csv (file,labels with '|' between labels)."""

    def __init__(self, root: Path, transform, labels=None):
        self.root, self.transform = Path(root), transform
        rows = list(csv.DictReader(open(self.root / "labels.csv", newline="")))
        names = sorted({l for r in rows for l in r["labels"].split("|") if l}) if labels is None else list(labels)
        self.classes, idx = names, {n: i for i, n in enumerate(names)}
        self.samples = []
        for r in rows:
            y = np.zeros(len(names), np.float32)
            for l in r["labels"].split("|"):
                if l in idx:
                    y[idx[l]] = 1.0
            self.samples.append((str(self.root / r["file"]), y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        return self.transform(Image.open(path).convert("RGB")), torch.from_numpy(y)


def calibrate_quality_gate(samples, limit=300):
    """Quality limits from the training images themselves: reject only what is clearly worse than
    anything the model learned from (much blurrier, darker or smaller)."""
    sharp, bright, side = [], [], []
    for path, _ in samples[:: max(1, len(samples) // limit)]:
        img = cv2.imread(path)
        if img is None:
            continue
        q = assess_quality(img, min_sharpness=0, min_side=0, min_brightness=0)
        sharp.append(q["sharpness"]); bright.append(q["brightness"]); side.append(min(img.shape[:2]))
    if not sharp:
        return {}
    return {
        "min_sharpness": round(float(min(25.0, max(2.0, 0.5 * np.percentile(sharp, 5)))), 1),
        "min_brightness": round(float(min(40.0, max(5.0, 0.5 * np.percentile(bright, 5)))), 1),
        "min_side": int(min(160, max(32, 0.5 * np.percentile(side, 5)))),
    }


def metrics_from(logits, y, n_classes, multilabel):
    """Per-class sensitivity / specificity / AUC at a 0.5 operating point (for the model card)."""
    if multilabel:
        p = 1 / (1 + np.exp(-logits))
        pred = (p >= 0.5).astype(int)
        yy = y.astype(int)
        per_class = {}
        for c in range(n_classes):
            tp = int(((pred[:, c] == 1) & (yy[:, c] == 1)).sum()); fn = int(((pred[:, c] == 0) & (yy[:, c] == 1)).sum())
            tn = int(((pred[:, c] == 0) & (yy[:, c] == 0)).sum()); fp = int(((pred[:, c] == 1) & (yy[:, c] == 0)).sum())
            per_class[c] = {"n": int(yy[:, c].sum()),
                            "sensitivity": tp / (tp + fn) if tp + fn else None,
                            "specificity": tn / (tn + fp) if tn + fp else None,
                            "auc": auc_binary(yy[:, c], p[:, c])}
        acc = float((pred == yy).mean())
    else:
        z = logits - logits.max(1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        pred, yy = p.argmax(1), y.astype(int)
        per_class = {}
        for c in range(n_classes):
            tp = int(((pred == c) & (yy == c)).sum()); fn = int(((pred != c) & (yy == c)).sum())
            tn = int(((pred != c) & (yy != c)).sum()); fp = int(((pred == c) & (yy != c)).sum())
            per_class[c] = {"n": int((yy == c).sum()),
                            "sensitivity": tp / (tp + fn) if tp + fn else None,
                            "specificity": tn / (tn + fp) if tn + fp else None,
                            "auc": auc_binary((yy == c).astype(int), p[:, c])}
        acc = float((pred == yy).mean())
    aucs = [v["auc"] for v in per_class.values() if v["auc"] is not None]
    return {"accuracy": acc, "macro_auc": float(np.mean(aucs)) if aucs else None, "per_class": per_class}


def concern_scores(logits, y, labels, concern, T, multilabel):
    """Calibrated 'any concerning finding' probability and the 0/1 truth, for threshold selection."""
    idx = [labels.index(c) for c in concern]
    z = logits / T
    if multilabel:
        p = 1 / (1 + np.exp(-z))
        return p[:, idx].max(1), (y[:, idx].max(1) > 0.5).astype(int)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    return p[:, idx].sum(1), np.isin(y.astype(int), idx).astype(int)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", required=True, help="module name, e.g. skin_lesion, retina_dr")
    ap.add_argument("--title", default=None)
    ap.add_argument("--arch", default="resnet50", choices=SUPPORTED_ARCHS)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--concern", nargs="+", required=True, help="class names that count as a potential finding")
    ap.add_argument("--multilabel", action="store_true", help="data uses labels.csv with several labels per image")
    ap.add_argument("--target-sensitivity", type=float, default=0.90)
    ap.add_argument("--target-specificity", type=float, default=0.95)
    ap.add_argument("--no-ood", action="store_true", help="skip the out-of-distribution guard")
    ap.add_argument("--dataset-note", default="")
    ap.add_argument("--out", default=str(ROOT / "models" / "screening"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    data = Path(args.data)
    norm = transforms.Normalize(MEAN, STD)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.7, 1.0)), transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12), transforms.ColorJitter(0.2, 0.2, 0.1), transforms.ToTensor(), norm])
    val_tf = transforms.Compose([transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor(), norm])
    if args.multilabel:
        train_ds = LabelCsvDataset(data / "train", train_tf)
        labels = train_ds.classes
        val_ds = LabelCsvDataset(data / "val", val_tf, labels)
        fit_ds = LabelCsvDataset(data / "train", val_tf, labels)          # train images, no augmentation
        counts = np.stack([y for _, y in train_ds.samples]).sum(0)
    else:
        train_ds = datasets.ImageFolder(data / "train", train_tf)
        val_ds = datasets.ImageFolder(data / "val", val_tf)
        fit_ds = datasets.ImageFolder(data / "train", val_tf)
        labels = train_ds.classes
        assert val_ds.classes == labels, "train and val must have the same class folders"
        counts = np.bincount([t for _, t in train_ds.samples], minlength=len(labels))
    missing = [c for c in args.concern if c not in labels]
    if missing:
        raise SystemExit(f"--concern {missing} not among classes {labels}")
    print(f"classes: {labels}  train counts: {np.asarray(counts).astype(int).tolist()}  val: {len(val_ds)}  device: {device}")

    train_dl = DataLoader(train_ds, args.batch, shuffle=True, num_workers=args.workers, drop_last=len(train_ds) > args.batch)
    val_dl = DataLoader(val_ds, args.batch, num_workers=args.workers)
    fit_dl = DataLoader(fit_ds, args.batch, num_workers=args.workers)

    model = build_arch(args.arch, len(labels), pretrained=True).to(device)
    if args.multilabel:
        n = len(train_ds)
        pw = torch.tensor(np.clip((n - counts) / np.maximum(counts, 1), 1, 20), dtype=torch.float32, device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        w = torch.tensor(counts.sum() / (len(labels) * np.maximum(counts, 1)), dtype=torch.float32, device=device)
        loss_fn = nn.CrossEntropyLoss(weight=w)                           # compensates for rare disease classes
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    best, out = -1.0, Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); run = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += loss.item() * len(x)
        sched.step()
        lg, _, yv = collect_outputs(model, val_dl, device)
        m = metrics_from(lg, yv, len(labels), args.multilabel)
        score = m["macro_auc"] if m["macro_auc"] is not None else m["accuracy"]
        print(f"epoch {epoch:2d}  loss {run / len(train_ds):.4f}  val acc {m['accuracy']:.3f}  "
              f"macro-AUC {m['macro_auc']}  ({time.time() - t0:.0f}s)")
        if score > best:
            best = score
            torch.save(model.state_dict(), out / f"{args.name}.pt")

    # ---- calibration, thresholds and OOD guard: fitted on VALIDATION outputs of the best checkpoint
    model.load_state_dict(torch.load(out / f"{args.name}.pt", map_location=device, weights_only=True))
    lg, feats_val, yv = collect_outputs(model, val_dl, device)
    best_metrics = metrics_from(lg, yv, len(labels), args.multilabel)
    T = fit_temperature(lg, yv, multilabel=args.multilabel)
    p_raw, y_bin = concern_scores(lg, yv, labels, args.concern, 1.0, args.multilabel)
    p_cal, _ = concern_scores(lg, yv, labels, args.concern, T, args.multilabel)
    ece_before, ece_after = expected_calibration_error(p_raw, y_bin), expected_calibration_error(p_cal, y_bin)
    thr = choose_triage_thresholds(y_bin, p_cal, args.target_sensitivity, args.target_specificity)
    print(f"\ntemperature T={T}  ECE {ece_before:.3f} -> {ece_after:.3f}   thresholds: {thr}")
    ood_note = "not fitted"
    if not args.no_ood:
        lt, feats_tr, ytr = collect_outputs(model, fit_dl, device)
        lab_tr = ytr.argmax(1) if args.multilabel else ytr
        if args.multilabel:                                      # normal vs any-finding is enough for a guard
            lab_tr = (ytr.sum(1) > 0).astype(int)
        ood = MahalanobisOOD.fit(feats_tr, lab_tr, feats_val)
        ood.save(out / f"{args.name}.ood.npz")
        ood_note = f"fitted, threshold {ood.threshold:.1f}"
    print(f"OOD guard: {ood_note}")

    spec = {
        "title": args.title or args.name.replace("_", " ").title(),
        "labels": labels, "concern_labels": args.concern, "arch": args.arch, "input_size": args.img_size,
        "mean": MEAN, "std": STD, "multilabel": args.multilabel, "min_confidence": 0.0 if args.multilabel else 0.40,
        "temperature": T, "tta": True,
        "moderate_threshold": thr["moderate_threshold"], "high_threshold": thr["high_threshold"],
        "threshold_source": thr["source"],
        "training_data": args.dataset_note,
        **calibrate_quality_gate(train_ds.samples),
        "validation": {
            "n_images": len(val_ds),
            "accuracy": best_metrics["accuracy"], "macro_auc": best_metrics["macro_auc"],
            "calibration_ece_before": round(ece_before, 4), "calibration_ece_after": round(ece_after, 4),
            "per_class": {labels[c]: v for c, v in best_metrics["per_class"].items()},
            "note": "Internal validation split only (the same split chose the checkpoint, temperature and thresholds, "
                    "so it is optimistic). Not clinical validation: run scripts/evaluate_classifier.py on a locked test set.",
        },
    }
    (out / f"{args.name}.json").write_text(json.dumps(spec, indent=2))
    print(f"\nSaved {out / (args.name + '.pt')} and .json")
    print("Per-class at 0.5 (sensitivity / specificity / AUC):")
    for c, v in best_metrics["per_class"].items():
        f = lambda z: "n/a" if z is None else f"{z:.3f}"  # noqa: E731
        print(f"  {labels[c]:20s} n={v['n']:5d}  {f(v['sensitivity'])} / {f(v['specificity'])} / {f(v['auc'])}")


if __name__ == "__main__":
    main()
