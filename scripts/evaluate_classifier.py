#!/usr/bin/env python
"""Honest evaluation of a screening module on a held-out TEST folder.

    python scripts/evaluate_classifier.py --name skin_lesion --data data/skin_lesion/test
    python scripts/evaluate_classifier.py --name skin_lesion --data data/skin_lesion/test \
        --subgroups data/skin_lesion/subgroups.csv        # fairness: sex, age band, skin type, site ...

The folder has either one sub-folder per class (the names the model was trained with) or, for
multi-label modules, a labels.csv (file,labels) next to the images.
Reports, for "any concerning finding" vs not:
  * AUC with a 95% bootstrap CI, and calibration error (ECE)
  * sensitivity and specificity at the MODERATE and HIGH triage thresholds, each with a 95% CI
  * how many images the system refused to judge (INCONCLUSIVE), counted against sensitivity
  * with --subgroups: the same numbers per group, and a warning when a group is clearly worse
A model is only as good as the lower end of these intervals, on data it has never seen, from
the same kind of camera and population you plan to use it on.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from _common import ROOT, list_images
from _metrics import auc_binary, bootstrap_ci, fmt, sens_spec

from app.screening.calibration import expected_calibration_error
from app.screening.classifier import ScreeningRegistry

GAP_WARN = 0.10          # flag a subgroup whose sensitivity is this much lower than overall
MIN_POS_FOR_CI = 10


def iter_test_images(root: Path, model):
    """yield (path, is_concerning). Folder-per-class or labels.csv."""
    csv_path = root / "labels.csv"
    if csv_path.exists():
        for r in csv.DictReader(open(csv_path, newline="")):
            labs = set(l for l in r["labels"].split("|") if l)
            yield root / r["file"], int(bool(labs & model.concern)), r["labels"]
        return
    for cls_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        if cls_dir.name not in model.labels:
            print(f"skip folder '{cls_dir.name}': not one of the model's classes {model.labels}")
            continue
        for path in list_images(cls_dir):
            yield path, int(cls_dir.name in model.concern), cls_dir.name


def rate_ci(y, pred, which, boot):
    se, sp = sens_spec(y, pred)
    v = se if which == 0 else sp
    ci = bootstrap_ci(y, pred, lambda a, b: sens_spec(a, b)[which], boot)
    return v, ci


def subgroup_table(sub_csv: Path, files, y, flagged, concl, boot, overall_sens):
    """Per attribute value: n, positives, sensitivity (with CI), specificity, inconclusive rate."""
    rows = {r["file"]: r for r in csv.DictReader(open(sub_csv, newline=""))}
    attrs = [c for c in next(iter(rows.values())).keys() if c not in ("file", "split")] if rows else []
    out, warnings = {}, []
    for a in attrs:
        vals = np.array([rows.get(f, {}).get(a, "unknown") for f in files])
        out[a] = {}
        print(f"\nSubgroup: {a}")
        print(f"  {'group':14s}{'n':>6s}{'pos':>6s}{'sensitivity [95% CI]':>28s}{'specificity':>14s}{'refused':>9s}")
        for v in sorted(set(vals)):
            m = vals == v
            yy, pp, cc = y[m], flagged[m], concl[m]
            n_pos = int(yy.sum())
            if n_pos >= MIN_POS_FOR_CI and (len(yy) - n_pos) >= MIN_POS_FOR_CI:
                se, se_ci = rate_ci(yy, pp, 0, boot)
                sp, _ = rate_ci(yy, pp, 1, boot)
                se_txt = f"{fmt(se)} [{fmt(se_ci[0])}, {fmt(se_ci[1])}]"
                if se_ci[1] < overall_sens - GAP_WARN or se < overall_sens - GAP_WARN:
                    warnings.append(f"{a}={v}: sensitivity {fmt(se)} is well below overall {fmt(overall_sens)} (n_pos={n_pos})")
                out[a][v] = {"n": int(m.sum()), "pos": n_pos, "sensitivity": se, "sensitivity_ci95": list(se_ci), "specificity": sp,
                             "refused": float((~cc).mean())}
            else:
                se_txt, sp = "too few cases", None
                out[a][v] = {"n": int(m.sum()), "pos": n_pos, "note": "too few cases to estimate"}
            print(f"  {str(v):14s}{int(m.sum()):6d}{n_pos:6d}{se_txt:>28s}{fmt(sp) if sp is not None else '-':>14s}{(~cc).mean() * 100:8.1f}%")
    return out, warnings


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "screening"))
    ap.add_argument("--subgroups", help="CSV with a 'file' column plus attribute columns (made by prepare_dataset.py)")
    ap.add_argument("--report", help="write the numbers to this JSON file")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    model = ScreeningRegistry(args.model_dir, "auto").get(args.name)
    if model is None:
        raise SystemExit(f"Module '{args.name}' not found in {args.model_dir}")
    root = Path(args.data)
    files, y, p, triage, per_class, ood_n = [], [], [], [], {}, 0
    for path, is_pos, lab in iter_test_images(root, model):
        img = cv2.imread(str(path))
        if img is None:
            continue
        r = model.predict(img)
        files.append(path.name); y.append(is_pos); p.append(r.concern_probability); triage.append(r.triage)
        ood_n += int(bool(r.ood and r.ood["out_of_distribution"]))
        per_class[lab] = per_class.get(lab, 0) + 1
    if not y:
        raise SystemExit("No images evaluated")
    y, p, triage = np.array(y), np.array(p), np.array(triage)
    concl = triage != "INCONCLUSIVE"
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    print(f"\n{len(y)} test images  (concerning: {n_pos}, not concerning: {n_neg})")
    if min(n_pos, n_neg) < 50:
        print("WARNING: fewer than 50 images in a group: intervals will be very wide.")
    print(f"INCONCLUSIVE (refused): {(~concl).sum()} ({(~concl).mean() * 100:.1f}%)   of which flagged out-of-distribution: {ood_n}")

    report = {"n": int(len(y)), "positives": n_pos, "negatives": n_neg, "inconclusive_rate": float((~concl).mean()),
              "calibration": model.provenance()["calibrated"]}
    auc = auc_binary(y[concl], p[concl])
    lo, hi = bootstrap_ci(y[concl], p[concl], auc_binary, args.boot)
    ece = expected_calibration_error(p[concl], y[concl])
    print(f"\nAUC (conclusive images): {fmt(auc)}  95% CI [{fmt(lo)}, {fmt(hi)}]    calibration error (ECE): {ece:.3f}"
          + ("" if report["calibration"] else "   (model is NOT temperature-calibrated)"))
    report["auc"] = {"value": auc, "ci95": [lo, hi]}
    report["ece"] = ece

    flagged_by_tier = {}
    for name, thr in (("MODERATE", model.t_moderate), ("HIGH", model.t_high)):
        pred = ((p >= thr) & concl).astype(int)       # refused images count as "not flagged"
        se, sp = sens_spec(y, pred)
        se_ci = bootstrap_ci(y, pred, lambda a, b: sens_spec(a, b)[0], args.boot)
        sp_ci = bootstrap_ci(y, pred, lambda a, b: sens_spec(a, b)[1], args.boot)
        print(f"flag at >= {name} ({thr:.2f}):  sensitivity {fmt(se)} [{fmt(se_ci[0])}, {fmt(se_ci[1])}]   "
              f"specificity {fmt(sp)} [{fmt(sp_ci[0])}, {fmt(sp_ci[1])}]")
        report[name.lower()] = {"threshold": thr, "sensitivity": se, "sensitivity_ci95": list(se_ci),
                                "specificity": sp, "specificity_ci95": list(sp_ci)}
        flagged_by_tier[name] = (pred, se)

    if args.subgroups:
        pred, se = flagged_by_tier["MODERATE"]
        table, warnings = subgroup_table(Path(args.subgroups), files, y, pred, concl, args.boot, se)
        report["subgroups"] = table
        report["subgroup_warnings"] = warnings
        if warnings:
            print("\nFAIRNESS WARNINGS (sensitivity at the MODERATE threshold):")
            for w in warnings:
                print("  - " + w)
            print("A model that works worse for one group must not be deployed to that group until fixed "
                  "(more data from that group, re-calibration per group, or a human always in the loop).")
        else:
            print("\nNo subgroup stood out, but groups with 'too few cases' are NOT shown to be fair, just untested.")

    print("\nFor screening, missing a real finding (low sensitivity) is usually the costlier error; tune the "
          "thresholds in the module's .json to trade specificity for sensitivity, then re-test on fresh data.")
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))
        print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
