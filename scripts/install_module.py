#!/usr/bin/env python
"""Train a screening module, test it on the locked test split, and install it ONLY if it clearly beats chance.

    python scripts/prepare_hf.py skin_lesion --out data/skin_lesion
    python scripts/install_module.py --name skin_lesion --data data/skin_lesion --concern suspicious \
        --arch resnet18 --img-size 160 --epochs 4 --dataset-note "HAM10000 (CC BY-NC)"

What it does:
  1. trains into models/_staging/ (never straight into models/screening/)
  2. runs scripts/evaluate_classifier.py on <data>/test, images the model never saw and that were not used to pick
     the checkpoint, temperature or thresholds
  3. installs (copies into models/screening/) only if the lower end of the 95% AUC interval is at least --min-auc-lower
     (default 0.70). Otherwise the module stays "not installed" and the reason is printed.
  4. writes the locked-test numbers into the module's .json, so the API, the dashboard and the PDF report can show them.

Passing this gate does NOT make a model clinically valid. It only means "clearly better than guessing on this
dataset". Read the leakage warning printed by prepare_hf.py / prepare_dataset.py for your data.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from _common import ROOT

PY = sys.executable
SCRIPTS = Path(__file__).resolve().parent


def run(cmd):
    print("\n$", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT)
    if r.returncode:
        raise SystemExit(f"step failed (exit {r.returncode}): {cmd[1]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--concern", nargs="+", required=True)
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--img-size", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--multilabel", action="store_true")
    ap.add_argument("--dataset-note", default="")
    ap.add_argument("--title", default=None)
    ap.add_argument("--min-auc-lower", type=float, default=0.70)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--skip-train", action="store_true", help="re-run the test and the gate on the staged model")
    args = ap.parse_args()

    staging, final, reports = ROOT / "models" / "_staging", ROOT / "models" / "screening", ROOT / "reports"
    data = Path(args.data)
    reports.mkdir(exist_ok=True)

    if not args.skip_train:
        cmd = [PY, SCRIPTS / "train_classifier.py", "--data", data, "--name", args.name, "--arch", args.arch,
               "--img-size", args.img_size, "--epochs", args.epochs, "--batch", args.batch, "--workers", args.workers,
               "--out", staging, "--concern", *args.concern, "--dataset-note", args.dataset_note]
        if args.multilabel:
            cmd.append("--multilabel")
        if args.title:
            cmd += ["--title", args.title]
        run(cmd)

    rep_path = reports / f"{args.name}.json"
    cmd = [PY, SCRIPTS / "evaluate_classifier.py", "--name", args.name, "--data", data / "test",
           "--model-dir", staging, "--report", rep_path]
    if (data / "subgroups.csv").exists():
        cmd += ["--subgroups", data / "subgroups.csv"]
    run(cmd)

    rep = json.loads(rep_path.read_text())
    lo = (rep.get("auc") or {}).get("ci95", [None, None])[0]
    auc = (rep.get("auc") or {}).get("value")
    ok = lo is not None and lo >= args.min_auc_lower
    print("\n" + "=" * 70)
    if not ok:
        print(f"NOT INSTALLED: {args.name}. Locked-test AUC {auc} with lower 95% bound {lo} is below the required "
              f"{args.min_auc_lower}. The model stays in {staging} and the module remains 'not installed'.")
        raise SystemExit(2)

    spec_path = staging / f"{args.name}.json"
    spec = json.loads(spec_path.read_text())
    spec.setdefault("validation", {})["locked_test"] = {
        "n_images": rep["n"], "positives": rep["positives"], "negatives": rep["negatives"],
        "auc": auc, "auc_ci95": rep["auc"]["ci95"], "ece": rep.get("ece"),
        "inconclusive_rate": rep.get("inconclusive_rate"),
        "moderate": rep.get("moderate"), "high": rep.get("high"),
        "gate": f"installed because the lower 95% AUC bound >= {args.min_auc_lower}",
        "note": "Measured on a held-out split of the SAME public dataset. Not clinical validation; accuracy on real patients "
                "or other cameras can be far lower.",
    }
    spec_path.write_text(json.dumps(spec, indent=2))
    final.mkdir(parents=True, exist_ok=True)
    for ext in (".pt", ".json", ".ood.npz"):
        src = staging / f"{args.name}{ext}"
        if src.exists():
            shutil.copy2(src, final / src.name)
    print(f"INSTALLED: {args.name}  (locked-test AUC {auc:.3f}, 95% CI {rep['auc']['ci95'][0]:.3f} to {rep['auc']['ci95'][1]:.3f}, "
          f"n={rep['n']})")


if __name__ == "__main__":
    main()
