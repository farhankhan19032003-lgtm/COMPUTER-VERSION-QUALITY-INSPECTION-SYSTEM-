"""Group-aware train/val/test splitting: the guard against data leakage.

If photos of the same patient / lesion / physical unit land in both train and test,
the model memorises that individual and the test score is inflated. Always split by group.
"""
from __future__ import annotations

import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def split_groups(group_label: Dict[str, str], val: float, test: float, seed: int = 0) -> Dict[str, str]:
    """Assign each group to train/val/test, stratified by the group's label."""
    rng = random.Random(seed)
    by_label: Dict[str, List[str]] = defaultdict(list)
    for g, lab in group_label.items():
        by_label[lab].append(g)
    out: Dict[str, str] = {}
    for lab, groups in by_label.items():
        groups.sort()
        rng.shuffle(groups)
        n = len(groups)
        n_test = int(round(n * test))
        n_val = int(round(n * val))
        if n >= 3:                      # make sure every class shows up in val and test
            n_test, n_val = max(1, n_test), max(1, n_val)
        for i, g in enumerate(groups):
            out[g] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return out


def materialise(items: Iterable[Tuple[Path, str, str, str]], out: Path, link: bool = False) -> Dict[str, Dict[str, int]]:
    """items = (source path, split, class name, group). Copies or symlinks into out/split/class/."""
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for src, split, cls, _ in items:
        dest = out / split / cls / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            continue
        if link:
            dest.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dest)
        counts[split][cls] += 1
    return counts


def assert_no_leakage(items: List[Tuple[Path, str, str, str]]) -> None:
    seen: Dict[str, str] = {}
    for _, split, _, group in items:
        if group in seen and seen[group] != split:
            raise AssertionError(f"Leakage: group {group!r} appears in both {seen[group]} and {split}")
        seen[group] = split


def report(counts, n_groups: int) -> None:
    print(f"\n{n_groups} groups (patients/lesions/units) split with no group shared between splits.")
    classes = sorted({c for s in counts.values() for c in s})
    print(f"{'split':8s}" + "".join(f"{c:>14s}" for c in classes) + f"{'total':>9s}")
    for split in ("train", "val", "test"):
        row = counts.get(split, {})
        print(f"{split:8s}" + "".join(f"{row.get(c, 0):>14d}" for c in classes) + f"{sum(row.values()):>9d}")
