#!/usr/bin/env python
"""Export reviewer-corrected photos from Supabase into folders you can train and test with.

    python scripts/export_feedback.py --out data/feedback
    python scripts/export_feedback.py --out data/feedback --only-overrides --product phone

Layout (the reviewer's verdict decides the folder, not the model's):

    data/feedback/<product>/good/<inspection_id>.jpg         reviewer said PASS
    data/feedback/<product>/defective/<inspection_id>.jpg    reviewer said FAIL
    data/feedback/<product>/borderline/<inspection_id>.jpg   reviewer said REVIEW
    data/feedback/labels.csv       one row per exported photo: model vs reviewer, note, defects
    data/feedback/conflicts.csv    inspections where reviewers disagree with each other (not exported)

Use it:
    * more good photos for the anomaly bank (false alarms are the most useful ones):
          python scripts/build_memory_bank.py --images data/feedback/phone/good --product phone
    * defective/ and borderline/ are labelled examples for a YOLO detector or for evaluation.

Keep in mind:
    * Photos are downloaded from the private bucket with the service-role key: needs Supabase set up in .env.
    * Do not measure accuracy on photos the bank was built from. Keep a locked test set aside.
    * If the same physical unit was photographed several times, split by unit, never by photo
      (see DATASETS.md); this export has no unit id.
    * Run again any time: photos already on disk are not downloaded twice.
"""
import argparse
import csv
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from _common import ROOT  # noqa: F401  (also fixes sys.path)

FOLDER = {"PASS": "good", "FAIL": "defective", "REVIEW": "borderline"}
FIELDS = ["file", "inspection_id", "product_type", "model_verdict", "reviewer_verdict", "agrees",
          "note", "defects", "source", "created_at"]


def safe_name(s: Optional[str]) -> str:
    return re.sub(r"[^a-z0-9_-]+", "_", (s or "unknown").lower()).strip("_") or "unknown"


def plan_export(
    feedback: Iterable[Dict], inspections: Dict[str, Dict], only_overrides: bool = False,
    product: Optional[str] = None,
) -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
    """Decide what to export. Returns (items, conflicts, skipped_counts). No file access."""
    by_inspection: Dict[str, List[Dict]] = {}
    for f in feedback:
        by_inspection.setdefault(f["inspection_id"], []).append(f)

    items, conflicts = [], []
    skipped = {"no_inspection": 0, "no_image": 0, "agreed": 0, "other_product": 0}
    for iid, reviews in by_inspection.items():
        insp = inspections.get(iid)
        if insp is None:
            skipped["no_inspection"] += 1
            continue
        verdicts = sorted({r["reviewer_verdict"] for r in reviews})
        if len(verdicts) > 1:  # reviewers disagree with each other: a human must settle it
            conflicts.append({"inspection_id": iid, "model_verdict": insp["verdict"],
                              "reviewer_verdicts": "|".join(verdicts),
                              "notes": " | ".join(r["note"] for r in reviews if r.get("note"))})
            continue
        if not insp.get("image_path"):
            skipped["no_image"] += 1
            continue
        prod = safe_name(insp.get("product_type"))
        if product and prod != safe_name(product):
            skipped["other_product"] += 1
            continue
        verdict = verdicts[0]
        if only_overrides and verdict == insp["verdict"]:
            skipped["agreed"] += 1
            continue
        notes = " | ".join(r["note"] for r in reviews if r.get("note"))
        items.append({
            "inspection_id": iid, "product": prod, "folder": FOLDER[verdict], "image_path": insp["image_path"],
            "row": {
                "inspection_id": iid, "product_type": prod, "model_verdict": insp["verdict"],
                "reviewer_verdict": verdict, "agrees": verdict == insp["verdict"], "note": notes,
                "defects": ",".join(sorted({d.get("type", "") for d in insp.get("defects") or []} - {""})),
                "source": insp.get("source"), "created_at": insp.get("created_at"),
            },
        })
    items.sort(key=lambda i: (i["product"], i["folder"], i["inspection_id"]))
    return items, conflicts, skipped


def write_export(items: List[Dict], conflicts: List[Dict], out: Path,
                 download: Callable[[str], bytes]) -> Dict[str, int]:
    """Download the photos and write labels.csv / conflicts.csv. Returns counts."""
    out.mkdir(parents=True, exist_ok=True)
    counts = {"downloaded": 0, "already_there": 0, "moved": 0, "failed": 0}
    rows = []
    for it in items:
        path = out / it["product"] / it["folder"] / f"{it['inspection_id']}.jpg"
        for other in set(FOLDER.values()) - {it["folder"]}:
            stale = out / it["product"] / other / path.name
            if stale.exists():  # the reviewer changed their verdict since the last export
                path.parent.mkdir(parents=True, exist_ok=True)
                stale.replace(path)
                counts["moved"] += 1
        if path.exists():
            counts["already_there"] += 1
        else:
            try:
                data = download(it["image_path"])
                if cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) is None:
                    raise ValueError("not a readable image")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                counts["downloaded"] += 1
            except Exception as exc:  # one bad file must not stop the export
                print(f"  failed {it['inspection_id']}: {type(exc).__name__}: {str(exc)[:100]}")
                counts["failed"] += 1
                continue
        rows.append({"file": path.relative_to(out).as_posix(), **it["row"]})

    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    if conflicts:
        with open(out / "conflicts.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(conflicts[0]))
            w.writeheader()
            w.writerows(conflicts)
    return counts


def fetch_all(query_factory, page: int = 1000) -> List[Dict]:
    """Read every row of a Supabase query (the API returns at most 1000 rows per request)."""
    rows, start = [], 0
    while True:
        batch = query_factory().range(start, start + page - 1).execute().data or []
        rows += batch
        if len(batch) < page:
            return rows
        start += page


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "data" / "feedback"))
    ap.add_argument("--only-overrides", action="store_true",
                    help="export only photos where the reviewer disagreed with the model")
    ap.add_argument("--product", help="export one product type only, e.g. phone")
    ap.add_argument("--user-id", help="export only the reviews of this user (Supabase user id)")
    args = ap.parse_args()

    from app.config import Settings
    from app.db import SupabaseStore

    settings = Settings()
    if not settings.supabase_enabled:
        raise SystemExit("Supabase is not configured in .env. Photos are only kept in Supabase "
                         "(the in-memory store drops them on restart), so there is nothing to export.")
    client = SupabaseStore(settings)._client
    bucket = client.storage.from_(settings.supabase_bucket)

    def fb_query():
        q = client.table("inspection_feedback").select("*").order("created_at").order("id")
        return q.eq("user_id", args.user_id) if args.user_id else q

    feedback = fetch_all(fb_query)
    ids = sorted({f["inspection_id"] for f in feedback})
    inspections: Dict[str, Dict] = {}
    for i in range(0, len(ids), 100):  # keep the request URL short
        res = client.table("inspections").select("*").in_("id", ids[i:i + 100]).execute().data or []
        inspections.update({r["id"]: r for r in res})

    items, conflicts, skipped = plan_export(feedback, inspections, args.only_overrides, args.product)
    print(f"{len(feedback)} review(s) on {len(ids)} inspection(s) -> {len(items)} photo(s) to export")
    counts = write_export(items, conflicts, Path(args.out), bucket.download)

    print(f"\nWrote {args.out}")
    print(f"  downloaded {counts['downloaded']}, already there {counts['already_there']}, "
          f"moved (verdict changed) {counts['moved']}, failed {counts['failed']}")
    for k, v in skipped.items():
        if v:
            print(f"  skipped {v}: {k.replace('_', ' ')}")
    if conflicts:
        print(f"  {len(conflicts)} inspection(s) have conflicting reviews: see {args.out}/conflicts.csv")
    tally: Dict[Tuple[str, str], int] = {}
    for it in items:
        tally[(it["product"], it["folder"])] = tally.get((it["product"], it["folder"]), 0) + 1
    for (prod, folder), n in sorted(tally.items()):
        print(f"  {prod}/{folder}: {n}")
    if any(f == "good" for _, f in tally):
        prod = next(p for p, f in tally if f == "good")
        print(f"\nNext: python scripts/build_memory_bank.py --images {args.out}/{prod}/good --product {prod}")


if __name__ == "__main__":
    main()
