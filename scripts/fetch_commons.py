#!/usr/bin/env python
"""Download openly licensed photos from Wikimedia Commons for testing (no login needed).

    python scripts/fetch_commons.py --query "cracked smartphone screen" --query "broken phone screen" \
        --out data/public/commons/candidates_bad

Photos are saved as <n>.jpg (resized to --width px) next to attribution.csv, which records title, author, licence
and the source page for each file. Most are CC BY / CC BY-SA: keep the attribution file with the photos and do not
republish them without the credit. Search results are noisy (screenshots, diagrams), so look at the photos and delete
the ones that do not fit before using them.
"""
import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

from _common import ROOT  # noqa: F401  (also fixes sys.path)

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "VisionInspect-test-fetch/1.0 (student project)"}
OPEN = re.compile(r"^(CC0|CC BY|CC BY-SA|Public domain|PD)", re.I)


def api(params):
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
        return json.load(r)


def search(query, limit, width):
    r = api({"action": "query", "generator": "search", "gsrsearch": query, "gsrnamespace": 6, "gsrlimit": limit,
             "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata", "iiurlwidth": width,
             "iiextmetadatafilter": "LicenseShortName|Artist"})
    for p in (r.get("query", {}).get("pages") or {}).values():
        ii = (p.get("imageinfo") or [{}])[0]
        meta = ii.get("extmetadata", {})
        lic = meta.get("LicenseShortName", {}).get("value", "")
        if ii.get("mime") not in ("image/jpeg", "image/png") or not OPEN.match(lic) or not ii.get("thumburl"):
            continue
        yield {
            "title": p["title"][5:], "license": lic, "source": ii.get("descriptionurl"),
            "author": re.sub(r"<[^>]+>", "", meta.get("Artist", {}).get("value", "")).strip()[:120],
            "url": ii["thumburl"] if ii["width"] > width else ii["url"], "w": ii["width"], "h": ii["height"],
        }


def fetch(queries, out, per_query=50, width=1000, min_side=600):
    """Download open-licence photos for the search texts into `out` (+ attribution.csv). Returns the number saved."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    seen, rows = set(), []
    for q in queries:
        try:
            found = list(search(q, per_query, width))
        except Exception as exc:                      # one failed search must not lose the others
            print(f"  search '{q}' failed: {type(exc).__name__}")
            continue
        for c in found:
            if c["title"] in seen or min(c["w"], c["h"]) < min_side:
                continue
            seen.add(c["title"])
            name = f"{len(rows) + 1:03d}.jpg"
            try:
                req = urllib.request.Request(c["url"], headers=UA)
                with urllib.request.urlopen(req, timeout=60) as r:
                    (out / name).write_bytes(r.read())
            except Exception as exc:
                print(f"  skip {c['title']}: {type(exc).__name__}")
                continue
            rows.append({"file": name, **{k: c[k] for k in ("title", "author", "license", "source")}})
            time.sleep(0.3)  # be polite to Wikimedia
    with open(out / "attribution.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "title", "author", "license", "source"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", action="append", required=True, help="search text (repeat for several)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-query", type=int, default=50)
    ap.add_argument("--width", type=int, default=1000, help="download size in px (originals can be 20 MB)")
    ap.add_argument("--min-side", type=int, default=600, help="skip originals smaller than this")
    args = ap.parse_args()
    n = fetch(args.query, args.out, args.per_query, args.width, args.min_side)
    print(f"Saved {n} photos + attribution.csv in {args.out}")


if __name__ == "__main__":
    main()
