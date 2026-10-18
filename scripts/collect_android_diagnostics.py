#!/usr/bin/env python
"""Read battery / storage / memory from a USB-connected Android phone via adb.

Setup on the phone: Settings > About phone > tap "Build number" 7 times, then
Developer options > enable USB debugging, connect the cable and accept the prompt.
Needs `adb` (Android platform-tools) on your PATH.

    python scripts/collect_android_diagnostics.py                      # print JSON
    python scripts/collect_android_diagnostics.py --symptoms overheating fast_drain
    python scripts/collect_android_diagnostics.py --post http://localhost:8000 --token <jwt>

Which values are available differs by phone maker and Android version (battery capacity
and cycle count are often hidden); missing values are simply left out.
"""
import argparse
import json
import shutil
import subprocess
import sys
import urllib.request

from _common import ROOT  # noqa: F401

from app.diagnostics.android import build_payload

SYSFS = "/sys/class/power_supply/battery"


def adb(serial, *args, timeout=15):
    cmd = ["adb"] + (["-s", serial] if serial else []) + ["shell"] + list(args)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return ""
    return r.stdout if r.returncode == 0 else ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serial", help="device serial if more than one is connected (adb devices)")
    ap.add_argument("--symptoms", nargs="*", default=[], help="e.g. overheating fast_drain bulging")
    ap.add_argument("--post", help="API base URL to send the result to, e.g. http://localhost:8000")
    ap.add_argument("--token", help="Supabase access token for --post")
    args = ap.parse_args()

    if not shutil.which("adb"):
        sys.exit("adb not found. Install Android platform-tools and add it to PATH.")
    devices = subprocess.run(["adb", "devices"], capture_output=True, text=True).stdout.splitlines()[1:]
    ready = [l.split()[0] for l in devices if l.strip().endswith("device")]
    if not ready:
        sys.exit("No authorised device. Enable USB debugging and accept the prompt on the phone.")
    serial = args.serial or (ready[0] if len(ready) == 1 else None)
    if serial is None:
        sys.exit(f"Several devices connected {ready}: choose one with --serial")

    payload = build_payload(
        model=adb(serial, "getprop", "ro.product.model"),
        release=adb(serial, "getprop", "ro.build.version.release"),
        battery_text=adb(serial, "dumpsys", "battery"),
        df_text=adb(serial, "df", "-k", "/data"),
        meminfo_text=adb(serial, "cat", "/proc/meminfo"),
        charge_full=adb(serial, "cat", f"{SYSFS}/charge_full"),
        charge_full_design=adb(serial, "cat", f"{SYSFS}/charge_full_design"),
        cycle_count=adb(serial, "cat", f"{SYSFS}/cycle_count"),
    )
    payload["symptoms"] = args.symptoms
    print(json.dumps(payload, indent=2))

    if args.post:
        req = urllib.request.Request(
            args.post.rstrip("/") + "/api/v1/diagnostics/phone",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {args.token}"} if args.token else {})},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            res = json.load(r)
        print(f"\nHealth score {res['health_score']}/100 -> {res['verdict']}")
        for f in res["findings"]:
            print(f"  [{f['severity']}] {f['title']}  {f.get('evidence', '')}")


if __name__ == "__main__":
    main()
