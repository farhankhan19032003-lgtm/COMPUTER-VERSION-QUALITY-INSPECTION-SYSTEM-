#!/usr/bin/env python
"""Standalone edge runner: inspect a webcam / RTSP / video file with no web server.

    python scripts/run_camera.py --source 0 --product phone --show
    python scripts/run_camera.py --source rtsp://user:pw@cam/stream --max-fps 5 --supabase

Frames are grabbed on a background thread so a slow model never causes a growing backlog
(for RTSP this keeps latency low). Verdicts are smoothed over several frames. Non-PASS
frames are saved to --save-dir and, with --supabase, logged to your Supabase project.
"""
import argparse
import threading
import time
from pathlib import Path

import cv2

from _common import ROOT  # noqa: F401

from app.config import Settings
from app.db import MemoryStore, build_store
from app.inspection.pipeline import InspectionPipeline
from app.inspection.temporal import TemporalSmoother
from app.utils import encode_jpeg


class LatestFrame:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
        if not self.cap.isOpened():
            raise SystemExit(f"Cannot open source {source}")
        self.frame, self.ok, self.stop, self._lock = None, True, False, threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self.stop:
            ok, f = self.cap.read()
            with self._lock:
                self.ok = ok
                if ok:
                    self.frame = f
            if not ok:
                time.sleep(0.05)

    def read(self):
        with self._lock:
            f, self.frame = self.frame, None
            return f

    def release(self):
        self.stop = True
        self.cap.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0")
    ap.add_argument("--product", default=None)
    ap.add_argument("--max-fps", type=float, default=5.0)
    ap.add_argument("--show", action="store_true", help="show an annotated preview window")
    ap.add_argument("--save-dir", default=str(ROOT / "data" / "rejects"))
    ap.add_argument("--supabase", action="store_true", help="log non-PASS results to Supabase (needs .env)")
    ap.add_argument("--cooldown", type=float, default=3.0, help="min seconds between saved rejects")
    args = ap.parse_args()

    settings = Settings()
    pipe = InspectionPipeline(settings)
    store = build_store(settings) if args.supabase else MemoryStore()
    smoother = TemporalSmoother()
    src = LatestFrame(args.source)
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running. detector={pipe.detector.backend} banks={pipe.anomaly.available()}  Ctrl+C to stop.")

    last_save, interval, n = 0.0, 1.0 / args.max_fps, 0
    try:
        while True:
            t0 = time.time()
            frame = src.read()
            if frame is None:
                if not src.ok:  # end of a video file or a dropped stream
                    print("\nSource ended.")
                    break
                time.sleep(0.01)
                continue
            res = pipe.inspect(frame, args.product, annotate=args.show or True)
            smooth = smoother.update(res.verdict)
            n += 1
            print(f"\r#{n} {smooth:6s} raw={res.verdict:6s} score={res.score:.2f} {res.latency_ms:5.0f} ms "
                  f"defects={[d.type for d in res.defects][:3]}      ", end="", flush=True)

            if smooth != "PASS" and time.time() - last_save >= args.cooldown:
                last_save = time.time()
                (save_dir / f"{int(last_save)}_{smooth}.jpg").write_bytes(res.annotated)
                store.insert_inspection({
                    "user_id": None, "source": "camera", "product_type": res.product_type,
                    "verdict": res.verdict, "score": res.score, "anomaly_ratio": res.anomaly_ratio,
                    "defects": [d.to_dict() for d in res.defects], "detections": [d.to_dict() for d in res.detections],
                    "quality": res.quality, "latency_ms": res.latency_ms,
                    "image_path": store.upload_image(None, encode_jpeg(frame)),
                    "annotated_path": store.upload_image(None, res.annotated),
                })
            if args.show:
                cv2.imshow("VisionInspect", cv2.imdecode(
                    __import__("numpy").frombuffer(res.annotated, "uint8"), cv2.IMREAD_COLOR))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            time.sleep(max(0.0, interval - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        src.release()
        cv2.destroyAllWindows()
        print("\nStopped.")


if __name__ == "__main__":
    main()
