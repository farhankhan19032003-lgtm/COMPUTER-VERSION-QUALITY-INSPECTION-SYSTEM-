"""In-process runtime metrics: latency, throughput, error rate, system warnings.

Feeds the dashboard's "inspection metrics / system warnings / edge stability" panels.
Counters reset on restart; durable history lives in Supabase.
"""
from __future__ import annotations

import threading
import time
from collections import Counter, deque
from typing import Any, Deque, Dict, List, Tuple

import psutil

from .utils import percentile


class Metrics:
    def __init__(self, device: str) -> None:
        self.started = time.time()
        self.device = device
        self._lock = threading.Lock()
        # (timestamp, verdict, latency_ms, quality_ok, source)
        self._insp: Deque[Tuple[float, str, float, bool, str]] = deque(maxlen=1000)
        self._requests: Deque[Tuple[float, bool]] = deque(maxlen=500)  # (ts, ok)
        self._defect_types: Counter = Counter()
        self._products: Counter = Counter()
        self.total_inspections = 0
        self.total_screenings = 0
        self.total_errors = 0
        psutil.cpu_percent(None)  # prime the counter

    # -- recording -----------------------------------------------------------------
    def record_inspection(self, verdict, latency_ms, quality_ok, product_type, defect_types, source="upload"):
        with self._lock:
            self._insp.append((time.time(), verdict, float(latency_ms), bool(quality_ok), source))
            self._requests.append((time.time(), True))
            self._products[product_type] += 1
            self._defect_types.update(defect_types)
            self.total_inspections += 1

    def record_screening(self):
        with self._lock:
            self._requests.append((time.time(), True))
            self.total_screenings += 1

    def record_error(self):
        with self._lock:
            self._requests.append((time.time(), False))
            self.total_errors += 1

    # -- reporting -----------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        now = time.time()
        with self._lock:
            insp = list(self._insp)
            reqs = list(self._requests)
            defects = dict(self._defect_types.most_common(10))
            products = dict(self._products.most_common(10))
            totals = (self.total_inspections, self.total_screenings, self.total_errors)

        lat = [r[2] for r in insp]
        recent = [r for r in insp if now - r[0] <= 10]
        verdicts = Counter(r[1] for r in insp)
        n = max(1, len(insp))
        mem = psutil.virtual_memory().percent
        cpu = psutil.cpu_percent(None)
        err_rate = (sum(1 for _, ok in reqs if not ok) / len(reqs)) if reqs else 0.0
        p95 = percentile(lat, 95)

        stability = 100.0
        stability -= min(40.0, err_rate * 400)
        if p95 is not None and p95 > 500:
            stability -= min(25.0, (p95 - 500) / 40)
        if mem > 80:
            stability -= min(20.0, (mem - 80) * 2)
        if cpu > 90:
            stability -= 10.0

        return {
            "uptime_s": int(now - self.started),
            "device": self.device,
            "totals": {"inspections": totals[0], "screenings": totals[1], "errors": totals[2]},
            "recent_window": {
                "count": len(insp),
                "pass_rate": round(verdicts["PASS"] / n, 3),
                "review_rate": round(verdicts["REVIEW"] / n, 3),
                "fail_rate": round(verdicts["FAIL"] / n, 3),
                "low_quality_rate": round(sum(1 for r in insp if not r[3]) / n, 3),
            },
            "latency_ms": {
                "p50": percentile(lat, 50), "p95": p95, "p99": percentile(lat, 99),
                "avg": (sum(lat) / len(lat)) if lat else None,
            },
            "throughput_fps": round(len(recent) / 10.0, 2),
            "defect_types": defects,
            "products": products,
            "edge": {
                "cpu_pct": round(cpu, 1),
                "mem_pct": round(mem, 1),
                "error_rate": round(err_rate, 4),
                "stability_score": round(max(0.0, stability), 1),
            },
        }

    def warnings(self, snap: Dict[str, Any], pipeline_status: Dict[str, Any], store_name: str) -> List[Dict[str, str]]:
        w: List[Dict[str, str]] = []
        rw, lat, edge = snap["recent_window"], snap["latency_ms"], snap["edge"]
        if lat["p95"] is not None and lat["p95"] > 800:
            w.append({"level": "WARNING", "message": f"High inference latency (p95 {lat['p95']:.0f} ms). "
                      "Lower ANOMALY_IMAGE_SIZE, use a smaller YOLO model, or enable GPU."})
        if edge["error_rate"] > 0.05:
            w.append({"level": "ERROR", "message": f"Request error rate {edge['error_rate'] * 100:.1f}% over the last requests."})
        if rw["count"] >= 20 and rw["fail_rate"] > 0.4:
            w.append({"level": "WARNING", "message": f"Unusually high FAIL rate ({rw['fail_rate'] * 100:.0f}%). "
                      "Check camera position, lighting or the production line."})
        if rw["count"] >= 20 and rw["low_quality_rate"] > 0.25:
            w.append({"level": "WARNING", "message": f"{rw['low_quality_rate'] * 100:.0f}% of images are low quality "
                      "(blur, darkness, low resolution). Verdicts are less reliable."})
        if edge["mem_pct"] > 85:
            w.append({"level": "WARNING", "message": f"Memory usage at {edge['mem_pct']}%."})
        if edge["cpu_pct"] > 95:
            w.append({"level": "WARNING", "message": f"CPU saturated ({edge['cpu_pct']}%)."})
        if pipeline_status["detector"] != "yolo":
            w.append({"level": "INFO", "message": "YOLO unavailable: using the classical fallback detector "
                      "(install ultralytics and set YOLO_WEIGHTS for multi-product detection)."})
        if not pipeline_status["anomaly_banks"]:
            w.append({"level": "INFO", "message": "No anomaly banks installed: only visible-damage checks run. "
                      "Build one with scripts/build_memory_bank.py."})
        if store_name != "supabase":
            w.append({"level": "WARNING", "message": "Supabase not configured: results are kept in memory and lost on restart."})
        return w
