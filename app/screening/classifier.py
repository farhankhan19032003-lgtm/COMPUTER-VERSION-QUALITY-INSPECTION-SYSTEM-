"""Image-based health screening classifiers (pluggable, bring-your-own trained weights).

Every module is a torchvision CNN fine-tuned on a medical image dataset. This file only
loads and runs them, adds an image-quality gate, a triage mapping and Grad-CAM, and wraps
every output in a "not a diagnosis" disclaimer.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision

from ..inspection.cv_defects import assess_quality
from ..utils import INFERENCE_GATE, encode_jpeg, resolve_device
from .calibration import MahalanobisOOD, final_linear
from .catalog import CATALOG, DISCLAIMER

log = logging.getLogger(__name__)

SUPPORTED_ARCHS = ("resnet18", "resnet50", "densenet121", "efficientnet_b0", "mobilenet_v3_large")


def build_arch(arch: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """torchvision backbone with its head replaced for `num_classes` outputs."""
    tv = torchvision.models
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"Unsupported arch '{arch}'. Choose from {SUPPORTED_ARCHS}")

    def make(fn, weights_enum):
        if pretrained:
            try:
                return fn(weights=weights_enum.DEFAULT)
            except Exception as exc:
                log.warning("Pretrained weights unavailable (%s); training from scratch", exc)
        return fn(weights=None)

    if arch == "resnet18":
        m = make(tv.resnet18, tv.ResNet18_Weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "resnet50":
        m = make(tv.resnet50, tv.ResNet50_Weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "densenet121":
        m = make(tv.densenet121, tv.DenseNet121_Weights)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    elif arch == "efficientnet_b0":
        m = make(tv.efficientnet_b0, tv.EfficientNet_B0_Weights)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:  # mobilenet_v3_large
        m = make(tv.mobilenet_v3_large, tv.MobileNet_V3_Large_Weights)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    return m


@dataclass
class ScreeningResult:
    module: str
    top_label: str
    probabilities: Dict[str, float]
    concern_probability: float
    triage: str                       # LOW | MODERATE | HIGH | INCONCLUSIVE
    message: str
    quality: Dict[str, Any]
    latency_ms: float = 0.0
    heatmap_jpeg: Optional[bytes] = None
    disclaimer: str = DISCLAIMER
    ood: Optional[Dict[str, Any]] = None          # {"score": .., "threshold": .., "out_of_distribution": bool}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module": self.module,
            "top_label": self.top_label,
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "concern_probability": round(self.concern_probability, 4),
            "triage": self.triage,
            "message": self.message,
            "quality": self.quality,
            "ood": self.ood,
            "latency_ms": round(self.latency_ms, 1),
            "not_a_diagnosis": True,
            "disclaimer": self.disclaimer,
        }


class ScreeningModel:
    def __init__(self, name: str, weights: Path, spec: Dict[str, Any], device: str,
                 ood_path: Optional[Path] = None) -> None:
        self.name = name
        self.spec = spec
        self.device = device
        self.labels: List[str] = list(spec["labels"])
        self.concern = set(spec.get("concern_labels", []))
        self.multilabel = bool(spec.get("multilabel", False))
        self.size = int(spec.get("input_size", 224))
        self.mean = np.array(spec.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        self.std = np.array(spec.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
        self.t_moderate = float(spec.get("moderate_threshold", 0.30))
        self.t_high = float(spec.get("high_threshold", 0.60))
        self.min_conf = float(spec.get("min_confidence", 0.0 if self.multilabel else 0.40))
        # Image-quality gate calibrated on the training data (see scripts/train_classifier.py).
        self.q_limits = {
            "min_sharpness": float(spec.get("min_sharpness", 25.0)),
            "min_side": int(spec.get("min_side", 160)),
            "min_brightness": float(spec.get("min_brightness", 40.0)),
        }

        # Calibration (fitted on validation data by the trainer). T=1 means "not calibrated".
        self.temperature = max(0.05, float(spec.get("temperature", 1.0)))
        self.tta = bool(spec.get("tta", True))
        self.ood: Optional[MahalanobisOOD] = None
        if ood_path is not None and Path(ood_path).exists():
            try:
                self.ood = MahalanobisOOD.load(ood_path)
            except Exception:
                log.exception("Could not load OOD guard for %s; continuing without it", name)

        net = build_arch(spec["arch"], len(self.labels), pretrained=False)
        net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.net = net.to(device).eval()
        self._lock = threading.Lock()
        self._last_conv = [m for m in self.net.modules() if isinstance(m, nn.Conv2d)][-1]
        self._head = final_linear(self.net)

    def provenance(self) -> Dict[str, Any]:
        """What this model is and how it was validated: printed on the patient report."""
        return {
            "module": self.name, "title": self.spec.get("title", self.name), "arch": self.spec.get("arch"),
            "training_data": self.spec.get("training_data"), "validation": self.spec.get("validation"),
            "calibrated": self.temperature != 1.0, "temperature": self.temperature,
            "thresholds": {"moderate": self.t_moderate, "high": self.t_high,
                           "source": self.spec.get("threshold_source", "set by hand")},
            "ood_guard": self.ood is not None, "tta": self.tta,
        }

    def _prep(self, img_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(cv2.resize(img_bgr, (self.size, self.size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - self.mean) / self.std
        return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def _gradcam(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        acts, grads = [], []
        h1 = self._last_conv.register_forward_hook(lambda m, i, o: acts.append(o))
        h2 = self._last_conv.register_full_backward_hook(lambda m, gi, go: grads.append(go[0]))
        try:
            with torch.enable_grad():
                xin = x.clone().requires_grad_(True)
                out = self.net(xin)
                self.net.zero_grad()
                out[0, class_idx].backward()
        finally:
            h1.remove()
            h2.remove()
        w = grads[0].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * acts[0]).sum(1, keepdim=True))[0, 0].detach().cpu().numpy()
        cam = cv2.resize(cam, (self.size, self.size))
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    def predict(self, img_bgr: np.ndarray, explain: bool = False) -> ScreeningResult:
        with INFERENCE_GATE:                       # shares the limit with photo inspection
            return self._predict(img_bgr, explain)

    def _predict(self, img_bgr: np.ndarray, explain: bool = False) -> ScreeningResult:
        import time

        t0 = time.perf_counter()
        quality = assess_quality(img_bgr, **self.q_limits)
        x = self._prep(img_bgr)
        with self._lock:
            feats: List[torch.Tensor] = []
            hook = self._head.register_forward_pre_hook(lambda m, inp: feats.append(inp[0].detach().float().cpu()))
            try:
                with torch.inference_mode():
                    logits = self.net(x)[0].float()
                    if self.tta:                                   # average with the mirrored image: steadier output
                        logits = (logits + self.net(torch.flip(x, dims=[3]))[0].float()) / 2
            finally:
                hook.remove()
            logits = logits / self.temperature
            probs_t = torch.sigmoid(logits) if self.multilabel else torch.softmax(logits, dim=0)
            probs = probs_t.cpu().numpy()
            top = int(probs.argmax())
            heat = self._gradcam(x, top) if explain else None

        ood_info = None
        if self.ood is not None and feats:
            sc = float(self.ood.score(feats[0].numpy())[0])
            ood_info = {"score": round(sc, 2), "threshold": round(self.ood.threshold, 2),
                        "out_of_distribution": bool(sc > self.ood.threshold)}

        prob_map = {lbl: float(p) for lbl, p in zip(self.labels, probs)}
        concern_p = (
            float(max([prob_map[l] for l in self.concern if l in prob_map] or [0.0]))
            if self.multilabel
            else float(sum(prob_map[l] for l in self.concern if l in prob_map))
        )

        if not quality["ok"]:
            triage = "INCONCLUSIVE"
            msg = "Image quality is too low for a reliable result. " + " ".join(quality["warnings"])
        elif ood_info and ood_info["out_of_distribution"]:
            triage = "INCONCLUSIVE"
            msg = ("This image looks unlike the images the model was trained on (wrong body area, camera or lighting?). "
                   "Retake it following the capture instructions.")
        elif float(probs.max()) < self.min_conf:
            triage = "INCONCLUSIVE"
            msg = "The model is not confident about this image. Try a clearer photo of the right body area or sample."
        elif concern_p >= self.t_high:
            triage = "HIGH"
            msg = "The model sees a pattern associated with a possible finding. Please arrange a professional evaluation soon."
        elif concern_p >= self.t_moderate:
            triage = "MODERATE"
            msg = "The model sees some features that may deserve attention. A professional check is advisable."
        else:
            triage = "LOW"
            msg = "No concerning pattern found by this model. This does not rule out disease; see a clinician if you have symptoms."

        jpeg = None
        if heat is not None:
            base = cv2.resize(img_bgr, (self.size * 2, self.size * 2))
            hm = cv2.applyColorMap((cv2.resize(heat, (base.shape[1], base.shape[0])) * 255).astype(np.uint8), cv2.COLORMAP_JET)
            jpeg = encode_jpeg(cv2.addWeighted(base, 0.6, hm, 0.4, 0))

        return ScreeningResult(
            module=self.name,
            top_label=self.labels[top],
            probabilities=prob_map,
            concern_probability=concern_p,
            triage=triage,
            message=msg,
            quality=quality,
            latency_ms=(time.perf_counter() - t0) * 1000,
            heatmap_jpeg=jpeg,
            ood=ood_info,
        )


class ScreeningRegistry:
    def __init__(self, directory: str, device: str = "auto") -> None:
        self.dir = Path(directory)
        self.device = resolve_device(device)
        self._models: Dict[str, ScreeningModel] = {}
        self._lock = threading.Lock()

    def installed(self) -> List[str]:
        if not self.dir.exists():
            return []
        return sorted(p.stem for p in self.dir.glob("*.json") if (self.dir / f"{p.stem}.pt").exists())

    def describe(self) -> List[Dict[str, Any]]:
        out = []
        installed = set(self.installed())
        for name in sorted(installed | set(CATALOG)):
            info: Dict[str, Any] = {"name": name, "installed": name in installed}
            if name in CATALOG:
                info.update({k: CATALOG[name].get(k) for k in
                             ("title", "input", "datasets", "body_area", "device", "evidence", "referral", "red_flags", "finding")})
            if name in installed:
                spec = json.loads((self.dir / f"{name}.json").read_text())
                info.update(
                    {
                        "title": spec.get("title", info.get("title", name)),
                        "labels": spec["labels"],
                        "validation": spec.get("validation"),
                        "training_data": spec.get("training_data"),
                    }
                )
            out.append(info)
        return out

    def get(self, name: str) -> Optional[ScreeningModel]:
        if "/" in name or "\\" in name or ".." in name:
            return None
        with self._lock:
            if name in self._models:
                return self._models[name]
            spec_path, w_path = self.dir / f"{name}.json", self.dir / f"{name}.pt"
            if not (spec_path.exists() and w_path.exists()):
                return None
            model = ScreeningModel(name, w_path, json.loads(spec_path.read_text()), self.device,
                                   ood_path=self.dir / f"{name}.ood.npz")
            self._models[name] = model
            return model
