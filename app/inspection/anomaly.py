"""PatchCore-style anomaly detection.

Train on photos of GOOD units only (no defect labels needed). At inference, every image
patch is compared with a memory bank of normal patches; patches far from anything seen in
good units are flagged, giving both a score and a heatmap.

One bank per product type is stored in `models/anomaly/<product_type>.pt`.
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ..utils import resolve_device

log = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------------------------------- backbone
class Backbone(nn.Module):
    """ResNet18 truncated after layer3; returns locally pooled, concatenated features."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        net = None
        self.pretrained = False
        if pretrained:
            try:
                net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
                self.pretrained = True
            except Exception as exc:
                log.warning("Could not load ImageNet weights (%s); using random init", exc)
        if net is None:
            net = torchvision.models.resnet18(weights=None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3 = net.layer1, net.layer2, net.layer3
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(self.stem(x))
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        f2 = F.avg_pool2d(f2, 3, 1, 1)
        f3 = F.avg_pool2d(f3, 3, 1, 1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3], dim=1)  # (B, 128+256, H/8, W/8)


# ---------------------------------------------------------------------------- helpers
def greedy_coreset(x: torch.Tensor, m: int, proj_dim: int = 64, seed: int = 0) -> torch.Tensor:
    """Approximate k-center greedy subsampling. Returns indices of m representative rows."""
    n = x.shape[0]
    if m >= n:
        return torch.arange(n)
    g = torch.Generator().manual_seed(seed)
    proj = torch.randn(x.shape[1], proj_dim, generator=g) / proj_dim**0.5
    z = x.float() @ proj
    idx = torch.empty(m, dtype=torch.long)
    idx[0] = int(torch.randint(0, n, (1,), generator=g))
    min_d = torch.full((n,), float("inf"))
    for i in range(1, m):
        d = ((z - z[idx[i - 1]]) ** 2).sum(1)
        min_d = torch.minimum(min_d, d)
        idx[i] = torch.argmax(min_d)
    return idx


def min_distance(feats: torch.Tensor, bank: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    """For every row of feats, the distance to its nearest neighbour in bank."""
    out = []
    for i in range(0, feats.shape[0], chunk):
        out.append(torch.cdist(feats[i : i + chunk], bank).min(dim=1).values)
    return torch.cat(out)


@dataclass
class AnomalyResult:
    score: float
    threshold: float
    ratio: float                      # score / threshold; >= 1.0 means anomalous
    heat: np.ndarray                  # (S, S) float32, anomaly ratio per pixel
    regions: List[Tuple[int, int, int, int]] = field(default_factory=list)  # in S x S coordinates
    size: int = 256


# ----------------------------------------------------------------------------- model
class AnomalyModel:
    def __init__(
        self,
        backbone: Backbone,
        bank: torch.Tensor,
        threshold: float,
        image_size: int,
        product_type: str,
        device: str,
        meta: Optional[dict] = None,
    ) -> None:
        self.device = device
        self.backbone = backbone.to(device).eval()
        if device == "cuda":
            self.backbone = self.backbone.to(memory_format=torch.channels_last)
        self.bank = bank.to(device).float()
        self.threshold = float(threshold)
        self.image_size = int(image_size)
        self.product_type = product_type
        self.meta = meta or {}
        self._lock = threading.Lock()

    # -- preprocessing ---------------------------------------------------------------
    def _prep(self, img_bgr: np.ndarray) -> torch.Tensor:
        s = self.image_size
        rgb = cv2.cvtColor(cv2.resize(img_bgr, (s, s), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)

    @torch.inference_mode()
    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        if self.device == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                f = self.backbone(x.contiguous(memory_format=torch.channels_last))
            return f.float()
        return self.backbone(x)

    # -- inference -------------------------------------------------------------------
    @torch.inference_mode()
    def score(self, img_bgr: np.ndarray) -> AnomalyResult:
        feats = self._features(self._prep(img_bgr))                    # (1, C, h, w)
        _, c, h, w = feats.shape
        flat = feats.squeeze(0).permute(1, 2, 0).reshape(-1, c)        # (h*w, C)
        with self._lock:
            d = min_distance(flat, self.bank).reshape(1, 1, h, w)
        s = self.image_size
        amap = F.interpolate(d, size=(s, s), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        amap = cv2.GaussianBlur(amap, (0, 0), sigmaX=4)
        score = float(amap.max())
        ratio_map = (amap / self.threshold).astype(np.float32)

        regions: List[Tuple[int, int, int, int]] = []
        hot = (ratio_map >= 1.0).astype(np.uint8)
        if hot.any():
            contours, _ = cv2.findContours(hot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
                x, y, bw, bh = cv2.boundingRect(cnt)
                if bw * bh >= 16:
                    regions.append((x, y, x + bw, y + bh))
        return AnomalyResult(score, self.threshold, score / self.threshold, ratio_map, regions, s)

    # -- training --------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        images: Sequence[np.ndarray],
        product_type: str,
        image_size: int = 256,
        device: str = "auto",
        coreset_ratio: float = 0.1,
        max_bank: int = 20000,
        cal_fraction: float = 0.2,
        threshold_margin: float = 1.1,
        pretrained: bool = True,
        seed: int = 0,
    ) -> "AnomalyModel":
        """Build a memory bank from GOOD images and calibrate the decision threshold.

        A fraction of the good images is held out to see how high normal scores can get;
        the threshold is set just above that.
        """
        if len(images) < 3:
            raise ValueError("Need at least 3 good images (aim for 30-200 per product type)")
        if len(images) < 20:
            log.warning("Only %d images: the threshold will be rough. Use 30+ for production.", len(images))

        device = resolve_device(device)
        backbone = Backbone(pretrained=pretrained).to(device)
        proto = cls(backbone, torch.zeros(1, 384), 1.0, image_size, product_type, device)

        rng = np.random.default_rng(seed)
        order = rng.permutation(len(images))
        n_cal = max(1, int(round(cal_fraction * len(images))))
        cal_idx, train_idx = order[:n_cal], order[n_cal:]

        def patches(i: int) -> torch.Tensor:
            f = proto._features(proto._prep(images[i]))
            return f.squeeze(0).permute(1, 2, 0).reshape(-1, f.shape[1]).cpu()

        train_feats = torch.cat([patches(i) for i in train_idx])
        cal_feats = [patches(i) for i in cal_idx]

        def make_bank(feats: torch.Tensor) -> torch.Tensor:
            m = min(max_bank, max(1000, int(feats.shape[0] * coreset_ratio)))
            if feats.shape[0] > 60000:  # pre-sample so greedy coreset stays fast
                sel = torch.randperm(feats.shape[0], generator=torch.Generator().manual_seed(seed))[:60000]
                feats = feats[sel]
            return feats[greedy_coreset(feats, m, seed=seed)].contiguous()

        cal_bank = make_bank(train_feats).to(device)
        cal_scores = []
        for f in cal_feats:
            d = min_distance(f.to(device), cal_bank).reshape(1, 1, int(f.shape[0] ** 0.5), -1)
            s = F.interpolate(d, size=(image_size, image_size), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
            cal_scores.append(float(cv2.GaussianBlur(s, (0, 0), sigmaX=4).max()))
        cal_scores_np = np.array(cal_scores)
        threshold = float(max(cal_scores_np.max() * threshold_margin, cal_scores_np.mean() + 3 * cal_scores_np.std()))

        final_bank = make_bank(torch.cat([train_feats] + cal_feats)).cpu()
        meta = {
            "n_images": len(images),
            "n_calibration": int(n_cal),
            "cal_scores": [round(s, 4) for s in cal_scores],
            "pretrained": backbone.pretrained,
        }
        log.info("Built bank for '%s': %d patches, threshold %.3f", product_type, final_bank.shape[0], threshold)
        model = cls(backbone, final_bank, threshold, image_size, product_type, device, meta)
        return model

    # -- persistence -----------------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "product_type": self.product_type,
            "image_size": self.image_size,
            "threshold": self.threshold,
            "bank": self.bank.cpu().half(),
            "meta": self.meta,
        }
        if not self.meta.get("pretrained", False):
            # Random-init backbones are not reproducible: ship the weights with the bank.
            payload["backbone_state"] = {k: v.cpu() for k, v in self.backbone.state_dict().items()}
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path, device: str = "auto") -> "AnomalyModel":
        device = resolve_device(device)
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        pretrained = bool(payload["meta"].get("pretrained", False))
        backbone = Backbone(pretrained=pretrained and "backbone_state" not in payload)
        if "backbone_state" in payload:
            backbone.load_state_dict(payload["backbone_state"])
        return cls(
            backbone,
            payload["bank"].float(),
            payload["threshold"],
            payload["image_size"],
            payload["product_type"],
            device,
            payload["meta"],
        )


# ---------------------------------------------------------------------------- registry
class AnomalyRegistry:
    """Lazy loader for per-product banks stored as <dir>/<product_type>.pt."""

    def __init__(self, directory: str, device: str = "auto") -> None:
        self.dir = Path(directory)
        self.device = device
        self._models: Dict[str, Optional[AnomalyModel]] = {}
        self._lock = threading.Lock()

    def available(self) -> List[str]:
        return sorted(p.stem for p in self.dir.glob("*.pt")) if self.dir.exists() else []

    def get(self, product_type: str) -> Optional[AnomalyModel]:
        """Bank for the product type, else the 'default' bank, else None."""
        for name in (product_type, "default"):
            if not re.fullmatch(r"[A-Za-z0-9_\-]{1,40}", name or ""):
                continue  # never build file paths from unvalidated names
            with self._lock:
                if name in self._models:
                    model = self._models[name]
                else:
                    path = self.dir / f"{name}.pt"
                    model = None
                    if path.exists():
                        try:
                            model = AnomalyModel.load(path, self.device)
                        except Exception:
                            log.exception("Failed to load anomaly bank %s", path)
                    self._models[name] = model
            if model is not None:
                return model
        return None

    def reload(self) -> None:
        with self._lock:
            self._models.clear()
