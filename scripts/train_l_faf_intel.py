#!/usr/bin/env python
"""
L-FAF-YOLOv11n Training Script for Intel Arc 130T GPU (DirectML backend).

Strategy: model forward/backward runs on DirectML GPU (accelerating heavy conv ops).
The loss preprocessor (torch.unique, scatter_add_ — tiny label tensors) is patched
to run on CPU where full op coverage exists. The CPU round-trip is negligible
because targets are only ~1000 rows × 6 columns.

Usage:
    python train_l_faf_intel.py
"""

import torch
import torch_directml

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.loss import v8DetectionLoss

# ===========================================================================
# Monkey-patch: v8DetectionLoss.preprocess → CPU-safe for DirectML
#
# DirectML does NOT support torch.unique(return_counts=True) or scatter_add_.
# These are only used on the tiny targets tensor (N×6). We intercept preprocess,
# run it on CPU, then move the resulting tensor back to the original device.
# ===========================================================================
_orig_preprocess = v8DetectionLoss.preprocess


def _dml_safe_preprocess(self, targets, batch_size, scale_tensor):
    """Run preprocess on CPU when targets are on a DirectML device."""
    if targets.device.type == "privateuseone":
        orig_device = self.device
        self.device = torch.device("cpu")
        try:
            cpu_targets = targets.to("cpu")
            cpu_scale = scale_tensor.to("cpu")
            result = _orig_preprocess(self, cpu_targets, batch_size, cpu_scale)
        finally:
            self.device = orig_device
        if isinstance(result, torch.Tensor):
            return result.to(targets.device)
        return result
    return _orig_preprocess(self, targets, batch_size, scale_tensor)


v8DetectionLoss.preprocess = _dml_safe_preprocess


# Also patch torch.unique / Tensor.unique as a global safety-net
# (used elsewhere in Ultralytics, e.g. data loading, metrics)
_orig_unique = torch.unique


def _safe_unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None):
    if isinstance(input, torch.Tensor) and input.device.type == "privateuseone":
        cpu_in = input.cpu()
        res = _orig_unique(cpu_in, sorted=sorted, return_inverse=return_inverse, return_counts=return_counts, dim=dim)
        if isinstance(res, tuple):
            return tuple(r.to(input.device) if isinstance(r, torch.Tensor) else r for r in res)
        return res.to(input.device)
    return _orig_unique(input, sorted=sorted, return_inverse=return_inverse, return_counts=return_counts, dim=dim)


torch.unique = _safe_unique
torch.Tensor.unique = lambda self, **kw: _safe_unique(self, **kw)

# Patch TaskAlignedAssigner to run on CPU for DirectML (uses scatter_add_ etc.)
from ultralytics.utils.tal import TaskAlignedAssigner


@torch.no_grad()
def _dml_safe_assigner_forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
    self.bs = pd_scores.shape[0]
    self.n_max_boxes = gt_bboxes.shape[1]
    if pd_scores.device.type == "privateuseone":
        orig_device = pd_scores.device
        cpu_tensors = [t.cpu() for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)]
        result = self._forward(*cpu_tensors)
        return tuple(t.to(orig_device) for t in result)
    return self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)


TaskAlignedAssigner.forward = _dml_safe_assigner_forward

# Patch trainer memory functions to not call torch.cuda on DML device
import gc

from ultralytics.models.yolo.detect.train import DetectionTrainer as _DT

_orig_get_memory = _DT._get_memory
_orig_clear_memory = _DT._clear_memory


def _dml_get_memory(self, fraction=False):
    if self.device.type == "privateuseone":
        return 0 if not fraction else 1.0
    return _orig_get_memory(self, fraction)


def _dml_clear_memory(self, threshold=None):
    if self.device.type == "privateuseone":
        gc.collect()
        return
    return _orig_clear_memory(self, threshold)


_DT._get_memory = _dml_get_memory
_DT._clear_memory = _dml_clear_memory

# Skip validation on DML (eval-mode BN + small spatial dims trigger oneDNN bugs)
_orig_validate = _DT.validate


def _dml_validate(self):
    if self.device.type == "privateuseone":
        LOGGER.info("Skipping validation on DirectML (eval-mode BN not supported)")
        return {}, 0.0
    return _orig_validate(self)


_DT.validate = _dml_validate

LOGGER.info("DirectML monkey-patches installed (preprocess + unique + assigner + memory + validate skip)")


# ===========================================================================
# Verify Intel GPU
# ===========================================================================
dml_device = torch_directml.device()
LOGGER.info(f"Intel GPU: {dml_device}  —  {torch_directml.device_name(0)}")

# Quick functional test
x = torch.randn(2, 3, 64, 64, device=dml_device, requires_grad=True)
w = torch.randn(5, 3, 3, 3, device=dml_device, requires_grad=True)
loss = torch.nn.functional.conv2d(x, w).sum()
loss.backward()
LOGGER.info(f"DirectML forward+backward: OK  (grad={'✓' if x.grad is not None else '✗'})")
del x, w, loss


# ===========================================================================
# Load Model & Launch Training
# ===========================================================================
model = YOLO("ultralytics/cfg/models/11/l_faf_yolov11n.yaml")
model.model.to(dml_device)
LOGGER.info(f"Model on {dml_device}")

LOGGER.info("Starting training — Intel Arc 130T GPU + DirectML ...")
LOGGER.info("Config: epochs=50  batch=16  imgsz=640  amp=False")

results = model.train(
    data="data.yaml",
    device=dml_device,
    epochs=50,
    batch=16,
    imgsz=640,
    workers=0,
    amp=False,
    # Optimization
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    cos_lr=True,
    # Augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
    # Logging
    project="runs/train",
    name="l_faf_intel_arc",
    exist_ok=True,
    pretrained=True,
    verbose=True,
    save=True,
    save_period=10,
    val=False,
)

LOGGER.info("=" * 60)
LOGGER.info("Training complete! Model saved to runs/detect/train/l_faf_intel_arc/")
