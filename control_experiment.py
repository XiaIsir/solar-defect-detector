#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 5: Control Experiment — Train official YOLO11n vs L-FAF-YOLOv11n
on the same dataset for 5 epochs, compare mAP.
"""
import sys
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import tempfile, os, json

DATA_YAML = "F:/ultralytics-main/data.yaml"
EPOCHS = 3
IMGSZ = 320
BATCH = 16
DEVICE = "cpu"  # Use CPU for fair comparison (no GPU dependency)

RESULTS = {}

def train_model(model_yaml, name, epochs=EPOCHS):
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info(f"Training {name} for {epochs} epochs...")
    LOGGER.info(f"{'='*60}")

    model = YOLO(model_yaml, task="detect")
    results = model.train(
        data=DATA_YAML,
        epochs=epochs,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=0,
        val=True,
        verbose=True,
        exist_ok=True,
        name=name,
        project="runs/control_experiment",
    )

    # Collect metrics
    metrics = {}
    if hasattr(results, 'results_dict'):
        metrics = dict(results.results_dict)
    elif isinstance(results, dict):
        metrics = results

    # Run validation
    val_results = model.val(data=DATA_YAML, imgsz=IMGSZ, device=DEVICE, verbose=False)
    if hasattr(val_results, 'results_dict'):
        metrics["val"] = dict(val_results.results_dict)
    elif isinstance(val_results, dict):
        metrics["val"] = val_results

    params = sum(p.numel() for p in model.model.parameters())
    metrics["params"] = params
    metrics["model_name"] = name

    LOGGER.info(f"\n{name} Results:")
    for k, v in metrics.items():
        LOGGER.info(f"  {k}: {v}")

    return metrics


if __name__ == "__main__":
    print("=" * 80)
    print("CONTROL EXPERIMENT: Official YOLO11n vs L-FAF-YOLOv11n")
    print("=" * 80)
    print(f"Dataset: {DATA_YAML}")
    print(f"Epochs: {EPOCHS}, Imgsz: {IMGSZ}, Batch: {BATCH}, Device: {DEVICE}")
    print()

    # Train official YOLO11n
    try:
        print("\n>>> Training Official YOLO11n...")
        official_metrics = train_model("yolo11n.yaml", "yolo11n_control")
        RESULTS["official"] = official_metrics
    except Exception as e:
        print(f"Official YOLO11n training failed: {e}")
        import traceback
        traceback.print_exc()

    # Train L-FAF-YOLOv11n
    try:
        print("\n>>> Training L-FAF-YOLOv11n...")
        lfaf_metrics = train_model(
            "ultralytics/cfg/models/11/l_faf_yolov11n.yaml",
            "lfaf_control"
        )
        RESULTS["lfaf"] = lfaf_metrics
    except Exception as e:
        print(f"L-FAF training failed: {e}")
        import traceback
        traceback.print_exc()

    # Compare
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    for key, metrics in RESULTS.items():
        print(f"\n{key}:")
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}" if isinstance(v, float) and abs(v) < 100 else f"  {k}: {v}")
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    print(f"    {kk}: {vv:.4f}" if isinstance(vv, float) else f"    {kk}: {vv}")
            else:
                print(f"  {k}: {v}")

    print("\nDone.")
