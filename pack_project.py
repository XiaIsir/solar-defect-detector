#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Package L-FAF-YOLOv11n project for cloud deployment."""
import zipfile
import os

base = os.path.dirname(os.path.abspath(__file__))
output = os.path.join(base, "project.zip")

# Core Ultralytics files needed (custom modules + their dependencies)
include_files = [
    "cloud_data.yaml",
    "train_cloud.py",
    "ultralytics/cfg/models/11/l_faf_yolov11n.yaml",
    "ultralytics/nn/modules/block.py",
    "ultralytics/nn/modules/__init__.py",
    "ultralytics/nn/tasks.py",
    "ultralytics/nn/modules/conv.py",
    "ultralytics/nn/modules/head.py",
    "ultralytics/engine/model.py",
    "ultralytics/engine/trainer.py",
    "ultralytics/models/yolo/detect/train.py",
    "ultralytics/cfg/__init__.py",
    "ultralytics/utils/__init__.py",
    "ultralytics/utils/ops.py",
    "ultralytics/utils/torch_utils.py",
    "ultralytics/utils/loss.py",
    "ultralytics/utils/metrics.py",
    "ultralytics/utils/plotting.py",
    "ultralytics/data/__init__.py",
    "ultralytics/data/dataset.py",
    "ultralytics/data/augment.py",
    "ultralytics/data/loaders.py",
    "ultralytics/data/build.py",
    "ultralytics/data/utils.py",
    "ultralytics/__init__.py",
]

code_count = 0
ds_count = 0

with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
    # 1. Code files
    for f in include_files:
        fp = os.path.join(base, f)
        if os.path.isfile(fp):
            zf.write(fp, f)
            code_count += 1
            print(f"  [CODE] {f}")
        else:
            print(f"  [MISS] {f}")

    # 2. Dataset
    ds_root = os.path.join(base, "datasets", "solar_dataset")
    for root, dirs, files in os.walk(ds_root):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fn in files:
            if fn.endswith(".pyc"):
                continue
            fp = os.path.join(root, fn)
            arc = os.path.relpath(fp, base)
            zf.write(fp, arc)
            ds_count += 1
            if ds_count % 300 == 0:
                print(f"  [DATA] {ds_count} files packed...")

print(f"\n=== Pack Complete ===")
print(f"Code files:  {code_count}")
print(f"Dataset files: {ds_count}")
size_mb = os.path.getsize(output) / (1024 * 1024)
print(f"Total size: {size_mb:.1f} MB")
print(f"Output: {output}")
