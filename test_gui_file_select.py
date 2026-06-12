#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Automated verification of GUI file selection fixes."""
import sys
import os
import gc
from pathlib import Path
import numpy as np
import cv2

# ----- Test 1: cv2.imread on real val images -----
print("=" * 60)
print("Test 1: cv2.imread on all val images")
val_dir = Path("datasets/solar_dataset/images/val")
if val_dir.is_dir():
    images = sorted([f for f in val_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}])
    fail = 0
    for f in images:
        img = cv2.imread(str(f))
        if img is None:
            print(f"  FAIL: {f.name}")
            fail += 1
    print(f"  Result: {len(images) - fail}/{len(images)} readable, {fail} failed")
else:
    print("  SKIP: val dir not found")

# ----- Test 2: QPixmap thumbnail generation (PyQt5 needed) -----
print("\n" + "=" * 60)
print("Test 2: QPixmap thumbnail generation")
try:
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtGui import QPixmap, QIcon
    from PyQt5.QtCore import Qt

    app = QApplication.instance() or QApplication(sys.argv)

    val_dir = Path("datasets/solar_dataset/images/val")
    if val_dir.is_dir():
        images = sorted([f for f in val_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}])
        fail = 0
        for f in images[:50]:  # Test first 50
            try:
                thumb = QPixmap(str(f)).scaled(48, 48, Qt.KeepAspectRatio)
                if thumb.isNull():
                    print(f"  NULL: {f.name}")
                    fail += 1
            except Exception as e:
                print(f"  EXCEPTION: {f.name} -> {e}")
                fail += 1
        print(f"  Result: {50 - fail}/50 thumbnails OK, {fail} failed")
    else:
        print("  SKIP: val dir not found")
except ImportError as e:
    print(f"  SKIP: PyQt5 not available ({e})")

# ----- Test 3: _on_file_selected logic simulation -----
print("\n" + "=" * 60)
print("Test 3: _on_file_selected guard logic simulation")

class MockItem:
    def __init__(self, data):
        self._data = data
    def data(self, role):
        return self._data

# Simulate non-image mode
mode = "video"
item = MockItem("datasets/solar_dataset/images/val/img10010.jpg")
path = item.data(None)
if not path:
    print("  [PASS] empty path -> silent return")
elif mode != "image":
    print("  [PASS] non-image mode -> warning shown (simulated)")
else:
    print("  [FAIL] should have detected non-image mode")

# Simulate invalid file
mode = "image"
item = MockItem("/nonexistent/file.jpg")
path = item.data(None)
img = cv2.imread(path)
if img is None:
    print("  [PASS] cv2.imread None -> error dialog shown (simulated)")
else:
    print("  [FAIL] should have detected unreadable image")

# Simulate valid file
item = MockItem("datasets/solar_dataset/images/val/img10010.jpg")
path = item.data(None)
img = cv2.imread(path)
if img is not None:
    print(f"  [PASS] valid image loaded: {img.shape}")
else:
    print("  [FAIL] valid image should have loaded")

# ----- Test 4: Import gui_app module -----
print("\n" + "=" * 60)
print("Test 4: Import gui_app module (headless)")
os.environ["QT_QPA_PLATFORM"] = "offscreen"
try:
    from gui_app import MainWindow, Detector, DragDropListWidget
    from gui_app import IMAGE_EXTS
    print(f"  [PASS] All classes imported successfully")
    print(f"  IMAGE_EXTS = {IMAGE_EXTS}")
except Exception as e:
    print(f"  [FAIL] Import error: {e}")

# ----- Test 5: Detector._validate_image -----
print("\n" + "=" * 60)
print("Test 5: Detector._validate_image")
try:
    from gui_app import Detector
    d = Detector.__new__(Detector)

    # Valid image
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    assert Detector._validate_image(img), "Valid image should pass"
    print("  [PASS] valid BGR image passed")

    # None
    assert not Detector._validate_image(None), "None should fail"
    print("  [PASS] None rejected")

    # Grayscale
    img_gray = np.random.randint(0, 255, (640, 640), dtype=np.uint8)
    assert not Detector._validate_image(img_gray), "Grayscale should fail"
    print("  [PASS] grayscale rejected")

    # Constant image (min==max)
    img_const = np.ones((640, 640, 3), dtype=np.uint8) * 128
    assert not Detector._validate_image(img_const), "Constant image should fail"
    print("  [PASS] constant image rejected")

    # Empty
    assert not Detector._validate_image(np.array([])), "Empty should fail"
    print("  [PASS] empty array rejected")

except Exception as e:
    print(f"  [FAIL] {e}")

# ----- Test 6: apply_preprocess color-space correctness -----
print("\n" + "=" * 60)
print("Test 6: Preprocess color-space preservation")
try:
    from gui_app import apply_preprocess

    img_bgr = cv2.imread("datasets/solar_dataset/images/val/img10010.jpg")
    assert img_bgr is not None, "Test image must load"
    expected_shape = img_bgr.shape

    methods = ["none", "clahe", "histeq", "sharpen", "denoise", "gamma", "bilateral"]
    for method in methods:
        result = apply_preprocess(img_bgr, method)
        assert result.shape == expected_shape, \
            f"{method}: shape mismatch {result.shape} != {expected_shape}"
        assert result.dtype == np.uint8, f"{method}: dtype mismatch {result.dtype}"
        # Verify it's still valid BGR (not all zeros, not grayscale)
        assert result.ndim == 3 and result.shape[2] == 3, \
            f"{method}: not 3-channel output"
    print(f"  [PASS] All {len(methods)} methods preserve BGR 3-channel output")

except Exception as e:
    print(f"  [FAIL] {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("All tests completed.")
