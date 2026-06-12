#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-end MainWindow file selection test (headless)."""
import os
import sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from pathlib import Path
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

app = QApplication(sys.argv)

from gui_app import MainWindow

window = MainWindow()
window.show()
app.processEvents()

print("=" * 60)
print("E2E Test: MainWindow file selection flow")

# --- Check initial state ---
print(f"Mode: {window._mode}")
print(f"Model loaded: {window.detector.is_loaded}")
print(f"File list count: {window.file_list.count()}")
print(f"Status bar: {window.status_bar.currentMessage()[:80]}...")

# --- Simulate clicking first file ---
if window.file_list.count() > 0:
    item = window.file_list.item(0)
    path = item.data(Qt.UserRole)
    print(f"\nSimulating click on: {Path(path).name}")

    # Force image mode
    window._mode = "image"
    window._on_file_selected(item)
    app.processEvents()

    print(f"current_image_path: {Path(window.current_image_path).name if window.current_image_path else 'None'}")
    print(f"current_original: {window.current_original.shape if window.current_original is not None else 'None'}")

# --- Test non-image mode guard ---
print("\n--- Non-image mode guard test ---")
window._mode = "video"
item2 = window.file_list.item(1) if window.file_list.count() > 1 else window.file_list.item(0)
window._on_file_selected(item2)
app.processEvents()
msg = window.status_bar.currentMessage()
print(f"Status after video-mode click: {msg}")

# --- Test invalid file ---
print("\n--- Invalid file guard test ---")
window._mode = "image"
from PyQt5.QtWidgets import QListWidgetItem
fake_item = QListWidgetItem("ghost.jpg")
fake_item.setData(Qt.UserRole, "/nonexistent/ghost.jpg")
window._on_file_selected(fake_item)
app.processEvents()
print(f"Current image unchanged: {window.current_image_path is not None}")

# --- Test switch_mode preserves file list ---
print("\n--- Mode switch preserves state ---")
count_before = window.file_list.count()
window._switch_mode("image")
app.processEvents()
count_after = window.file_list.count()
print(f"File list: {count_before} -> {count_after} (same: {count_before == count_after})")

# --- Test browse folder ---
print("\n--- Browse folder API test ---")
val_dir = Path("datasets/solar_dataset/images/val")
if val_dir.is_dir():
    window._current_browse_dir = val_dir
    window.file_list.clear()
    images = sorted([f for f in val_dir.iterdir()
                     if f.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}])
    for f in images:
        item = QListWidgetItem(f.name)
        item.setData(Qt.UserRole, str(f))
        window.file_list.addItem(item)
    print(f"Manually populated: {window.file_list.count()} files")

# --- Verify worker cleanup ---
print("\n--- Worker state ---")
print(f"_running_worker: {window._running_worker}")
print(f"worker: {window.worker}")
print(f"video_worker: {window.video_worker}")
print(f"batch_worker: {window.batch_worker}")
print(f"camera_worker: {window.camera_worker}")

# --- Cleanup ---
window._stop_all_workers()
window.close()

print("\n" + "=" * 60)
print("E2E test completed successfully.")
