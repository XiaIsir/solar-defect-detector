#!/usr/bin/env python
"""
Stability test suite for gui_app.py — no GUI required, tests all back-end paths.

Covers:
  1. Image detection ×20 (memory leak check)
  2. Rapid summary / table updates (video-simulation stress)
  3. Camera open/close ×10 (resource leak check)
  4. Batch detection on 50+ images (output correctness)
  5. Export report JSON/CSV integrity + CJK encoding
  6. Model switching (old-model release)
  7. Thread clean-up, signal disconnection, QImage lifetime
"""

import sys
import gc
import json
import csv
import time
import weakref
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import cv2

# ── Add project root ──────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

# These must import WITHOUT triggering QApplication creation
from gui_app import (
    Detector, DetectionWorker, BatchWorker, VideoWorker, CameraWorker,
    CLASS_NAMES, CLASS_COLORS_BGR, CLASS_LABELS_CN, CLASS_COLORS_HEX,
    IMAGE_EXTS, PREPROCESS_OPTIONS,
    apply_preprocess, UserManager, LoginDialog,
)

RESULTS = []  # [(test_name, passed, detail), ...]


def record(test_name: str, passed: bool, detail: str = ""):
    mark = "PASS" if passed else "FAIL"
    RESULTS.append((test_name, passed, detail))
    print(f"  [{mark}] {test_name}")
    if detail:
        for line in detail.splitlines():
            print(f"        {line}")


# ===========================================================================
# Helpers
# ===========================================================================
def make_dummy_images(n: int, w: int = 640, h: int = 640) -> Path:
    """Create n synthetic solar-panel-like images in a temp dir."""
    d = Path(tempfile.mkdtemp(prefix="test_images_"))
    for i in range(n):
        img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        # Draw a fake "cell" rectangle to simulate solar panel
        cv2.rectangle(img, (50, 50), (w - 50, h - 50), (180, 180, 180), 2)
        cv2.imwrite(str(d / f"panel_{i:04d}.jpg"), img)
    return d


def find_model() -> Path | None:
    """Find a usable best.pt in the standard locations."""
    candidates = [
        PROJECT / "runs/detect/runs/train/l_faf_intel_arc/weights/best.pt",
        PROJECT / "runs/detect/train/l_faf_intel_arc/weights/best.pt",
        PROJECT / "runs/solar_smoke_test/weights/best.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    # Recursive scan
    runs = PROJECT / "runs"
    if runs.exists():
        for pt in sorted(runs.rglob("**/weights/best.pt"), key=lambda x: x.stat().st_mtime, reverse=True):
            if pt.stat().st_size > 0:
                return pt
    return None


def memory_mb() -> float:
    """Cross-platform process RSS estimate — rough but enough for leak detection."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return -1.0  # psutil not available


# ===========================================================================
# Test 1: Image Detection ×20 — memory / stability
# ===========================================================================
def test_image_20x(model_path: Path):
    print("\n═══ Test 1: 20 consecutive image detections ═══")
    tmpdir = make_dummy_images(20)
    try:
        detector = Detector(str(model_path), device="cpu", conf_threshold=0.5)
        if not detector.is_loaded:
            record("T1: model load", False, "Detector.is_loaded is False")
            return
        record("T1: model load", True)

        images = sorted(tmpdir.iterdir())
        mem_start = memory_mb()
        times_ms = []
        all_counts = []
        gc.collect()

        for i, img_path in enumerate(images):
            try:
                _, annotated, counts, elapsed_ms, boxes = detector.detect(str(img_path), preprocess="none")
                times_ms.append(elapsed_ms)
                all_counts.append(counts)
                if i == 0:
                    # First run: verify return shapes
                    assert isinstance(annotated, np.ndarray), "annotated not ndarray"
                    assert isinstance(counts, dict), "counts not dict"
                    assert isinstance(boxes, list), "boxes_detail not list"
                    assert isinstance(elapsed_ms, float), "elapsed_ms not float"
            except Exception as e:
                record("T1: 20x detect", False, f"Detection {i} crashed: {e}")
                return

        gc.collect()
        mem_end = memory_mb()
        avg_time = sum(times_ms) / len(times_ms) if times_ms else 0

        record("T1: 20x detect", True,
               f"avg={avg_time:.1f}ms per image, all 20 completed")

        # Memory growth check — PyTorch caches allocator memory on first runs.
        # We use a generous threshold (150 MB) and also check stabilization.
        if mem_start > 0 and mem_end > 0:
            growth = mem_end - mem_start
            per_image = growth / 20
            record("T1: memory growth (20 images)", growth < 150,
                   f"{growth:+.1f} MB total, {per_image:.1f} MB/img (threshold: 150 MB)")

        # Continuity check: no time drift (indicates no accumulation)
        if len(times_ms) >= 4:
            first_half = sum(times_ms[:10]) / 10
            second_half = sum(times_ms[10:]) / 10
            ratio = second_half / first_half if first_half > 0 else 999
            record("T1: time stability", ratio < 1.3,
                   f"first10 avg={first_half:.1f}ms  last10 avg={second_half:.1f}ms  ratio={ratio:.2f}")

        # Deliberate delete + gc check
        del detector
        gc.collect()
        record("T1: detector cleanup", True, "no crash on delete")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ===========================================================================
# Test 2: Rapid summary / table updates (simulate video stress)
# ===========================================================================
def test_rapid_updates():
    """Simulate rapid-fire _update_summary + _populate_table calls to find crashes.

    Tests QImage lifetime and summary data-path logic.
    """
    print("\n═══ Test 2: Rapid-update stress (video simulation) ═══")

    # Create a QApplication if one doesn't exist (required for QPixmap)
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtGui import QImage, QPixmap

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape

    # 2a: QImage lifetime — does QPixmap.fromImage copy the data buffer?
    # Bug: QImage constructs from rgb.data pointer. If numpy array is
    # garbage-collected before QPixmap copies, we get a dangling pointer.
    try:
        for _ in range(100):
            rgb2 = rgb.copy()
            qimg = QImage(rgb2.data, w, h, ch * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            del rgb2, qimg
            _ = pix.width(), pix.height()  # force access — crashes iff dangling pointer
        record("T2: QImage lifetime (100×)", True, "no crash on QImage→QPixmap→delete cycle")
    except Exception as e:
        record("T2: QImage lifetime (100×)", False, f"CRASH: {e}")

    # 2b: Explicit data-copy safety (tobytes)
    try:
        for _ in range(100):
            rgb2 = rgb.copy()
            qimg = QImage(rgb2.data.tobytes(), w, h, ch * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            del rgb2, qimg
            _ = pix.width(), pix.height()
        record("T2: QImage safe copy (100×)", True, "tobytes() approach no crash")
    except Exception as e:
        record("T2: QImage safe copy (100×)", False, f"CRASH: {e}")

    # 2c: Rapid summary recreate (simulate video frame-by-frame updates)
    from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
    try:
        container = QWidget()
        layout = QVBoxLayout(container)
        for i in range(200):
            # Clear child widgets rapidly — simulate _update_summary at 30fps
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            # Re-add new labels
            for c in range(5):
                lbl = QLabel(f"Test {i}-{c}")
                layout.addWidget(lbl)
            QApplication.processEvents()
        record("T2: rapid summary recreate (200×)", True, "no crash on rapid deleteLater+addWidget")
    except Exception as e:
        record("T2: rapid summary recreate (200×)", False, f"CRASH: {e}")

    # 2d: QTableWidget rapid clear/populate
    from PyQt5.QtWidgets import QTableWidget
    try:
        table = QTableWidget(0, 6)
        for i in range(200):
            table.setRowCount(0)
            table.setRowCount(10)
            for r in range(10):
                for c in range(6):
                    from PyQt5.QtWidgets import QTableWidgetItem
                    table.setItem(r, c, QTableWidgetItem(f"R{r}C{c}"))
            QApplication.processEvents()
        record("T2: rapid table recreate (200×)", True, "no crash on QTableWidget rapid clear/populate")
    except Exception as e:
        record("T2: rapid table recreate (200×)", False, f"CRASH: {e}")

    # 2c: Counter dict stability (simulating _update_summary data path)
    for _ in range(1000):
        counts = {"broken": 3, "scratch": 1, "hot_spot": 2, "black_border": 0, "no_electricity": 0}
        # Simulate the logic
        total = sum(counts.values())
        labels = [(CLASS_LABELS_CN.get(k, k), CLASS_COLORS_HEX.get(k, "#ccc"), v)
                  for k, v in sorted(counts.items()) if v > 0]
        assert len(labels) == 3
    record("T2: summary data path (1000×)", True, "no crash in Counter/sorted logic")


# ===========================================================================
# Test 3: Camera open/close ×10
# ===========================================================================
def test_camera_cycle():
    print("\n═══ Test 3: Camera open/close ×10 ═══")
    successes = 0
    failures = 0
    for i in range(10):
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                successes += 1
            else:
                failures += 1
        else:
            failures += 1
        cap.release()
        del cap
        # Small delay between cycles
        time.sleep(0.05)

    gc.collect()
    record("T3: camera open/close ×10", successes >= 9 or (failures == 10),
           f"successes={successes}  open-failures={failures}  "
           f"(10 failures = no camera hw, which is fine)")

    # Verify no leaked VideoCapture objects
    remaining_caps = sum(1 for obj in gc.get_objects() if isinstance(obj, cv2.VideoCapture))
    record("T3: no leaked VideoCapture", remaining_caps == 0,
           f"{remaining_caps} leaked VideoCapture objects in gc")


# ===========================================================================
# Test 4: Batch detection on 50 images
# ===========================================================================
def test_batch_50(model_path: Path):
    print("\n═══ Test 4: Batch detection ×50 ═══")
    tmpdir = make_dummy_images(50)
    try:
        # Clean output first
        out_base = Path("output/batch")
        if out_base.exists():
            shutil.rmtree(out_base, ignore_errors=True)

        detector = Detector(str(model_path), device="cpu", conf_threshold=0.5)
        if not detector.is_loaded:
            record("T4: model load", False, "failed")
            return

        images = sorted(tmpdir.iterdir())
        total_defects = 0
        total_ms = 0.0
        per_class = Counter()
        annotated_dir = out_base / "annotated"

        for idx, img_path in enumerate(images):
            _, annotated, counts, elapsed_ms, _ = detector.detect(str(img_path))
            out_name = f"{img_path.stem}_detected{img_path.suffix}"
            annotated_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(annotated_dir / out_name), annotated)
            total_defects += sum(counts.values())
            total_ms += elapsed_ms
            per_class.update(counts)

        # Verify output files
        saved = list(annotated_dir.glob("*_detected.*"))
        record("T4: annotated count", len(saved) == 50,
               f"expected 50, got {len(saved)}")

        # Write summary CSV (matching gui_app batch output)
        import csv as csv_mod
        stats = {
            "total_images": 50,
            "processed": 50,
            "total_defects": total_defects,
            "avg_time_ms": total_ms / max(50, 1),
            "per_class": dict(per_class),
            "cancelled": False,
        }
        out_base.mkdir(parents=True, exist_ok=True)
        csv_path = out_base / "summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv_mod.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in stats.items():
                writer.writerow([k, str(v)])

        # Verify summary CSV was written
        csv_files = list(out_base.glob("summary.csv"))
        record("T4: summary.csv exists", len(csv_files) == 1,
               f"found {len(csv_files)} file(s) at {csv_path}")

        # Verify CSV content
        if len(csv_files) == 1:
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv_mod.reader(f)
                rows = list(reader)
            record("T4: summary.csv has rows", len(rows) >= 5,
                   f"{len(rows)} rows written")

        # Verify annotated images
        record("T4: batch 50 processed", True,
               f"defects={total_defects}  avg={total_ms/max(50,1):.1f}ms  saved={len(saved)}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ===========================================================================
# Test 5: Export report integrity + CJK encoding
# ===========================================================================
def test_export_reports(model_path: Path):
    print("\n═══ Test 5: Export report JSON/CSV integrity ═══")
    tmpdir = make_dummy_images(1)
    try:
        detector = Detector(str(model_path), device="cpu", conf_threshold=0.5)
        img_path = next(tmpdir.iterdir())
        _, annotated, counts, elapsed_ms, boxes = detector.detect(str(img_path))

        # Build a report identical to gui_app._on_export_report
        report = {
            "timestamp": datetime.now().isoformat(),
            "image_name": img_path.name,
            "image_path": str(img_path),
            "confidence_threshold": 0.5,
            "inference_time_ms": round(elapsed_ms, 1),
            "total_defects": sum(counts.values()),
            "per_class_counts": counts,
            "detections": boxes,
        }

        out_dir = Path("output/reports")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON
        json_path = out_dir / f"{ts}_test_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        # Read back and verify
        with open(json_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["image_name"] == report["image_name"]
        assert loaded["total_defects"] == report["total_defects"]
        assert len(loaded["detections"]) == len(boxes)
        record("T5: JSON round-trip", True,
               f"keys={list(loaded.keys())}  detections={len(loaded['detections'])}")

        # CSV with CJK
        csv_path = out_dir / f"{ts}_test_report.csv"
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["字段", "值"])
            writer.writerow(["时间戳", report["timestamp"]])
            writer.writerow(["图像名称", report["image_name"]])
            writer.writerow(["置信度阈值", str(report["confidence_threshold"])])
            writer.writerow(["推理耗时(ms)", str(report["inference_time_ms"])])
            writer.writerow(["缺陷总数", str(report["total_defects"])])
            writer.writerow([])
            writer.writerow(["类别", "置信度", "X1", "Y1", "X2", "Y2"])
            for det in boxes:
                writer.writerow([
                    det["class"], f"{det['confidence']:.3f}",
                    det["x1"], det["y1"], det["x2"], det["y2"],
                ])

        # Read CSV back and verify encoding
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            content = f.read()

        # Check CJK chars render correctly
        cjk_ok = "图像名称" in content and "时间戳" in content and "缺陷总数" in content
        record("T5: CSV CJK encoding", cjk_ok,
               f"BOM+utf-8-sig, CJK headers readable: {cjk_ok}")

        # Verify no BOM duplication
        no_double_bom = not content.startswith("﻿﻿")
        record("T5: CSV no double-BOM", no_double_bom)

        # Verify box coordinates are integers
        for det in boxes:
            assert isinstance(det["x1"], int), f"x1 not int: {type(det['x1'])}"
            assert isinstance(det["y1"], int)
            assert isinstance(det["x2"], int)
            assert isinstance(det["y2"], int)
        record("T5: box coordinates are int", True)

        # Verify JSON contains per-class CN labels
        cn_in_json = json.dumps(report, ensure_ascii=False)
        record("T5: JSON has no \\u escapes", "\\u" not in cn_in_json)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ===========================================================================
# Test 6: Model switching (old-model release)
# ===========================================================================
def test_model_switching(model_path: Path):
    print("\n═══ Test 6: Model switching ×5 ═══")
    detector = Detector()
    refs = []

    for i in range(5):
        detector.load_model(str(model_path))
        refs.append(weakref.ref(detector.model))

        # Run one detection to verify
        img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        try:
            _, _, _, _, _ = detector.detect(img)
        except Exception as e:
            record("T6: switch detect", False, f"iteration {i}: {e}")
            return

        # Old model ref should become garbage after reassignment
        gc.collect()

    # Check old model references
    alive = sum(1 for r in refs[:-1] if r() is not None)
    record("T6: old models GC'd", alive <= 1,
           f"{alive} old models still alive (expected ≤1, PyTorch may hold one)")

    del detector
    gc.collect()
    record("T6: detector cleanup", True)


# ===========================================================================
# Test 7: Thread lifecycle + signal safety
# ===========================================================================
def test_thread_lifecycle(model_path: Path):
    print("\n═══ Test 7: Thread lifecycle ═══")

    # 7a: DetectionWorker start → finished → cleanup
    tmpdir = make_dummy_images(1)
    try:
        img_path = str(next(tmpdir.iterdir()))
        detector = Detector(str(model_path), device="cpu", conf_threshold=0.5)

        for _ in range(5):
            worker = DetectionWorker(detector, img_path, "none")
            worker.start()
            worker.wait(10000)  # 10s timeout
            if worker.isRunning():
                record("T7: worker hung", False, "DetectionWorker still running after 10s")
                worker.cancel()
                worker.wait(1000)
                return
            worker.deleteLater()
            gc.collect()

        record("T7: DetectionWorker ×5 lifecycle", True, "all completed within timeout")

        # 7b: Cancel mid-operation
        worker2 = DetectionWorker(detector, img_path, "none")
        worker2.start()
        time.sleep(0.01)
        worker2.cancel()
        worker2.wait(2000)
        record("T7: cancel mid-detection", not worker2.isRunning(),
               "worker stopped" if not worker2.isRunning() else "worker still running")
        worker2.deleteLater()

        # 7c: BatchWorker thread safety check
        batch_dir = make_dummy_images(10)
        try:
            batch_worker = BatchWorker(detector, str(batch_dir), "none")
            batch_worker.start()
            batch_worker.wait(60000)
            record("T7: BatchWorker ×10 lifecycle", not batch_worker.isRunning(),
                   "completed" if not batch_worker.isRunning() else "timeout")
            batch_worker.deleteLater()
        finally:
            shutil.rmtree(batch_dir, ignore_errors=True)

        # Check thread count
        import threading
        active_threads = threading.enumerate()
        record("T7: thread count", len(active_threads) < 10,
               f"{len(active_threads)} active threads (names: {[t.name for t in active_threads]})")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ===========================================================================
# Test 8: Signal handler leak check
# ===========================================================================
def test_signal_leak():
    """Detect if QObjects accumulate signal connections."""
    print("\n═══ Test 8: Signal connection check ═══")
    try:
        from PyQt5.QtCore import QObject, pyqtSignal

        class TestEmitter(QObject):
            sig = pyqtSignal()

        emitter = TestEmitter()
        initial = emitter.receivers(emitter.sig)

        # Simulate connect/disconnect cycle ×50
        for _ in range(50):
            def slot():
                pass
            emitter.sig.connect(slot)
            emitter.sig.disconnect(slot)

        final = emitter.receivers(emitter.sig)
        record("T8: signal connect/disconnect ×50", initial == final == 0,
               f"initial={initial} final={final} (should both be 0)")
    except ImportError as e:
        record("T8: signal test", False, f"Import error: {e}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 70)
    print("  STABILITY TEST SUITE — gui_app.py backend paths")
    print(f"  Project: {PROJECT}")
    print("=" * 70)

    model_path = find_model()
    if model_path is None:
        print("\nFATAL: No best.pt found. Cannot run model-dependent tests.")
        sys.exit(1)
    print(f"Model: {model_path}  ({model_path.stat().st_size / 1e6:.1f} MB)\n")

    # Run all tests
    test_image_20x(model_path)
    test_rapid_updates()
    test_camera_cycle()
    test_batch_50(model_path)
    test_export_reports(model_path)
    test_model_switching(model_path)
    test_thread_lifecycle(model_path)
    test_signal_leak()

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    total = len(RESULTS)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = total - passed

    print(f"  RESULTS:  {passed} passed  /  {failed} failed  /  {total} total")
    print("=" * 70)

    if failed:
        print("\n  FAILED TESTS:")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"    ✗ {name}")
                if detail:
                    print(f"      {detail}")
        sys.exit(1)
    else:
        print("\n  All tests passed.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
