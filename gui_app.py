#!/usr/bin/env python
"""
solar-defect-detector — Industrial Visual Inspection Software
PyQt5 + L-FAF-YOLOv11n — matching paper Chapter 5 specifications.

Paper: 基于改进YOLOv11n算法的太阳能电池板缺陷检测技术研究 (王逸凡, 2025)
Chapter 5: 太阳能电池板缺陷检测软件的实现

Features per paper §5.4:
  - Login/registration system (§5.4.1)
  - Model loading + status (§5.4.2)
  - File browsing with thumbnails (§5.4.3)
  - Image / Video / Camera detection modes (§5.4.2)
  - Image preprocessing (§5.4.2)
  - Dual-view results + per-box results table (§5.4.4)
  - Stop detection button (§5.4)
  - Enlarged fonts, per-class colors, zoom/pan, conf slider, batch, export

Usage:
    python gui_app.py
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PyQt5.QtCore import QSize, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ultralytics import YOLO

# ===========================================================================
# Constants
# ===========================================================================
CLASS_NAMES = {
    0: "black_border",
    1: "broken",
    2: "hot_spot",
    3: "no_electricity",
    4: "scratch",
}

CLASS_LABELS_CN = {
    "black_border": "黑边 Black Border",
    "broken": "隐裂 Broken",
    "hot_spot": "热点 Hot Spot",
    "no_electricity": "无电 No Electricity",
    "scratch": "划痕 Scratch",
}

CLASS_COLORS_BGR = {
    "broken": (0, 0, 255),
    "scratch": (0, 255, 255),
    "hot_spot": (0, 255, 0),
    "black_border": (255, 0, 0),
    "no_electricity": (255, 0, 255),
}
_DEFAULT_BGR = (0, 0, 255)

CLASS_COLORS_HEX = {
    "broken": "#E74C3C",
    "scratch": "#F1C40F",
    "hot_spot": "#27AE60",
    "black_border": "#3498DB",
    "no_electricity": "#9B59B6",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
USERS_FILE = Path("config/users.json")


# ===========================================================================
# User Manager — simple file-based credential store
# ===========================================================================
class UserManager:
    @staticmethod
    def _load():
        if USERS_FILE.exists():
            with open(USERS_FILE, encoding="utf-8") as f:
                return json.load(f)
        return {}

    @staticmethod
    def _save(data):
        USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def authenticate(cls, username: str, password: str) -> bool:
        users = cls._load()
        h = hashlib.sha256(password.encode()).hexdigest()
        return username in users and users[username] == h

    @classmethod
    def register(cls, username: str, password: str) -> tuple[bool, str]:
        users = cls._load()
        if not username.strip():
            return False, "用户名不能为空。"
        if username in users:
            return False, "用户名已存在。"
        if len(password) < 3:
            return False, "密码至少 3 位。"
        users[username] = hashlib.sha256(password.encode()).hexdigest()
        cls._save(users)
        return True, "注册成功！"


# ===========================================================================
# Login / Register Dialogs — paper §5.4.1
# ===========================================================================
class LoginDialog(QDialog):
    """Login dialog matching paper Fig 5.1 / 5.2."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("用户登录 — solar-defect-detector")
        self.setMinimumSize(420, 320)
        self.resize(460, 360)
        self.setStyleSheet(self._style())
        self._init_ui()

    def _style(self):
        return """
            QDialog { background-color: #2b2b2b; }
            QLabel { color: #ccc; font-size: 15px; }
            QLineEdit {
                font-size: 15px; padding: 8px 12px;
                background-color: #1e1e1e; color: #fff;
                border: 1px solid #444; border-radius: 4px;
            }
            QLineEdit:focus { border-color: #4A90E2; }
        """

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(40, 30, 40, 20)

        title = QLabel("solar-defect-detector")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #4A90E2;")
        layout.addWidget(title)

        layout.addWidget(QLabel("用户名  |  Username:"))
        self.txt_user = QLineEdit()
        self.txt_user.setPlaceholderText("请输入用户名")
        layout.addWidget(self.txt_user)

        layout.addWidget(QLabel("密码  |  Password:"))
        self.txt_pass = QLineEdit()
        self.txt_pass.setEchoMode(QLineEdit.Password)
        self.txt_pass.setPlaceholderText("请输入密码")
        layout.addWidget(self.txt_pass)

        btn_row = QHBoxLayout()
        btn_login = QPushButton("登录  |  Login")
        btn_login.setStyleSheet(self._btn_style("#4A90E2", "#5BA0F2"))
        btn_login.clicked.connect(self._on_login)
        btn_row.addWidget(btn_login)

        btn_reg = QPushButton("注册  |  Register")
        btn_reg.setStyleSheet(self._btn_style("#27AE60", "#2ECC71"))
        btn_reg.clicked.connect(self._on_register)
        btn_row.addWidget(btn_reg)
        layout.addLayout(btn_row)

        self.lbl_msg = QLabel("")
        self.lbl_msg.setAlignment(Qt.AlignCenter)
        self.lbl_msg.setStyleSheet("font-size: 13px;")
        layout.addWidget(self.lbl_msg)

        # Enter key triggers login
        self.txt_pass.returnPressed.connect(self._on_login)

    @staticmethod
    def _btn_style(bg, hover):
        return (
            f"QPushButton {{ font-size: 16px; font-weight: bold; padding: 10px 24px; "
            f"background-color: {bg}; color: #fff; border-radius: 4px; }}"
            f"QPushButton:hover {{ background-color: {hover}; }}"
        )

    def _on_login(self):
        user = self.txt_user.text().strip()
        pwd = self.txt_pass.text()
        if UserManager.authenticate(user, pwd):
            self.accept()
        else:
            self.lbl_msg.setText("用户名或密码错误  |  Invalid credentials")
            self.lbl_msg.setStyleSheet("color: #E74C3C; font-size: 13px;")

    def _on_register(self):
        user = self.txt_user.text().strip()
        pwd = self.txt_pass.text()
        ok, msg = UserManager.register(user, pwd)
        self.lbl_msg.setText(msg)
        if ok:
            self.lbl_msg.setStyleSheet("color: #27AE60; font-size: 13px;")
        else:
            self.lbl_msg.setStyleSheet("color: #E74C3C; font-size: 13px;")


# ===========================================================================
# Image Preprocessing — paper §5.4.2 "图像预处理"
# ===========================================================================
# ---------------------------------------------------------------------------
# 预处理映射表 — UI 标签与处理函数统一绑定，消除分离常量/if-elif 硬编码
# 每个条目: (UI显示文本, 处理函数)
# 后续添加新预处理只需在此字典追加一行，UI 下拉框和推理流程自动同步
# ---------------------------------------------------------------------------
def _prep_none(img):
    return img


def _prep_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _prep_histeq(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = cv2.equalizeHist(y)
    return cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)


def _prep_sharpen(img):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
    return cv2.filter2D(img, -1, kernel)


def _prep_denoise(img):
    return cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)


def _prep_bilateral(img):
    return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)


def _prep_gamma(img, gamma=1.5):
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


# key → (UI标签, 处理函数) — 单一数据源，UI 与逻辑通过此字典保持同步
PREPROCESS_MAP = {
    "none": ("无  |  None", _prep_none),
    "clahe": ("CLAHE 自适应直方图", _prep_clahe),
    "histeq": ("直方图均衡化", _prep_histeq),
    "sharpen": ("锐化  |  Sharpen", _prep_sharpen),
    "denoise": ("降噪  |  Denoise", _prep_denoise),
    "gamma": ("Gamma 校正", _prep_gamma),
    "bilateral": ("双边滤波  |  Bilateral", _prep_bilateral),
}


def apply_preprocess(img_bgr: np.ndarray, method: str) -> np.ndarray:
    """通过预处理映射表分发，不再使用 if/elif 硬编码。.

    所有对比度增强方法仅操作亮度通道（LAB-L 或 YCrCb-Y）， 避免灰度转换导致的色度信息丢失和训练/推理域差异。
    """
    entry = PREPROCESS_MAP.get(method)
    if entry is None:
        return img_bgr
    return entry[1](img_bgr)


# ===========================================================================
# Detector
# ===========================================================================
class Detector:
    """YOLO wrapper with preprocessing, timing, per-class colors."""

    @staticmethod
    def _detect_device() -> str:
        """Auto-detect the best available torch device."""
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def __init__(self, model_path: str | None = None, device: str | None = None, conf_threshold: float = 0.5):
        self.model = None
        self.model_path = None
        self.device = device or self._detect_device()
        self.conf_threshold = conf_threshold
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        if self.model is not None:
            del self.model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.model_path = str(model_path)
        self.model = YOLO(self.model_path, task="detect")
        # Warm-up: move model to target device so first real inference isn't
        # penalized by CPU→GPU transfer.
        try:
            self.model.model.eval().to(self.device)
        except Exception:
            pass

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def model_name(self) -> str:
        return Path(self.model_path).name if self.model_path else "N/A"

    @staticmethod
    def _validate_image(img: np.ndarray) -> bool:
        """Reject clearly corrupted frames before feeding to model."""
        if img is None or img.size == 0:
            return False
        if img.ndim != 3 or img.shape[2] != 3:
            return False
        if img.min() == img.max():  # single-value (blank) image
            return False
        return True

    def detect(self, image_path_or_array, preprocess: str = "none"):
        """Returns (original, annotated, counts, elapsed_ms, boxes_detail)."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded.")

        if isinstance(image_path_or_array, (str, Path)):
            original = cv2.imread(str(image_path_or_array))
            if original is None:
                raise FileNotFoundError(f"Cannot read image: {image_path_or_array}")
        else:
            original = image_path_or_array
            if not self._validate_image(original):
                raise ValueError("Input frame is invalid (None, empty, or single-value).")

        # Preprocessing
        processed = apply_preprocess(original, preprocess)

        t0 = time.perf_counter()
        results = self.model(processed, device=self.device, verbose=False, conf=self.conf_threshold)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        annotated, counts, boxes_detail = self._draw_boxes(original.copy(), results[0])
        return original, annotated, counts, elapsed_ms, boxes_detail

    def _draw_boxes(self, img, result):
        """Returns (annotated_img, counts_dict, boxes_detail_list)."""
        counts = Counter()
        boxes_detail = []
        if result.boxes is None or len(result.boxes) == 0:
            return img, dict(counts), boxes_detail

        boxes = result.boxes.xyxy.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        for box, cls_id, conf in zip(boxes, clss, confs):
            if conf < self.conf_threshold:
                continue
            class_name = CLASS_NAMES.get(cls_id, str(cls_id))
            color = CLASS_COLORS_BGR.get(class_name, _DEFAULT_BGR)
            counts[class_name] += 1

            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)

            label = f"{class_name} {conf:.2f}"
            font_scale = 1.0
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(img, (x1, y1 - th - baseline - 6), (x1 + tw + 8, y1), color, -1)
            cv2.putText(
                img,
                label,
                (x1 + 4, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
            )

            boxes_detail.append(
                {
                    "class": class_name,
                    "confidence": float(conf),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )
        return img, dict(counts), boxes_detail


# ===========================================================================
# Workers — detection threads with cancel support (paper: §5.4 停止检测)
# ===========================================================================
class DetectionWorker(QThread):
    """Single-image detection in background thread. Supports cancel."""

    detection_done = pyqtSignal(np.ndarray, np.ndarray, dict, float, list)
    error_occurred = pyqtSignal(str)

    def __init__(self, detector: Detector, image_path: str, preprocess: str = "none"):
        super().__init__()
        self.detector = detector
        self.image_path = image_path
        self.preprocess = preprocess
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            if self._cancelled:
                return
            original, annotated, counts, elapsed_ms, boxes_detail = self.detector.detect(
                self.image_path, self.preprocess
            )
            if not self._cancelled:
                self.detection_done.emit(original, annotated, counts, elapsed_ms, boxes_detail)
        except Exception as e:
            if not self._cancelled:
                self.error_occurred.emit(str(e))


class VideoWorker(QThread):
    """Video file detection — paper §5.4.2 视频检测."""

    frame_ready = pyqtSignal(np.ndarray, np.ndarray, dict, float, list, int, int)
    video_done = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, detector: Detector, video_path: str, preprocess: str = "none"):
        super().__init__()
        self.detector = detector
        self.video_path = video_path
        self.preprocess = preprocess
        self._cancelled = False
        self._cap = None  # keep reference for immediate release on cancel
        self._cap_released = False

    def cancel(self):
        self._cancelled = True
        if self._cap is not None and not self._cap_released:
            try:
                self._cap.release()
                self._cap_released = True
            except Exception:
                pass

    def run(self):
        try:
            self._cap = cv2.VideoCapture(self.video_path)
            if not self._cap.isOpened():
                self.error_occurred.emit(f"Cannot open video: {self.video_path}")
                self._cap = None
                return
            total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx = 0
            while not self._cancelled:
                t_start = time.perf_counter()
                ret, frame = self._cap.read()
                if not ret:
                    break
                try:
                    _, annotated, counts, elapsed_ms, boxes_detail = self.detector.detect(frame, self.preprocess)
                    if not self._cancelled:
                        self.frame_ready.emit(
                            frame, annotated, counts, elapsed_ms, boxes_detail, frame_idx, total_frames
                        )
                except Exception as e:
                    if not self._cancelled:
                        self.error_occurred.emit(f"Frame {frame_idx}: {e}")
                frame_idx += 1
                elapsed = time.perf_counter() - t_start
                target_interval = 1.0 / 30
                if elapsed < target_interval:
                    self.msleep(int((target_interval - elapsed) * 1000))
            self.video_done.emit()
        finally:
            if self._cap is not None and not self._cap_released:
                self._cap.release()
                self._cap_released = True
                self._cap = None


class CameraWorker(QThread):
    """Live camera detection — paper §5.4.2 摄像头检测.

    - Heartbeat: 50 consecutive read failures → emit error + stop
    - Cancel: immediate cap.release() via cancel(block=True)
    - Adaptive pacing so detection time doesn't stack frames
    """

    frame_ready = pyqtSignal(np.ndarray, np.ndarray, dict, float, list)
    error_occurred = pyqtSignal(str)
    camera_opened = pyqtSignal(int, int)  # actual_width, actual_height

    _HEARTBEAT_LIMIT = 50  # 50 frames = ~2.5 s of continuous failure
    _TARGET_FPS = 25  # cap render rate to keep UI responsive

    def __init__(self, detector: Detector, camera_id: int = 0, preprocess: str = "none"):
        super().__init__()
        self.detector = detector
        self.camera_id = camera_id
        self.preprocess = preprocess
        self._cancelled = False
        self._cap = None
        self._cap_released = False

    def cancel(self):
        self._cancelled = True
        if self._cap is not None and not self._cap_released:
            try:
                self._cap.release()
                self._cap_released = True
            except Exception:
                pass

    def run(self):
        try:
            # Try multiple backends: DirectShow → MSMF → default
            for api_name, api_id in (("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("ANY", cv2.CAP_ANY)):
                self._cap = cv2.VideoCapture(self.camera_id + api_id)
                if self._cap.isOpened():
                    break
                self._cap.release()
                self._cap = None
            if self._cap is None or not self._cap.isOpened():
                self.error_occurred.emit(f"无法打开摄像头 Camera {self.camera_id} — 请检查连接或隐私权限")
                self._cap = None
                return
            actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.camera_opened.emit(actual_w, actual_h)
            consecutive_failures = 0
            t_frame_start = time.perf_counter()
            first_frame = True
            while not self._cancelled:
                ret, frame = self._cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures >= self._HEARTBEAT_LIMIT:
                        self.error_occurred.emit(
                            f"摄像头 Camera {self.camera_id} 已断开连接（连续 {self._HEARTBEAT_LIMIT} 帧读取失败），请检查设备。"
                        )
                        break
                    self.msleep(20)
                    continue
                consecutive_failures = 0
                if first_frame:
                    first_frame = False
                    if frame.max() < 20:
                        self.error_occurred.emit(
                            f"摄像头 Camera {self.camera_id} 画面全黑 — 请检查镜头盖或隐私快门是否关闭"
                        )
                try:
                    _, annotated, counts, elapsed_ms, boxes_detail = self.detector.detect(frame, self.preprocess)
                    if not self._cancelled:
                        self.frame_ready.emit(frame, annotated, counts, elapsed_ms, boxes_detail)
                except Exception as e:
                    if not self._cancelled:
                        self.error_occurred.emit(f"Camera {self.camera_id}: {e}")
                # Pace to at most _TARGET_FPS
                elapsed = time.perf_counter() - t_frame_start
                target_interval = 1.0 / self._TARGET_FPS
                if elapsed < target_interval:
                    self.msleep(int((target_interval - elapsed) * 1000))
                t_frame_start = time.perf_counter()
        finally:
            if self._cap is not None and not self._cap_released:
                self._cap.release()
                self._cap_released = True
            self._cap = None


class BatchWorker(QThread):
    """Folder batch detection."""

    progress = pyqtSignal(int, int, str)
    image_done = pyqtSignal(str, dict, float)
    batch_finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, detector: Detector, folder_path: str, preprocess: str = "none"):
        super().__init__()
        self.detector = detector
        self.folder_path = folder_path
        self.preprocess = preprocess
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        folder = Path(self.folder_path)
        image_files = sorted([f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTS and f.is_file()])
        if not image_files:
            self.error_occurred.emit("文件夹中没有图像文件。")
            return

        total = len(image_files)
        total_defects = 0
        total_time_ms = 0.0
        per_class_totals = Counter()
        out_dir = Path("output/batch/annotated")
        out_dir.mkdir(parents=True, exist_ok=True)

        for idx, img_path in enumerate(image_files):
            if self._cancelled:
                break
            self.progress.emit(idx + 1, total, img_path.name)
            try:
                _, annotated, counts, elapsed_ms, _ = self.detector.detect(str(img_path), self.preprocess)
                out_name = f"{img_path.stem}_detected{img_path.suffix}"
                cv2.imwrite(str(out_dir / out_name), annotated)
                self.image_done.emit(img_path.name, counts, elapsed_ms)
                total_defects += sum(counts.values())
                total_time_ms += elapsed_ms
                per_class_totals.update(counts)
            except Exception as e:
                self.error_occurred.emit(f"{img_path.name}: {e}")

        processed = idx + 1 if not self._cancelled else idx
        stats = {
            "folder": str(folder),
            "total_images": total,
            "processed": processed,
            "total_defects": total_defects,
            "avg_time_ms": total_time_ms / max(processed, 1),
            "total_time_ms": total_time_ms,
            "per_class": dict(per_class_totals),
            "cancelled": self._cancelled,
        }
        self._write_batch_summary(stats)
        self.batch_finished.emit(stats)

    def _write_batch_summary(self, stats: dict):
        out_dir = Path("output/batch")
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "summary.csv"
        tmp_path = out_dir / ".summary.csv.tmp"
        try:
            with open(tmp_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "value"])
                for k, v in stats.items():
                    writer.writerow([k, str(v)])
            tmp_path.replace(csv_path)
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            # Don't re-raise — let batch_finished still fire so the UI resets.
            stats["summary_error"] = str(e)


# ===========================================================================
# ZoomableImageLabel (unchanged)
# ===========================================================================
class ZoomableImageLabel(QLabel):
    """Image viewer with mouse-wheel zoom, drag-to-pan, double-click reset."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._base_pixmap = None
        self._zoom_factor = 1.0
        self._fit_scale = 1.0
        self._user_zoomed = False
        self._dragging = False
        self._drag_global = None
        self._in_apply_zoom = False
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(1, 1)
        self.setMouseTracking(True)

    def setBasePixmap(self, pixmap):
        self._base_pixmap = pixmap
        self._user_zoomed = False
        self._fit_to_viewport()

    def resetZoom(self):
        self._user_zoomed = False
        self._fit_to_viewport()

    def currentZoomPercent(self) -> int:
        return round(self._zoom_factor * 100)

    def _scroll_area(self):
        p = self.parent()
        while p:
            if isinstance(p, QScrollArea):
                return p
            p = p.parent()
        return None

    def _fit_to_viewport(self):
        if self._base_pixmap is None or self._in_apply_zoom:
            return
        sa = self._scroll_area()
        if sa:
            sa.setWidgetResizable(True)
            vp = sa.viewport().size()
            if vp.width() < 2 or vp.height() < 2:
                return
            scaled = self._base_pixmap.scaled(vp, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self._fit_scale = scaled.width() / self._base_pixmap.width()
        else:
            self._fit_scale = 1.0
        self._zoom_factor = 1.0
        self._apply_zoom()

    def _apply_zoom(self):
        if self._base_pixmap is None:
            return
        self._in_apply_zoom = True
        try:
            scale = self._fit_scale * self._zoom_factor
            w = max(1, int(self._base_pixmap.width() * scale))
            h = max(1, int(self._base_pixmap.height() * scale))
            if self.width() == w and self.height() == h:
                return
            scaled = self._base_pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            super().setPixmap(scaled)
            if self._user_zoomed:
                sa = self._scroll_area()
                if sa:
                    sa.setWidgetResizable(False)
            self.resize(w, h)
        finally:
            self._in_apply_zoom = False

    def wheelEvent(self, event):
        if self._base_pixmap is None:
            return
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 0.87
        new_zoom = self._zoom_factor * factor
        if 0.05 <= new_zoom <= 20.0:
            self._zoom_factor = new_zoom
            self._user_zoomed = True
            self._apply_zoom()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._drag_global = event.globalPos()
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging and self._drag_global is not None:
            delta = event.globalPos() - self._drag_global
            self._drag_global = event.globalPos()
            sa = self._scroll_area()
            if sa:
                sa.horizontalScrollBar().setValue(sa.horizontalScrollBar().value() - delta.x())
                sa.verticalScrollBar().setValue(sa.verticalScrollBar().value() - delta.y())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = False
            self._drag_global = None
            self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.resetZoom()

    def resizeEvent(self, event):
        if not self._user_zoomed:
            self._fit_to_viewport()
        super().resizeEvent(event)


# ===========================================================================
# DragDropListWidget — accepts filesystem folder drops
# ===========================================================================
class DragDropListWidget(QListWidget):
    """QListWidget that accepts drag-and-drop of folders from the filesystem."""

    folder_dropped = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DropOnly)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if not path:
            return
        # Accept either a folder or a file (use its parent dir)
        p = Path(path)
        if p.is_file():
            p = p.parent
        if p.is_dir():
            self.folder_dropped.emit(str(p))
            event.acceptProposedAction()


# ===========================================================================
# MainWindow — full industrial detection system
# ===========================================================================
class MainWindow(QMainWindow):
    """Main GUI matching paper Fig 5.5 layout."""

    # 部署后 PyInstaller 把数据文件放在 _internal/ 下，通过 sys._MEIPASS 定位
    @staticmethod
    def _bundle_dir() -> Path:
        if getattr(sys, "frozen", False):
            return Path(sys._MEIPASS)
        return Path(__file__).resolve().parent

    @classmethod
    def _resolve_candidate(cls, rel: str) -> Path:
        """优先在 bundle 内查找，其次搜索运行目录下类似路径。."""
        bundled = cls._bundle_dir() / rel
        if bundled.exists():
            return bundled
        cwd = Path.cwd() / rel
        if cwd.exists():
            return cwd
        return bundled  # 返回 bundle 版本，后续检查时会报明确错误

    DEFAULT_MODEL_CANDIDATES = [
        "weights/best.pt",
        "weights/last.pt",
    ]

    def __init__(self):
        super().__init__()
        self.detector = Detector()
        self.current_image_path = None
        self.current_original = None
        self.video_path = None

        # Workers
        self.worker = None
        self.video_worker = None
        self.camera_worker = None
        self.batch_worker = None
        self._running_worker = None  # which worker currently owns UI running state

        # Current detection mode: "image", "video", "camera"
        self._mode = "image"

        # Last result
        self._last_annotated = None
        self._last_counts = {}
        self._last_elapsed_ms = 0.0
        self._last_boxes_detail = []

        # File browser data
        self._current_browse_dir = None

        # Table throttle — limit QTableWidget rebuild to ~4 Hz in video/camera mode
        self._last_table_update = 0.0

        self._init_ui()
        self._update_device_status()
        self._try_auto_load_model()
        self._try_auto_browse_val()

    # =========================================================================
    # UI Construction
    # =========================================================================
    def _init_ui(self):
        self.setWindowTitle("solar-defect-detector")
        self.setMinimumSize(900, 600)
        self.resize(1400, 880)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setSpacing(0)
        root.setContentsMargins(4, 4, 4, 4)

        # =====================================================================
        # Left sidebar — fixed width, outside of any splitter
        # =====================================================================
        sidebar = self._build_sidebar()
        sidebar.setFixedWidth(260)

        # =====================================================================
        # Main content
        # =====================================================================
        main_widget = QWidget()
        main_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_area = QVBoxLayout(main_widget)
        main_area.setSpacing(6)
        main_area.setContentsMargins(0, 0, 0, 0)

        # Top status bar
        main_area.addLayout(self._build_status_bar())

        # =====================================================================
        # Image area QSplitter: Original (L) | Result + Table (R)
        # =====================================================================
        self.image_splitter = QSplitter(Qt.Horizontal)
        self.image_splitter.setHandleWidth(3)
        self.image_splitter.setStyleSheet(
            "QSplitter::handle { background-color: #444; }QSplitter::handle:hover { background-color: #4A90E2; }"
        )

        # Left — Original image group
        left_grp = QGroupBox("原始图像  |  Original Image")
        left_grp.setFont(self._group_font())
        left_grp.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lv = QVBoxLayout(left_grp)
        lv.setContentsMargins(4, 20, 4, 4)
        self.scroll_original = QScrollArea()
        self.scroll_original.setWidgetResizable(True)
        self.scroll_original.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.scroll_original.setStyleSheet("QScrollArea { background-color: #1a1a1a; border: 2px dashed #444; }")
        self.lbl_original = ZoomableImageLabel()
        self.lbl_original.setText("未加载图像\nNo image loaded")
        self.lbl_original.setStyleSheet("color: #777; font-size: 18px;")
        self.lbl_original.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.scroll_original.setWidget(self.lbl_original)
        lv.addWidget(self.scroll_original)
        self.lbl_zoom_orig = QLabel("100%")
        self.lbl_zoom_orig.setStyleSheet("color: #888; font-size: 12px;")
        self.lbl_zoom_orig.setAlignment(Qt.AlignCenter)
        lv.addWidget(self.lbl_zoom_orig)
        self.image_splitter.addWidget(left_grp)

        # Right — vertical splitter for result image + table
        self.right_splitter = QSplitter(Qt.Vertical)
        self.right_splitter.setHandleWidth(3)
        self.right_splitter.setStyleSheet(
            "QSplitter::handle { background-color: #444; }QSplitter::handle:hover { background-color: #4A90E2; }"
        )

        # Result image group
        right_grp = QGroupBox("检测结果  |  Detection Result")
        right_grp.setFont(self._group_font())
        right_grp.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        rv = QVBoxLayout(right_grp)
        rv.setContentsMargins(4, 20, 4, 4)
        self.scroll_result = QScrollArea()
        self.scroll_result.setWidgetResizable(True)
        self.scroll_result.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.scroll_result.setStyleSheet("QScrollArea { background-color: #1a1a1a; border: 2px dashed #444; }")
        self.lbl_result = ZoomableImageLabel()
        self.lbl_result.setText("检测结果将显示在此处\nResult will appear here")
        self.lbl_result.setStyleSheet("color: #777; font-size: 18px;")
        self.lbl_result.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.scroll_result.setWidget(self.lbl_result)
        rv.addWidget(self.scroll_result)
        self.lbl_zoom_res = QLabel("100%")
        self.lbl_zoom_res.setStyleSheet("color: #888; font-size: 12px;")
        self.lbl_zoom_res.setAlignment(Qt.AlignCenter)
        rv.addWidget(self.lbl_zoom_res)
        self.right_splitter.addWidget(right_grp)

        # Detection Results Table (paper §5.4.4, Fig 5.9-5.11)
        table_grp = QGroupBox("检测结果表  |  Detection Results Table")
        table_grp.setFont(self._group_font())
        table_grp.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tv = QVBoxLayout(table_grp)
        tv.setContentsMargins(4, 20, 4, 4)
        self.results_table = QTableWidget(0, 6)
        self.results_table.setHorizontalHeaderLabels(["类别 Class", "置信度 Conf", "X1", "Y1", "X2", "Y2"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.results_table.setStyleSheet("""
            QTableWidget {
                background-color: #1a1a1a; color: #ccc; font-size: 13px;
                border: 1px solid #444; gridline-color: #333;
            }
            QTableWidget::item { padding: 4px; }
            QHeaderView::section {
                background-color: #2b2b2b; color: #4A90E2;
                font-weight: bold; font-size: 13px; padding: 4px;
                border: 1px solid #444;
            }
        """)
        tv.addWidget(self.results_table)
        self.right_splitter.addWidget(table_grp)

        self.image_splitter.addWidget(self.right_splitter)
        main_area.addWidget(self.image_splitter, stretch=1)

        # Confidence slider
        main_area.addWidget(self._build_conf_slider())

        # Perf + Summary row (compact)
        bottom_row = QHBoxLayout()

        self.summary_group = QGroupBox("检测统计  |  Summary")
        self.summary_group.setFont(self._group_font())
        self.summary_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.summary_layout = QVBoxLayout()
        self.summary_layout.setSpacing(2)
        self.summary_group.setLayout(self.summary_layout)
        bottom_row.addWidget(self.summary_group, stretch=1)

        self.perf_frame = QFrame()
        self.perf_frame.setStyleSheet("QFrame { background-color: #1e1e1e; border-radius: 4px; padding: 6px; }")
        perf_layout = QVBoxLayout(self.perf_frame)
        self.lbl_inference_time = QLabel("Inference: — ms")
        self.lbl_inference_time.setStyleSheet("color: #888; font-size: 14px; font-weight: bold;")
        perf_layout.addWidget(self.lbl_inference_time)
        self.lbl_fps = QLabel("FPS: —")
        self.lbl_fps.setStyleSheet("color: #888; font-size: 14px; font-weight: bold;")
        perf_layout.addWidget(self.lbl_fps)
        self.lbl_video_progress = QLabel("")
        self.lbl_video_progress.setStyleSheet("color: #888; font-size: 12px;")
        perf_layout.addWidget(self.lbl_video_progress)
        bottom_row.addWidget(self.perf_frame)

        main_area.addLayout(bottom_row)

        # Progress bar
        self.batch_progress = QProgressBar()
        self.batch_progress.setVisible(False)
        self.batch_progress.setMinimumHeight(22)
        self.batch_progress.setStyleSheet("""
            QProgressBar { border: 1px solid #444; border-radius: 4px;
                background-color: #1e1e1e; text-align: center;
                color: #ccc; font-size: 13px; font-weight: bold; }
            QProgressBar::chunk { background-color: #8E44AD; border-radius: 3px; }
        """)
        main_area.addWidget(self.batch_progress)
        self.lbl_batch_status = QLabel("")
        self.lbl_batch_status.setStyleSheet("color: #888; font-size: 12px;")
        self.lbl_batch_status.setAlignment(Qt.AlignCenter)
        self.lbl_batch_status.setVisible(False)
        main_area.addWidget(self.lbl_batch_status)

        # Buttons
        main_area.addLayout(self._build_button_row())

        self.image_splitter.setSizes([600, 800])
        self.right_splitter.setSizes([450, 200])

        root.addWidget(sidebar)
        root.addWidget(main_widget)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪 — 请加载模型并打开图像  |  Ready")

        # Dark theme
        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; }
            QGroupBox {
                font-size: 18px; font-weight: bold; color: #ccc;
                border: 1px solid #444; border-radius: 5px;
                margin-top: 12px; padding-top: 22px;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 12px; padding: 0 6px;
            }
        """)

        # Initialize summary panel with placeholder
        self._reset_summary()

    def _build_sidebar(self) -> QWidget:
        """Left sidebar with mode buttons, file browser, preprocess options."""
        sidebar = QFrame()
        sidebar.setStyleSheet("QFrame { background-color: #1e1e1e; border-radius: 6px; padding: 4px; }")
        layout = QVBoxLayout(sidebar)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 12, 8, 12)

        # Mode selection
        mode_grp = QGroupBox("检测模式  |  Mode")
        mode_grp.setFont(self._group_font())
        mode_layout = QVBoxLayout(mode_grp)
        mode_layout.setSpacing(4)

        self.btn_mode_image = QPushButton("图片检测  |  Image")
        self.btn_mode_image.setCheckable(True)
        self.btn_mode_image.setChecked(True)
        self.btn_mode_image.setMinimumHeight(36)
        self.btn_mode_image.setStyleSheet(self._sidebar_btn_style(True))
        self.btn_mode_image.clicked.connect(lambda: self._switch_mode("image"))
        mode_layout.addWidget(self.btn_mode_image)

        self.btn_mode_video = QPushButton("视频检测  |  Video")
        self.btn_mode_video.setCheckable(True)
        self.btn_mode_video.setMinimumHeight(36)
        self.btn_mode_video.setStyleSheet(self._sidebar_btn_style(False))
        self.btn_mode_video.clicked.connect(lambda: self._switch_mode("video"))
        mode_layout.addWidget(self.btn_mode_video)

        self.btn_mode_camera = QPushButton("摄像头检测  |  Camera")
        self.btn_mode_camera.setCheckable(True)
        self.btn_mode_camera.setMinimumHeight(36)
        self.btn_mode_camera.setStyleSheet(self._sidebar_btn_style(False))
        self.btn_mode_camera.clicked.connect(lambda: self._switch_mode("camera"))
        mode_layout.addWidget(self.btn_mode_camera)

        layout.addWidget(mode_grp)

        # Camera selector (paper §5.4.2)
        cam_grp = QGroupBox("摄像头选择  |  Camera ID")
        cam_grp.setFont(self._group_font())
        cam_layout = QVBoxLayout(cam_grp)
        self.combo_camera = QComboBox()
        self.combo_camera.setStyleSheet(
            "QComboBox { font-size: 13px; padding: 4px; background: #2b2b2b; "
            "color: #ccc; border: 1px solid #444; border-radius: 3px; }"
            "QComboBox::drop-down { border: none; }"
            "QComboBox QAbstractItemView { background: #2b2b2b; color: #ccc; }"
        )
        self.combo_camera.currentIndexChanged.connect(self._on_camera_changed)
        # 阻断信号，避免 addItem 触发 _on_camera_changed 时
        # self.status_bar 尚未创建导致 AttributeError 闪退
        self.combo_camera.blockSignals(True)
        self.combo_camera.addItem("点击检测后自动扫描  |  Auto-scan on detect", -1)
        self.combo_camera.blockSignals(False)
        cam_layout.addWidget(self.combo_camera)
        layout.addWidget(cam_grp)

        # 预处理选项 — 从 PREPROCESS_MAP 动态构建，确保 UI 标签与处理函数同步
        prep_grp = QGroupBox("图像预处理  |  Preprocess")
        prep_grp.setFont(self._group_font())
        prep_layout = QVBoxLayout(prep_grp)
        self.combo_preprocess = QComboBox()
        for key, (label, _func) in PREPROCESS_MAP.items():
            self.combo_preprocess.addItem(label, key)
        self.combo_preprocess.setStyleSheet(
            "QComboBox { font-size: 13px; padding: 4px; background: #2b2b2b; "
            "color: #ccc; border: 1px solid #444; border-radius: 3px; }"
            "QComboBox::drop-down { border: none; }"
            "QComboBox QAbstractItemView { background: #2b2b2b; color: #ccc; }"
        )
        prep_layout.addWidget(self.combo_preprocess)
        layout.addWidget(prep_grp)

        # File browser (paper: directory tree + thumbnails §5.4.3)
        browse_grp = QGroupBox("文件浏览  |  Browse")
        browse_grp.setFont(self._group_font())
        browse_layout = QVBoxLayout(browse_grp)
        browse_layout.setSpacing(4)

        btn_browse = QPushButton("选择文件夹  |  Open Folder")
        btn_browse.setMinimumHeight(32)
        btn_browse.setStyleSheet(self._sidebar_btn_style(True))
        btn_browse.clicked.connect(self._on_browse_folder)
        browse_layout.addWidget(btn_browse)

        self.file_list = DragDropListWidget()
        self.file_list.setStyleSheet(
            "QListWidget { background-color: #1a1a1a; color: #ccc; font-size: 12px; "
            "border: 1px solid #333; }"
            "QListWidget::item:selected { background-color: #4A90E2; }"
            "QListWidget::item:hover { background-color: #333; }"
        )
        self.file_list.itemClicked.connect(self._on_file_selected)
        self.file_list.folder_dropped.connect(self._on_folder_dropped)
        self.file_list.setIconSize(QSize(48, 48))
        browse_layout.addWidget(self.file_list)
        layout.addWidget(browse_grp, stretch=1)

        return sidebar

    @staticmethod
    def _sidebar_btn_style(checked: bool) -> str:
        if checked:
            return (
                "QPushButton { font-size: 14px; font-weight: bold; padding: 6px 12px; "
                "background-color: #4A90E2; color: #fff; border-radius: 3px; }"
                "QPushButton:hover { background-color: #5BA0F2; }"
            )
        return (
            "QPushButton { font-size: 14px; font-weight: bold; padding: 6px 12px; "
            "background-color: #2b2b2b; color: #888; border: 1px solid #444; border-radius: 3px; }"
            "QPushButton:hover { background-color: #333; color: #ccc; }"
        )

    def _build_status_bar(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(24)
        self.lbl_model_status = QLabel("● 模型: 未加载  |  Model: Not Loaded")
        self.lbl_model_status.setStyleSheet(
            "color: #E74C3C; font-size: 15px; font-weight: bold;"
            "padding: 6px 14px; background-color: #1e1e1e; border-radius: 4px;"
        )
        row.addWidget(self.lbl_model_status)
        self.lbl_device = QLabel("● 设备: CPU")
        self.lbl_device.setStyleSheet(
            "color: #F39C12; font-size: 15px; font-weight: bold;"
            "padding: 6px 14px; background-color: #1e1e1e; border-radius: 4px;"
        )
        row.addWidget(self.lbl_device)
        row.addStretch()
        return row

    def _build_conf_slider(self) -> QWidget:
        container = QFrame()
        container.setStyleSheet("QFrame { background-color: #1e1e1e; border-radius: 4px; padding: 6px; }")
        layout = QHBoxLayout(container)
        layout.setContentsMargins(12, 6, 12, 6)
        lbl = QLabel("置信度阈值  |  Confidence:")
        lbl.setStyleSheet("color: #ccc; font-size: 16px; font-weight: bold;")
        layout.addWidget(lbl)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 10)
        self.conf_slider.setValue(5)
        self.conf_slider.setMinimumWidth(200)
        self.conf_slider.setStyleSheet("""
            QSlider::groove:horizontal { height: 8px; background: #333; border-radius: 4px; }
            QSlider::handle:horizontal { width: 20px; height: 20px; margin: -6px 0;
                background: #4A90E2; border-radius: 10px; }
            QSlider::sub-page:horizontal { background: #4A90E2; border-radius: 4px; }
        """)
        self.conf_slider.valueChanged.connect(self._on_conf_changed)
        layout.addWidget(self.conf_slider)
        self.lbl_conf_value = QLabel("0.50")
        self.lbl_conf_value.setStyleSheet("color: #4A90E2; font-size: 18px; font-weight: bold; min-width: 50px;")
        self.lbl_conf_value.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_conf_value)
        layout.addStretch()
        return container

    def _build_button_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)

        self.btn_open = QPushButton("选择文件")
        self.btn_open.setToolTip("Select File")
        self.btn_open.setMinimumHeight(44)
        self.btn_open.setFont(self._btn_font())
        self.btn_open.setStyleSheet(self._btn_style("#4A90E2", "#5BA0F2"))
        self.btn_open.clicked.connect(self._on_open_image)
        self.btn_open.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row.addWidget(self.btn_open)

        self.btn_load = QPushButton("加载模型")
        self.btn_load.setToolTip("Load Model")
        self.btn_load.setMinimumHeight(44)
        self.btn_load.setFont(self._btn_font())
        self.btn_load.setStyleSheet(self._btn_style("#4A90E2", "#5BA0F2"))
        self.btn_load.clicked.connect(self._on_load_model)
        self.btn_load.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row.addWidget(self.btn_load)

        self.btn_detect = QPushButton("开始检测")
        self.btn_detect.setToolTip("Start Detection")
        self.btn_detect.setMinimumHeight(44)
        self.btn_detect.setFont(self._btn_font())
        self.btn_detect.setEnabled(False)
        self.btn_detect.setStyleSheet(self._btn_style("#27AE60", "#2ECC71"))
        self.btn_detect.clicked.connect(self._on_detect)
        self.btn_detect.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row.addWidget(self.btn_detect)

        # Stop button (paper: §5.4 停止检测)
        self.btn_stop = QPushButton("停止检测")
        self.btn_stop.setToolTip("Stop Detection")
        self.btn_stop.setMinimumHeight(44)
        self.btn_stop.setFont(self._btn_font())
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet(self._btn_style("#E74C3C", "#F06050"))
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_stop.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row.addWidget(self.btn_stop)

        self.btn_save = QPushButton("保存结果")
        self.btn_save.setToolTip("Save Result")
        self.btn_save.setMinimumHeight(44)
        self.btn_save.setFont(self._btn_font())
        self.btn_save.setEnabled(False)
        self.btn_save.setStyleSheet(self._btn_style("#F39C12", "#F5B041"))
        self.btn_save.clicked.connect(self._on_save_result)
        self.btn_save.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row.addWidget(self.btn_save)

        self.btn_export = QPushButton("导出报告")
        self.btn_export.setToolTip("Export Report")
        self.btn_export.setMinimumHeight(44)
        self.btn_export.setFont(self._btn_font())
        self.btn_export.setEnabled(False)
        self.btn_export.setStyleSheet(self._btn_style("#F39C12", "#F5B041"))
        self.btn_export.clicked.connect(self._on_export_report)
        self.btn_export.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row.addWidget(self.btn_export)

        self.btn_batch = QPushButton("批量检测")
        self.btn_batch.setToolTip("Batch Detection")
        self.btn_batch.setMinimumHeight(44)
        self.btn_batch.setFont(self._btn_font())
        self.btn_batch.setStyleSheet(self._btn_style("#8E44AD", "#A569BD"))
        self.btn_batch.clicked.connect(self._on_batch_detect)
        self.btn_batch.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row.addWidget(self.btn_batch)

        row.addStretch()

        self.detect_progress = QProgressBar()
        self.detect_progress.setMinimum(0)
        self.detect_progress.setMaximum(0)  # indeterminate by default
        self.detect_progress.setMaximumHeight(16)
        self.detect_progress.setMaximumWidth(180)
        self.detect_progress.setVisible(False)
        self.detect_progress.setStyleSheet(
            "QProgressBar { background: #2b2b2b; border: 1px solid #444; border-radius: 3px; }"
            "QProgressBar::chunk { background: #27AE60; }"
        )
        row.addWidget(self.detect_progress)

        return row

    # =========================================================================
    # Style helpers
    # =========================================================================
    @staticmethod
    def _group_font():
        f = QFont()
        f.setPointSize(11)
        f.setBold(True)
        return f

    @staticmethod
    def _btn_font():
        f = QFont()
        f.setPointSize(9)
        f.setBold(True)
        return f

    @staticmethod
    def _btn_style(bg, hover):
        return (
            f"QPushButton {{ font-weight: bold; padding: 4px 8px; "
            f"background-color: {bg}; color: #fff; border: 1px solid {hover}; "
            f"border-radius: 6px; }}"
            f"QPushButton:hover {{ background-color: {hover}; }}"
            "QPushButton:disabled { background-color: #333; color: #555; border-color: #3a3a3a; }"
        )

    # =========================================================================
    # Mode switching (paper: 图片检测/视频检测/摄像头检测)
    # =========================================================================
    def _scan_cameras(self):
        """懒加载扫描可用摄像头索引 0..7，填充下拉框。 仅在用户进入摄像头模式时调用，不在启动时触发。 依次尝试 DirectShow → MSMF → 默认后端，单个设备失败不影响整体扫描。.
        """
        current = self.combo_camera.currentData() if self.combo_camera.count() > 0 else None
        self.combo_camera.blockSignals(True)
        self.combo_camera.clear()

        backends = [
            ("DSHOW", cv2.CAP_DSHOW),
            ("MSMF", cv2.CAP_MSMF),
            ("ANY", cv2.CAP_ANY),
        ]
        seen = set()

        for idx in range(8):
            for name, api in backends:
                cap = None
                try:
                    cap = cv2.VideoCapture(idx + api)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            if frame.max() < 20:  # 黑帧过滤
                                continue
                            label = f"Camera {idx} ({name})" if name != "ANY" else f"Camera {idx}"
                            if idx not in seen:
                                self.combo_camera.addItem(label, idx)
                                seen.add(idx)
                            break
                except Exception:
                    pass  # 单个摄像头异常不影响后续扫描
                finally:
                    if cap is not None:
                        cap.release()

            if idx - len(seen) >= 3:  # 连续 3 个索引无设备则停止
                break

        self.combo_camera.blockSignals(False)
        # 恢复之前选中的设备
        for i in range(self.combo_camera.count()):
            if self.combo_camera.itemData(i) == current:
                self.combo_camera.setCurrentIndex(i)
                break

    def _on_camera_changed(self, index: int):
        if index < 0 or self.combo_camera.count() == 0:
            return
        cam_id = self.combo_camera.itemData(index)
        self.status_bar.showMessage(f"已选择 Camera {cam_id}  |  Camera {cam_id} selected")
        if self._mode == "camera":
            self._update_ui_state()

    def _switch_mode(self, mode: str):
        self._stop_all_workers()
        self._mode = mode
        for btn, m in [
            (self.btn_mode_image, "image"),
            (self.btn_mode_video, "video"),
            (self.btn_mode_camera, "camera"),
        ]:
            btn.setChecked(m == mode)
            btn.setStyleSheet(self._sidebar_btn_style(m == mode))

        if mode == "image":
            self.btn_open.setText("选择文件  |  Select File")
            self.btn_detect.setText("开始检测  |  Start")
        elif mode == "video":
            self.btn_open.setText("选择视频  |  Select Video")
            self.btn_detect.setText("开始检测  |  Start")
        elif mode == "camera":
            self.btn_open.setText("选择摄像头  |  Camera")
            self.btn_detect.setText("开始检测  |  Start")

        # Reset display
        self._clear_results()
        self._reset_summary()
        self._reset_perf_stats()
        self._clear_table()
        self.lbl_original.setText("未加载图像\nNo image loaded")
        self.lbl_original.setStyleSheet("color: #777; font-size: 18px;")
        self.lbl_result.setText("检测结果将显示在此处\nResult will appear here")
        self.lbl_result.setStyleSheet("color: #777; font-size: 18px;")
        self._update_ui_state()

    # =========================================================================
    # File browser (paper §5.4.3)
    # =========================================================================
    def _on_browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择图像文件夹  |  Select Image Folder", "")
        if not folder:
            return
        self._current_browse_dir = Path(folder)
        self.file_list.clear()
        image_files = sorted(
            [f for f in self._current_browse_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS and f.is_file()]
        )
        for f in image_files:
            item = QListWidgetItem(f.name)
            item.setData(Qt.UserRole, str(f))
            try:
                thumb = QPixmap(str(f)).scaled(48, 48, Qt.KeepAspectRatio)
                item.setIcon(QIcon(thumb))
            except Exception:
                pass
            self.file_list.addItem(item)
        self.status_bar.showMessage(f"已浏览: {folder}  ({len(image_files)} 张图像)  |  {len(image_files)} images")

    def _on_folder_dropped(self, folder: str):
        """Handle drag-and-drop of a folder onto the file browser."""
        self._current_browse_dir = Path(folder)
        self.file_list.clear()
        image_files = sorted(
            [f for f in self._current_browse_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS and f.is_file()]
        )
        for f in image_files:
            item = QListWidgetItem(f.name)
            item.setData(Qt.UserRole, str(f))
            try:
                thumb = QPixmap(str(f)).scaled(48, 48, Qt.KeepAspectRatio)
                item.setIcon(QIcon(thumb))
            except Exception:
                pass
            self.file_list.addItem(item)
        self.status_bar.showMessage(f"已拖入: {folder}  ({len(image_files)} 张图像)  |  {len(image_files)} images")

    def _on_file_selected(self, item: QListWidgetItem):
        path = item.data(Qt.UserRole)
        if not path:
            return
        if self._mode != "image":
            self.status_bar.showMessage("提示: 当前非图像模式，请切换到图像模式后再选择文件")
            return
        img = cv2.imread(path)
        if img is None:
            QMessageBox.warning(self, "读取失败", f"无法读取图像文件:\n{path}")
            return
        self.current_image_path = path
        self.current_original = img
        self._show_image(self.lbl_original, img)
        self._clear_results()
        self._reset_summary()
        self._reset_perf_stats()
        self._clear_table()
        self.lbl_result.setText("检测结果将显示在此处\nResult will appear here")
        self.lbl_result.setStyleSheet("color: #777; font-size: 18px;")
        self.lbl_zoom_orig.setText("100%")
        self.lbl_zoom_res.setText("100%")
        self.status_bar.showMessage(f"已加载: {Path(path).name}  ({img.shape[1]}×{img.shape[0]})")
        self._update_ui_state()

    # =========================================================================
    # Auto-load model
    # =========================================================================
    @staticmethod
    def _scan_all_best_pt():
        candidates = []
        # 部署后不存在 runs/ 目录，仅源码运行时扫描
        base = Path("runs")
        if base.exists():
            for pt_file in base.rglob("**/weights/best.pt"):
                stat = pt_file.stat()
                if stat.st_size > 0:
                    candidates.append((pt_file.resolve(), stat.st_mtime, stat.st_size))
        # 也扫描 bundle 内的 weights/
        bundle_weights = MainWindow._bundle_dir() / "weights"
        if bundle_weights.is_dir():
            for pt_file in bundle_weights.glob("*.pt"):
                stat = pt_file.stat()
                if stat.st_size > 0:
                    candidates.append((pt_file.resolve(), stat.st_mtime, stat.st_size))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    def _try_auto_load_model(self):
        candidates = self._scan_all_best_pt()
        for rel in self.DEFAULT_MODEL_CANDIDATES:
            p = self._resolve_candidate(rel)
            if p.exists() and p.stat().st_size > 0:
                if not any(p == c[0] for c in candidates):
                    candidates.insert(0, (p, p.stat().st_mtime, p.stat().st_size))
        if not candidates:
            self.status_bar.showMessage("自动加载失败: 未找到任何 best.pt — 请手动加载模型")
            self._update_model_status(False)
            return
        errors = []
        for path, _mtime, size in candidates:
            try:
                self.detector.load_model(str(path))
                self._update_model_status(True)
                self._update_device_status()
                try:
                    rel_str = str(path.relative_to(Path.cwd()))
                except ValueError:
                    rel_str = str(path)
                self.status_bar.showMessage(f"已加载模型: {rel_str}  ({size / 1e6:.1f} MB)")
                self._update_ui_state()
                return
            except Exception as e:
                errors.append(f"  {path.name}: {e}")
                continue
        self._update_model_status(False)
        detail = "\n".join(errors) if errors else "(unknown)"
        QMessageBox.warning(self, "模型加载失败", f"尝试了 {len(candidates)} 个 checkpoint，全部失败:\n\n{detail}")

    def _try_auto_browse_val(self):
        """Auto-load the default val folder if it exists on disk."""
        default_val = Path("datasets/solar_dataset/images/val")
        if default_val.is_dir():
            self._current_browse_dir = default_val
            self.file_list.clear()
            image_files = sorted([f for f in default_val.iterdir() if f.suffix.lower() in IMAGE_EXTS and f.is_file()])
            for f in image_files:
                item = QListWidgetItem(f.name)
                item.setData(Qt.UserRole, str(f))
                try:
                    thumb = QPixmap(str(f)).scaled(48, 48, Qt.KeepAspectRatio)
                    item.setIcon(QIcon(thumb))
                except Exception:
                    pass
                self.file_list.addItem(item)
            self.status_bar.showMessage(
                f"已自动加载验证集: {default_val}  ({len(image_files)} 张)  |  "
                f"Auto-loaded val set: {len(image_files)} images"
            )

    # =========================================================================
    # Conversion helpers
    # =========================================================================
    @staticmethod
    @staticmethod
    def _cv_to_qpixmap(cv_bgr, target_w=None, target_h=None):
        rgb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)
        _ok, buf = cv2.imencode(".png", rgb)
        pix = QPixmap()
        pix.loadFromData(buf.tobytes())
        if target_w and target_h:
            pix = pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return pix

    def _show_image(self, label: ZoomableImageLabel, cv_bgr):
        pix = self._cv_to_qpixmap(cv_bgr)
        label.setBasePixmap(pix)

    # =========================================================================
    # Status updates
    # =========================================================================
    def _update_model_status(self, loaded: bool):
        if loaded:
            name = self.detector.model_name
            self.lbl_model_status.setText(f"● 模型: {name} 已加载")
            self.lbl_model_status.setStyleSheet(
                "color: #27AE60; font-size: 15px; font-weight: bold;"
                "padding: 6px 14px; background-color: #1e1e1e; border-radius: 4px;"
            )
        else:
            self.lbl_model_status.setText("● 模型: 未加载")
            self.lbl_model_status.setStyleSheet(
                "color: #E74C3C; font-size: 15px; font-weight: bold;"
                "padding: 6px 14px; background-color: #1e1e1e; border-radius: 4px;"
            )

    def _update_device_status(self):
        device = self.detector.device
        if device == "cpu":
            self.lbl_device.setText("● 设备: CPU")
            self.lbl_device.setStyleSheet(
                "color: #F39C12; font-size: 15px; font-weight: bold;"
                "padding: 6px 14px; background-color: #1e1e1e; border-radius: 4px;"
            )
        elif device.startswith("cuda"):
            self.lbl_device.setText(f"● 设备: NVIDIA GPU  |  {device}")
            self.lbl_device.setStyleSheet(
                "color: #27AE60; font-size: 15px; font-weight: bold;"
                "padding: 6px 14px; background-color: #1e1e1e; border-radius: 4px;"
            )
        elif device.startswith("xpu"):
            self.lbl_device.setText(f"● 设备: Intel Arc GPU  |  {device}")
            self.lbl_device.setStyleSheet(
                "color: #27AE60; font-size: 15px; font-weight: bold;"
                "padding: 6px 14px; background-color: #1e1e1e; border-radius: 4px;"
            )
        elif device == "mps":
            self.lbl_device.setText("● 设备: Apple MPS GPU")
            self.lbl_device.setStyleSheet(
                "color: #27AE60; font-size: 15px; font-weight: bold;"
                "padding: 6px 14px; background-color: #1e1e1e; border-radius: 4px;"
            )
        else:
            self.lbl_device.setText(f"● 设备: {device}")
            self.lbl_device.setStyleSheet(
                "color: #27AE60; font-size: 15px; font-weight: bold;"
                "padding: 6px 14px; background-color: #1e1e1e; border-radius: 4px;"
            )

    def _update_perf_stats(self, elapsed_ms: float):
        fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0
        self.lbl_inference_time.setText(f"Inference: {elapsed_ms:.1f} ms")
        self.lbl_inference_time.setStyleSheet("color: #4A90E2; font-size: 14px; font-weight: bold;")
        self.lbl_fps.setText(f"FPS: {fps:.1f}")
        self.lbl_fps.setStyleSheet("color: #27AE60; font-size: 14px; font-weight: bold;")

    def _reset_perf_stats(self):
        self.lbl_inference_time.setText("Inference: — ms")
        self.lbl_inference_time.setStyleSheet("color: #888; font-size: 14px; font-weight: bold;")
        self.lbl_fps.setText("FPS: —")
        self.lbl_fps.setStyleSheet("color: #888; font-size: 14px; font-weight: bold;")
        self.lbl_video_progress.setText("")

    # =========================================================================
    # Results Table (paper §5.4.4: per-box table)
    # =========================================================================
    def _clear_table(self):
        self.results_table.setRowCount(0)

    def _populate_table(self, boxes_detail: list):
        """Fill the results table with per-box details."""
        self._clear_table()
        self.results_table.setRowCount(len(boxes_detail))
        for row, det in enumerate(boxes_detail):
            cn_label = CLASS_LABELS_CN.get(det["class"], det["class"])
            items = [
                (cn_label, CLASS_COLORS_HEX.get(det["class"], "#ccc")),
                (f"{det['confidence']:.3f}", "#F1C40F"),
                (str(det["x1"]), "#888"),
                (str(det["y1"]), "#888"),
                (str(det["x2"]), "#888"),
                (str(det["y2"]), "#888"),
            ]
            for col, (text, color) in enumerate(items):
                item = QTableWidgetItem(text)
                item.setForeground(QColor(color))
                item.setTextAlignment(Qt.AlignCenter)
                self.results_table.setItem(row, col, item)

    # =========================================================================
    # Summary panel
    # =========================================================================
    def _reset_summary(self):
        # Reuse a single label to avoid widget churn (important for video mode).
        self._summary_label = QLabel("等待检测 …\nAwaiting detection …")
        self._summary_label.setAlignment(Qt.AlignCenter)
        self._summary_label.setStyleSheet("color: #666; font-size: 14px;")
        # Remove old widgets (deleteLater bypassed — instant removal)
        while self.summary_layout.count():
            child = self.summary_layout.takeAt(0)
            w = child.widget()
            if w is not None:
                w.setParent(None)
        self.summary_layout.addWidget(self._summary_label)

    def _update_summary(self, counts: dict):
        # Build single HTML string to avoid per-frame widget recreation.
        total = sum(counts.values())
        if total == 0:
            text = '<div style="color:#27AE60;font-size:14px;text-align:center;">'
            text += "未检测到缺陷<br>No defects detected"
            text += "</div>"
        else:
            text = ""
            for class_name in sorted(counts.keys()):
                count = counts[class_name]
                if count == 0:
                    continue
                cn_label = CLASS_LABELS_CN.get(class_name, class_name)
                hex_color = CLASS_COLORS_HEX.get(class_name, "#E74C3C")
                text += (
                    f'<div style="color:{hex_color};font-size:14px;font-weight:bold;">  ● {cn_label}:  {count}</div>'
                )
            text += (
                f'<div style="color:#fff;font-size:16px;font-weight:bold;margin-top:4px;">  总计 Total:  {total}</div>'
            )
        # Ensure label exists (should have been created by _reset_summary)
        if not hasattr(self, "_summary_label") or self._summary_label is None:
            self._reset_summary()
        self._summary_label.setText(text)
        self._summary_label.setAlignment(Qt.AlignCenter)
        self._summary_label.setWordWrap(True)

    # =========================================================================
    # Clear results
    # =========================================================================
    def _clear_results(self):
        self._last_annotated = None
        self._last_counts = {}
        self._last_elapsed_ms = 0.0
        self._last_boxes_detail = []

    # =========================================================================
    # Stop all workers (paper: §5.4 停止检测)
    # =========================================================================
    def _stop_all_workers(self):
        """Stop all running workers. Uses short timeouts to avoid blocking UI."""
        _SIGNAL_NAMES = (
            "detection_done",
            "video_done",
            "error_occurred",
            "frame_ready",
            "progress",
            "image_done",
            "batch_finished",
        )
        for attr, w in [
            ("worker", self.worker),
            ("video_worker", self.video_worker),
            ("camera_worker", self.camera_worker),
            ("batch_worker", self.batch_worker),
        ]:
            if w and w.isRunning():
                w.cancel()
                # Short timeout: if the worker is in the middle of model inference
                # (which is not interruptible), we only wait a brief moment. The
                # thread stays alive but the ::run loop will exit at the next
                # `if self._cancelled` check.
                w.wait(500)
                for sig_name in _SIGNAL_NAMES:
                    sig = getattr(w, sig_name, None)
                    if sig is not None:
                        try:
                            sig.disconnect()
                        except (TypeError, RuntimeError):
                            pass  # signal was never connected or already disconnected
        self.worker = None
        self.video_worker = None
        self.camera_worker = None
        self.batch_worker = None
        self._running_worker = None
        self.batch_progress.setVisible(False)
        self.lbl_batch_status.setVisible(False)

    def _on_stop(self):
        """Stop button handler."""
        self.status_bar.showMessage("正在停止 …  |  Stopping …")
        self._stop_all_workers()
        self._set_ui_running(False)
        self.btn_stop.setEnabled(False)
        self.status_bar.showMessage("已停止  |  Stopped")
        self._update_ui_state()

    # =========================================================================
    # Save / Export
    # =========================================================================
    def _on_save_result(self):
        if self._last_annotated is None:
            QMessageBox.warning(self, "提示", "请先执行检测。")
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        orig_name = Path(self.current_image_path).stem if self.current_image_path else "result"
        default_name = f"{ts}_{orig_name}.jpg"
        path, _ = QFileDialog.getSaveFileName(
            self, "保存检测结果", default_name, "JPEG (*.jpg);;PNG (*.png);;All Files (*)"
        )
        if not path:
            return
        try:
            ok = cv2.imwrite(path, self._last_annotated)
            if not ok:
                raise OSError(f"cv2.imwrite returned False for {path}")
            self.status_bar.showMessage(f"结果已保存: {path}")
            QMessageBox.information(self, "保存成功", f"检测结果已保存至:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))

    def _on_export_report(self):
        if not self._last_counts or not self.current_image_path:
            QMessageBox.warning(self, "提示", "请先执行检测。")
            return
        folder = QFileDialog.getExistingDirectory(self, "选择报告保存文件夹  |  Select Report Folder", "output/reports")
        if not folder:
            return
        try:
            out_dir = Path(folder)
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            image_name = Path(self.current_image_path).name
            total = sum(self._last_counts.values())
            stem = Path(image_name).stem

            report = {
                "timestamp": datetime.now().isoformat(),
                "image_name": image_name,
                "image_path": str(self.current_image_path),
                "confidence_threshold": self.detector.conf_threshold,
                "inference_time_ms": round(self._last_elapsed_ms, 1),
                "total_defects": total,
                "per_class_counts": self._last_counts,
                "detections": self._last_boxes_detail,
            }

            json_path = out_dir / f"{ts}_{stem}_report.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            csv_path = out_dir / f"{ts}_{stem}_report.csv"
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["字段", "值"])
                writer.writerow(["时间戳", report["timestamp"]])
                writer.writerow(["图像名称", report["image_name"]])
                writer.writerow(["置信度阈值", report["confidence_threshold"]])
                writer.writerow(["推理耗时(ms)", report["inference_time_ms"]])
                writer.writerow(["缺陷总数", report["total_defects"]])
                writer.writerow([])
                writer.writerow(["类别", "置信度", "X1", "Y1", "X2", "Y2"])
                for det in self._last_boxes_detail:
                    writer.writerow(
                        [
                            det["class"],
                            f"{det['confidence']:.3f}",
                            det["x1"],
                            det["y1"],
                            det["x2"],
                            det["y2"],
                        ]
                    )

            self.status_bar.showMessage(f"报告已导出: {json_path.name}, {csv_path.name}")
            QMessageBox.information(self, "导出成功", f"报告已保存至:\n{json_path}\n{csv_path}")
        except PermissionError:
            QMessageBox.warning(self, "文件占用", "无法写入报告文件，请关闭占用该文件的程序后重试。")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    # =========================================================================
    # Batch detection
    # =========================================================================
    def _on_batch_detect(self):
        if not self.detector.is_loaded:
            QMessageBox.warning(self, "提示", "请先加载模型。")
            return
        folder = QFileDialog.getExistingDirectory(self, "选择图像文件夹", "")
        if not folder:
            return

        self._set_ui_running(True)
        self.batch_progress.setVisible(True)
        self.batch_progress.setValue(0)
        self.lbl_batch_status.setVisible(True)
        self.lbl_batch_status.setText("正在批量检测 …")

        prep = self.combo_preprocess.currentData()
        self.batch_worker = BatchWorker(self.detector, folder, prep)
        self._running_worker = self.batch_worker
        self.batch_worker.progress.connect(self._on_batch_progress)
        self.batch_worker.image_done.connect(self._on_batch_image_done)
        self.batch_worker.batch_finished.connect(self._on_batch_finished)
        self.batch_worker.error_occurred.connect(self._on_batch_error)
        self.batch_worker.start()

    def _on_batch_progress(self, current: int, total: int, filename: str):
        pct = int(current / total * 100) if total > 0 else 0
        self.batch_progress.setValue(pct)
        self.lbl_batch_status.setText(f"[{current}/{total}] {filename}")

    def _on_batch_image_done(self, filename: str, counts: dict, elapsed_ms: float):
        total = sum(counts.values())
        self.status_bar.showMessage(f"{filename}: {total} defects, {elapsed_ms:.0f}ms")

    def _on_batch_finished(self, stats: dict):
        if self.sender() is not self._running_worker:
            return
        self._running_worker = None
        self.batch_progress.setVisible(False)
        self.lbl_batch_status.setVisible(False)
        self._set_ui_running(False)
        self.batch_worker = None
        self._update_ui_state()
        status = "已取消" if stats["cancelled"] else "完成"
        avg_ms = stats["avg_time_ms"]
        msg = (
            f"批量检测{status} — {stats['processed']}/{stats['total_images']} 张  |  "
            f"缺陷: {stats['total_defects']}  |  平均: {avg_ms:.1f}ms"
        )
        self.status_bar.showMessage(msg)
        QMessageBox.information(self, f"批量检测{status}", msg)

    def _on_batch_error(self, msg: str):
        self.status_bar.showMessage(f"批量检测错误: {msg[:150]}")

    # =========================================================================
    # Detection slots
    # =========================================================================
    def _on_open_image(self):
        if self._mode == "image":
            path, _ = QFileDialog.getOpenFileName(
                self, "选择 EL 图像", "", "Images (*.jpg *.jpeg *.png *.bmp *.tiff);;All Files (*)"
            )
            if not path:
                return
            img = cv2.imread(path)
            if img is None:
                QMessageBox.warning(self, "错误", f"无法读取图像:\n{path}")
                return
            self.current_image_path = path
            self.current_original = img
            self._show_image(self.lbl_original, img)
            self._clear_results()
            self._reset_summary()
            self._reset_perf_stats()
            self._clear_table()
            self.lbl_result.setText("检测结果将显示在此处\nResult will appear here")
            self.lbl_result.setStyleSheet("color: #777; font-size: 18px;")
            self.lbl_zoom_orig.setText("100%")
            self.lbl_zoom_res.setText("100%")
            self.status_bar.showMessage(f"已加载: {Path(path).name}  ({img.shape[1]}×{img.shape[0]})")
            self._update_ui_state()

        elif self._mode == "video":
            path, _ = QFileDialog.getOpenFileName(
                self, "选择视频文件", "", "Videos (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
            )
            if not path:
                return
            self.video_path = path
            self.current_image_path = None
            cap = cv2.VideoCapture(path)
            ret, frame = cap.read()
            if ret:
                self._show_image(self.lbl_original, frame)
                self.lbl_original.setStyleSheet("")
            cap.release()
            self._clear_results()
            self._reset_summary()
            self._reset_perf_stats()
            self._clear_table()
            self.lbl_result.setText("视频检测结果\nVideo result")
            self.lbl_result.setStyleSheet("color: #777; font-size: 18px;")
            self.status_bar.showMessage(f"已选择视频: {Path(path).name}")
            self._update_ui_state()

        elif self._mode == "camera":
            # Re-scan cameras and show available devices
            self._scan_cameras()
            n = self.combo_camera.count()
            if n > 0:
                # Build list of available cameras for the dialog
                cam_list = []
                for i in range(n):
                    cid = self.combo_camera.itemData(i)
                    cam_list.append(f"  • Camera {cid}")
                cam_id = self.combo_camera.currentData()
                self.status_bar.showMessage(
                    f"已选择 Camera {cam_id} — 共 {n} 个摄像头  |  Camera {cam_id} ({n} available)"
                )
                # Pop open the combo dropdown so user can see cameras
                self.combo_camera.showPopup()
                QMessageBox.information(
                    self,
                    "摄像头扫描结果",
                    f"已找到 {n} 个摄像头设备:\n\n" + "\n".join(cam_list) + f"\n\n当前选中: Camera {cam_id}\n"
                    "可在左侧栏 Camera 下拉框中切换摄像头。",
                )
            else:
                self.status_bar.showMessage("未检测到摄像头  |  No camera found")
                QMessageBox.warning(
                    self,
                    "摄像头未找到",
                    "未检测到任何摄像头设备。\n\n"
                    "可能原因:\n"
                    "1. 未连接摄像头\n"
                    "2. Windows 隐私设置禁止了摄像头访问\n"
                    "   (设置 → 隐私 → 摄像头 → 允许应用访问摄像头)\n"
                    "3. 摄像头驱动未安装",
                )
            self._update_ui_state()

    def _on_load_model(self):
        # Smart default directory: current model → bundle weights/ → local weights/ → runs/ → cwd
        default_dir = ""
        if self.detector.model_path:
            default_dir = str(Path(self.detector.model_path).parent)
        else:
            bundled_weights = self._bundle_dir() / "weights"
            if bundled_weights.is_dir():
                default_dir = str(bundled_weights)
            elif Path("weights").is_dir():
                default_dir = "weights"
            elif Path("runs").is_dir():
                default_dir = "runs"
            else:
                default_dir = "."
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", default_dir, "Model Files (*.pt *.pth *.onnx *.yaml *.yml);;All Files (*)"
        )
        if not path:
            return
        try:
            self.status_bar.showMessage(f"正在加载 {Path(path).name} …")
            QApplication.processEvents()
            self.detector.load_model(path)
            self._update_model_status(True)
            self._update_device_status()
            self.status_bar.showMessage(f"模型已加载: {Path(path).name}")
            self._update_ui_state()
        except Exception as e:
            QMessageBox.critical(self, "加载失败", str(e))

    def _on_detect(self):
        """Start detection based on current mode."""
        if not self.detector.is_loaded:
            QMessageBox.warning(self, "提示", "请先加载模型。")
            return

        prep = self.combo_preprocess.currentData()

        if self._mode == "image":
            if self.current_image_path is None:
                QMessageBox.warning(self, "提示", "请先打开图像。")
                return
            self._set_ui_running(True)
            self.status_bar.showMessage("正在检测中 …")
            self.worker = DetectionWorker(self.detector, self.current_image_path, prep)
            self._running_worker = self.worker
            self.worker.detection_done.connect(self._on_image_finished)
            self.worker.error_occurred.connect(self._on_error)
            self.worker.start()

        elif self._mode == "video":
            if self.video_path is None:
                QMessageBox.warning(self, "提示", "请先选择视频文件。")
                return
            self._set_ui_running(True)
            self.status_bar.showMessage("正在检测视频 …")
            self.video_worker = VideoWorker(self.detector, self.video_path, prep)
            self._running_worker = self.video_worker
            self.video_worker.frame_ready.connect(self._on_video_frame)
            self.video_worker.video_done.connect(self._on_video_finished)
            self.video_worker.error_occurred.connect(self._on_error)
            self.video_worker.start()

        elif self._mode == "camera":
            # 懒加载：首次进入摄像头模式才扫描设备
            if self.combo_camera.count() == 0 or self.combo_camera.itemData(0) == -1:
                self._scan_cameras()
            cam_id = self.combo_camera.currentData() if self.combo_camera.count() > 0 else 0
            if self.combo_camera.count() == 0 or cam_id == -1:
                QMessageBox.warning(
                    self,
                    "摄像头未找到",
                    "未检测到任何摄像头设备。\n\n"
                    "可能原因:\n"
                    "1. 未连接摄像头\n"
                    "2. Windows 隐私设置禁止了摄像头访问\n"
                    "   (设置 → 隐私 → 摄像头 → 允许应用访问摄像头)\n"
                    "3. 摄像头驱动未安装",
                )
                return
            try:
                self._set_ui_running(True)
                self.status_bar.showMessage(f"正在检测摄像头 Camera {cam_id} …")
                self.camera_worker = CameraWorker(self.detector, cam_id, prep)
                self._running_worker = self.camera_worker
                self.camera_worker.frame_ready.connect(self._on_camera_frame)
                self.camera_worker.camera_opened.connect(self._on_camera_opened)
                self.camera_worker.error_occurred.connect(self._on_error)
                self.camera_worker.start()
            except Exception as e:
                self._set_ui_running(False)
                QMessageBox.critical(self, "摄像头启动失败", f"无法启动摄像头 Camera {cam_id}:\n{e!s}")

    def _on_image_finished(self, original, annotated, counts, elapsed_ms, boxes_detail):
        if self.sender() is not self._running_worker:
            return  # stale signal from a cancelled/overwritten worker
        self._running_worker = None
        self.current_original = original
        self._last_annotated = annotated
        self._last_counts = counts
        self._last_elapsed_ms = elapsed_ms
        self._last_boxes_detail = boxes_detail

        self._show_image(self.lbl_original, original)
        self._show_image(self.lbl_result, annotated)
        self.lbl_zoom_orig.setText(f"{self.lbl_original.currentZoomPercent()}%")
        self.lbl_zoom_res.setText(f"{self.lbl_result.currentZoomPercent()}%")
        self._update_summary(counts)
        self._update_perf_stats(elapsed_ms)
        self._populate_table(boxes_detail)
        self._set_ui_running(False)
        self.worker = None

        total = sum(counts.values())
        self.status_bar.showMessage(
            f"检测完成 — 发现 {total} 处缺陷 ({elapsed_ms:.0f}ms)"
            if total > 0
            else f"检测完成 — 未发现缺陷 ({elapsed_ms:.0f}ms)"
        )
        self._update_ui_state()

    def _on_video_frame(self, frame, annotated, counts, elapsed_ms, boxes_detail, frame_idx: int, total_frames: int):
        if self.sender() is not self.video_worker:
            return  # stale signal from a cancelled worker
        self._show_image(self.lbl_original, frame)
        self._show_image(self.lbl_result, annotated)
        self._update_summary(counts)
        self._update_perf_stats(elapsed_ms)
        now = time.perf_counter()
        if now - self._last_table_update > 0.25:  # 4 Hz max
            self._populate_table(boxes_detail)
            self._last_table_update = now
        self.lbl_video_progress.setText(f"Frame: {frame_idx + 1}/{total_frames}")
        total = sum(counts.values())
        self.status_bar.showMessage(f"视频帧 {frame_idx + 1}/{total_frames}  |  {total} defects  |  {elapsed_ms:.0f}ms")

        # Save latest for export
        self._last_annotated = annotated
        self._last_counts = counts
        self._last_elapsed_ms = elapsed_ms
        self._last_boxes_detail = boxes_detail

    def _on_video_finished(self):
        if self.sender() is not self._running_worker:
            return
        self._running_worker = None
        self._set_ui_running(False)
        self.video_worker = None
        self.lbl_video_progress.setText("视频检测完成  |  Video complete")
        self.status_bar.showMessage("视频检测完成  |  Video detection finished")
        self._update_ui_state()

    def _on_camera_opened(self, width: int, height: int):
        self.status_bar.showMessage(f"摄像头已就绪: {width}×{height}  |  Camera ready: {width}×{height}")

    def _on_camera_frame(self, frame, annotated, counts, elapsed_ms, boxes_detail):
        if self.sender() is not self.camera_worker:
            return  # stale signal from a cancelled worker
        self._show_image(self.lbl_original, frame)
        self._show_image(self.lbl_result, annotated)
        self._update_summary(counts)
        self._update_perf_stats(elapsed_ms)
        now = time.perf_counter()
        if now - self._last_table_update > 0.25:  # 4 Hz max
            self._populate_table(boxes_detail)
            self._last_table_update = now
        self._last_annotated = annotated
        self._last_counts = counts
        self._last_elapsed_ms = elapsed_ms
        self._last_boxes_detail = boxes_detail

    def _on_error(self, msg):
        self._running_worker = None
        QMessageBox.critical(self, "检测失败  |  Detection Error", msg)
        self._set_ui_running(False)
        self.status_bar.showMessage(f"错误: {msg[:100]}")
        self._update_ui_state()

    def _on_conf_changed(self, value):
        threshold = value / 10.0
        self.detector.conf_threshold = threshold
        self.lbl_conf_value.setText(f"{threshold:.2f}")

    # =========================================================================
    # UI state helpers
    # =========================================================================
    def _set_ui_running(self, running: bool):
        """Set UI state during detection (enable stop, disable start/open/load)."""
        self.btn_open.setEnabled(not running)
        self.btn_load.setEnabled(not running)
        self.btn_batch.setEnabled(not running)
        self.conf_slider.setEnabled(not running)
        self.detect_progress.setVisible(running)
        if running:
            self.btn_detect.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.btn_save.setEnabled(False)
            self.btn_export.setEnabled(False)
        else:
            self.btn_stop.setEnabled(False)
            self._update_ui_state()

    def _update_ui_state(self):
        """Update button states based on current mode and model state."""
        has_model = self.detector.is_loaded

        if self._mode == "image":
            has_source = self.current_image_path is not None
        elif self._mode == "video":
            has_source = self.video_path is not None
        else:  # camera
            has_source = True  # camera is always available

        self.btn_detect.setEnabled(has_model and has_source)
        has_result = self._last_annotated is not None
        self.btn_save.setEnabled(has_result)
        self.btn_export.setEnabled(has_result)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        h = self.height()
        sz = max(7, min(11, h // 75))
        btn_h = max(24, sz * 3 + 8)
        f = QFont()
        f.setPointSize(sz)
        f.setBold(True)
        for btn in [
            self.btn_open,
            self.btn_load,
            self.btn_detect,
            self.btn_stop,
            self.btn_save,
            self.btn_export,
            self.btn_batch,
        ]:
            btn.setFont(f)
            btn.setMinimumHeight(btn_h)


# ===========================================================================
# Entry point
# ===========================================================================
def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Login dialog (paper §5.4.1)
    login = LoginDialog()
    if login.exec_() != QDialog.Accepted:
        sys.exit(0)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
