#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Blind inference script for L-FAF-YOLOv11n cloud-trained model."""
import argparse
import os
import cv2
import torch
from ultralytics import YOLO


CLASS_NAMES = {
    0: "black_border",
    1: "broken",
    2: "hot_spot",
    3: "no_electricity",
    4: "scratch",
}

CLASS_COLORS = {
    0: (0, 165, 255),    # orange
    1: (0, 0, 255),      # red
    2: (0, 255, 255),    # yellow
    3: (255, 0, 0),      # blue
    4: (0, 255, 0),      # green
}


def draw_boxes(img, results):
    """Draw prediction boxes on image."""
    if results[0].boxes is None:
        return img
    boxes = results[0].boxes.xyxy.cpu().numpy()
    clss = results[0].boxes.cls.cpu().numpy().astype(int)
    confs = results[0].boxes.conf.cpu().numpy()
    for box, cls_id, conf in zip(boxes, clss, confs):
        x1, y1, x2, y2 = map(int, box)
        color = CLASS_COLORS.get(cls_id, (0, 0, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES.get(cls_id, str(cls_id))} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
    return img


def main():
    parser = argparse.ArgumentParser(description="L-FAF-YOLOv11n Blind Inference")
    parser.add_argument("--weights", default="weights/best.pt", help="model weights path")
    parser.add_argument("--input", default="datasets/solar_dataset/images/val",
                        help="input image folder")
    parser.add_argument("--output", default="output_predictions", help="output folder")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--device", default="", help="cuda device (empty=auto)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu")

    model = YOLO(args.weights)
    model.to(device)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    img_files = sorted(
        f for f in os.listdir(args.input)
        if os.path.splitext(f)[1].lower() in exts
    )
    print(f"Found {len(img_files)} images in {args.input}")
    print(f"Model: {args.weights}  |  Device: {device}  |  Conf: {args.conf}")

    total = len(img_files)
    det_count = 0
    for i, fn in enumerate(img_files, 1):
        impath = os.path.join(args.input, fn)
        img = cv2.imread(impath)
        if img is None:
            print(f"  [{i}/{total}] SKIP {fn} (unreadable)")
            continue

        results = model(img, imgsz=args.imgsz, conf=args.conf, verbose=False)
        annotated = draw_boxes(img, results)

        out_path = os.path.join(args.output, fn)
        cv2.imwrite(out_path, annotated)

        n_obj = 0 if results[0].boxes is None else len(results[0].boxes)
        if n_obj > 0:
            det_count += 1
            classes_found = {CLASS_NAMES.get(int(c), str(int(c)))
                             for c in results[0].boxes.cls}
            print(f"  [{i}/{total}] {fn} -> {n_obj} defects: {classes_found}")
        else:
            print(f"  [{i}/{total}] {fn} -> no defect")

    print(f"\nDone. {det_count}/{total} images have defects.")
    print(f"Output: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
