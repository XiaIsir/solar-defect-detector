#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
L-FAF-YOLOv11n Cloud Training Script
Usage: python train_cloud.py [--epochs 100] [--batch 16] [--imgsz 640] [--device 0]
"""
import argparse
import sys
sys.path.insert(0, '.')

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description='L-FAF-YOLOv11n Training')
    parser.add_argument('--epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    parser.add_argument('--device', type=str, default='0', help='cuda device (0,1,2... or cpu)')
    parser.add_argument('--data', type=str, default='cloud_data.yaml', help='data yaml path')
    parser.add_argument('--model', type=str, default='ultralytics/cfg/models/11/l_faf_yolov11n.yaml')
    parser.add_argument('--project', type=str, default='runs/train', help='save project dir')
    parser.add_argument('--name', type=str, default='lfaf_v2', help='experiment name')
    parser.add_argument('--resume', action='store_true', help='resume from last checkpoint')
    args = parser.parse_args()

    model = YOLO(args.model, task='detect')
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=8,
        val=True,
        cls_pw=1.0,
        mosaic=1.0,
        mixup=0.15,
        project=args.project,
        name=args.name,
        resume=args.resume,
        pretrained=True,
        optimizer='auto',
        cos_lr=True,
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,
    )


if __name__ == '__main__':
    main()
