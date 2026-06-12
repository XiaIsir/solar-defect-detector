"""Convert PV-Multi-Defect VOC XML annotations to YOLO txt format with 80/20 split."""
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import random

random.seed(42)

# Paths
SRC_DIR = Path(r"F:/ultralytics-main/datasets/PV-Multi-Defect")
XML_DIR = SRC_DIR / "Annotations"
IMG_DIR = SRC_DIR / "JPEGImages"
DST_DIR = Path(r"F:/ultralytics-main/datasets/solar_dataset")

# Class mapping from XML <name> to index
CLASSES = {
    "black_border": 0,
    "broken": 1,
    "hot_spot": 2,
    "no_electricity": 3,
    "scratch": 4,
}

# Create directory structure
for split in ("train", "val"):
    (DST_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (DST_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def parse_xml(xml_path):
    """Parse VOC XML and return list of (class_id, xc, yc, w, h) normalized."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    w_img = int(size.find("width").text)
    h_img = int(size.find("height").text)

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in CLASSES:
            continue
        cls_id = CLASSES[name]
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        # YOLO format: class_id x_center y_center width height (normalized)
        xc = (xmin + xmax) / 2.0 / w_img
        yc = (ymin + ymax) / 2.0 / h_img
        w = (xmax - xmin) / w_img
        h = (ymax - ymin) / h_img

        objects.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    return objects


def main():
    xml_files = sorted(XML_DIR.glob("*.xml"))

    # Pair XML with corresponding JPG, verify both exist
    pairs = []
    for xml_path in xml_files:
        jpg_path = IMG_DIR / (xml_path.stem + ".jpg")
        if jpg_path.exists():
            pairs.append((xml_path, jpg_path))

    print(f"Found {len(pairs)} valid image-annotation pairs")

    # Shuffle and split 80/20
    random.shuffle(pairs)
    split_idx = int(len(pairs) * 0.80)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    # Process each split
    for split_name, split_pairs in (("train", train_pairs), ("val", val_pairs)):
        imgs_dst = DST_DIR / "images" / split_name
        lbls_dst = DST_DIR / "labels" / split_name

        for xml_path, jpg_path in split_pairs:
            # Convert annotations
            yolo_objects = parse_xml(xml_path)
            txt_path = lbls_dst / (xml_path.stem + ".txt")
            with open(txt_path, "w") as f:
                f.write("\n".join(yolo_objects) + "\n" if yolo_objects else "")

            # Copy image
            shutil.copy2(jpg_path, imgs_dst / (xml_path.stem + ".jpg"))

    # Class distribution summary
    class_counts = {name: 0 for name in CLASSES}
    all_label_dirs = [DST_DIR / "labels" / s for s in ("train", "val")]
    for lbl_dir in all_label_dirs:
        for txt_file in lbl_dir.glob("*.txt"):
            with open(txt_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        cls_id = int(line.split()[0])
                        for name, idx in CLASSES.items():
                            if idx == cls_id:
                                class_counts[name] += 1

    print("\nClass distribution:")
    for name, count in class_counts.items():
        print(f"  {name}: {count} instances")

    print(f"\nDataset ready at: {DST_DIR}")
    print(f"  images/train: {len(train_pairs)} images")
    print(f"  images/val:   {len(val_pairs)} images")
    print(f"  labels/train: {len(train_pairs)} labels")
    print(f"  labels/val:   {len(val_pairs)} labels")


if __name__ == "__main__":
    main()
