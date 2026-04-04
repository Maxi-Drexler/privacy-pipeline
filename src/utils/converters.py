"""
Annotation format conversion utilities.

Supports conversion between COCO JSON, YOLO TXT, and CVAT XML formats
for interoperability between annotation tools and training pipelines.

Reference:
    Lin, T.-Y. et al. (2014) 'Microsoft COCO: Common Objects in Context',
    Proceedings of the European Conference on Computer Vision (ECCV),
    pp. 740-755.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime


def detections_to_yolo_txt(
    detections: List[Dict],
    image_width: int,
    image_height: int
) -> str:
    """
    Convert detection list to YOLO TXT format.

    YOLO format uses normalised centre coordinates and dimensions:
    <class_id> <x_center> <y_center> <width> <height>

    Args:
        detections: List of detection dicts with 'bbox' [x1,y1,x2,y2]
            and 'class_id'.
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        String content for a YOLO .txt label file.
    """
    lines = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cx = ((x1 + x2) / 2) / image_width
        cy = ((y1 + y2) / 2) / image_height
        w = (x2 - x1) / image_width
        h = (y2 - y1) / image_height

        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))

        lines.append(f"{det['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return "\n".join(lines)


def detections_to_coco(
    all_detections: Dict[str, List[Dict]],
    image_info_list: List[Dict],
    class_names: List[str]
) -> dict:
    """
    Convert all detections to COCO JSON format.

    Args:
        all_detections: Dict mapping image filenames to detection lists.
        image_info_list: List of dicts with 'file_name', 'width', 'height', 'id'.
        class_names: Ordered list of class names (index = class_id).

    Returns:
        COCO-format dictionary with 'images', 'annotations', 'categories'.
    """
    categories = [
        {"id": i, "name": name, "supercategory": "object"}
        for i, name in enumerate(class_names)
    ]

    annotations = []
    ann_id = 1

    for img_info in image_info_list:
        fname = img_info["file_name"]
        dets = all_detections.get(fname, [])

        for det in dets:
            x1, y1, x2, y2 = det["bbox"]
            w = x2 - x1
            h = y2 - y1
            area = w * h

            ann = {
                "id": ann_id,
                "image_id": img_info["id"],
                "category_id": det["class_id"],
                "bbox": [x1, y1, w, h],
                "area": area,
                "iscrowd": 0,
                "segmentation": []
            }

            if "attributes" in det:
                ann["attributes"] = det["attributes"]

            annotations.append(ann)
            ann_id += 1

    return {
        "info": {
            "description": "TUM Kita Construction Site Auto-Annotations",
            "date_created": datetime.now().isoformat(),
            "version": "1.0"
        },
        "images": image_info_list,
        "annotations": annotations,
        "categories": categories
    }


def coco_to_yolo_files(
    coco_json_path: str,
    output_dir: str,
    image_dir: Optional[str] = None
) -> None:
    """
    Convert COCO JSON annotations to YOLO TXT label files.

    Creates one .txt file per image in the output directory.

    Args:
        coco_json_path: Path to COCO JSON file.
        output_dir: Directory for output .txt files.
        image_dir: Optional directory to symlink images for training.
    """
    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    img_lookup = {img["id"]: img for img in coco["images"]}

    anns_by_image = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in anns_by_image:
            anns_by_image[img_id] = []
        anns_by_image[img_id].append(ann)

    for img_id, img_info in img_lookup.items():
        anns = anns_by_image.get(img_id, [])
        w = img_info["width"]
        h = img_info["height"]

        lines = []
        for ann in anns:
            bx, by, bw, bh = ann["bbox"]
            cx = (bx + bw / 2) / w
            cy = (by + bh / 2) / h
            nw = bw / w
            nh = bh / h
            lines.append(f"{ann['category_id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        stem = Path(img_info["file_name"]).stem
        txt_path = os.path.join(output_dir, f"{stem}.txt")
        with open(txt_path, "w") as f:
            f.write("\n".join(lines))


def prepare_training_split(
    image_dir: str,
    label_dir: str,
    output_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Dict[str, int]:
    """
    Split annotated images into train/val sets for YOLO training.

    Creates the directory structure expected by Ultralytics YOLO:
    output_dir/images/train/, output_dir/images/val/,
    output_dir/labels/train/, output_dir/labels/val/

    Args:
        image_dir: Directory containing images.
        label_dir: Directory containing YOLO .txt label files.
        output_dir: Root directory for the training dataset.
        val_ratio: Fraction of images for validation (default 0.2).
        seed: Random seed for reproducible splits.

    Returns:
        Dict with 'train' and 'val' counts.
    """
    import random
    import shutil

    random.seed(seed)

    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    label_files = sorted(Path(label_dir).glob("*.txt"))
    image_exts = [".jpg", ".jpeg", ".png", ".bmp"]

    paired = []
    for lf in label_files:
        for ext in image_exts:
            img_path = Path(image_dir) / f"{lf.stem}{ext}"
            if img_path.exists():
                paired.append((str(img_path), str(lf)))
                break

    random.shuffle(paired)
    split_idx = int(len(paired) * (1 - val_ratio))
    train_pairs = paired[:split_idx]
    val_pairs = paired[split_idx:]

    for split_name, pairs in [("train", train_pairs), ("val", val_pairs)]:
        for img_path, lbl_path in pairs:
            shutil.copy2(img_path, os.path.join(output_dir, "images", split_name))
            shutil.copy2(lbl_path, os.path.join(output_dir, "labels", split_name))

    return {"train": len(train_pairs), "val": len(val_pairs)}
