#!/usr/bin/env python3
"""
Stage 4: Train YOLO11 object detection model.

Converts CVAT YOLO export into YOLO training directory structure,
creates train/val splits, and launches YOLO11 training with transfer
learning from COCO pretrained weights.

Expected CVAT export structure:
    cvat_export/
        obj.names              # class names
        train.txt              # image list
        obj_train_data/...     # label .txt files

Usage:
    python scripts/04_train_detector.py \
        --cvat-export /home/mdrexler/data/cvat_export \
        --images-dir /home/mdrexler/data/construction_images_selected/selected \
        --data-dir /workspace/data/training \
        --epochs 100 \
        --project /workspace/output \
        --name construction_v1

    # Resume from checkpoint:
    python scripts/04_train_detector.py \
        --dataset-yaml /workspace/data/training/dataset.yaml \
        --resume /workspace/output/construction_v1/weights/last.pt

Author: Maximilian Drexler
License: MIT
"""

import os
import sys
import random
import shutil
import argparse
import logging
import yaml
from pathlib import Path
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

CLASS_NAMES = [
    "face", "person", "vehicle", "text_or_logo",
    "crane", "container", "scaffolding", "material_stack",
]


def find_label_files(cvat_export_dir):
    """Recursively find all .txt label files in the CVAT export.

    Args:
        cvat_export_dir: Root of CVAT YOLO export.

    Returns:
        Dict mapping stem (e.g. 'Kamera1_00_20250101072120') to label path.
    """
    labels = {}
    obj_dir = os.path.join(cvat_export_dir, "obj_train_data")
    for root, _, files in os.walk(obj_dir):
        for f in files:
            if f.endswith(".txt"):
                stem = Path(f).stem
                labels[stem] = os.path.join(root, f)
    return labels


def prepare_training_split(cvat_export_dir, images_dir, output_dir,
                           val_ratio=0.2, seed=42):
    """Convert CVAT YOLO export to YOLO training directory structure.

    Creates:
        output_dir/
            images/train/  images/val/
            labels/train/  labels/val/
            dataset.yaml

    Args:
        cvat_export_dir: Root of CVAT YOLO export.
        images_dir: Directory containing the source images.
        output_dir: Where to create the training dataset.
        val_ratio: Fraction of data for validation.
        seed: Random seed for reproducible splits.

    Returns:
        Dict with split statistics.
    """
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    label_map = find_label_files(cvat_export_dir)
    log.info("Found %d label files in CVAT export", len(label_map))

    image_exts = (".jpg", ".jpeg", ".png", ".bmp")
    image_map = {}
    for f in os.listdir(images_dir):
        if f.lower().endswith(image_exts):
            image_map[Path(f).stem] = os.path.join(images_dir, f)

    matched = sorted(set(label_map.keys()) & set(image_map.keys()))
    missing_images = set(label_map.keys()) - set(image_map.keys())
    missing_labels = set(image_map.keys()) - set(label_map.keys())

    if missing_images:
        log.warning("%d labels without images (skipped)", len(missing_images))
    if missing_labels:
        log.info("%d images without labels (skipped — no annotations)", len(missing_labels))

    log.info("Matched %d image-label pairs", len(matched))

    empty_labels = 0
    for stem in matched:
        label_path = label_map[stem]
        if os.path.getsize(label_path) == 0:
            empty_labels += 1
    if empty_labels:
        log.info("%d images have empty label files (background / no objects)", empty_labels)

    random.seed(seed)
    random.shuffle(matched)
    split_idx = int(len(matched) * (1 - val_ratio))
    train_stems = matched[:split_idx]
    val_stems = matched[split_idx:]

    class_counts = {"train": Counter(), "val": Counter()}

    for split_name, stems in [("train", train_stems), ("val", val_stems)]:
        for stem in stems:
            img_src = image_map[stem]
            lbl_src = label_map[stem]
            img_ext = Path(img_src).suffix

            img_dst = os.path.join(output_dir, "images", split_name, stem + img_ext)
            lbl_dst = os.path.join(output_dir, "labels", split_name, stem + ".txt")

            if not os.path.exists(img_dst):
                os.symlink(os.path.abspath(img_src), img_dst)
            shutil.copy2(lbl_src, lbl_dst)

            with open(lbl_src) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        if cls_id < len(CLASS_NAMES):
                            class_counts[split_name][CLASS_NAMES[cls_id]] += 1

    yaml_config = {
        "path": os.path.abspath(output_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_config, f, default_flow_style=False)

    stats = {
        "train": len(train_stems),
        "val": len(val_stems),
        "total": len(matched),
        "empty_labels": empty_labels,
        "class_counts": class_counts,
        "yaml_path": yaml_path,
    }
    return stats


def print_dataset_summary(stats):
    """Print formatted dataset summary."""
    print("\n" + "=" * 60)
    print("DATASET PREPARED")
    print("=" * 60)
    print(f"  Total pairs:     {stats['total']}")
    print(f"  Train:           {stats['train']}")
    print(f"  Val:             {stats['val']}")
    print(f"  Empty labels:    {stats['empty_labels']}")
    print(f"  Config:          {stats['yaml_path']}")
    print()
    print("  Annotations per class:")
    print(f"  {'Class':20s} {'Train':>8s} {'Val':>8s} {'Total':>8s}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
    for cls in CLASS_NAMES:
        tr = stats["class_counts"]["train"].get(cls, 0)
        va = stats["class_counts"]["val"].get(cls, 0)
        print(f"  {cls:20s} {tr:8d} {va:8d} {tr+va:8d}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Stage 4: Train YOLO11 on construction site images."
    )

    prep = parser.add_argument_group("Dataset preparation")
    prep.add_argument("--cvat-export", type=str, default=None,
                      help="Path to CVAT YOLO export directory.")
    prep.add_argument("--images-dir", type=str, default=None,
                      help="Directory with source images.")
    prep.add_argument("--data-dir", type=str,
                      default="/workspace/data/training",
                      help="Output directory for training dataset.")
    prep.add_argument("--val-ratio", type=float, default=0.2,
                      help="Validation split ratio (default: 0.2).")
    prep.add_argument("--dataset-yaml", type=str, default=None,
                      help="Skip preparation, use existing dataset.yaml.")

    train = parser.add_argument_group("Training")
    train.add_argument("--model", type=str, default="yolo11n.pt",
                       help="Base model (default: yolo11n.pt).")
    train.add_argument("--epochs", type=int, default=100,
                       help="Training epochs (default: 100).")
    train.add_argument("--imgsz", type=int, default=640,
                       help="Input image size (default: 640).")
    train.add_argument("--batch", type=int, default=16,
                       help="Batch size (default: 16).")
    train.add_argument("--device", type=str, default="0",
                       help="CUDA device (default: 0).")
    train.add_argument("--project", type=str, default="/workspace/output",
                       help="Output project directory.")
    train.add_argument("--name", type=str, default="construction_v1",
                       help="Experiment name.")
    train.add_argument("--patience", type=int, default=20,
                       help="Early stopping patience (default: 20).")
    train.add_argument("--resume", type=str, default=None,
                       help="Resume training from checkpoint.")
    train.add_argument("--skip-train", action="store_true",
                       help="Only prepare dataset, skip training.")

    args = parser.parse_args()

    if args.dataset_yaml:
        dataset_yaml = args.dataset_yaml
        log.info("Using existing dataset config: %s", dataset_yaml)
    else:
        if not args.cvat_export or not args.images_dir:
            parser.error("--cvat-export and --images-dir are required "
                         "when --dataset-yaml is not provided.")

        log.info("Preparing training dataset from CVAT export...")
        stats = prepare_training_split(
            cvat_export_dir=args.cvat_export,
            images_dir=args.images_dir,
            output_dir=args.data_dir,
            val_ratio=args.val_ratio,
        )
        print_dataset_summary(stats)
        dataset_yaml = stats["yaml_path"]

    if args.skip_train:
        print("\n  --skip-train set, stopping here.")
        return

    from ultralytics import YOLO

    if args.resume:
        log.info("Resuming training from: %s", args.resume)
        model = YOLO(args.resume)
        results = model.train(resume=True)
    else:
        log.info("Loading base model: %s", args.model)
        model = YOLO(args.model)
        results = model.train(
            data=dataset_yaml,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
            patience=args.patience,
            pretrained=True,
            verbose=True,
        )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Project:    {args.project}/{args.name}")
    print(f"  Best model: {args.project}/{args.name}/weights/best.pt")
    print(f"  Last model: {args.project}/{args.name}/weights/last.pt")
    print(f"\n  Next steps:")
    print(f"    05_anonymize.py --model {args.project}/{args.name}/weights/best.pt")
    print(f"    06_evaluate.py  --model {args.project}/{args.name}/weights/best.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()c
