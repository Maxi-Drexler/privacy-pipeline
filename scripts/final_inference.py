#!/usr/bin/env python3
"""
Multi-model inference with configurable model-class assignments.

Combines 2-5 specialised YOLO models at inference time. Each model is
assigned a set of classes it is responsible for. Overlapping class
assignments are resolved via Weighted Boxes Fusion (WBF). The merged
detections are optionally anonymised using a three-tier strategy.

Configuration is provided via a YAML file specifying models, class
assignments, confidence thresholds, and anonymisation parameters.

Usage:
    python final_inference.py --config config/inference.yaml --input-dir /path/to/images
    python final_inference.py --config config/inference.yaml --input-dir /path/to/images --anonymise --vis
    python final_inference.py --generate-config config/inference.yaml

Author: Maximilian Drexler
License: MIT
"""

import argparse
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "class_names": [
        "face", "person", "vehicle", "text_or_logo",
        "crane", "container", "scaffolding", "material_stack",
    ],
    "models": [
        {
            "path": "weights/v12_general.pt",
            "classes": ["person", "vehicle", "text_or_logo", "crane",
                        "container", "scaffolding", "material_stack"],
            "confidence": 0.15,
        },
        {
            "path": "weights/face_only.pt",
            "classes": ["face"],
            "confidence": 0.10,
        },
        {
            "path": "weights/v11_person_supplement.pt",
            "classes": ["person"],
            "confidence": 0.20,
        },
    ],
    "wbf": {
        "enabled": True,
        "iou_threshold": 0.30,
    },
    "face_clipping": {
        "enabled": True,
        "max_height_ratio": 0.40,
        "clip_to_ratio": 0.30,
    },
    "anonymisation": {
        "blur_kernel": 51,
        "face_high_conf": 0.50,
        "person_fallback_ratio": 0.33,
        "anonymise_classes": ["face", "person", "text_or_logo"],
    },
}


def load_config(config_path):
    """Load inference configuration from YAML file.

    Falls back to DEFAULT_CONFIG for any missing keys.

    Args:
        config_path: Path to YAML config file, or None for defaults.

    Returns:
        Configuration dict.
    """
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            user_cfg = yaml.safe_load(f)
        cfg = DEFAULT_CONFIG.copy()
        cfg.update(user_cfg)
        return cfg
    return DEFAULT_CONFIG.copy()


def compute_iou(a, b):
    """IoU between two [x1,y1,x2,y2] boxes."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


def weighted_boxes_fusion_single_class(detections, iou_threshold=0.30):
    """Apply WBF to a list of same-class detections from multiple models.

    Fuses overlapping boxes by averaging coordinates weighted by
    confidence. Non-overlapping boxes are kept as-is.

    Args:
        detections: List of dicts with bbox, conf, model keys.
        iou_threshold: IoU threshold for merging.

    Returns:
        List of fused detection dicts.
    """
    if len(detections) <= 1:
        return detections

    used = [False] * len(detections)
    fused = []

    for i in range(len(detections)):
        if used[i]:
            continue
        cluster = [detections[i]]
        used[i] = True
        for j in range(i + 1, len(detections)):
            if used[j]:
                continue
            if compute_iou(detections[i]["bbox"], detections[j]["bbox"]) > iou_threshold:
                cluster.append(detections[j])
                used[j] = True

        if len(cluster) == 1:
            fused.append(cluster[0])
        else:
            total_conf = sum(d["conf"] for d in cluster)
            avg_bbox = [0.0, 0.0, 0.0, 0.0]
            for d in cluster:
                w = d["conf"] / total_conf
                for k in range(4):
                    avg_bbox[k] += d["bbox"][k] * w
            fused.append({
                "class_id": cluster[0]["class_id"],
                "bbox": avg_bbox,
                "conf": max(d["conf"] for d in cluster),
                "model": "wbf_fused",
            })

    return fused


def clip_face_to_person(detections, cfg, class_names):
    """Clip oversized face boxes to the head region of containing persons.

    Args:
        detections: List of detection dicts.
        cfg: Face clipping configuration dict.
        class_names: List of class name strings.

    Returns:
        Modified list with clipped face boxes.
    """
    if not cfg.get("enabled", False):
        return detections

    face_id = class_names.index("face") if "face" in class_names else None
    person_id = class_names.index("person") if "person" in class_names else None

    if face_id is None or person_id is None:
        return detections

    persons = [d for d in detections if d["class_id"] == person_id]
    max_ratio = cfg.get("max_height_ratio", 0.40)
    clip_ratio = cfg.get("clip_to_ratio", 0.30)

    for d in detections:
        if d["class_id"] != face_id:
            continue
        fh = d["bbox"][3] - d["bbox"][1]
        fcx = (d["bbox"][0] + d["bbox"][2]) / 2
        fcy = (d["bbox"][1] + d["bbox"][3]) / 2

        for p in persons:
            pb = p["bbox"]
            if pb[0] <= fcx <= pb[2] and pb[1] <= fcy <= pb[3]:
                ph = pb[3] - pb[1]
                if fh > ph * max_ratio:
                    d["bbox"][3] = pb[1] + ph * clip_ratio
                    d["bbox"][0] = max(d["bbox"][0], pb[0])
                    d["bbox"][2] = min(d["bbox"][2], pb[2])
                break

    return [d for d in detections if d["bbox"][3] - d["bbox"][1] >= 5]


def det_to_yolo(cls_id, bbox, img_w, img_h):
    """Convert absolute coords to YOLO format string."""
    cx = ((bbox[0] + bbox[2]) / 2) / img_w
    cy = ((bbox[1] + bbox[3]) / 2) / img_h
    bw = (bbox[2] - bbox[0]) / img_w
    bh = (bbox[3] - bbox[1]) / img_h
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def run_multi_model_inference(img_path, models, class_names, wbf_cfg):
    """Run all models on a single image and merge results.

    Args:
        img_path: Path to image file.
        models: List of (yolo_model, class_ids, confidence) tuples.
        class_names: List of class name strings.
        wbf_cfg: WBF configuration dict.

    Returns:
        List of detection dicts.
    """
    all_detections = []

    for yolo_model, allowed_ids, conf in models:
        results = yolo_model(str(img_path), conf=conf, verbose=False)
        if not results or results[0].boxes is None:
            continue

        model_class_names = yolo_model.names

        for box in results[0].boxes:
            model_cls = int(box.cls[0])
            model_cls_name = model_class_names.get(model_cls, "")

            if model_cls_name in class_names:
                mapped_id = class_names.index(model_cls_name)
            elif model_cls < len(class_names):
                mapped_id = model_cls
            else:
                continue

            if mapped_id not in allowed_ids:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            all_detections.append({
                "class_id": mapped_id,
                "bbox": [x1, y1, x2, y2],
                "conf": float(box.conf[0]),
                "model": str(getattr(yolo_model, "ckpt_path", "unknown")),
            })

    if wbf_cfg.get("enabled", False):
        by_class = {}
        for d in all_detections:
            by_class.setdefault(d["class_id"], []).append(d)

        merged = []
        for cls_id, dets in by_class.items():
            merged.extend(weighted_boxes_fusion_single_class(
                dets, iou_threshold=wbf_cfg.get("iou_threshold", 0.30)))
        return merged

    return dedup_detections(all_detections)


def dedup_detections(detections, iou_threshold=0.30):
    """Remove duplicate same-class detections, keeping highest confidence.

    Args:
        detections: List of detection dicts.
        iou_threshold: IoU threshold for considering duplicates.

    Returns:
        Deduplicated list.
    """
    if len(detections) <= 1:
        return detections

    detections = sorted(detections, key=lambda d: -d["conf"])
    keep = []
    for d in detections:
        is_dup = False
        for k in keep:
            if k["class_id"] == d["class_id"] and compute_iou(d["bbox"], k["bbox"]) > iou_threshold:
                is_dup = True
                break
        if not is_dup:
            keep.append(d)
    return keep


def anonymise_image(img_path, detections, output_path, class_names, anon_cfg):
    """Apply three-tier anonymisation to an image.

    Args:
        img_path: Path to source image.
        detections: List of detection dicts.
        output_path: Path to save anonymised image.
        class_names: List of class name strings.
        anon_cfg: Anonymisation configuration dict.
    """
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    blur_k = anon_cfg.get("blur_kernel", 51)
    face_high = anon_cfg.get("face_high_conf", 0.50)
    fallback_ratio = anon_cfg.get("person_fallback_ratio", 0.33)
    anon_classes = anon_cfg.get("anonymise_classes", ["face", "person", "text_or_logo"])

    face_id = class_names.index("face") if "face" in class_names else None
    person_id = class_names.index("person") if "person" in class_names else None

    faces = [d for d in detections if d["class_id"] == face_id] if face_id is not None else []
    persons = [d for d in detections if d["class_id"] == person_id] if person_id is not None else []

    def blur_region(x1, y1, x2, y2):
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 > x1 and y2 > y1:
            img[y1:y2, x1:x2] = cv2.GaussianBlur(
                img[y1:y2, x1:x2], (blur_k, blur_k), 0)

    for det in faces:
        b = det["bbox"]
        if det["conf"] >= face_high:
            blur_region(*b)
        else:
            buf = int(max(b[2] - b[0], b[3] - b[1]) * 0.2)
            blur_region(b[0] - buf, b[1] - buf, b[2] + buf, b[3] + buf)

    if "person" in anon_classes:
        person_has_face = [False] * len(persons)
        for i, p in enumerate(persons):
            pb = p["bbox"]
            for f in faces:
                fb = f["bbox"]
                fcx = (fb[0] + fb[2]) / 2
                fcy = (fb[1] + fb[3]) / 2
                if pb[0] <= fcx <= pb[2] and pb[1] <= fcy <= pb[3]:
                    person_has_face[i] = True
                    break

        for i, det in enumerate(persons):
            if person_has_face[i]:
                continue
            b = det["bbox"]
            head_y2 = b[1] + (b[3] - b[1]) * fallback_ratio
            blur_region(b[0], b[1], b[2], head_y2)

    anon_class_ids = set()
    for name in anon_classes:
        if name in class_names and name not in ("face", "person"):
            anon_class_ids.add(class_names.index(name))

    for det in detections:
        if det["class_id"] in anon_class_ids:
            blur_region(*det["bbox"])

    cv2.imwrite(str(output_path), img)


def generate_default_config(output_path):
    """Write a default configuration YAML file.

    Args:
        output_path: Path to write the config file.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False, sort_keys=False)
    log.info("Default config written to %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-model inference with configurable class assignments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config (3 models)
  python final_inference.py --input-dir images/ --output-dir output/

  # Run with custom config
  python final_inference.py --config my_config.yaml --input-dir images/ --output-dir output/

  # Generate a default config file to customise
  python final_inference.py --generate-config config/inference.yaml

  # Run with anonymisation and visualisation
  python final_inference.py --input-dir images/ --output-dir output/ --anonymise --vis
        """,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file. Uses defaults if not provided.")
    parser.add_argument("--generate-config", type=str, default=None,
                        help="Generate a default config file at this path and exit.")
    parser.add_argument("--input-dir", type=str)
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--anonymise", action="store_true")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    if args.generate_config:
        generate_default_config(args.generate_config)
        return

    if not args.input_dir:
        parser.error("--input-dir is required")

    cfg = load_config(args.config)
    class_names = cfg["class_names"]

    labels_dir = Path(args.output_dir) / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    log.info("Loading %d models...", len(cfg["models"]))
    loaded_models = []
    for mcfg in cfg["models"]:
        model = YOLO(mcfg["path"])
        allowed_ids = set()
        for cls_name in mcfg["classes"]:
            if cls_name in class_names:
                allowed_ids.add(class_names.index(cls_name))
        loaded_models.append((model, allowed_ids, mcfg.get("confidence", 0.15)))
        log.info("  Loaded %s -> classes %s (conf=%.2f)",
                 mcfg["path"], mcfg["classes"], mcfg.get("confidence", 0.15))

    img_dir = Path(args.input_dir)
    images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    if args.max_images:
        images = images[:args.max_images]

    if args.anonymise:
        anon_dir = Path(args.output_dir) / "anonymised"
        anon_dir.mkdir(parents=True, exist_ok=True)

    if args.vis:
        vis_dir = Path(args.output_dir) / "vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        palette = [(0,0,255),(0,255,0),(255,0,0),(0,255,255),
                    (255,0,255),(255,255,0),(128,128,0),(0,128,128)]
        colors = {i: palette[i % len(palette)] for i in range(len(class_names))}

    stats = {n: 0 for n in class_names}

    for img_path in tqdm(images, desc="Inference"):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        dets = run_multi_model_inference(
            img_path, loaded_models, class_names, cfg.get("wbf", {}))

        dets = clip_face_to_person(dets, cfg.get("face_clipping", {}), class_names)

        lines = []
        for d in dets:
            lines.append(det_to_yolo(d["class_id"], d["bbox"], w, h))
            if d["class_id"] < len(class_names):
                stats[class_names[d["class_id"]]] += 1

        with open(labels_dir / (img_path.stem + ".txt"), "w") as f:
            f.write("\n".join(lines) + "\n" if lines else "")

        if args.anonymise:
            anonymise_image(
                img_path, dets, anon_dir / img_path.name,
                class_names, cfg.get("anonymisation", {}))

        if args.vis:
            vis_img = cv2.imread(str(img_path))
            for d in dets:
                cls = d["class_id"]
                x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
                col = colors.get(cls, (255, 255, 255))
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), col, 2)
                label = class_names[cls] if cls < len(class_names) else str(cls)
                cv2.putText(vis_img, f'{label} {d["conf"]:.2f}',
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
            cv2.imwrite(str(vis_dir / img_path.name), vis_img)

    n_imgs = len(images)
    print("\n" + "=" * 60)
    print("INFERENCE RESULTS")
    print("=" * 60)
    print(f"  Images:  {n_imgs}")
    print(f"  Models:  {len(cfg['models'])}")
    print(f"  WBF:     {'ON' if cfg.get('wbf', {}).get('enabled') else 'OFF'}")
    print(f"  Per class:")
    for cls in class_names:
        print(f"    {cls:20s}: {stats[cls]:6d} ({stats[cls]/max(n_imgs,1):.1f}/img)")
    print(f"  Labels:  {labels_dir}")
    if args.anonymise:
        print(f"  Anonymised: {anon_dir}")
    if args.vis:
        print(f"  Visualisations: {vis_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
