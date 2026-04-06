#!/usr/bin/env python3
"""
Multi-model inference with configurable model-class assignments.

Combines 2-5 specialised YOLO models at inference time. Each model is
assigned a set of classes it is responsible for. Overlapping class
assignments are resolved via Weighted Boxes Fusion (WBF). The merged
detections are optionally anonymised using a three-tier strategy
identical to Stage 5 (05_anonymize.py).

Configuration is provided via a YAML file specifying models, class
assignments, confidence thresholds, and anonymisation parameters.

Usage:
    python final_inference.py --config config/inference.yaml --input-dir /path/to/images
    python final_inference.py --config config/inference.yaml --input-dir /path/to/images --vis
    python final_inference.py --generate-config config/inference.yaml

Author: Maximilian Drexler
License: MIT
"""

import argparse
import json
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
        "base_kernel": 51,
        "body_kernel": 31,
        "face_high_conf": 0.50,
        "padding_pct": 10,
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


def run_multi_model_inference(image, models, class_names, wbf_cfg):
    """Run all models on a single image and merge results.

    Args:
        image: Numpy BGR image array (already zone-blurred if applicable).
        models: List of (yolo_model, class_ids, confidence) tuples.
        class_names: List of class name strings.
        wbf_cfg: WBF configuration dict.

    Returns:
        List of detection dicts.
    """
    all_detections = []

    for yolo_model, allowed_ids, conf in models:
        results = yolo_model(image, conf=conf, verbose=False)
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


def dynamic_kernel(box_width, box_height, base_kernel=51):
    """Scale kernel size based on bounding box dimensions.

    Larger regions get stronger blur to ensure anonymisation holds
    even when viewed at full resolution.

    Args:
        box_width: Width of the bounding box.
        box_height: Height of the bounding box.
        base_kernel: Minimum kernel size.

    Returns:
        Odd integer kernel size.
    """
    size = max(box_width, box_height)
    k = max(base_kernel, int(size * 0.4))
    if k % 2 == 0:
        k += 1
    return k


def add_padding(x1, y1, x2, y2, img_w, img_h, padding_pct=10):
    """Expand bounding box by a percentage for safety margin.

    Args:
        x1, y1, x2, y2: Original box coordinates.
        img_w, img_h: Image dimensions.
        padding_pct: Padding as percentage of box size.

    Returns:
        Tuple of padded (x1, y1, x2, y2).
    """
    bw, bh = x2 - x1, y2 - y1
    px = bw * padding_pct / 100
    py = bh * padding_pct / 100
    return (
        max(0, x1 - px), max(0, y1 - py),
        min(img_w, x2 + px), min(img_h, y2 + py),
    )


def apply_gaussian_blur(image, x1, y1, x2, y2, kernel_size=51):
    """Apply Gaussian blur to a rectangular region of the image.

    Args:
        image: OpenCV image (modified in-place).
        x1, y1, x2, y2: Bounding box coordinates.
        kernel_size: Gaussian kernel size (must be odd).

    Returns:
        The modified image.
    """
    h, w = image.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return image

    k = max(kernel_size, 3)
    if k % 2 == 0:
        k += 1

    roi = image[y1:y2, x1:x2]
    blurred = cv2.GaussianBlur(roi, (k, k), 0)
    image[y1:y2, x1:x2] = blurred
    return image


def overlay_metadata(image, filename):
    """Extract timestamp from filename and overlay on image.

    Parses the filename pattern Kamera[N]_00_YYYYMMDDHHMMSS.jpg
    and renders date and time in the bottom-right corner with a
    semi-transparent black background for readability.

    Args:
        image: OpenCV BGR image (modified in-place).
        filename: Image filename containing timestamp.

    Returns:
        The modified image, or unmodified if parsing fails.
    """
    stem = Path(filename).stem
    try:
        timestamp_str = stem.split("_")[-1]
        if len(timestamp_str) != 14 or not timestamp_str.isdigit():
            return image
        date_part = f"{timestamp_str[0:4]}-{timestamp_str[4:6]}-{timestamp_str[6:8]}"
        time_part = f"{timestamp_str[8:10]}:{timestamp_str[10:12]}:{timestamp_str[12:14]}"
        text = f"{date_part}  {time_part}"
    except (IndexError, ValueError):
        return image

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.8, image.shape[1] / 1500)
    thickness = max(2, int(scale * 2))

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    margin = 10
    x = image.shape[1] - tw - margin
    y = image.shape[0] - margin

    overlay = image.copy()
    cv2.rectangle(overlay, (x - 6, y - th - 6), (x + tw + 6, y + baseline + 4),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    cv2.putText(image, text, (x, y), font, scale, (0, 255, 255), thickness,
                cv2.LINE_AA)

    return image


def load_zone_config(config_path):
    """Load zone-based privacy mask configuration from JSON.

    Args:
        config_path: Path to zones JSON file.

    Returns:
        Dict mapping setup names to zone definitions, or empty dict.
    """
    if not config_path or not os.path.exists(config_path):
        return {}
    with open(config_path) as f:
        return json.load(f)


def parse_zone_polygon(polygon_def, img_w, img_h):
    """Convert a zone polygon definition to numpy coordinates.

    Args:
        polygon_def: Polygon definition (string or list).
        img_w: Image width.
        img_h: Image height.

    Returns:
        Numpy array of shape (N, 2) with polygon points.
    """
    if isinstance(polygon_def, str):
        if polygon_def.startswith("full_width_top_"):
            pct = int(polygon_def.split("_")[-1]) / 100
            return np.array([[0, 0], [img_w, 0],
                             [img_w, int(img_h * pct)], [0, int(img_h * pct)]])
        elif polygon_def.startswith("full_width_bottom_"):
            pct = int(polygon_def.split("_")[-1]) / 100
            y_start = int(img_h * (1 - pct))
            return np.array([[0, y_start], [img_w, y_start],
                             [img_w, img_h], [0, img_h]])
        elif polygon_def.startswith("right_"):
            pct = int(polygon_def.split("_")[-1]) / 100
            x_start = int(img_w * (1 - pct))
            return np.array([[x_start, 0], [img_w, 0],
                             [img_w, img_h], [x_start, img_h]])
        elif polygon_def.startswith("left_"):
            pct = int(polygon_def.split("_")[-1]) / 100
            return np.array([[0, 0], [int(img_w * pct), 0],
                             [int(img_w * pct), img_h], [0, img_h]])
        else:
            raise ValueError(f"Unknown zone shorthand: {polygon_def}")

    coords = polygon_def
    if coords and not isinstance(coords[0], (list, tuple)):
        coords = [[coords[i], coords[i + 1]] for i in range(0, len(coords), 2)]
    return np.array(coords, dtype=np.int32)


def apply_zone_masks(image, zones, kernel_size=99):
    """Blur static exclusion zones before detection.

    Applied to the input image before models run. Personal data in
    excluded regions is never processed by the detector.

    Args:
        image: OpenCV BGR image (modified in-place).
        zones: List of zone dicts with 'polygon' key.
        kernel_size: Gaussian kernel for zone blur.

    Returns:
        Tuple of (modified image, number of zones applied).
    """
    if not zones:
        return image, 0

    h, w = image.shape[:2]
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    count = 0

    for zone in zones:
        poly = parse_zone_polygon(zone["polygon"], w, h)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)

        blurred = cv2.GaussianBlur(image, (k, k), 0)
        image = np.where(mask[:, :, np.newaxis] == 255, blurred, image)
        count += 1

    return image, count


def filter_detections_in_zones(detections, zones, img_w, img_h):
    """Remove detections whose center falls within an exclusion zone.

    Args:
        detections: List of detection dicts with 'bbox' key.
        zones: List of zone dicts with 'polygon' key.
        img_w: Image width.
        img_h: Image height.

    Returns:
        Filtered list of detections.
    """
    if not zones:
        return detections

    zone_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for zone in zones:
        poly = parse_zone_polygon(zone["polygon"], img_w, img_h)
        cv2.fillPoly(zone_mask, [poly], 255)

    filtered = []
    for d in detections:
        cx = int((d["bbox"][0] + d["bbox"][2]) / 2)
        cy = int((d["bbox"][1] + d["bbox"][3]) / 2)
        cx = max(0, min(cx, img_w - 1))
        cy = max(0, min(cy, img_h - 1))
        if zone_mask[cy, cx] == 0:
            filtered.append(d)

    return filtered


def detect_setup_from_filename(filename):
    """Detect camera setup from filename pattern.

    Args:
        filename: Image filename.

    Returns:
        Setup key string ('setup_1', 'setup_2', ...) or None.
    """
    import re
    name = filename.lower()
    match = re.search(r'(?:kamera|setup_?|cam)(\d+)', name)
    if match:
        return f"setup_{match.group(1)}"
    return None


def anonymise_image(image, detections, class_names, anon_cfg):
    """Apply three-tier confidence-based anonymisation to an image.

    Tier 1: High-confidence face — targeted blur with padding.
    Tier 2: Low-confidence face — blur with doubled padding.
    Tier 3: Person without detected face — blur upper third.
    Additionally: text_or_logo always blurred.

    Before applying tiers, small person detections in the head region
    of a larger person are reclassified as faces.

    Args:
        image: OpenCV BGR image (modified in-place).
        detections: List of detection dicts with class_id, bbox, conf.
        class_names: List of class name strings.
        anon_cfg: Anonymisation configuration dict.

    Returns:
        The modified image.
    """
    h, w = image.shape[:2]

    base_kernel = anon_cfg.get("base_kernel", 51)
    body_kernel = anon_cfg.get("body_kernel", 31)
    face_high = anon_cfg.get("face_high_conf", 0.50)
    padding_pct = anon_cfg.get("padding_pct", 10)
    fallback_ratio = anon_cfg.get("person_fallback_ratio", 0.33)

    face_id = class_names.index("face") if "face" in class_names else None
    person_id = class_names.index("person") if "person" in class_names else None
    text_id = class_names.index("text_or_logo") if "text_or_logo" in class_names else None

    faces = [d for d in detections if d["class_id"] == face_id] if face_id is not None else []
    persons = [d for d in detections if d["class_id"] == person_id] if person_id is not None else []
    text_logos = [d for d in detections if d["class_id"] == text_id] if text_id is not None else []

    reclassified_faces = []
    small_person_indices = set()
    for i, small_p in enumerate(persons):
        sb = small_p["bbox"]
        sa = (sb[2] - sb[0]) * (sb[3] - sb[1])
        for j, large_p in enumerate(persons):
            if i == j:
                continue
            lb = large_p["bbox"]
            la = (lb[2] - lb[0]) * (lb[3] - lb[1])
            if sa >= la * 0.5:
                continue
            ix1 = max(sb[0], lb[0])
            iy1 = max(sb[1], lb[1])
            ix2 = min(sb[2], lb[2])
            iy2 = min(sb[3], lb[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if sa > 0 and inter / sa < 0.5:
                continue
            head_cutoff = lb[1] + (lb[3] - lb[1]) * 0.40
            if (sb[1] + sb[3]) / 2 <= head_cutoff:
                reclassified_faces.append(small_p)
                small_person_indices.add(i)
                break

    persons = [p for i, p in enumerate(persons) if i not in small_person_indices]

    person_has_face = [False] * len(persons)
    for face in faces + reclassified_faces:
        fb = face["bbox"]
        fx_center = (fb[0] + fb[2]) / 2
        fy_center = (fb[1] + fb[3]) / 2
        for i, person in enumerate(persons):
            pb = person["bbox"]
            if pb[0] <= fx_center <= pb[2] and pb[1] <= fy_center <= pb[3]:
                person_has_face[i] = True
                break

    for face in reclassified_faces:
        fb = face["bbox"]
        bw, bh = fb[2] - fb[0], fb[3] - fb[1]
        k = dynamic_kernel(bw, bh, base_kernel)
        x1, y1, x2, y2 = add_padding(fb[0], fb[1], fb[2], fb[3], w, h, padding_pct)
        apply_gaussian_blur(image, x1, y1, x2, y2, k)

    for face in faces:
        fb = face["bbox"]
        bw, bh = fb[2] - fb[0], fb[3] - fb[1]
        k = dynamic_kernel(bw, bh, base_kernel)

        if face["conf"] >= face_high:
            x1, y1, x2, y2 = add_padding(fb[0], fb[1], fb[2], fb[3], w, h, padding_pct)
            apply_gaussian_blur(image, x1, y1, x2, y2, k)
        else:
            x1, y1, x2, y2 = add_padding(fb[0], fb[1], fb[2], fb[3], w, h, padding_pct * 2)
            apply_gaussian_blur(image, x1, y1, x2, y2, k)

    for i, person in enumerate(persons):
        if person_has_face[i]:
            continue
        pb = person["bbox"]
        top_y2 = pb[1] + (pb[3] - pb[1]) * fallback_ratio
        x1, y1, x2, y2 = add_padding(pb[0], pb[1], pb[2], top_y2, w, h, padding_pct // 2)
        k = dynamic_kernel(pb[2] - pb[0], top_y2 - pb[1], body_kernel)
        apply_gaussian_blur(image, x1, y1, x2, y2, k)

    for tl in text_logos:
        tb = tl["bbox"]
        bw, bh = tb[2] - tb[0], tb[3] - tb[1]
        k = dynamic_kernel(bw, bh, base_kernel)
        x1, y1, x2, y2 = add_padding(tb[0], tb[1], tb[2], tb[3], w, h, padding_pct)
        apply_gaussian_blur(image, x1, y1, x2, y2, k)

    return image


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

  # Run with bounding box visualisation overlay
  python final_inference.py --input-dir images/ --output-dir output/ --vis
        """,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file. Uses defaults if not provided.")
    parser.add_argument("--generate-config", type=str, default=None,
                        help="Generate a default config file at this path and exit.")
    parser.add_argument("--input-dir", type=str)
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--vis", action="store_true",
                        help="Save additional images with bounding box overlays.")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--no-metadata", action="store_true",
                        help="Disable timestamp overlay on output images.")
    parser.add_argument("--zones-config", type=str, default=None,
                        help="Path to JSON config with static zone masks per camera setup.")
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

    zone_config = load_zone_config(args.zones_config)

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

    anon_dir = Path(args.output_dir) / "anonymised"
    anon_dir.mkdir(parents=True, exist_ok=True)

    if args.vis:
        vis_dir = Path(args.output_dir) / "vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        palette = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
                    (255, 0, 255), (255, 255, 0), (128, 128, 0), (0, 128, 128)]
        colors = {i: palette[i % len(palette)] for i in range(len(class_names))}

    stats = {n: 0 for n in class_names}
    show_metadata = not args.no_metadata

    for img_path in tqdm(images, desc="Inference"):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        zones = []
        if zone_config:
            setup = detect_setup_from_filename(img_path.name)
            zones = zone_config.get(setup, {}).get("zones", []) if setup else []
            img, _ = apply_zone_masks(img, zones)

        dets = run_multi_model_inference(
            img, loaded_models, class_names, cfg.get("wbf", {}))

        dets = clip_face_to_person(dets, cfg.get("face_clipping", {}), class_names)
        dets = filter_detections_in_zones(dets, zones, w, h)

        lines = []
        for d in dets:
            lines.append(det_to_yolo(d["class_id"], d["bbox"], w, h))
            if d["class_id"] < len(class_names):
                stats[class_names[d["class_id"]]] += 1

        with open(labels_dir / (img_path.stem + ".txt"), "w") as f:
            f.write("\n".join(lines) + "\n" if lines else "")

        anon_img = img.copy()
        anon_img = anonymise_image(
            anon_img, dets, class_names, cfg.get("anonymisation", {}))

        if show_metadata:
            overlay_metadata(anon_img, img_path.name)

        cv2.imwrite(str(anon_dir / img_path.name), anon_img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])

        if args.vis:
            vis_img = anon_img.copy()
            for d in dets:
                cls = d["class_id"]
                x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
                col = colors.get(cls, (255, 255, 255))
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), col, 2)
                label = class_names[cls] if cls < len(class_names) else str(cls)
                cv2.putText(vis_img, f'{label} {d["conf"]:.2f}',
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

            cv2.imwrite(str(vis_dir / img_path.name), vis_img,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

    n_imgs = len(images)
    print("\n" + "=" * 60)
    print("INFERENCE RESULTS")
    print("=" * 60)
    print(f"  Images:  {n_imgs}")
    print(f"  Models:  {len(cfg['models'])}")
    print(f"  WBF:     {'ON' if cfg.get('wbf', {}).get('enabled') else 'OFF'}")
    print(f"  Per class:")
    for cls in class_names:
        print(f"    {cls:20s}: {stats[cls]:6d} ({stats[cls] / max(n_imgs, 1):.1f}/img)")
    print(f"  Labels:  {labels_dir}")
    print(f"  Anonymised: {anon_dir}")
    if args.vis:
        print(f"  Visualisations: {vis_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
