#!/usr/bin/env python3
"""
Stage 5: Anonymise construction site images.

Detects privacy-sensitive regions using the trained YOLO11 model and
applies Gaussian blur for GDPR-compliant anonymisation. Implements a
3-tier confidence-based escalation strategy that increases blur
coverage proportionally with detection uncertainty.

Anonymisation tiers:
  Tier 1 — High-confidence face detection (conf >= face_high_thresh):
           Apply targeted Gaussian blur to face bounding box only.
  Tier 2 — Low-confidence face detection (conf >= model threshold but
           below face_high_thresh): Apply face blur plus a safety
           buffer around the face region.
  Tier 3 — No face detected on a person: Blur the upper 33 percent of
           the person bounding box as a precautionary measure.

Additional rules:
  - text_or_logo: Always blurred regardless of confidence tier.
  - vehicle: Not blurred (license plates are text_or_logo).
  - crane, container, scaffolding, material_stack: Never blurred.

This proportionality principle ensures anonymisation strength
increases with detection uncertainty while preserving maximum
construction-relevant information (GDPR Article 5 data minimisation).

Usage:
    python scripts/05_anonymize.py \
        --input-dir /workspace/data/TUM_KITA_imgs \
        --output-dir /workspace/data/anonymized \
        --model /workspace/output/construction_v1/weights/best.pt

    # Test on a few images first:
    python scripts/05_anonymize.py \
        --input-dir /workspace/data/construction_images_selected/selected \
        --output-dir /workspace/data/anonymized_test \
        --model /workspace/output/construction_v1/weights/best.pt \
        --max-images 20

Author: Maximilian Drexler
License: MIT
"""

import os
import cv2
import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
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

ALWAYS_BLUR = {"face", "text_or_logo"}
NEVER_BLUR = {"vehicle", "crane", "container", "scaffolding", "material_stack"}

CLASS_COLORS = {
    "face": (0, 0, 255), "person": (0, 200, 0),
    "vehicle": (0, 165, 255), "text_or_logo": (255, 0, 255),
    "crane": (0, 255, 255), "container": (255, 255, 0),
    "scaffolding": (0, 128, 128), "material_stack": (128, 128, 0),
}


def load_zone_config(config_path):
    """Load zone-based privacy mask configuration from JSON.

    Args:
        config_path: Path to zones JSON file.

    Returns:
        Dict mapping setup names to zone definitions, or empty dict.
    """
    if not config_path or not os.path.exists(config_path):
        return {}
    import json
    with open(config_path) as f:
        return json.load(f)


def parse_zone_polygon(polygon_def, img_w, img_h):
    """Convert a zone polygon definition to numpy coordinates.

    Supports shorthand strings like "full_width_top_20" and
    explicit coordinate lists.

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

    coords = polygon_def
    if coords and not isinstance(coords[0], (list, tuple)):
        coords = [[coords[i], coords[i+1]] for i in range(0, len(coords), 2)]
    return np.array(coords, dtype=np.int32)


def apply_zone_masks(image, zones, kernel_size=99):
    """Apply static zone blurring as the first anonymisation step.

    This implements the data minimisation principle of GDPR Article 5(1)(c)
    by obscuring image regions irrelevant to monitoring before any detection.

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


def detect_setup_from_filename(filename):
    """Detect camera setup from filename pattern.

    Matches patterns like 'Kamera1', 'Kamera2', 'setup_1', 'cam3', etc.
    and maps them to 'setup_N' keys used in the zones config.

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


def draw_detection(image, x1, y1, x2, y2, class_name, confidence):
    """Draw bounding box and label on image.

    Args:
        image: OpenCV image (modified in-place).
        x1, y1, x2, y2: Box coordinates.
        class_name: Detection class name.
        confidence: Detection confidence score.
    """
    color = CLASS_COLORS.get(class_name, (200, 200, 200))
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    label = f"{class_name} {confidence:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (int(x1), int(y1) - th - 6),
                  (int(x1) + tw + 4, int(y1)), color, -1)
    cv2.putText(image, label, (int(x1) + 2, int(y1) - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


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


def anonymise_image(image, detections, face_high_thresh=0.5,
                    base_kernel=51, body_kernel=31, padding_pct=10,
                    body_blur_top_pct=0.33, enable_body_blur=True):
    """Apply three-tier anonymisation to a single image.

    Before applying tiers, small person detections in the head region
    of a larger person are reclassified as faces (preprocessing step
    addressing the known misclassification pattern).

    Tiers:
      1   — High-confidence face (conf >= face_high_thresh): targeted blur.
      2   — Low-confidence face: blur with doubled safety buffer.
      3   — Person without any face: blur upper 33% of person box.
      +   — text_or_logo: always blurred.

    Args:
        image: OpenCV image (BGR).
        detections: List of dicts with bbox, confidence, class_name.
        face_high_thresh: Tier-1 face confidence threshold.
        base_kernel: Base Gaussian kernel size.
        body_kernel: Kernel size for tier-3 body blur.
        padding_pct: Safety padding percentage.
        body_blur_top_pct: Fraction of person box to blur from top (tier 3).
        enable_body_blur: If False, skip tier-3 body blur.

    Returns:
        Tuple of (anonymised image, stats dict).
    """
    h, w = image.shape[:2]
    stats = Counter()

    faces = [d for d in detections if d["class_name"] == "face"]
    persons = [d for d in detections if d["class_name"] == "person"]
    text_logos = [d for d in detections if d["class_name"] == "text_or_logo"]

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
        x1, y1, x2, y2 = add_padding(
            fb[0], fb[1], fb[2], fb[3], w, h, padding_pct
        )
        apply_gaussian_blur(image, x1, y1, x2, y2, k)
        stats["reclassified_person_to_face"] += 1

    for face in faces:
        fb = face["bbox"]
        bw, bh = fb[2] - fb[0], fb[3] - fb[1]
        k = dynamic_kernel(bw, bh, base_kernel)

        if face["confidence"] >= face_high_thresh:
            x1, y1, x2, y2 = add_padding(
                fb[0], fb[1], fb[2], fb[3], w, h, padding_pct
            )
            apply_gaussian_blur(image, x1, y1, x2, y2, k)
            stats["tier1_face"] += 1
        else:
            x1, y1, x2, y2 = add_padding(
                fb[0], fb[1], fb[2], fb[3], w, h, padding_pct * 2
            )
            apply_gaussian_blur(image, x1, y1, x2, y2, k)
            stats["tier2_face_buffered"] += 1

    if enable_body_blur:
        for i, person in enumerate(persons):
            if person_has_face[i]:
                continue
            pb = person["bbox"]
            top_y2 = pb[1] + (pb[3] - pb[1]) * body_blur_top_pct
            x1, y1, x2, y2 = add_padding(
                pb[0], pb[1], pb[2], top_y2, w, h, padding_pct // 2
            )
            k = dynamic_kernel(pb[2] - pb[0], top_y2 - pb[1], body_kernel)
            apply_gaussian_blur(image, x1, y1, x2, y2, k)
            stats["tier3_body_blur"] += 1

    for tl in text_logos:
        tb = tl["bbox"]
        bw, bh = tb[2] - tb[0], tb[3] - tb[1]
        k = dynamic_kernel(bw, bh, base_kernel)
        x1, y1, x2, y2 = add_padding(
            tb[0], tb[1], tb[2], tb[3], w, h, padding_pct
        )
        apply_gaussian_blur(image, x1, y1, x2, y2, k)
        stats["text_or_logo"] += 1

    return image, stats


def process_directory(model, input_dir, output_dir, confidence=0.25,
                      face_high_thresh=0.5, base_kernel=51, body_kernel=31,
                      padding_pct=10, max_images=None, enable_body_blur=True,
                      draw_detections_flag=False, draw_blur_only=False,
                      zone_config=None):
    """Process all images in a directory.

    Args:
        model: Loaded YOLO model.
        input_dir: Source image directory.
        output_dir: Destination for anonymised images.
        confidence: Detection confidence threshold.
        face_high_thresh: Tier-1 face confidence threshold.
        base_kernel: Base blur kernel size.
        body_kernel: Tier-3 body blur kernel size.
        padding_pct: Box padding percentage.
        max_images: Limit number of images (for testing).
        enable_body_blur: Enable tier-3 body blur.
        draw_detections_flag: Draw bounding boxes on output.
        draw_blur_only: Only draw boxes for blurred classes.
        zone_config: Dict mapping setup names to zone definitions.

    Returns:
        Dict with processing statistics.
    """
    os.makedirs(output_dir, exist_ok=True)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted([
        f for f in os.listdir(input_dir)
        if Path(f).suffix.lower() in image_exts
    ])

    if max_images:
        image_files = image_files[:max_images]

    total_stats = Counter()
    total_stats["total_images"] = len(image_files)

    for filename in tqdm(image_files, desc="Anonymising"):
        img_path = os.path.join(input_dir, filename)
        image = cv2.imread(img_path)
        if image is None:
            log.warning("Could not read: %s", img_path)
            total_stats["errors"] += 1
            continue

        if zone_config:
            setup = detect_setup_from_filename(filename)
            zones = zone_config.get(setup, {}).get("zones", []) if setup else []
            image, n_zones = apply_zone_masks(image, zones)
            total_stats["zones_applied"] += n_zones

        results = model(img_path, conf=confidence, verbose=False)

        detections = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(box.conf[0]),
                    "class_name": cls_name,
                    "class_id": cls_id,
                })
                total_stats[f"det_{cls_name}"] += 1

        image, img_stats = anonymise_image(
            image, detections,
            face_high_thresh=face_high_thresh,
            base_kernel=base_kernel,
            body_kernel=body_kernel,
            padding_pct=padding_pct,
            enable_body_blur=enable_body_blur,
        )
        total_stats.update(img_stats)

        if draw_detections_flag:
            for det in detections:
                cls = det["class_name"]
                if draw_blur_only and cls in NEVER_BLUR:
                    continue
                b = det["bbox"]
                draw_detection(image, b[0], b[1], b[2], b[3],
                               cls, det["confidence"])

        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        total_stats["processed"] += 1

    return dict(total_stats)


def main():
    parser = argparse.ArgumentParser(
        description="Stage 5: Anonymise construction site images (3-tier)."
    )
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory with images to anonymise.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory for anonymised output images.")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained YOLO11 weights (.pt).")
    parser.add_argument("--confidence", type=float, default=0.25,
                        help="Detection confidence threshold (default: 0.25).")
    parser.add_argument("--face-high-thresh", type=float, default=0.5,
                        help="Tier-1 face confidence threshold (default: 0.5).")
    parser.add_argument("--blur-strength", type=int, default=51,
                        help="Base Gaussian kernel size (default: 51).")
    parser.add_argument("--body-blur-strength", type=int, default=31,
                        help="Tier-3 body blur kernel size (default: 31).")
    parser.add_argument("--padding", type=int, default=10,
                        help="Box padding percentage (default: 10).")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Process at most N images (for testing).")
    parser.add_argument("--no-body-blur", action="store_true",
                        help="Disable tier-3 body blur for faceless persons.")
    parser.add_argument("--draw-detections", action="store_true",
                        help="Draw bounding boxes and class labels on output.")
    parser.add_argument("--draw-blur-only", action="store_true",
                        help="Only draw boxes for blurred (privacy) classes.")
    parser.add_argument("--zones-config", type=str, default=None,
                        help="Path to JSON config with static zone masks per camera setup.")
    args = parser.parse_args()

    from ultralytics import YOLO

    log.info("Loading model: %s", args.model)
    model = YOLO(args.model)

    log.info("Starting anonymisation...")
    stats = process_directory(
        model=model,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        confidence=args.confidence,
        face_high_thresh=args.face_high_thresh,
        base_kernel=args.blur_strength,
        body_kernel=args.body_blur_strength,
        padding_pct=args.padding,
        max_images=args.max_images,
        enable_body_blur=not args.no_body_blur,
        draw_detections_flag=args.draw_detections,
        draw_blur_only=args.draw_blur_only,
        zone_config=load_zone_config(args.zones_config),
    )

    print("\n" + "=" * 60)
    print("ANONYMISATION RESULTS")
    print("=" * 60)
    print(f"  Images processed:     {stats.get('processed', 0)}")
    print(f"  Errors:               {stats.get('errors', 0)}")
    zones = stats.get('zones_applied', 0)
    if zones:
        print(f"  Static zones applied: {zones}")
    print()
    print("  Anonymisation tiers:")
    print(f"    Tier 1   (face, high conf):       {stats.get('tier1_face', 0)}")
    reclassified = stats.get('reclassified_person_to_face', 0)
    if reclassified:
        print(f"    Reclassified person→face:          {reclassified}")
    print(f"    Tier 2   (face, low + buffer):    {stats.get('tier2_face_buffered', 0)}")
    print(f"    Tier 3   (person, no face):       {stats.get('tier3_body_blur', 0)}")
    print(f"    Text/logo blurred:                {stats.get('text_or_logo', 0)}")
    t_total = sum(stats.get(k, 0) for k in [
        "tier1_face", "reclassified_person_to_face",
        "tier2_face_buffered", "tier3_body_blur", "text_or_logo",
    ])
    print(f"    Total blur operations:            {t_total}")
    print()
    print("  Detections per class:")
    for cls in CLASS_NAMES:
        count = stats.get(f"det_{cls}", 0)
        if count > 0:
            print(f"    {cls:20s}: {count}")
    print()
    print(f"  Output: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
