#!/usr/bin/env python3
"""
Stage 1: Pre-filter construction site images by quality metrics.

Applies three assessment dimensions to remove unsuitable images from the
raw dataset. No machine learning model is used at this stage; all metrics
are computed from pixel statistics alone.

Assessment dimensions (aligned with thesis Chapter 4):
  1. Exposure analysis: Combines mean brightness and RMS contrast to
     reject underexposed (night), overexposed (glare), and flat images.
  2. Sharpness: Laplacian variance measures high-frequency content;
     low values indicate motion blur or defocus.
  3. Edge density: Canny edge pixel ratio detects structureless frames
     (e.g. fog, lens obstruction, uniform sky).

Instead of copying accepted images, this module produces index files:
  - accepted_images.csv: Path, brightness, contrast, sharpness, edge_density
  - rejected_images.csv: Path, rejection reason(s), metrics
  - prefilter_stats.json: Aggregated statistics

Subsequent pipeline stages read accepted_images.csv to locate images
in their original directory, avoiding duplication of ~437k files.

Usage:
    python scripts/01_prefilter.py \
        --input-dir /workspace/data/TUM_KITA_imgs \
        --output-dir /workspace/data/prefiltered

Author: Maximilian Drexler
License: MIT
"""

import os
import csv
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial

import cv2
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def load_zone_config(config_path):
    """Load zone config and extract crop percentages per setup.

    Args:
        config_path: Path to zones JSON file.

    Returns:
        Dict mapping setup names to crop percentages.
    """
    if not config_path or not os.path.exists(config_path):
        return {}

    with open(config_path) as f:
        raw = json.load(f)

    crops = {}
    for setup, cfg in raw.items():
        c = {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0}
        for zone in cfg.get("zones", []):
            poly = zone.get("polygon", "")
            if isinstance(poly, str):
                if poly.startswith("full_width_top_"):
                    c["top"] = max(c["top"], int(poly.split("_")[-1]) / 100)
                elif poly.startswith("full_width_bottom_"):
                    c["bottom"] = max(c["bottom"], int(poly.split("_")[-1]) / 100)
                elif poly.startswith("left_"):
                    c["left"] = max(c["left"], int(poly.split("_")[-1]) / 100)
                elif poly.startswith("right_"):
                    c["right"] = max(c["right"], int(poly.split("_")[-1]) / 100)
        crops[setup] = c
    return crops


def detect_setup(filename):
    """Detect camera setup from filename.

    Args:
        filename: Image filename.

    Returns:
        Setup key string or None.
    """
    import re
    match = re.search(r'(?:kamera|setup_?|cam)(\d+)', filename.lower())
    if match:
        return f"setup_{match.group(1)}"
    return None


def crop_zones(image, crop_pcts):
    """Crop image by removing zone margins.

    Args:
        image: OpenCV BGR image.
        crop_pcts: Dict with top/bottom/left/right as fractions.

    Returns:
        Cropped image.
    """
    h, w = image.shape[:2]
    y1 = int(h * crop_pcts.get("top", 0))
    y2 = h - int(h * crop_pcts.get("bottom", 0))
    x1 = int(w * crop_pcts.get("left", 0))
    x2 = w - int(w * crop_pcts.get("right", 0))
    return image[y1:y2, x1:x2]


def compute_brightness(image):
    """Compute mean brightness from the V channel of HSV.

    Args:
        image: OpenCV BGR image.

    Returns:
        Mean brightness as float (0-255).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 2]))


def compute_contrast(image):
    """Compute RMS contrast from grayscale image.

    Args:
        image: OpenCV BGR image.

    Returns:
        RMS contrast as float.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray.astype(np.float64)))


def compute_sharpness(image):
    """Compute sharpness via Laplacian variance.

    Args:
        image: OpenCV BGR image.

    Returns:
        Laplacian variance as float.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_edge_density(image, low_threshold=50, high_threshold=150):
    """Compute edge pixel ratio via Canny edge detection.

    Args:
        image: OpenCV BGR image.
        low_threshold: Canny low threshold.
        high_threshold: Canny high threshold.

    Returns:
        Ratio of edge pixels to total pixels (0.0 - 1.0).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return float(np.count_nonzero(edges) / edges.size)


def assess_image(image_path, min_brightness=50.0, max_brightness=230.0,
                 min_contrast=25.0, min_sharpness=200.0, min_edge_density=0.02,
                 crop_pcts=None):
    """Assess a single image across all three quality dimensions.

    If crop_pcts is provided, zone margins are removed before computing
    metrics. This ensures sky, edges, and irrelevant areas do not
    influence the quality assessment.

    Args:
        image_path: Path to image file.
        min_brightness: Minimum mean brightness threshold.
        max_brightness: Maximum mean brightness threshold.
        min_contrast: Minimum RMS contrast threshold.
        min_sharpness: Minimum Laplacian variance threshold.
        min_edge_density: Minimum Canny edge pixel ratio.
        crop_pcts: Dict with top/bottom/left/right crop fractions.

    Returns:
        Tuple of (accepted: bool, metrics: dict, reasons: list).
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return False, {}, ["unreadable"]

    if crop_pcts:
        image = crop_zones(image, crop_pcts)

    brightness = compute_brightness(image)
    contrast = compute_contrast(image)
    sharpness = compute_sharpness(image)
    edge_density = compute_edge_density(image)

    metrics = {
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 1),
        "sharpness": round(sharpness, 1),
        "edge_density": round(edge_density, 4),
    }

    reasons = []

    if brightness < min_brightness:
        reasons.append("too_dark")
    if brightness > max_brightness:
        reasons.append("overexposed")
    if contrast < min_contrast:
        reasons.append("low_contrast")
    if sharpness < min_sharpness:
        reasons.append("blurry")
    if edge_density < min_edge_density:
        reasons.append("low_edge_density")

    return len(reasons) == 0, metrics, reasons


def collect_images(input_dir):
    """Recursively collect all image paths from input directory.

    Args:
        input_dir: Root directory to scan.

    Returns:
        Sorted list of Path objects.
    """
    input_path = Path(input_dir)
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(input_path.rglob(f"*{ext}"))
        images.extend(input_path.rglob(f"*{ext.upper()}"))
    return sorted(set(images))


def _assess_wrapper(args_tuple):
    """Wrapper for multiprocessing.

    Args:
        args_tuple: (image_path, min_brightness, max_brightness,
                     min_contrast, min_sharpness, min_edge_density, crop_pcts).

    Returns:
        Tuple of (image_path_str, accepted, metrics, reasons).
    """
    img_path, min_b, max_b, min_c, min_s, min_e, crop = args_tuple
    accepted, metrics, reasons = assess_image(
        img_path, min_b, max_b, min_c, min_s, min_e, crop_pcts=crop
    )
    return str(img_path), accepted, metrics, reasons


def run_prefilter(input_dir, output_dir, min_brightness=50.0, max_brightness=230.0,
                  min_contrast=25.0, min_sharpness=200.0, min_edge_density=0.02,
                  max_images=None, workers=None, zones_config_path=None):
    """Run pre-filtering on all images using multiprocessing.

    Args:
        input_dir: Directory with raw images (supports nested structure).
        output_dir: Directory for index files and stats.
        min_brightness: Minimum brightness threshold.
        max_brightness: Maximum brightness threshold.
        min_contrast: Minimum RMS contrast threshold.
        min_sharpness: Minimum Laplacian variance threshold.
        min_edge_density: Minimum Canny edge pixel ratio.
        max_images: Process at most N images (for testing).
        workers: Number of parallel workers (default: cpu_count - 1).
        zones_config_path: Path to zones.json for cropping before assessment.

    Returns:
        Dict with filtering statistics.
    """
    os.makedirs(output_dir, exist_ok=True)

    images = collect_images(input_dir)
    if max_images:
        images = images[:max_images]

    if workers is None:
        workers = max(1, cpu_count() - 1)

    zone_crops = load_zone_config(zones_config_path)
    if zone_crops:
        log.info("Zone crops loaded for %d setup(s): %s",
                 len(zone_crops), list(zone_crops.keys()))

    log.info("Found %d images in %s", len(images), input_dir)
    log.info("Using %d worker processes", workers)

    stats = {
        "total": len(images),
        "accepted": 0,
        "rejected_too_dark": 0,
        "rejected_overexposed": 0,
        "rejected_low_contrast": 0,
        "rejected_blurry": 0,
        "rejected_low_edge_density": 0,
        "rejected_unreadable": 0,
    }

    acc_path = os.path.join(output_dir, "accepted_images.csv")
    rej_path = os.path.join(output_dir, "rejected_images.csv")

    task_args = []
    for img in images:
        setup = detect_setup(img.name) if zone_crops else None
        crop = zone_crops.get(setup) if setup else None
        task_args.append((
            img, min_brightness, max_brightness, min_contrast,
            min_sharpness, min_edge_density, crop
        ))

    with open(acc_path, "w", newline="") as acc_f, \
         open(rej_path, "w", newline="") as rej_f:

        acc_writer = csv.writer(acc_f)
        acc_writer.writerow(["path", "brightness", "contrast", "sharpness", "edge_density"])

        rej_writer = csv.writer(rej_f)
        rej_writer.writerow(["path", "reason", "brightness", "contrast",
                             "sharpness", "edge_density"])

        with Pool(processes=workers) as pool:
            for path_str, accepted, metrics, reasons in tqdm(
                pool.imap_unordered(_assess_wrapper, task_args, chunksize=64),
                total=len(images),
                desc=f"Pre-filtering ({workers} workers)",
            ):
                if accepted:
                    stats["accepted"] += 1
                    acc_writer.writerow([
                        path_str,
                        metrics["brightness"],
                        metrics["contrast"],
                        metrics["sharpness"],
                        metrics["edge_density"],
                    ])
                else:
                    for reason in reasons:
                        key = f"rejected_{reason}"
                        if key in stats:
                            stats[key] += 1
                    rej_writer.writerow([
                        path_str,
                        ";".join(reasons),
                        metrics.get("brightness", ""),
                        metrics.get("contrast", ""),
                        metrics.get("sharpness", ""),
                        metrics.get("edge_density", ""),
                    ])

    stats_path = os.path.join(output_dir, "prefilter_stats.json")
    with open(stats_path, "w") as f:
        json.dump({
            "stats": stats,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "min_brightness": min_brightness,
                "max_brightness": max_brightness,
                "min_contrast": min_contrast,
                "min_sharpness": min_sharpness,
                "min_edge_density": min_edge_density,
            },
            "output_files": {
                "accepted_index": acc_path,
                "rejected_index": rej_path,
            },
        }, f, indent=2)

    log.info("Pre-filtering complete: %s", stats)
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Pre-filter construction site images by quality metrics."
    )
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory with raw images (supports nested structure).")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory for index files and stats.")
    parser.add_argument("--min-brightness", type=float, default=50.0,
                        help="Minimum brightness threshold (default: 50).")
    parser.add_argument("--max-brightness", type=float, default=230.0,
                        help="Maximum brightness threshold (default: 230).")
    parser.add_argument("--min-contrast", type=float, default=25.0,
                        help="Minimum RMS contrast threshold (default: 25).")
    parser.add_argument("--min-sharpness", type=float, default=200.0,
                        help="Minimum Laplacian variance (default: 200).")
    parser.add_argument("--min-edge-density", type=float, default=0.02,
                        help="Minimum Canny edge pixel ratio (default: 0.02).")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Process at most N images (for testing).")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: cpu_count - 1).")
    parser.add_argument("--zones-config", type=str, default=None,
                        help="Path to zones.json — crops zone margins before assessment.")
    args = parser.parse_args()

    stats = run_prefilter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_brightness=args.min_brightness,
        max_brightness=args.max_brightness,
        min_contrast=args.min_contrast,
        min_sharpness=args.min_sharpness,
        min_edge_density=args.min_edge_density,
        max_images=args.max_images,
        workers=args.workers,
        zones_config_path=args.zones_config,
    )

    print("\n" + "=" * 60)
    print("PRE-FILTER RESULTS")
    print("=" * 60)
    print(f"  Total images:             {stats['total']}")
    print(f"  Accepted:                 {stats['accepted']}")
    print(f"  Acceptance rate:          {stats['accepted']/max(stats['total'],1)*100:.1f}%")
    print()
    print("  Rejection breakdown:")
    print(f"    Too dark:               {stats['rejected_too_dark']}")
    print(f"    Overexposed:            {stats['rejected_overexposed']}")
    print(f"    Low contrast:           {stats['rejected_low_contrast']}")
    print(f"    Blurry:                 {stats['rejected_blurry']}")
    print(f"    Low edge density:       {stats['rejected_low_edge_density']}")
    print(f"    Unreadable:             {stats['rejected_unreadable']}")
    print()
    print(f"  Output: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
