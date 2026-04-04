"""
Pre-filtering module for construction site image datasets.

Applies three quality assessment dimensions to remove unsuitable images.
No machine learning model is used at this stage; all metrics are computed
from pixel statistics alone.

Assessment dimensions:
  1. Exposure analysis: Mean brightness and RMS contrast reject
     underexposed (night), overexposed (glare), and flat images.
  2. Sharpness: Laplacian variance measures high-frequency content;
     low values indicate motion blur or defocus.
  3. Edge density: Canny edge pixel ratio detects structureless frames
     (fog, lens obstruction, uniform sky).

Produces an index file (accepted_images.csv) rather than copying images.
"""

import os
import csv
import json
import logging
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


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


class ImagePrefilter:
    """Filters construction site images by quality metrics only.

    No ML model is used. Three assessment dimensions: exposure
    (brightness + contrast), sharpness (Laplacian), edge density (Canny).

    Attributes:
        input_dir: Source directory containing raw images.
        output_dir: Destination for index files and logs.
        min_brightness: Minimum brightness threshold.
        max_brightness: Maximum brightness threshold.
        min_contrast: Minimum RMS contrast threshold.
        min_sharpness: Minimum sharpness threshold.
        min_edge_density: Minimum Canny edge pixel ratio.
    """

    def __init__(self, input_dir, output_dir, min_brightness=30.0,
                 max_brightness=250.0, min_contrast=15.0,
                 min_sharpness=50.0, min_edge_density=0.01):
        """Initialise the pre-filter.

        Args:
            input_dir: Path to directory with raw images (supports nested).
            output_dir: Path for output index files and logs.
            min_brightness: Reject images below this brightness.
            max_brightness: Reject images above this brightness.
            min_contrast: Reject images below this RMS contrast.
            min_sharpness: Reject images below this Laplacian variance.
            min_edge_density: Reject images below this Canny edge ratio.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast
        self.min_sharpness = min_sharpness
        self.min_edge_density = min_edge_density

        self.stats = {
            "total": 0,
            "accepted": 0,
            "rejected_too_dark": 0,
            "rejected_overexposed": 0,
            "rejected_low_contrast": 0,
            "rejected_blurry": 0,
            "rejected_low_edge_density": 0,
            "rejected_unreadable": 0,
        }

        os.makedirs(self.output_dir, exist_ok=True)

    def _collect_images(self):
        """Recursively collect all image files from the input directory.

        Returns:
            Sorted list of image file paths.
        """
        images = []
        for ext in IMAGE_EXTENSIONS:
            images.extend(self.input_dir.rglob(f"*{ext}"))
            images.extend(self.input_dir.rglob(f"*{ext.upper()}"))
        return sorted(set(images))

    def _assess_image(self, image_path):
        """Assess a single image across all three quality dimensions.

        Args:
            image_path: Path to image file.

        Returns:
            Tuple of (accepted, metrics_dict, rejection_reasons).
        """
        image = cv2.imread(str(image_path))
        if image is None:
            return False, {}, ["unreadable"]

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
        if brightness < self.min_brightness:
            reasons.append("too_dark")
        if brightness > self.max_brightness:
            reasons.append("overexposed")
        if contrast < self.min_contrast:
            reasons.append("low_contrast")
        if sharpness < self.min_sharpness:
            reasons.append("blurry")
        if edge_density < self.min_edge_density:
            reasons.append("low_edge_density")

        return len(reasons) == 0, metrics, reasons

    def run(self, max_images=None):
        """Execute the pre-filtering pipeline.

        Args:
            max_images: Process at most this many images (for testing).

        Returns:
            Statistics dictionary with filtering results.
        """
        images = self._collect_images()
        if max_images:
            images = images[:max_images]

        self.stats["total"] = len(images)
        logger.info("Total images: %d", len(images))

        csv_path = self.output_dir / "accepted_images.csv"
        rejected_path = self.output_dir / "rejected_images.csv"

        with open(csv_path, "w", newline="") as f_acc, \
             open(rejected_path, "w", newline="") as f_rej:

            acc_writer = csv.writer(f_acc)
            acc_writer.writerow(["path", "brightness", "contrast", "sharpness", "edge_density"])

            rej_writer = csv.writer(f_rej)
            rej_writer.writerow(["path", "reason", "brightness", "contrast",
                                 "sharpness", "edge_density"])

            for img_path in tqdm(images, desc="Pre-filtering"):
                accepted, metrics, reasons = self._assess_image(img_path)

                if accepted:
                    self.stats["accepted"] += 1
                    acc_writer.writerow([
                        str(img_path),
                        metrics["brightness"], metrics["contrast"],
                        metrics["sharpness"], metrics["edge_density"],
                    ])
                else:
                    for reason in reasons:
                        key = f"rejected_{reason}"
                        if key in self.stats:
                            self.stats[key] += 1
                    rej_writer.writerow([
                        str(img_path), ";".join(reasons),
                        metrics.get("brightness", ""), metrics.get("contrast", ""),
                        metrics.get("sharpness", ""), metrics.get("edge_density", ""),
                    ])

        stats_path = self.output_dir / "prefilter_stats.json"
        with open(stats_path, "w") as f:
            json.dump({
                "stats": self.stats,
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "min_brightness": self.min_brightness,
                    "max_brightness": self.max_brightness,
                    "min_contrast": self.min_contrast,
                    "min_sharpness": self.min_sharpness,
                    "min_edge_density": self.min_edge_density,
                },
            }, f, indent=2)

        logger.info("Pre-filtering complete: %s", self.stats)
        return self.stats
