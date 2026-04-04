"""
Intelligent dataset curation through score-based image selection.

Implements detection-based scoring with temporal diversity constraints
to select optimal training images from large construction site datasets.

A pretrained YOLO11 model serves as a proxy scorer using batch inference.
A random baseline is included for comparison to validate that intelligent
scoring outperforms naive random subsampling at the same budget.
"""

import os
import json
import csv
import shutil
import random
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class IntelligentImageSelector:
    """Score-based image selection for optimal training set construction.

    Scores each image based on detection relevance (person/vehicle counts)
    and temporal diversity (minimum gap, max per hour). Quality filtering
    is handled by Stage 1; this stage focuses on content relevance.

    Attributes:
        source_dir: Directory containing candidate images.
        target_dir: Directory for selected images.
        num_samples: Target number of images to select.
        min_gap_minutes: Minimum time gap between selected images.
        max_per_hour: Maximum images from any single hour.
        person_weight: Scoring weight for person detections.
        vehicle_weight: Scoring weight for vehicle detections.
    """

    def __init__(self, source_dir, target_dir, num_samples=2500,
                 min_gap_minutes=3, max_per_hour=5,
                 person_weight=10.0, vehicle_weight=5.0):
        """Initialise the selector.

        Args:
            source_dir: Path to pre-filtered images.
            target_dir: Path for output selected images.
            num_samples: Number of images to select.
            min_gap_minutes: Minimum minutes between selections.
            max_per_hour: Maximum selections per clock hour.
            person_weight: Score multiplier for person count.
            vehicle_weight: Score multiplier for vehicle count.
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.num_samples = num_samples
        self.min_gap = timedelta(minutes=min_gap_minutes)
        self.max_per_hour = max_per_hour
        self.person_weight = person_weight
        self.vehicle_weight = vehicle_weight

        self.model = None
        self.relevant_classes = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    def _load_model(self):
        """Load YOLO11 COCO pretrained model for relevance scoring."""
        if self.model is None:
            self.model = YOLO("yolo11n.pt")
            logger.info("Loaded YOLO11 nano for scoring")

    def _parse_timestamp(self, filename):
        """Extract timestamp from construction site image filename.

        Expected pattern: Kamera[N]_00_YYYYMMDDHHMMSS.jpg

        Args:
            filename: Image filename.

        Returns:
            datetime object or None if parsing fails.
        """
        try:
            parts = filename.replace(".jpg", "").replace(".jpeg", "").replace(".png", "")
            ts_part = parts.split("_")[-1]
            return datetime.strptime(ts_part, "%Y%m%d%H%M%S")
        except (ValueError, IndexError):
            return None

    def _apply_temporal_diversity(self, scored):
        """Apply temporal constraints to ensure diverse selection.

        Enforces minimum time gap and maximum selections per hour.

        Args:
            scored: List of scored image dicts, sorted by score descending.

        Returns:
            Filtered list respecting temporal constraints.
        """
        selected = []
        hour_counts = defaultdict(int)

        for item in scored:
            ts = item.get("timestamp")
            if ts is None:
                selected.append(item)
                if len(selected) >= self.num_samples:
                    break
                continue

            hour_key = ts.strftime("%Y%m%d_%H")
            if hour_counts[hour_key] >= self.max_per_hour:
                continue

            too_close = False
            for s in selected:
                s_ts = s.get("timestamp")
                if s_ts and abs((ts - s_ts).total_seconds()) < self.min_gap.total_seconds():
                    too_close = True
                    break

            if not too_close:
                selected.append(item)
                hour_counts[hour_key] += 1

            if len(selected) >= self.num_samples:
                break

        return selected

    def _select_random_baseline(self, images):
        """Select images randomly as a comparison baseline.

        Args:
            images: Full list of candidate image paths.

        Returns:
            List of randomly selected image info dicts.
        """
        sample_size = min(self.num_samples, len(images))
        indices = random.sample(list(range(len(images))), sample_size)

        return [{
            "path": str(images[idx]),
            "score": 0.0,
            "person_count": 0,
            "vehicle_count": 0,
            "timestamp": self._parse_timestamp(images[idx].name),
        } for idx in indices]

    def run(self, max_score_images=None, index_file=None):
        """Execute the intelligent selection pipeline.

        Args:
            max_score_images: Score at most this many images (for testing).
            index_file: Path to accepted_images.csv from Stage 1.
                Reads image paths from CSV instead of scanning directory.

        Returns:
            Statistics dictionary with selection results.
        """
        self._load_model()
        self.target_dir.mkdir(parents=True, exist_ok=True)

        if index_file and Path(index_file).exists():
            images = []
            with open(index_file) as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if row:
                        images.append(Path(row[0]))
            logger.info("Loaded %d images from index file: %s", len(images), index_file)
        else:
            extensions = {".jpg", ".jpeg", ".png", ".bmp"}
            images = []
            for ext in extensions:
                images.extend(self.source_dir.rglob(f"*{ext}"))
            images = sorted(set(images))

        if max_score_images:
            images = images[:max_score_images]

        logger.info("Scoring %d images...", len(images))
        scored = []
        batch_size = 64
        image_strs = [str(p) for p in images]

        from concurrent.futures import ThreadPoolExecutor
        import cv2

        def load_image(path):
            """Load and return image from disk."""
            img = cv2.imread(path)
            return img

        with ThreadPoolExecutor(max_workers=8) as executor:
            for batch_start in tqdm(range(0, len(images), batch_size),
                                    desc=f"Scoring (batch={batch_size}, 8 loaders)",
                                    total=(len(images) + batch_size - 1) // batch_size):
                batch_end = min(batch_start + batch_size, len(images))
                batch_paths = image_strs[batch_start:batch_end]
                batch_images_meta = images[batch_start:batch_end]

                loaded = list(executor.map(load_image, batch_paths))

                valid_imgs = []
                valid_meta = []
                for img, meta in zip(loaded, batch_images_meta):
                    if img is not None:
                        valid_imgs.append(img)
                        valid_meta.append(meta)

                if not valid_imgs:
                    continue

                results = self.model(valid_imgs, conf=0.25, verbose=False)

                for img_path, result in zip(valid_meta, results):
                    person_count = 0
                    vehicle_count = 0
                    if result.boxes is not None:
                        for box in result.boxes:
                            cls_id = int(box.cls[0])
                            if cls_id == 0:
                                person_count += 1
                            elif cls_id in [2, 3, 5, 7]:
                                vehicle_count += 1

                    scored.append({
                        "path": str(img_path),
                        "score": (person_count * self.person_weight +
                                  vehicle_count * self.vehicle_weight),
                        "person_count": person_count,
                        "vehicle_count": vehicle_count,
                        "timestamp": self._parse_timestamp(img_path.name),
                    })

        scored.sort(key=lambda x: x["score"], reverse=True)

        logger.info("Applying temporal diversity constraints...")
        selected = self._apply_temporal_diversity(scored)

        logger.info("Copying %d selected images...", len(selected))
        for item in tqdm(selected, desc="Copying"):
            src = Path(item["path"])
            shutil.copy2(str(src), str(self.target_dir / src.name))

        stats = {
            "total_scored": len(scored),
            "total_selected": len(selected),
            "avg_score": float(np.mean([s["score"] for s in selected])) if selected else 0,
            "avg_persons": float(np.mean([s["person_count"] for s in selected])) if selected else 0,
            "avg_vehicles": float(np.mean([s["vehicle_count"] for s in selected])) if selected else 0,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info("Generating random baseline for comparison...")
        random_baseline = self._select_random_baseline(images)
        stats["random_baseline"] = {"count": len(random_baseline)}

        stats_path = self.target_dir / "selection_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2, default=str)

        baseline_path = self.target_dir / "random_baseline_paths.txt"
        with open(baseline_path, "w") as f:
            for item in random_baseline:
                f.write(item["path"] + "\n")

        logger.info("Selection complete: %s", stats)
        return stats
