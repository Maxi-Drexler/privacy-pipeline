"""
Privacy-preserving anonymisation engine for construction site imagery.

Detects privacy-sensitive regions (faces, text, logos) using trained
YOLO11 models and applies Gaussian blur to protect personal data
in accordance with GDPR Article 5(1)(c).

Three-tier confidence-based anonymisation:
  Tier 1: High-confidence face — targeted face blur.
  Tier 2: Low-confidence face — face blur with doubled safety buffer.
  Tier 3: Person without detected face — blur upper 33% of person box.

Preprocessing: Small person detections in the head region of a larger
person are reclassified as faces before tier assignment.

Additional rules:
  - text_or_logo: Always blurred regardless of tier.
  - vehicle, crane, container, scaffolding, material_stack: Never blurred.
"""

import os
import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

logger = logging.getLogger(__name__)

CLASSES_ALWAYS_BLUR = {"face", "text_or_logo"}
CLASSES_NEVER_BLUR = {"vehicle", "crane", "container", "scaffolding", "material_stack"}


class Anonymizer:
    """Detection-based image anonymisation with three-tier blur strategy.

    Attributes:
        model_path: Path to trained YOLO11 weights.
        blur_method: Blurring method ('gaussian', 'pixelate', 'black').
        blur_strength: Kernel size for Gaussian blur (must be odd).
        body_blur_strength: Kernel size for tier-3 body blur (lighter).
        padding_percent: Percentage to expand detection boxes.
        confidence: Minimum detection confidence.
        body_anonymize: Enable tier-3 body blur for persons without faces.
        face_high_thresh: Confidence threshold separating tier 1 and 2.
        body_blur_top_pct: Fraction of person box to blur from top in tier 3.
    """

    def __init__(self, model_path, blur_method="gaussian", blur_strength=51,
                 body_blur_strength=31, padding_percent=10, confidence=0.25,
                 body_anonymize=True, face_high_thresh=0.5,
                 body_blur_top_pct=0.33):
        """Initialise the anonymiser.

        Args:
            model_path: Path to trained YOLO11 .pt weights file.
            blur_method: One of 'gaussian', 'pixelate', 'black'.
            blur_strength: Gaussian kernel for faces/text (strong).
            body_blur_strength: Gaussian kernel for tier-3 body blur.
            padding_percent: Expand boxes by this percentage.
            confidence: Detection confidence threshold.
            body_anonymize: Enable tier-3 body blur.
            face_high_thresh: Tier-1 face confidence threshold.
            body_blur_top_pct: Fraction of person box to blur in tier 3.
        """
        self.model_path = model_path
        self.blur_method = blur_method
        self.blur_strength = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
        self.body_blur_strength = body_blur_strength if body_blur_strength % 2 == 1 else body_blur_strength + 1
        self.padding_percent = padding_percent
        self.confidence = confidence
        self.body_anonymize = body_anonymize
        self.face_high_thresh = face_high_thresh
        self.body_blur_top_pct = body_blur_top_pct
        self.model = None

    def load_model(self):
        """Load the trained YOLO11 detection model."""
        logger.info("Loading model from %s", self.model_path)
        self.model = YOLO(self.model_path)
        logger.info("Model loaded. Classes: %s", self.model.names)

    def _pad_box(self, box, img_w, img_h, padding_override=None):
        """Expand a bounding box by padding percentage.

        Args:
            box: [x1, y1, x2, y2] coordinates.
            img_w: Image width.
            img_h: Image height.
            padding_override: Override default padding percentage.

        Returns:
            Padded and clamped (x1, y1, x2, y2) as integers.
        """
        x1, y1, x2, y2 = box
        pct = padding_override if padding_override is not None else self.padding_percent
        bw, bh = x2 - x1, y2 - y1
        px = bw * pct / 100
        py = bh * pct / 100
        return (
            max(0, int(x1 - px)), max(0, int(y1 - py)),
            min(img_w, int(x2 + px)), min(img_h, int(y2 + py)),
        )

    def _apply_blur(self, image, x1, y1, x2, y2, strength=None):
        """Apply blurring to a region of the image.

        Args:
            image: Input image (modified in place).
            x1, y1, x2, y2: Region coordinates.
            strength: Override blur kernel size.

        Returns:
            Image with blurred region.
        """
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return image

        k = strength if strength else self.blur_strength

        if self.blur_method == "gaussian":
            blurred = cv2.GaussianBlur(roi, (k, k), 0)
        elif self.blur_method == "pixelate":
            h, w = roi.shape[:2]
            factor = max(1, min(w, h) // 10)
            small = cv2.resize(roi, (max(1, w // factor), max(1, h // factor)),
                               interpolation=cv2.INTER_LINEAR)
            blurred = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        elif self.blur_method == "black":
            blurred = np.zeros_like(roi)
        else:
            blurred = cv2.GaussianBlur(roi, (k, k), 0)

        image[y1:y2, x1:x2] = blurred
        return image

    def _reclassify_head_persons(self, detections):
        """Reclassify small person boxes in head region as faces.

        Args:
            detections: List of detection dicts.

        Returns:
            Updated detections with reclassified entries.
        """
        persons = [(i, d) for i, d in enumerate(detections) if d["cls_name"] == "person"]
        reclassify_indices = set()

        for idx_s, small_p in persons:
            sb = small_p["bbox"]
            sa = (sb[2] - sb[0]) * (sb[3] - sb[1])
            for idx_l, large_p in persons:
                if idx_s == idx_l:
                    continue
                lb = large_p["bbox"]
                la = (lb[2] - lb[0]) * (lb[3] - lb[1])
                if sa >= la * 0.5:
                    continue
                ix1, iy1 = max(sb[0], lb[0]), max(sb[1], lb[1])
                ix2, iy2 = min(sb[2], lb[2]), min(sb[3], lb[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                if sa > 0 and inter / sa < 0.5:
                    continue
                head_cutoff = lb[1] + (lb[3] - lb[1]) * 0.40
                if (sb[1] + sb[3]) / 2 <= head_cutoff:
                    reclassify_indices.add(idx_s)
                    break

        for idx in reclassify_indices:
            detections[idx]["cls_name"] = "face"

        return detections, len(reclassify_indices)

    def anonymize_image(self, image_path, output_path):
        """Detect and anonymise privacy-sensitive regions in a single image.

        Args:
            image_path: Path to input image.
            output_path: Path for anonymised output image.

        Returns:
            Dict with detection counts and anonymisation statistics.
        """
        image = cv2.imread(image_path)
        if image is None:
            logger.warning("Could not read: %s", image_path)
            return {"error": "unreadable"}

        h, w = image.shape[:2]
        results = self.model(image_path, conf=self.confidence, verbose=False)

        stats = {
            "total_detections": 0,
            "tier1_face": 0,
            "tier2_face_buffered": 0,
            "tier3_body_blur": 0,
            "text_or_logo": 0,
            "reclassified_person_to_face": 0,
            "classes": {},
        }

        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                detections.append({
                    "cls_name": cls_name,
                    "bbox": box.xyxy[0].tolist(),
                    "conf": float(box.conf[0]),
                })
                stats["total_detections"] += 1
                stats["classes"][cls_name] = stats["classes"].get(cls_name, 0) + 1

        detections, n_reclassified = self._reclassify_head_persons(detections)
        stats["reclassified_person_to_face"] = n_reclassified

        faces = [d for d in detections if d["cls_name"] == "face"]
        persons = [d for d in detections if d["cls_name"] == "person"]
        text_logos = [d for d in detections if d["cls_name"] == "text_or_logo"]

        person_has_face = [False] * len(persons)
        for face in faces:
            fb = face["bbox"]
            fcx, fcy = (fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2
            for i, person in enumerate(persons):
                pb = person["bbox"]
                if pb[0] <= fcx <= pb[2] and pb[1] <= fcy <= pb[3]:
                    person_has_face[i] = True
                    break

        for face in faces:
            fb = face["bbox"]
            if face["conf"] >= self.face_high_thresh:
                x1, y1, x2, y2 = self._pad_box(fb, w, h)
                self._apply_blur(image, x1, y1, x2, y2)
                stats["tier1_face"] += 1
            else:
                x1, y1, x2, y2 = self._pad_box(fb, w, h,
                                                 padding_override=self.padding_percent * 2)
                self._apply_blur(image, x1, y1, x2, y2)
                stats["tier2_face_buffered"] += 1

        if self.body_anonymize:
            for i, person in enumerate(persons):
                if person_has_face[i]:
                    continue
                pb = person["bbox"]
                top_y2 = pb[1] + (pb[3] - pb[1]) * self.body_blur_top_pct
                x1, y1, x2, y2 = self._pad_box(
                    [pb[0], pb[1], pb[2], top_y2], w, h,
                    padding_override=self.padding_percent // 2,
                )
                self._apply_blur(image, x1, y1, x2, y2,
                                 strength=self.body_blur_strength)
                stats["tier3_body_blur"] += 1

        for tl in text_logos:
            x1, y1, x2, y2 = self._pad_box(tl["bbox"], w, h)
            self._apply_blur(image, x1, y1, x2, y2)
            stats["text_or_logo"] += 1

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image)

        return stats

    def run(self, input_dir, output_dir, max_images=None):
        """Anonymise all images in a directory.

        Args:
            input_dir: Source image directory.
            output_dir: Destination for anonymised images.
            max_images: Process at most this many (for testing).

        Returns:
            Aggregated statistics.
        """
        self.load_model()

        input_path = Path(input_dir)
        output_path = Path(output_dir)

        extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        images = []
        for ext in extensions:
            images.extend(input_path.rglob(f"*{ext}"))
        images = sorted(set(images))

        if max_images:
            images = images[:max_images]

        logger.info("Anonymising %d images...", len(images))

        total_stats = {
            "processed": 0,
            "blurred_regions": 0,
            "body_blurred_regions": 0,
            "class_totals": {},
        }

        for img in tqdm(images, desc="Anonymising"):
            rel = img.relative_to(input_path)
            out = output_path / rel

            stats = self.anonymize_image(str(img), str(out))

            if "error" not in stats:
                total_stats["processed"] += 1
                total_stats["blurred_regions"] += (stats["tier1_face"] +
                                                    stats["tier2_face_buffered"] +
                                                    stats["text_or_logo"])
                total_stats["body_blurred_regions"] += stats["tier3_body_blur"]
                for cls, count in stats.get("classes", {}).items():
                    total_stats["class_totals"][cls] = \
                        total_stats["class_totals"].get(cls, 0) + count

        logger.info("Anonymisation complete: %s", total_stats)
        return total_stats
