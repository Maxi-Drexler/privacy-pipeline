#!/usr/bin/env python3
"""
Stage 3: Multi-model auto-annotation with WBF fusion and attribute enrichment.

Generates comprehensive pseudo-labels for construction site images in a
single pass per image. Combines four detection models, fuses overlapping
boxes via Weighted Boxes Fusion (WBF), and enriches all annotations with
zero-shot attribute classification.

Per-image pipeline:
  1. Run detection models (YOLO11 COCO, YOLO-Face, YOLO-World, Grounding DINO)
  2. Weighted Boxes Fusion per class (with person merge guard)
  3. Post-fusion cleanup (reclassify head-region persons as face,
     cross-class duplicate removal with allowed containment)
  4. Per-class confidence filtering + aspect ratio + max-area filtering
  5. Zone filtering (exclude detections in masked image regions)
  6. Attribute enrichment (PPE, CLIP, OCR)
  7. Failsafes:
     a. Face rescue: YOLO-Face on person head crops + CLIP fallback
     b. License plate rescue: OCR on vehicle bottom crops
     c. Head detection: GDino with high confidence for human heads
  8. CLIP verification — remove false positives for vehicle, material_stack,
     text_or_logo, and container
  9. Output COCO JSON + YOLO TXT + CVAT import ZIP

Optional SAHI mode (--use-sahi):
  Slices each image into overlapping tiles (default 640x640, 25% overlap),
  runs the full multi-model pipeline per tile, maps coordinates back to
  the original image, then runs a second WBF pass across all tiles.
  Enrichment runs on the merged full-image detections only. Also runs
  a full-image pass and merges those detections with the tile results.

Models:
    - Grounding DINO — open-vocabulary, Apache-2.0
    - YOLO-World — open-vocabulary, GPL-3.0
    - YOLO-Face — face specialist, GPL-3.0
    - YOLO11 — COCO pretrained, AGPL-3.0
    - OpenCLIP — zero-shot classification
    - EasyOCR — text recognition

Fusion:
    Weighted Boxes Fusion (WBF) replaces standard NMS. WBF averages
    coordinates across models weighted by confidence, producing more
    precise bounding boxes.

Author: Maximilian Drexler
License: MIT
"""

import json
import os
import sys
import time
import shutil
import zipfile
import argparse
import logging
import tempfile
import re
import numpy as np
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from collections import Counter

_GDINO_PATH = "/workspace/GroundingDINO"
if _GDINO_PATH not in sys.path:
    sys.path.insert(0, _GDINO_PATH)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# TAXONOMY
# ═══════════════════════════════════════════════════════════════════════════

CLASS_NAMES = [
    "face", "person", "vehicle", "text_or_logo",
    "crane", "container", "scaffolding", "material_stack",
]
CLASS_NAME_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

PER_CLASS_CONFIDENCE = {
    "face": 0.15,
    "person": 0.20,
    "vehicle": 0.36,
    "text_or_logo": 0.25,
    "crane": 0.35,
    "container": 0.35,
    "scaffolding": 0.30,
    "material_stack": 0.225,
}

PER_CLASS_MIN_AREA = {
    "face": 100,
    "person": 400,
    "vehicle": 400,
    "text_or_logo": 100,
    "crane": 400,
    "container": 400,
    "scaffolding": 400,
    "material_stack": 400,
}

MAX_ASPECT_RATIO = 6.0

COCO_TO_CLASS = {
    0: ("person", 1),
    1: ("vehicle", 2),
    2: ("vehicle", 2),
    7: ("vehicle", 2),
}

GROUNDING_DINO_PERSON_PROMPTS = "person . construction worker . pedestrian"
GROUNDING_DINO_VEHICLE_PROMPTS = (
    "car . truck . excavator . bicycle . van . "
    "concrete mixer truck . dump truck . forklift . "
    "delivery truck . flatbed truck"
)
GROUNDING_DINO_FACE_PROMPTS = (
    "human face . person's face . worker face"
)
GROUNDING_DINO_HEAD_PROMPTS = (
    "human head . person's head . worker head with helmet"
)
GROUNDING_DINO_TEXT_PROMPTS = (
    "sign . logo . company name . license plate . text . safety sign . "
    "banner . label . sticker . warning sign . number plate . brand name . "
    "information board . poster . nameplate . company logo on vehicle . "
    "company logo on container . text on fence . text on banner . "
    "construction company sign"
)
GROUNDING_DINO_CONSTRUCTION_PROMPTS = (
    "crane . tower crane . scaffolding . scaffold . "
    "large portable office container . large storage container . "
    "construction site container . portable cabin . "
    "stacked wood planks . stacked lumber . pile of wooden boards . "
    "stacked iron pipes . stacked metal tubes . "
    "pile of bricks . pallet of concrete blocks . "
    "stacked building materials on pallets"
)

GDINO_PHRASE_TO_CLASS = {
    "human head": ("face", 0), "person's head": ("face", 0),
    "worker head": ("face", 0), "head": ("face", 0),
    "excavator": ("vehicle", 2), "bicycle": ("vehicle", 2),
    "truck": ("vehicle", 2), "car": ("vehicle", 2),
    "information board": ("text_or_logo", 3),
    "license plate": ("text_or_logo", 3), "number plate": ("text_or_logo", 3),
    "company name": ("text_or_logo", 3), "warning sign": ("text_or_logo", 3),
    "safety sign": ("text_or_logo", 3), "brand name": ("text_or_logo", 3),
    "nameplate": ("text_or_logo", 3), "banner": ("text_or_logo", 3),
    "sticker": ("text_or_logo", 3), "poster": ("text_or_logo", 3),
    "label": ("text_or_logo", 3), "sign": ("text_or_logo", 3),
    "logo": ("text_or_logo", 3), "text": ("text_or_logo", 3),
    "shipping container": ("container", 5), "container": ("container", 5),
    "portable office container": ("container", 5),
    "large portable office container": ("container", 5),
    "large storage container": ("container", 5),
    "construction site container": ("container", 5),
    "portable cabin": ("container", 5),
    "storage container": ("container", 5),
    "tower crane": ("crane", 4), "crane": ("crane", 4),
    "scaffolding": ("scaffolding", 6), "scaffold": ("scaffolding", 6),
    "building materials": ("material_stack", 7),
    "material stack": ("material_stack", 7),
    "stacked wood planks": ("material_stack", 7),
    "stacked lumber": ("material_stack", 7),
    "pile of wooden boards": ("material_stack", 7),
    "stacked iron pipes": ("material_stack", 7),
    "stacked metal tubes": ("material_stack", 7),
    "pile of bricks": ("material_stack", 7),
    "pallet of concrete blocks": ("material_stack", 7),
    "stacked building materials on pallets": ("material_stack", 7),
    "van": ("vehicle", 2), "concrete mixer": ("vehicle", 2),
    "concrete mixer truck": ("vehicle", 2), "dump truck": ("vehicle", 2),
    "forklift": ("vehicle", 2), "delivery truck": ("vehicle", 2),
    "flatbed truck": ("vehicle", 2),
    "company logo": ("text_or_logo", 3),
    "construction company sign": ("text_or_logo", 3),
    "company logo on vehicle": ("text_or_logo", 3),
    "company logo on container": ("text_or_logo", 3),
    "text on fence": ("text_or_logo", 3),
    "text on banner": ("text_or_logo", 3),
    "storage container": ("container", 5),
    "pile of wood": ("material_stack", 7),
    "stacked pipes": ("material_stack", 7),
    "stacked bricks": ("material_stack", 7),
    "pile of construction materials": ("material_stack", 7),
    "pallets with materials": ("material_stack", 7),
    "stacked wood": ("material_stack", 7),
    "wooden boards": ("material_stack", 7),
    "iron pipes": ("material_stack", 7),
    "metal tubes": ("material_stack", 7),
    "concrete blocks": ("material_stack", 7),
}

GDINO_PHRASE_SORTED = sorted(
    GDINO_PHRASE_TO_CLASS.items(), key=lambda x: -len(x[0])
)

PHRASE_TO_CONTENT_TYPE = {
    "license plate": "license_plate", "number plate": "license_plate",
    "plate": "license_plate",
    "safety sign": "safety_sign", "warning sign": "safety_sign",
    "warning": "safety_sign",
    "logo": "company_name", "company": "company_name",
    "company name": "company_name", "brand": "company_name",
    "brand name": "company_name", "nameplate": "company_name",
    "sign": "other", "banner": "other", "label": "other",
    "sticker": "other", "text": "other",
    "information board": "other", "poster": "other",
}
PHRASE_IS_LOGO = {
    "logo", "company", "company name", "brand",
    "brand name", "banner", "nameplate",
}

ALLOWED_CONTAINMENT = {
    ("person", "face"),
    ("person", "text_or_logo"),
    ("vehicle", "text_or_logo"),
}

DEFAULT_ZONES = {
    "setup_1": {"top_pct": 0.25, "right_pct": 0.0, "bottom_pct": 0.0},
    "setup_2": {"top_pct": 0.25, "right_pct": 0.0, "bottom_pct": 0.0},
}

ZONE_FILTERED_CLASSES = {"face", "text_or_logo"}


# ═══════════════════════════════════════════════════════════════════════════
# GEOMETRY HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def bbox_xyxy_to_coco(bbox):
    """Convert [x1, y1, x2, y2] to COCO [x, y, w, h]."""
    return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]


def compute_iou(box1, box2):
    """IoU between two [x1,y1,x2,y2] boxes."""
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def box_contains(outer, inner, threshold=0.5):
    """Check if inner box is mostly contained within outer box."""
    x1, y1 = max(outer[0], inner[0]), max(outer[1], inner[1])
    x2, y2 = min(outer[2], inner[2]), min(outer[3], inner[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    inner_area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    if inner_area <= 0:
        return False
    return (inter / inner_area) >= threshold


def crop_bbox(image, bbox_xyxy, padding=10):
    """Crop region from PIL Image with padding."""
    w, h = image.size
    x1 = max(0, int(bbox_xyxy[0]) - padding)
    y1 = max(0, int(bbox_xyxy[1]) - padding)
    x2 = min(w, int(bbox_xyxy[2]) + padding)
    y2 = min(h, int(bbox_xyxy[3]) + padding)
    return image.crop((x1, y1, x2, y2))


def box_aspect_ratio(bbox):
    """Return max(w/h, h/w) for a [x1,y1,x2,y2] box."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    if w <= 0 or h <= 0:
        return 999.0
    return max(w / h, h / w)


def detection_in_exclusion_zone(bbox, img_w, img_h, zones):
    """Check if a detection center falls in an exclusion zone.

    Args:
        bbox: [x1, y1, x2, y2] detection box.
        img_w: Image width.
        img_h: Image height.
        zones: Dict with top_pct, right_pct, bottom_pct.

    Returns:
        True if the detection center is in the exclusion zone.
    """
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2

    top_pct = zones.get("top_pct", 0)
    right_pct = zones.get("right_pct", 0)
    bottom_pct = zones.get("bottom_pct", 0)

    if cy < img_h * top_pct:
        return True
    if cy > img_h * (1 - bottom_pct):
        return True
    if cx > img_w * (1 - right_pct):
        return True
    return False


def get_setup_from_filename(filename):
    """Derive setup name from image filename.

    Args:
        filename: Image filename (e.g. 'Kamera1_00_20250101131948.jpg').

    Returns:
        Setup key string (e.g. 'setup_1') or None.
    """
    fn = filename.lower()
    if "kamera1" in fn or "setup_1" in fn or "setup1" in fn:
        return "setup_1"
    if "kamera2" in fn or "setup_2" in fn or "setup2" in fn:
        return "setup_2"
    return None


def load_zones(zones_file):
    """Load exclusion zones from JSON file.

    Args:
        zones_file: Path to zones.json, or None for defaults.

    Returns:
        Dict mapping setup names to zone dicts.
    """
    if zones_file and os.path.exists(zones_file):
        with open(zones_file) as f:
            raw = json.load(f)
        zones = {}
        for key, val in raw.items():
            if isinstance(val, dict):
                zones[key] = val
        if zones:
            log.info("Loaded zones from %s: %s", zones_file, list(zones.keys()))
            return zones
    return DEFAULT_ZONES


# ═══════════════════════════════════════════════════════════════════════════
# WEIGHTED BOXES FUSION
# ═══════════════════════════════════════════════════════════════════════════

def weighted_boxes_fusion(detections, iou_threshold=0.35):
    """Apply Weighted Boxes Fusion across detections from multiple models.

    Operates per class. Clusters overlapping boxes and computes
    confidence-weighted coordinate averages. Uses best-IoU matching.
    Per-class minimum area filtering applied via PER_CLASS_MIN_AREA.

    Args:
        detections: List of dicts with bbox, confidence, class_id, etc.
        iou_threshold: IoU above which boxes are fused.

    Returns:
        List of fused detection dicts.
    """
    if not detections:
        return []

    filtered = []
    for d in detections:
        area = (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1])
        min_area = PER_CLASS_MIN_AREA.get(d["class_name"], 400)
        if area >= min_area:
            filtered.append(d)

    by_class = {}
    for det in filtered:
        by_class.setdefault(det["class_id"], []).append(det)

    fused = []
    for cls_id, class_dets in by_class.items():
        class_dets.sort(key=lambda d: d["confidence"], reverse=True)
        clusters = []

        for det in class_dets:
            box = np.array(det["bbox"], dtype=np.float64)
            conf = det["confidence"]

            best_iou, best_cluster = 0.0, None
            for cluster in clusters:
                rep_box = cluster["box"] / cluster["wsum"]
                iou = compute_iou(box.tolist(), rep_box.tolist())
                if iou > best_iou:
                    best_iou = iou
                    best_cluster = cluster

            if best_cluster is not None and best_iou > iou_threshold:
                rep_box = best_cluster["box"] / best_cluster["wsum"]
                merged_w = max(box[2], rep_box[2]) - min(box[0], rep_box[0])
                merged_h = max(box[3], rep_box[3]) - min(box[1], rep_box[1])
                det_w = box[2] - box[0]
                det_h = box[3] - box[1]
                rep_w = rep_box[2] - rep_box[0]
                if det["class_name"] == "person" and merged_w > max(det_w, rep_w) * 1.5:
                    clusters.append({
                        "box": box * conf, "wsum": conf, "csum": conf, "n": 1,
                        "best_conf": conf,
                        "best_attrs": det.get("attributes", {}),
                        "best_phrase": det.get("phrase", ""),
                        "class_id": cls_id, "class_name": det["class_name"],
                        "models": {det.get("model", "unknown")},
                    })
                else:
                    best_cluster["box"] += box * conf
                    best_cluster["wsum"] += conf
                    best_cluster["csum"] += conf
                    best_cluster["n"] += 1
                    best_cluster["models"].add(det.get("model", "unknown"))
                    if conf > best_cluster["best_conf"]:
                        best_cluster["best_conf"] = conf
                        best_cluster["best_attrs"] = det.get("attributes", {})
                        best_cluster["best_phrase"] = det.get("phrase", "")
            else:
                clusters.append({
                    "box": box * conf, "wsum": conf, "csum": conf, "n": 1,
                    "best_conf": conf,
                    "best_attrs": det.get("attributes", {}),
                    "best_phrase": det.get("phrase", ""),
                    "class_id": cls_id, "class_name": det["class_name"],
                    "models": {det.get("model", "unknown")},
                })

        for c in clusters:
            fused.append({
                "bbox": (c["box"] / c["wsum"]).tolist(),
                "confidence": min(c["csum"] / c["n"], 1.0),
                "class_id": c["class_id"],
                "class_name": c["class_name"],
                "model": "+".join(sorted(c["models"])),
                "attributes": dict(c["best_attrs"]),
                "phrase": c.get("best_phrase", ""),
                "num_fused": c["n"],
            })

    return fused


# ═══════════════════════════════════════════════════════════════════════════
# POST-FUSION CLEANUP
# ═══════════════════════════════════════════════════════════════════════════

def post_fusion_cleanup(detections, iou_threshold=0.5):
    """Fix cross-class issues after WBF fusion.

    1. Reclassify small person-boxes in the head region (top 35%) of a
       larger person as face (fixes Grounding DINO misclassification).
    2. Remove cross-class duplicates keeping higher confidence, but
       preserve ALLOWED_CONTAINMENT pairs.

    Args:
        detections: List of fused detection dicts.
        iou_threshold: IoU above which cross-class duplicates are removed.

    Returns:
        Cleaned list of detections.
    """
    persons = [d for d in detections if d["class_name"] == "person"]
    others = [d for d in detections if d["class_name"] != "person"]

    reclassified, keep_persons = [], []
    for sp in persons:
        sb = sp["bbox"]
        sa = (sb[2] - sb[0]) * (sb[3] - sb[1])
        done = False
        for lp in persons:
            if sp is lp:
                continue
            lb = lp["bbox"]
            la = (lb[2] - lb[0]) * (lb[3] - lb[1])
            if sa >= la * 0.4 or not box_contains(lb, sb, 0.4):
                continue
            if (sb[1] + sb[3]) / 2 <= lb[1] + (lb[3] - lb[1]) * 0.45:
                sp["class_id"] = 0
                sp["class_name"] = "face"
                sp["attributes"] = {"visibility": "occluded"}
                reclassified.append(sp)
                done = True
                break
        if not done:
            keep_persons.append(sp)

    all_dets = sorted(
        keep_persons + others + reclassified,
        key=lambda d: d["confidence"], reverse=True,
    )
    keep = [True] * len(all_dets)
    for i in range(len(all_dets)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(all_dets)):
            if not keep[j]:
                continue
            ci, cj = all_dets[i]["class_name"], all_dets[j]["class_name"]
            if ci == cj:
                continue
            if (ci, cj) in ALLOWED_CONTAINMENT or (cj, ci) in ALLOWED_CONTAINMENT:
                continue
            if compute_iou(all_dets[i]["bbox"], all_dets[j]["bbox"]) > iou_threshold:
                keep[j] = False

    return [d for d, k in zip(all_dets, keep) if k]


def dedup_same_class(detections, iou_threshold=0.30):
    """Remove remaining same-class duplicates after WBF.

    SAHI tiles can produce near-duplicates that WBF misses when boxes
    straddle tile boundaries. This pass removes the lower-confidence
    duplicate for any same-class pair above the IoU threshold.

    Args:
        detections: List of detection dicts.
        iou_threshold: IoU above which same-class boxes are deduplicated.

    Returns:
        Deduplicated list.
    """
    dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    keep = [True] * len(dets)
    for i in range(len(dets)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(dets)):
            if not keep[j]:
                continue
            if dets[i]["class_name"] != dets[j]["class_name"]:
                continue
            if compute_iou(dets[i]["bbox"], dets[j]["bbox"]) > iou_threshold:
                keep[j] = False
    return [d for d, k in zip(dets, keep) if k]


def filter_aspect_ratio(detections, max_ratio=MAX_ASPECT_RATIO):
    """Remove detections with extreme aspect ratios.

    Very elongated boxes (e.g. 1:5 or 5:1) are almost always false
    positives from edge artefacts or tile boundaries.

    Args:
        detections: List of detection dicts.
        max_ratio: Maximum allowed max(w/h, h/w).

    Returns:
        Filtered list.
    """
    return [d for d in detections if box_aspect_ratio(d["bbox"]) <= max_ratio]


def filter_per_class_confidence(detections):
    """Apply per-class minimum confidence thresholds.

    Uses the PER_CLASS_CONFIDENCE dict. Privacy-critical classes (face)
    have lower thresholds to maximise recall; noisy classes (text_or_logo,
    scaffolding) have higher thresholds.

    Args:
        detections: List of detection dicts.

    Returns:
        Filtered list.
    """
    return [
        d for d in detections
        if d["confidence"] >= PER_CLASS_CONFIDENCE.get(d["class_name"], 0.20)
    ]


def filter_max_area(detections, img_w, img_h, max_pct=0.15):
    """Remove person detections that exceed a percentage of image area.

    Oversized person boxes are almost always false positives from
    Grounding DINO misclassifying buildings or structures.

    Args:
        detections: List of detection dicts.
        img_w: Image width.
        img_h: Image height.
        max_pct: Maximum allowed fraction of image area for person class.

    Returns:
        Filtered list.
    """
    img_area = img_w * img_h
    return [
        d for d in detections
        if d["class_name"] != "person"
        or (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]) <= img_area * max_pct
    ]


def filter_exclusion_zones(detections, img_w, img_h, zones):
    """Remove detections whose center falls in an exclusion zone.

    Only applies to classes in ZONE_FILTERED_CLASSES (face, text_or_logo).
    Crane, scaffolding, and other construction classes are exempt because
    they legitimately appear in the upper image region.

    Args:
        detections: List of detection dicts.
        img_w: Image width.
        img_h: Image height.
        zones: Zone dict with top_pct, right_pct, bottom_pct.

    Returns:
        Filtered list.
    """
    if not zones:
        return detections
    return [
        d for d in detections
        if d["class_name"] not in ZONE_FILTERED_CLASSES
        or not detection_in_exclusion_zone(d["bbox"], img_w, img_h, zones)
    ]



# ═══════════════════════════════════════════════════════════════════════════
# LAZY MODEL LOADERS
# ═══════════════════════════════════════════════════════════════════════════

_models = {}


_PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_DIR = os.path.join(_PIPELINE_DIR, "models")


def resolve_model_path(filename):
    """Find a model file in standard locations.

    Search order: models/ dir, pipeline root, /workspace/privacy_pipeline/models/,
    current working dir. Falls back to filename alone (ultralytics auto-download).

    Args:
        filename: Model filename (e.g. 'yolo11n.pt').

    Returns:
        Resolved path string.
    """
    candidates = [
        os.path.join(_MODELS_DIR, filename),
        os.path.join(_PIPELINE_DIR, filename),
        os.path.join("/workspace/privacy_pipeline/models", filename),
        filename,
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return filename


def get_yolo_coco():
    """Load YOLO11n pretrained on COCO."""
    if "yolo_coco" not in _models:
        from ultralytics import YOLO
        _models["yolo_coco"] = YOLO(resolve_model_path("yolo11n.pt"))
        log.info("YOLO11n COCO loaded")
    return _models["yolo_coco"]


def get_yolo_face():
    """Load YOLO-Face for face detection."""
    if "yolo_face" not in _models:
        from ultralytics import YOLO
        _models["yolo_face"] = YOLO(resolve_model_path("yolov8n-face.pt"))
        log.info("YOLO-Face loaded")
    return _models["yolo_face"]


def get_yolo_world():
    """Load YOLO-World for open-vocabulary detection.

    Configured for text/logo and construction objects only.
    PPE detection runs separately via get_yolo_world_ppe().
    """
    if "yolo_world" not in _models:
        from ultralytics import YOLO
        model = YOLO(resolve_model_path("yolov8s-worldv2.pt"))
        model.set_classes([
            "sign", "logo", "text", "safety sign", "license plate",
            "banner", "company logo", "brand name", "sticker",
            "information board", "warning sign", "number plate",
            "crane", "tower crane", "scaffolding",
            "shipping container", "storage container",
            "pile of materials", "stacked wood", "stacked pipes",
            "concrete blocks", "pallets with materials",
        ])
        _models["yolo_world"] = model
        log.info("YOLO-World loaded")
    return _models["yolo_world"]


def _patch_gdino_deform_attn():
    """Patch Grounding DINO ms_deform_attn to use pure-Python fallback.

    The CUDA C++ extension (_C) often fails to compile due to version
    mismatches between nvcc, PyTorch, and NumPy. This patches both call
    sites in ms_deform_attn.py to use the 4-argument Python fallback
    function instead of the 6-argument _C.ms_deform_attn_forward.
    """
    attn_file = os.path.join(
        "/workspace/GroundingDINO/groundingdino/models/GroundingDINO",
        "ms_deform_attn.py",
    )
    if not os.path.exists(attn_file):
        return

    with open(attn_file) as fh:
        code = fh.read()

    if "_C.ms_deform_attn_forward(" not in code and \
       "value_level_start_index, sampling_locations, attention_weights, im2col_step" not in code:
        return

    code = code.replace(
        "_C.ms_deform_attn_forward(",
        "multi_scale_deformable_attn_pytorch(",
    )

    old_6arg = (
        "multi_scale_deformable_attn_pytorch(\n"
        "            value, value_spatial_shapes, value_level_start_index,"
        " sampling_locations, attention_weights, im2col_step\n"
        "        )"
    )
    new_4arg = (
        "multi_scale_deformable_attn_pytorch(\n"
        "            value, value_spatial_shapes,"
        " sampling_locations, attention_weights\n"
        "        )"
    )
    code = code.replace(old_6arg, new_4arg)

    old_6arg_v2 = (
        "multi_scale_deformable_attn_pytorch(\n"
        "                value, value_spatial_shapes, value_level_start_index,"
        " sampling_locations, attention_weights, im2col_step\n"
        "            )"
    )
    new_4arg_v2 = (
        "multi_scale_deformable_attn_pytorch(\n"
        "                value, value_spatial_shapes,"
        " sampling_locations, attention_weights\n"
        "            )"
    )
    code = code.replace(old_6arg_v2, new_4arg_v2)

    code = re.sub(
        r'multi_scale_deformable_attn_pytorch\(\s*'
        r'value\s*,\s*value_spatial_shapes\s*,\s*'
        r'value_level_start_index\s*,\s*'
        r'sampling_locations\s*,\s*'
        r'attention_weights\s*,\s*'
        r'im2col_step\s*\)',
        'multi_scale_deformable_attn_pytorch(\n'
        '            value, value_spatial_shapes,'
        ' sampling_locations, attention_weights\n'
        '        )',
        code,
    )

    with open(attn_file, "w") as fh:
        fh.write(code)
    log.info("Patched ms_deform_attn.py -> pure-Python fallback (4 args)")


def get_grounding_dino(config_path, weights_path):
    """Load Grounding DINO SwinT-OGC.

    Automatically patches the ms_deform_attn CUDA extension fallback
    before loading, to avoid _C compilation issues.
    """
    if "grounding_dino" not in _models:
        _patch_gdino_deform_attn()
        from groundingdino.util.inference import load_model
        _models["grounding_dino"] = load_model(config_path, weights_path)
        log.info("Grounding DINO loaded")
    return _models["grounding_dino"]


def get_yolo_world_ppe():
    """Load YOLO-World configured for PPE detection on person crops."""
    if "yolo_world_ppe" not in _models:
        from ultralytics import YOLO
        model = YOLO(resolve_model_path("yolov8s-worldv2.pt"))
        model.set_classes([
            "safety helmet", "hard hat", "construction helmet",
            "yellow safety vest", "orange safety vest",
            "high visibility vest", "reflective safety jacket",
        ])
        _models["yolo_world_ppe"] = model
        log.info("YOLO-World PPE loaded")
    return _models["yolo_world_ppe"]


def get_clip():
    """Load OpenCLIP ViT-B-32."""
    if "clip" not in _models:
        import open_clip
        import torch
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        _models["clip"] = (model, preprocess, tokenizer, device)
        log.info("CLIP loaded on %s", device)
    return _models["clip"]


def get_ocr():
    """Load EasyOCR (en+de)."""
    if "ocr" not in _models:
        import easyocr
        _models["ocr"] = easyocr.Reader(["en", "de"], gpu=True)
        log.info("EasyOCR loaded")
    return _models["ocr"]


# ═══════════════════════════════════════════════════════════════════════════
# DETECTION RUNNERS
# ═══════════════════════════════════════════════════════════════════════════

def detect_yolo_coco(img_path, conf=0.25):
    """Run YOLO11 COCO and map relevant classes to our taxonomy."""
    try:
        model = get_yolo_coco()
        results = model(img_path, conf=conf, verbose=False)
        dets = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                if cls_id not in COCO_TO_CLASS:
                    continue
                class_name, our_id = COCO_TO_CLASS[cls_id]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                dets.append({
                    "bbox": [x1, y1, x2, y2], "confidence": float(box.conf[0]),
                    "class_id": our_id, "class_name": class_name,
                    "model": "yolo_coco", "attributes": {},
                })
        return dets
    except Exception as e:
        log.warning("YOLO COCO failed on %s: %s", img_path, e)
        return []


def detect_yolo_face(img_path, conf=0.10):
    """Run YOLO-Face."""
    try:
        model = get_yolo_face()
        results = model(img_path, conf=conf, verbose=False)
        dets = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                dets.append({
                    "bbox": [x1, y1, x2, y2], "confidence": float(box.conf[0]),
                    "class_id": 0, "class_name": "face",
                    "model": "yolo_face", "attributes": {},
                })
        return dets
    except Exception as e:
        log.warning("YOLO-Face failed on %s: %s", img_path, e)
        return []


def detect_yolo_world(img_path, conf=0.15):
    """Run YOLO-World for text/logo and construction objects."""
    try:
        model = get_yolo_world()
        results = model(img_path, conf=conf, verbose=False)
        dets = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_name_raw = results[0].names[int(box.cls[0])].lower()
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                if any(k in cls_name_raw for k in [
                    "sign", "logo", "text", "license", "banner", "brand",
                    "sticker", "board", "warning", "number plate", "company",
                ]):
                    our_class, our_id = "text_or_logo", 3
                elif "crane" in cls_name_raw:
                    our_class, our_id = "crane", 4
                elif "scaffold" in cls_name_raw:
                    our_class, our_id = "scaffolding", 6
                elif "container" in cls_name_raw:
                    our_class, our_id = "container", 5
                elif any(k in cls_name_raw for k in [
                    "material", "pile", "stacked", "concrete", "pallet",
                ]):
                    our_class, our_id = "material_stack", 7
                else:
                    continue

                dets.append({
                    "bbox": [x1, y1, x2, y2], "confidence": float(box.conf[0]),
                    "class_id": our_id, "class_name": our_class,
                    "model": "yolo_world", "attributes": {},
                })
        return dets
    except Exception as e:
        log.warning("YOLO-World failed on %s: %s", img_path, e)
        return []


def detect_grounding_dino(img_path, model, prompts_and_thresholds):
    """Run Grounding DINO with multiple prompt groups.

    Phrase matching uses GDINO_PHRASE_SORTED (longest key first) so
    multi-word phrases match before single words. Unmatched phrases
    are discarded to avoid false positives.

    Args:
        img_path: Path to image.
        model: Loaded Grounding DINO model.
        prompts_and_thresholds: List of (prompt, box_thresh, text_thresh).

    Returns:
        List of detection dicts.
    """
    from groundingdino.util.inference import load_image, predict

    try:
        image_source, image_tensor = load_image(img_path)
        h, w = image_source.shape[:2]

        dets = []
        for prompt, box_thresh, text_thresh in prompts_and_thresholds:
            boxes, logits, phrases = predict(
                model=model, image=image_tensor, caption=prompt,
                box_threshold=box_thresh, text_threshold=text_thresh,
            )
            for box, logit, phrase in zip(boxes, logits, phrases):
                cx, cy, bw, bh = box.tolist()
                x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
                x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
                if (x2 - x1) < 5 or (y2 - y1) < 5:
                    continue

                phrase_clean = phrase.lower().strip()
                for key, (cls_name, cls_id) in GDINO_PHRASE_SORTED:
                    if key in phrase_clean:
                        dets.append({
                            "bbox": [x1, y1, x2, y2], "confidence": float(logit),
                            "class_id": cls_id, "class_name": cls_name,
                            "model": "grounding_dino", "attributes": {},
                            "phrase": phrase_clean,
                        })
                        break
        return dets
    except Exception as e:
        log.warning("Grounding DINO failed on %s: %s", img_path, e)
        return []


# ═══════════════════════════════════════════════════════════════════════════
# ATTRIBUTE ENRICHMENT + CLIP VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def clip_classify(image_crop, candidates, threshold=0.25):
    """Zero-shot classify a crop using CLIP."""
    import torch
    model, preprocess, tokenizer, device = get_clip()
    image_input = preprocess(image_crop).unsqueeze(0).to(device)
    all_prompts, label_map = [], []
    for label, prompts in candidates:
        for p in prompts:
            all_prompts.append(p)
            label_map.append(label)
    text_tokens = tokenizer(all_prompts).to(device)
    with torch.no_grad():
        img_f = model.encode_image(image_input)
        txt_f = model.encode_text(text_tokens)
        img_f /= img_f.norm(dim=-1, keepdim=True)
        txt_f /= txt_f.norm(dim=-1, keepdim=True)
        sims = (img_f @ txt_f.T).squeeze(0).cpu().numpy()
    scores = {}
    for i, s in enumerate(sims):
        scores[label_map[i]] = max(scores.get(label_map[i], 0.0), float(s))
    best = max(scores, key=scores.get)
    return (best, scores[best]) if scores[best] >= threshold else ("unspecified", 0.0)


def run_ocr_on_crop(crop):
    """Run EasyOCR. Returns True/False/None."""
    try:
        reader = get_ocr()
        results = reader.readtext(np.array(crop), detail=1)
        return len(results) > 0 and any(r[2] > 0.3 for r in results)
    except Exception:
        return None


def classify_gdino_phrase(phrase):
    """Derive text_or_logo attributes from Grounding DINO phrase."""
    phrase = phrase.strip().lower()
    is_logo = "yes" if any(k in phrase for k in PHRASE_IS_LOGO) else "no"
    content_type = "other"
    for key, val in sorted(PHRASE_TO_CONTENT_TYPE.items(), key=lambda x: -len(x[0])):
        if key in phrase:
            content_type = val
            break
    return {"is_logo": is_logo, "is_text": "yes", "content_type": content_type}


def enrich_annotations(fused_dets, image, skip_ocr=False, skip_clip=False):
    """Enrich fused detections with attributes in dependency order."""
    persons = [d for d in fused_dets if d["class_name"] == "person"]
    vehicles = [d for d in fused_dets if d["class_name"] == "vehicle"]
    text_logos = [d for d in fused_dets if d["class_name"] == "text_or_logo"]
    faces = [d for d in fused_dets if d["class_name"] == "face"]

    # --- TEXT/LOGO PHRASE ATTRIBUTES ---
    for det in text_logos:
        phrase = det.get("phrase", "")
        if phrase:
            det["attributes"].update(classify_gdino_phrase(phrase))

    # --- PERSON PPE ENRICHMENT ---
    for det in persons:
        crop = crop_bbox(image, det["bbox"], padding=20)
        if crop.width <= 10 or crop.height <= 10:
            continue

        has_helmet, has_vest = "no", "no"

        ppe_model = get_yolo_world_ppe()
        results = ppe_model(crop, conf=0.10, verbose=False)
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cn = results[0].names[int(box.cls[0])].lower()
                if "helmet" in cn or "hard hat" in cn:
                    has_helmet = "yes"
                elif "vest" in cn or "visibility" in cn or "reflective" in cn or "safety jacket" in cn:
                    has_vest = "yes"

        if not skip_clip:
            if has_helmet == "no":
                h_label, h_score = clip_classify(crop, [
                    ("yes", [
                        "a construction worker wearing a hard hat",
                        "a person wearing a white safety helmet",
                        "a person wearing a yellow hard hat",
                        "a person wearing a red safety helmet",
                        "a person wearing a blue hard hat",
                        "a person wearing an orange safety helmet",
                        "a worker with a helmet on their head",
                        "a round hard hat on top of a head",
                    ]),
                    ("no", [
                        "a person without a helmet, bare head visible",
                        "a person with hair visible and no hard hat",
                        "a bald head without any helmet",
                    ]),
                ], threshold=0.21)
                if h_label == "yes" and h_score > 0.22:
                    has_helmet = "yes"

            if has_vest == "no":
                upper_body = crop_bbox(image, [
                    det["bbox"][0],
                    det["bbox"][1] + (det["bbox"][3] - det["bbox"][1]) * 0.15,
                    det["bbox"][2],
                    det["bbox"][1] + (det["bbox"][3] - det["bbox"][1]) * 0.65,
                ], padding=5)
                if upper_body.width >= 15 and upper_body.height >= 15:
                    v_label, v_score = clip_classify(upper_body, [
                        ("yes", [
                            "a person wearing a bright yellow high visibility vest",
                            "a person wearing a bright orange reflective safety vest",
                            "a person wearing a neon green safety vest",
                            "a fluorescent safety vest over dark clothing",
                            "a bright colored safety vest with reflective stripes",
                            "a worker in a high-vis jacket",
                        ]),
                        ("no", [
                            "a person in dark clothing without any safety vest",
                            "a person wearing a regular jacket or coat",
                            "a person in normal work clothes without fluorescent vest",
                            "dark colored upper body clothing",
                        ]),
                    ], threshold=0.20)
                    if v_label == "yes" and v_score > 0.21:
                        has_vest = "yes"

            posture, _ = clip_classify(crop, [
                ("standing", [
                    "a person standing upright",
                    "a worker standing at a construction site",
                ]),
                ("walking", [
                    "a person walking",
                    "a worker walking across a construction site",
                ]),
                ("crouching", [
                    "a person crouching or kneeling down",
                    "a worker bending down working on the ground",
                ]),
                ("sitting", [
                    "a person sitting down",
                    "a worker sitting on a chair or bench",
                ]),
                ("climbing", [
                    "a person climbing a ladder or scaffolding",
                    "a worker climbing up a structure",
                ]),
                ("carrying", [
                    "a person carrying materials or tools",
                    "a worker lifting or moving heavy objects",
                ]),
                ("operating", [
                    "a person operating machinery or equipment",
                    "a worker using a power tool or machine",
                ]),
            ], threshold=0.20)
            det["attributes"]["posture"] = posture

        det["attributes"]["helmet"] = has_helmet
        det["attributes"]["vest"] = has_vest

    # --- FACE FAILSAFE ---
    for det in persons:
        det["attributes"]["face_visible"] = "no"
        for face in faces:
            if box_contains(det["bbox"], face["bbox"], threshold=0.3):
                det["attributes"]["face_visible"] = "yes"
                break

    rescued_faces = []
    for det in persons:
        if det["attributes"]["face_visible"] == "yes":
            continue
        pb = det["bbox"]
        head_y2 = pb[1] + (pb[3] - pb[1]) * 0.35
        head_crop = image.crop((
            max(0, int(pb[0])), max(0, int(pb[1])),
            min(image.width, int(pb[2])), min(image.height, int(head_y2)),
        ))
        if head_crop.width < 10 or head_crop.height < 10:
            continue

        face_model = get_yolo_face()
        results = face_model(np.array(head_crop), conf=0.05, verbose=False)
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            best_box = max(results[0].boxes, key=lambda b: float(b.conf[0]))
            fx1, fy1, fx2, fy2 = best_box.xyxy[0].tolist()
            abs_x1 = pb[0] + fx1
            abs_y1 = pb[1] + fy1
            abs_x2 = pb[0] + fx2
            abs_y2 = pb[1] + fy2

            face_conf = float(best_box.conf[0])
            if face_conf >= 0.15:
                visibility = "clear"
            elif face_conf >= 0.08:
                visibility = "partial"
            else:
                visibility = "occluded"

            rescued_faces.append({
                "bbox": [abs_x1, abs_y1, abs_x2, abs_y2],
                "confidence": face_conf,
                "class_id": 0, "class_name": "face",
                "model": "yolo_face_rescue",
                "attributes": {"visibility": visibility},
            })
            det["attributes"]["face_visible"] = "yes"
        else:
            if not skip_clip:
                head_crop_clip = crop_bbox(image, [
                    pb[0], pb[1], pb[2],
                    pb[1] + (pb[3] - pb[1]) * 0.30
                ], padding=5)
                if head_crop_clip.width >= 10 and head_crop_clip.height >= 10:
                    face_label, _ = clip_classify(head_crop_clip, [
                        ("face_visible", [
                            "a person's face visible from the front or side",
                            "a human face looking at the camera",
                            "a face partially visible under a helmet",
                        ]),
                        ("no_face", [
                            "the back of a person's head",
                            "a hard hat seen from behind",
                            "a person facing away from the camera",
                        ]),
                    ], threshold=0.20)
                    if face_label == "face_visible":
                        rescued_faces.append({
                            "bbox": [pb[0], pb[1],
                                     pb[2], pb[1] + (pb[3] - pb[1]) * 0.25],
                            "confidence": 0.20,
                            "class_id": 0, "class_name": "face",
                            "model": "clip_face_rescue",
                            "attributes": {"visibility": "partial"},
                        })
                        det["attributes"]["face_visible"] = "yes"

    if rescued_faces:
        fused_dets.extend(rescued_faces)
        faces.extend(rescued_faces)
        log.debug("Rescued %d faces from person head crops", len(rescued_faces))

    # --- LICENSE PLATE FAILSAFE ---
    rescued_plates = []
    for det in vehicles:
        vb = det["bbox"]
        has_plate = False
        for tl in text_logos:
            if (tl["attributes"].get("content_type") == "license_plate"
                    and box_contains(vb, tl["bbox"], threshold=0.3)):
                has_plate = True
                break
        if has_plate:
            continue

        plate_y1 = vb[1] + (vb[3] - vb[1]) * 0.5
        plate_crop = image.crop((
            max(0, int(vb[0])), max(0, int(plate_y1)),
            min(image.width, int(vb[2])), min(image.height, int(vb[3])),
        ))
        if plate_crop.width < 20 or plate_crop.height < 10:
            continue

        found_plate = False
        if not skip_ocr:
            ocr_result = run_ocr_on_crop(plate_crop)
            if ocr_result:
                found_plate = True

        if not found_plate and not skip_clip:
            label, _ = clip_classify(plate_crop, [
                ("plate", [
                    "a vehicle license plate with numbers and letters",
                    "a car registration plate",
                    "a number plate on a vehicle",
                ]),
                ("no_plate", [
                    "a plain vehicle surface without any plate",
                    "a wheel or tire",
                    "the ground under a vehicle",
                ]),
            ], threshold=0.22)
            if label == "plate":
                found_plate = True

        if found_plate:
            rescued_plates.append({
                "bbox": [vb[0], plate_y1, vb[2], vb[3]],
                "confidence": 0.30,
                "class_id": 3, "class_name": "text_or_logo",
                "model": "plate_rescue",
                "attributes": {
                    "content_type": "license_plate",
                    "is_text": "yes",
                    "is_logo": "no",
                },
            })

    if rescued_plates:
        fused_dets.extend(rescued_plates)
        text_logos.extend(rescued_plates)
        log.debug("Rescued %d license plates from vehicle crops", len(rescued_plates))

    # --- VEHICLE ATTRIBUTE ENRICHMENT ---
    if not skip_clip:
        for det in vehicles:
            crop = crop_bbox(image, det["bbox"], padding=15)
            if crop.width <= 10 or crop.height <= 10:
                continue
            vtype, _ = clip_classify(crop, [
                ("car", ["a photo of a car", "a sedan", "a passenger car", "a parked car"]),
                ("truck", ["a photo of a truck", "a large truck", "a delivery truck", "a lorry", "a dump truck"]),
                ("excavator", ["a photo of an excavator", "a digger", "construction excavator", "a backhoe"]),
                ("bicycle", ["a photo of a bicycle", "a bike", "a parked bicycle"]),
            ], threshold=0.22)
            det["attributes"]["vehicle_type"] = vtype

            ownership, _ = clip_classify(crop, [
                ("construction", [
                    "a construction vehicle on a building site",
                    "a yellow excavator", "heavy construction equipment",
                ]),
                ("public", [
                    "a regular car on a public road",
                    "a passenger car parked on the street",
                ]),
                ("delivery", [
                    "a delivery truck", "a truck delivering building materials",
                    "a cargo van making a delivery",
                ]),
            ], threshold=0.22)
            det["attributes"]["ownership"] = ownership

    # --- TEXT/LOGO OCR + CLIP ENRICHMENT ---
    for det in text_logos:
        attrs = det["attributes"]
        crop = crop_bbox(image, det["bbox"], padding=5)
        if crop.width < 5 or crop.height < 5:
            continue
        if not skip_ocr and det["confidence"] >= 0.35 and attrs.get("is_text", "unspecified") == "unspecified":
            result = run_ocr_on_crop(crop)
            if result is not None:
                attrs["is_text"] = "yes" if result else "no"
        if not skip_clip and attrs.get("content_type", "unspecified") == "unspecified":
            ct, _ = clip_classify(crop, [
                ("company_name", ["a company logo", "a brand logo", "a business sign"]),
                ("license_plate", ["a license plate", "a vehicle registration plate"]),
                ("safety_sign", ["a safety sign", "a warning sign"]),
            ], threshold=0.22)
            attrs["content_type"] = ct
        if attrs.get("content_type") == "company_name":
            attrs["is_logo"] = "yes"
        elif attrs.get("is_text") == "yes" and attrs.get("is_logo", "unspecified") == "unspecified":
            attrs["is_logo"] = "no"

    # --- FACE ATTRIBUTE ENRICHMENT ---
    for det in faces:
        attrs = det["attributes"]
        if not skip_clip:
            crop = crop_bbox(image, det["bbox"], padding=5)
            if crop.width >= 5 and crop.height >= 5:
                vis, _ = clip_classify(crop, [
                    ("clear", ["a clear photo of a human face", "a frontal view of a face"]),
                    ("partial", ["a partially visible face", "a side profile of a face"]),
                    ("occluded", ["an occluded face", "the back of a head"]),
                ], threshold=0.20)
                attrs["visibility"] = vis
        attrs["has_helmet"] = "no"
        for person in persons:
            if box_contains(person["bbox"], det["bbox"], threshold=0.3):
                if person["attributes"].get("helmet") == "yes":
                    attrs["has_helmet"] = "yes"
                break

    return fused_dets


def verify_detections(detections, image, skip_clip=False):
    """Remove false positives via CLIP verification.

    Verified classes:
    - vehicle: removed if CLIP cannot identify any vehicle_type.
    - material_stack: removed if CLIP scores 'not_material' higher.
    - text_or_logo: removed if CLIP identifies sky, clouds, building,
      or ground instead of actual text/signage.
    - container: removed if CLIP identifies building, vehicle, or
      empty area instead of a container.

    Args:
        detections: Enriched detection dicts.
        image: PIL Image.
        skip_clip: Skip verification if True.

    Returns:
        Filtered detections.
    """
    if skip_clip:
        return detections

    keep = []
    for det in detections:
        if det["class_name"] == "vehicle":
            crop = crop_bbox(image, det["bbox"], padding=10)
            if crop.width > 10 and crop.height > 10:
                label, _ = clip_classify(crop, [
                    ("vehicle", [
                        "a car or truck on a road",
                        "a construction vehicle or excavator",
                        "a parked vehicle",
                        "a delivery van or truck",
                        "a bicycle or motorbike",
                    ]),
                    ("not_vehicle", [
                        "a building or permanent structure",
                        "a wall or fence",
                        "an empty area or ground",
                        "the sky or clouds",
                        "a pile of construction materials",
                        "a person or group of people",
                    ]),
                ], threshold=0.20)
                if label == "not_vehicle":
                    continue

        elif det["class_name"] == "material_stack":
            crop = crop_bbox(image, det["bbox"], padding=10)
            if crop.width > 10 and crop.height > 10:
                label, _ = clip_classify(crop, [
                    ("material", [
                        "a pile of stacked wood or lumber",
                        "stacked iron pipes or metal tubes",
                        "a pile of bricks or concrete blocks",
                        "stacked building materials on a palette",
                        "construction material storage area",
                    ]),
                    ("not_material", [
                        "a building or structure",
                        "a vehicle or machine",
                        "an empty area",
                        "a wall or fence",
                        "a person or group of people",
                        "the sky or clouds",
                    ]),
                ], threshold=0.20)
                if label == "not_material":
                    continue

        elif det["class_name"] == "text_or_logo":
            crop = crop_bbox(image, det["bbox"], padding=5)
            if crop.width > 10 and crop.height > 10:
                label, _ = clip_classify(crop, [
                    ("text_or_sign", [
                        "a sign with text or writing",
                        "a company logo or brand name",
                        "a license plate with numbers",
                        "a safety warning sign",
                        "a sticker or label with text",
                    ]),
                    ("not_text", [
                        "the sky or clouds",
                        "a building wall or facade",
                        "the ground or road surface",
                        "a tree or vegetation",
                        "a plain surface with no text",
                        "a shadow on the ground",
                    ]),
                ], threshold=0.20)
                if label == "not_text":
                    continue

        elif det["class_name"] == "container":
            crop = crop_bbox(image, det["bbox"], padding=10)
            if crop.width > 10 and crop.height > 10:
                label, _ = clip_classify(crop, [
                    ("container", [
                        "a large portable office container on a construction site",
                        "a large metal storage container",
                        "a construction site cabin or portable building",
                        "a large waste skip or dumpster",
                        "a rectangular metal container used on construction sites",
                    ]),
                    ("not_container", [
                        "a building or permanent structure",
                        "a vehicle or truck",
                        "a wall or fence",
                        "an empty area or ground",
                        "a small box or crate",
                        "a window or door",
                    ]),
                ], threshold=0.20)
                if label == "not_container":
                    continue

        keep.append(det)
    return keep


# ═══════════════════════════════════════════════════════════════════════════
# OUTPUT FORMATTING
# ═══════════════════════════════════════════════════════════════════════════

def build_coco_structure():
    """Create empty COCO annotation structure with our taxonomy."""
    return {
        "images": [], "annotations": [],
        "categories": [
            {"id": i + 1, "name": n, "supercategory": "none"}
            for i, n in enumerate(CLASS_NAMES)
        ],
    }


def det_to_coco_ann(det, ann_id, image_id):
    """Convert a detection dict to a COCO annotation."""
    b = bbox_xyxy_to_coco(det["bbox"])
    return {
        "id": ann_id, "image_id": image_id,
        "category_id": det["class_id"] + 1,
        "bbox": b, "area": b[2] * b[3], "iscrowd": 0,
        "confidence": det["confidence"],
        "attributes": det.get("attributes", {}),
        "source": det.get("model", "unknown"),
        "num_fused": det.get("num_fused", 1),
    }


def det_to_yolo_line(det, img_w, img_h):
    """Convert detection to YOLO format line: cls cx cy w h."""
    x1, y1, x2, y2 = det["bbox"]
    cx, cy = ((x1 + x2) / 2) / img_w, ((y1 + y2) / 2) / img_h
    bw, bh = (x2 - x1) / img_w, (y2 - y1) / img_h
    return f"{det['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def create_cvat_zip(coco_data, output_dir):
    """Create CVAT import ZIP from COCO data."""
    zip_path = os.path.join(output_dir, "cvat_import.zip")
    json_path = os.path.join(output_dir, "instances_default.json")
    with open(json_path, "w") as f:
        json.dump(coco_data, f)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, "annotations/instances_default.json")
    os.remove(json_path)
    return zip_path


# ═══════════════════════════════════════════════════════════════════════════
# SAHI + PROCESS IMAGE
# ═══════════════════════════════════════════════════════════════════════════

def generate_sahi_slices(img_w, img_h, slice_size=640, overlap=0.25):
    """Generate tile coordinates for SAHI slicing.

    Args:
        img_w: Image width.
        img_h: Image height.
        slice_size: Tile dimension in pixels.
        overlap: Overlap ratio between tiles.

    Returns:
        List of (x1, y1, x2, y2) tile coordinates.
    """
    step = int(slice_size * (1 - overlap))
    slices = []
    for y in range(0, img_h, step):
        for x in range(0, img_w, step):
            x2 = min(x + slice_size, img_w)
            y2 = min(y + slice_size, img_h)
            x1 = max(0, x2 - slice_size)
            y1 = max(0, y2 - slice_size)
            slices.append((x1, y1, x2, y2))
    return slices


def offset_detections(detections, offset_x, offset_y):
    """Map tile-local detection coordinates back to full image.

    Args:
        detections: List of detection dicts with tile-local coordinates.
        offset_x: Tile x offset in original image.
        offset_y: Tile y offset in original image.

    Returns:
        Detections with full-image coordinates.
    """
    for d in detections:
        d["bbox"] = [
            d["bbox"][0] + offset_x, d["bbox"][1] + offset_y,
            d["bbox"][2] + offset_x, d["bbox"][3] + offset_y,
        ]
    return detections


def detect_single_pass(img_input, use_face, use_yolo_world, use_gdino,
                       gdino_model, gdino_prompts):
    """Run all detection models on a single image or tile path.

    Args:
        img_input: Path string to image.
        use_face: Enable YOLO-Face.
        use_yolo_world: Enable YOLO-World.
        use_gdino: Enable Grounding DINO.
        gdino_model: Loaded GDino model.
        gdino_prompts: GDino prompt configurations.

    Returns:
        List of raw detection dicts (before WBF).
    """
    all_dets = []
    all_dets.extend(detect_yolo_coco(img_input, conf=0.25))
    if use_face:
        all_dets.extend(detect_yolo_face(img_input, conf=0.10))
    if use_yolo_world:
        all_dets.extend(detect_yolo_world(img_input, conf=0.15))
    if use_gdino and gdino_model is not None:
        all_dets.extend(detect_grounding_dino(img_input, gdino_model, gdino_prompts))
    return all_dets


def process_image(img_path, image, gdino_model, gdino_prompts,
                  use_face, use_yolo_world, use_gdino,
                  skip_ocr, skip_clip, wbf_iou,
                  use_sahi=False, sahi_slice_size=640, sahi_overlap=0.25,
                  zones=None):
    """Run full pipeline: detect -> fuse -> cleanup -> filter -> enrich -> verify.

    When use_sahi is True, images are sliced into overlapping tiles,
    each tile is processed independently, coordinates are mapped back,
    and a second WBF pass merges tile-level detections. Enrichment
    runs on the merged full-image detections only.

    Args:
        img_path: Path to image file.
        image: PIL Image object.
        gdino_model: Loaded Grounding DINO model.
        gdino_prompts: GDino prompt configurations.
        use_face: Enable YOLO-Face.
        use_yolo_world: Enable YOLO-World.
        use_gdino: Enable Grounding DINO.
        skip_ocr: Skip OCR enrichment.
        skip_clip: Skip CLIP enrichment and verification.
        wbf_iou: WBF IoU threshold.
        use_sahi: Enable SAHI tile-based inference.
        sahi_slice_size: SAHI tile size in pixels.
        sahi_overlap: SAHI tile overlap ratio.
        zones: Exclusion zone dict for this image's setup, or None.

    Returns:
        List of enriched, verified detection dicts.
    """
    img_w, img_h = image.size

    if use_sahi and (img_w > sahi_slice_size or img_h > sahi_slice_size):
        slices = generate_sahi_slices(img_w, img_h, sahi_slice_size, sahi_overlap)
        all_tile_dets = []

        for sx1, sy1, sx2, sy2 in slices:
            tile = image.crop((sx1, sy1, sx2, sy2))
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tile.save(tmp.name)
                tile_dets = detect_single_pass(
                    tmp.name, use_face, use_yolo_world,
                    use_gdino, gdino_model, gdino_prompts
                )
                os.unlink(tmp.name)
            tile_dets = offset_detections(tile_dets, sx1, sy1)
            all_tile_dets.extend(tile_dets)

        full_dets = detect_single_pass(
            str(img_path), use_face, use_yolo_world,
            use_gdino, gdino_model, gdino_prompts
        )
        all_tile_dets.extend(full_dets)

        fused = weighted_boxes_fusion(all_tile_dets, iou_threshold=wbf_iou)
    else:
        all_dets = detect_single_pass(
            str(img_path), use_face, use_yolo_world,
            use_gdino, gdino_model, gdino_prompts
        )
        fused = weighted_boxes_fusion(all_dets, iou_threshold=wbf_iou)

    fused = post_fusion_cleanup(fused, iou_threshold=0.5)
    fused = dedup_same_class(fused)
    fused = filter_per_class_confidence(fused)
    fused = filter_max_area(fused, img_w, img_h)
    fused = filter_aspect_ratio(fused)

    if zones:
        fused = filter_exclusion_zones(fused, img_w, img_h, zones)

    fused = enrich_annotations(fused, image, skip_ocr=skip_ocr, skip_clip=skip_clip)
    fused = verify_detections(fused, image, skip_clip=skip_clip)

    for det in fused:
        if det["class_name"] == "text_or_logo":
            det["attributes"].setdefault("is_logo", "unspecified")
            det["attributes"].setdefault("is_text", "unspecified")
            det["attributes"].setdefault("content_type", "unspecified")

    return fused


def print_attribute_summary(coco_data):
    """Print detailed attribute fill rates."""
    cat_map = {c["id"]: c["name"] for c in coco_data["categories"]}
    by_class = {}
    for ann in coco_data["annotations"]:
        by_class.setdefault(cat_map.get(ann["category_id"], "?"), []).append(ann)

    print("\n  ATTRIBUTE SUMMARY:")
    for cat_name in sorted(by_class):
        anns = by_class[cat_name]
        all_keys = set()
        for a in anns:
            all_keys.update(a.get("attributes", {}).keys())
        if not all_keys:
            continue
        print(f"\n    {cat_name} ({len(anns)} annotations)")
        for key in sorted(all_keys):
            vals = Counter(a.get("attributes", {}).get(key, "MISSING") for a in anns)
            total = len(anns)
            unspec = vals.get("unspecified", 0)
            pct = ((total - unspec) / total * 100) if total > 0 else 0
            print(f"      {key} ({pct:.0f}% filled):", end="")
            for v, c in vals.most_common(4):
                print(f" {v}={c}", end="")
            print()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Stage 3: Auto-annotation")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--no-grounding-dino", action="store_true")
    parser.add_argument("--no-face", action="store_true")
    parser.add_argument("--no-yolo-world", action="store_true")
    parser.add_argument("--skip-ocr", action="store_true")
    parser.add_argument("--skip-clip", action="store_true")
    parser.add_argument("--grounding-dino-config", type=str,
                        default="/workspace/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--grounding-dino-weights", type=str,
                        default="/workspace/GroundingDINO/weights/groundingdino_swint_ogc.pth")
    parser.add_argument("--wbf-iou", type=float, default=0.35)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Process at most N images (for testing).")
    parser.add_argument("--use-sahi", action="store_true",
                        help="Enable SAHI tile-based inference for small objects.")
    parser.add_argument("--sahi-slice-size", type=int, default=640,
                        help="SAHI tile size in pixels (default: 640).")
    parser.add_argument("--sahi-overlap", type=float, default=0.25,
                        help="SAHI tile overlap ratio (default: 0.25).")
    parser.add_argument("--zones-file", type=str, default=None,
                        help="Path to zones.json for exclusion zone filtering.")
    args = parser.parse_args()

    if args.no_resume:
        for subdir in ["annotations", "yolo_labels"]:
            p = os.path.join(args.output_dir, subdir)
            if os.path.exists(p):
                shutil.rmtree(p)
        for f in ["cvat_import.zip"]:
            p = os.path.join(args.output_dir, f)
            if os.path.exists(p):
                os.remove(p)

    os.makedirs(os.path.join(args.output_dir, "annotations"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "yolo_labels"), exist_ok=True)

    if not os.path.isdir(args.input_dir):
        log.error("Input directory does not exist: %s", args.input_dir)
        sys.exit(1)

    use_gdino = not args.no_grounding_dino
    use_face = not args.no_face
    use_yolo_world = not args.no_yolo_world

    zones_map = load_zones(args.zones_file)

    gdino_model = None
    if use_gdino:
        gdino_model = get_grounding_dino(args.grounding_dino_config, args.grounding_dino_weights)

    gdino_prompts = [
        (GROUNDING_DINO_VEHICLE_PROMPTS, 0.25, 0.20),
        (GROUNDING_DINO_TEXT_PROMPTS, 0.25, 0.20),
        (GROUNDING_DINO_CONSTRUCTION_PROMPTS, 0.25, 0.20),
        (GROUNDING_DINO_HEAD_PROMPTS, 0.40, 0.35),
    ]

    image_files = sorted([
        f for f in os.listdir(args.input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ])

    if not image_files:
        log.error("No images found in %s", args.input_dir)
        sys.exit(0)

    if args.max_images:
        image_files = image_files[:args.max_images]

    coco_data = build_coco_structure()
    ckpt = os.path.join(args.output_dir, "annotations", "checkpoint.json")
    start_idx = 0
    if not args.no_resume and os.path.exists(ckpt):
        with open(ckpt) as f:
            coco_data = json.load(f)
        start_idx = len(coco_data["images"])
        log.info("Resuming from checkpoint: %d images done", start_idx)

    next_ann_id = max((a["id"] for a in coco_data["annotations"]), default=0) + 1

    models_used = ["yolo_coco"]
    if use_face: models_used.append("yolo_face")
    if use_yolo_world: models_used.append("yolo_world")
    if use_gdino: models_used.append("grounding_dino")

    coco_data["info"] = {
        "description": "Auto-annotated by 03_auto_annotate.py",
        "date_created": datetime.now().isoformat(),
        "models": models_used,
        "wbf_iou": args.wbf_iou,
        "sahi": args.use_sahi,
        "confidence_thresholds": dict(PER_CLASS_CONFIDENCE),
    }

    print("=" * 60)
    print("STAGE 3: AUTO-ANNOTATION + WBF + CLEANUP + ENRICHMENT")
    print("=" * 60)
    print(f"  Images:         {len(image_files)}")
    print(f"  Models:         {', '.join(models_used)}")
    print(f"  WBF IoU:        {args.wbf_iou}")
    print(f"  SAHI:           {'ON (%dx%d, %.0f%% overlap)' % (args.sahi_slice_size, args.sahi_slice_size, args.sahi_overlap * 100) if args.use_sahi else 'OFF'}")
    print(f"  Cross-class:    IoU=0.5 with allowed containment")
    print(f"  Dedup:          IoU=0.50 same-class post-WBF")
    print(f"  Aspect ratio:   max {MAX_ASPECT_RATIO}:1")
    print(f"  Zone filtering: {'ON' if args.zones_file else 'defaults (top 25%, right 5%, bottom 5%)'}")
    print(f"  Verification:   CLIP gating for vehicle + material_stack + text_or_logo + container")
    print(f"  CLIP:           {'OFF' if args.skip_clip else 'ON'}")
    print(f"  OCR:            {'OFF' if args.skip_ocr else 'ON (conf >= 0.35 only)'}")
    print(f"  Confidence:     per-class {dict(PER_CLASS_CONFIDENCE)}")
    print(f"  Resume from:    {start_idx}")
    print("=" * 60)

    t_start = time.time()
    stats = Counter()

    for i, filename in enumerate(tqdm(image_files[start_idx:], desc="Annotating",
                                       initial=start_idx, total=len(image_files))):
        img_idx = start_idx + i
        img_path = os.path.join(args.input_dir, filename)
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        coco_data["images"].append({
            "id": img_idx + 1, "file_name": filename,
            "width": img_w, "height": img_h,
        })

        setup = get_setup_from_filename(filename)
        img_zones = zones_map.get(setup) if setup else None

        fused_dets = process_image(
            img_path, image, gdino_model, gdino_prompts,
            use_face, use_yolo_world, use_gdino,
            args.skip_ocr, args.skip_clip, args.wbf_iou,
            use_sahi=args.use_sahi,
            sahi_slice_size=args.sahi_slice_size,
            sahi_overlap=args.sahi_overlap,
            zones=img_zones,
        )

        yolo_lines = []
        for det in fused_dets:
            coco_data["annotations"].append(det_to_coco_ann(det, next_ann_id, img_idx + 1))
            next_ann_id += 1
            yolo_lines.append(det_to_yolo_line(det, img_w, img_h))
            stats[det["class_name"]] += 1

        with open(os.path.join(args.output_dir, "yolo_labels", Path(filename).stem + ".txt"), "w") as f:
            f.write("\n".join(yolo_lines) + "\n" if yolo_lines else "")

        stats["total"] += len(fused_dets)

        if args.checkpoint_interval > 0 and (img_idx + 1) % args.checkpoint_interval == 0:
            with open(ckpt, "w") as f:
                json.dump(coco_data, f)
            log.info("Checkpoint: %d/%d", img_idx + 1, len(image_files))

    coco_path = os.path.join(args.output_dir, "annotations", "annotations.json")
    with open(coco_path, "w") as f:
        json.dump(coco_data, f)
    cvat_zip = create_cvat_zip(coco_data, args.output_dir)
    if os.path.exists(ckpt):
        os.remove(ckpt)

    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print("ANNOTATION RESULTS")
    print("=" * 60)
    print(f"  Time:           {elapsed/60:.1f} min")
    print(f"  Images:         {len(coco_data['images'])}")
    print(f"  Annotations:    {len(coco_data['annotations'])}")
    print(f"  WBF IoU:        {args.wbf_iou}")
    print()
    print("  Per class:")
    for cls in CLASS_NAMES:
        c = stats.get(cls, 0)
        if c > 0:
            print(f"    {cls:20s}: {c}")
    print_attribute_summary(coco_data)
    print()
    print(f"  COCO JSON:  {coco_path}")
    print(f"  YOLO labels: {args.output_dir}/yolo_labels/")
    print(f"  CVAT ZIP:   {cvat_zip}")
    print("=" * 60)


if __name__ == "__main__":
    main()
