"""
Multi-model detection fusion using Weighted Boxes Fusion (WBF).

When multiple detection models process the same image, their predictions
must be merged. This module provides two strategies:

1. Weighted Boxes Fusion (WBF) — default, recommended.
   Computes confidence-weighted averages of bounding box coordinates
   across all models. Unlike NMS, WBF does not discard overlapping boxes
   but fuses them into improved predictions. Top results in Open Images
   and COCO detection challenges.

2. Cross-model NMS — fallback for simpler use cases.
   Keeps the highest-confidence detection among overlapping boxes.

Both methods operate PER CLASS to prevent suppression of legitimate
overlaps between different classes (e.g., face inside person).

References:
    Solovyev, R., Wang, W. and Gabruseva, T. (2021) 'Weighted boxes
    fusion: Ensembling boxes from different object detection models',
    Image and Vision Computing, 107, p. 104117.

    Neubeck, A. and Van Gool, L. (2006) 'Efficient Non-Maximum
    Suppression', Proceedings of the 18th International Conference on
    Pattern Recognition (ICPR), pp. 850-855.
"""

import numpy as np
from typing import List, Dict


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: Array [x1, y1, x2, y2] for the first box.
        box2: Array [x1, y1, x2, y2] for the second box.

    Returns:
        IoU value between 0.0 and 1.0.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0
    return intersection / union


def weighted_boxes_fusion(
    detections: List[Dict],
    iou_threshold: float = 0.5,
    skip_box_threshold: float = 0.0
) -> List[Dict]:
    """
    Apply Weighted Boxes Fusion across detections from multiple models, per class.

    WBF fuses overlapping boxes by computing confidence-weighted averages of
    their coordinates. This preserves information from all models rather than
    discarding lower-confidence detections as NMS does.

    Algorithm per class:
        1. Sort detections by confidence (descending).
        2. For each detection, find existing clusters with IoU > threshold.
        3. If match found: merge into cluster (weighted coordinate average).
        4. If no match: create new cluster.
        5. Output one fused detection per cluster.

    Args:
        detections: List of detection dicts, each containing:
            - 'bbox': [x1, y1, x2, y2] in pixel coordinates
            - 'confidence': float detection score
            - 'class_id': int class identifier
            - 'class_name': str class label
            - 'model': str source model identifier
        iou_threshold: IoU threshold above which detections of the
            SAME class are fused into one.
        skip_box_threshold: Ignore detections below this confidence.

    Returns:
        List of fused detections with improved coordinate accuracy.
    """
    if not detections:
        return []

    by_class = {}
    for det in detections:
        if det["confidence"] < skip_box_threshold:
            continue
        cid = det["class_id"]
        if cid not in by_class:
            by_class[cid] = []
        by_class[cid].append(det)

    fused_all = []

    for cid, class_dets in by_class.items():
        sorted_dets = sorted(class_dets, key=lambda d: d["confidence"], reverse=True)

        clusters = []
        cluster_members = []

        for det in sorted_dets:
            det_box = np.array(det["bbox"])
            matched_cluster = -1
            best_iou = 0.0

            for ci, cluster in enumerate(clusters):
                iou = compute_iou(det_box, np.array(cluster["bbox"]))
                if iou > iou_threshold and iou > best_iou:
                    best_iou = iou
                    matched_cluster = ci

            if matched_cluster >= 0:
                members = cluster_members[matched_cluster]
                members.append(det)

                total_conf = sum(m["confidence"] for m in members)
                fused_bbox = [0.0, 0.0, 0.0, 0.0]
                for m in members:
                    w = m["confidence"] / total_conf
                    for i in range(4):
                        fused_bbox[i] += m["bbox"][i] * w

                clusters[matched_cluster]["bbox"] = fused_bbox
                clusters[matched_cluster]["confidence"] = total_conf / len(members)
                models = list(set(
                    clusters[matched_cluster].get("fused_models", []) + [det["model"]]
                ))
                clusters[matched_cluster]["fused_models"] = models
                clusters[matched_cluster]["model"] = "+".join(models)
                clusters[matched_cluster]["num_fused"] = len(members)
            else:
                new_cluster = dict(det)
                new_cluster["fused_models"] = [det["model"]]
                new_cluster["num_fused"] = 1
                clusters.append(new_cluster)
                cluster_members.append([det])

        fused_all.extend(clusters)

    return fused_all


def cross_model_nms(
    detections: List[Dict],
    iou_threshold: float = 0.5
) -> List[Dict]:
    """
    Apply NMS across detections from multiple models, per class.

    Retained as fallback. For production use, prefer weighted_boxes_fusion().

    Args:
        detections: List of detection dicts.
        iou_threshold: IoU threshold above which detections of the
            SAME class are considered duplicates.

    Returns:
        Filtered list of detections with duplicates removed.
    """
    if not detections:
        return []

    by_class = {}
    for det in detections:
        cid = det["class_id"]
        if cid not in by_class:
            by_class[cid] = []
        by_class[cid].append(det)

    keep = []
    for cid, class_dets in by_class.items():
        sorted_dets = sorted(class_dets, key=lambda d: d["confidence"], reverse=True)

        while sorted_dets:
            best = sorted_dets.pop(0)
            keep.append(best)

            remaining = []
            for det in sorted_dets:
                iou = compute_iou(
                    np.array(best["bbox"]),
                    np.array(det["bbox"])
                )
                if iou < iou_threshold:
                    remaining.append(det)
            sorted_dets = remaining

    return keep


def link_faces_to_persons(
    detections: List[Dict],
    containment_threshold: float = 0.7
) -> List[Dict]:
    """
    Link face detections to their parent person detections.

    Two-way linking:
    1. If a face box is largely inside a person box: person gets
       face_visible=yes attribute.
    2. If a person has NO linked face: person gets face_visible=no,
       signaling potential need for body-level anonymization when
       face detection failed (e.g., helmet occlusion).

    This addresses the indirect-identifier risk identified in the Oxford
    Academic analysis (International Data Privacy Law, 2022): clothing
    patterns, tattoos, and body shape can identify individuals even when
    faces are not visible.

    Args:
        detections: List of detection dicts (modified in place).
        containment_threshold: Minimum fraction of face box area
            inside person box to establish a link.

    Returns:
        Modified detections list with face-person links applied.
    """
    faces = [d for d in detections if d["class_name"] == "face"]
    persons = [d for d in detections if d["class_name"] == "person"]

    linked_persons = set()

    for face in faces:
        fb = face["bbox"]
        face_area = (fb[2] - fb[0]) * (fb[3] - fb[1])
        if face_area <= 0:
            continue

        best_person = None
        best_containment = 0.0
        best_idx = -1

        for idx, person in enumerate(persons):
            pb = person["bbox"]
            ix1 = max(fb[0], pb[0])
            iy1 = max(fb[1], pb[1])
            ix2 = min(fb[2], pb[2])
            iy2 = min(fb[3], pb[3])
            intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            containment = intersection / face_area

            if containment > best_containment:
                best_containment = containment
                best_person = person
                best_idx = idx

        if best_person and best_containment >= containment_threshold:
            if "attributes" not in best_person:
                best_person["attributes"] = {}
            best_person["attributes"]["face_visible"] = "yes"
            linked_persons.add(best_idx)

    for idx, person in enumerate(persons):
        if idx not in linked_persons:
            if "attributes" not in person:
                person["attributes"] = {}
            if "face_visible" not in person["attributes"]:
                person["attributes"]["face_visible"] = "no"

    return detections
