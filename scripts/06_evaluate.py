#!/usr/bin/env python3
"""
Stage 6: Evaluate detection and anonymisation performance.

Runs the trained model on the validation set and computes mAP,
precision, recall, and F1 per class. Computes a privacy compliance
score that weights face and text_or_logo recall heavily — a missed
face is a GDPR violation, so recall matters more than precision for
privacy-critical classes.

Privacy score formula:
    privacy_score = 0.40 * face_recall
                  + 0.30 * person_recall
                  + 0.10 * vehicle_recall
                  + 0.10 * text_or_logo_recall
                  + 0.10 * mean_recall_other_classes

Outputs:
    - evaluation_results.json (full metrics)
    - evaluation_results.csv (for thesis tables)
    - val_results/ (confusion matrix, PR curves from Ultralytics)

Usage:
    python scripts/06_evaluate.py \
        --model /workspace/output/construction_v1/weights/best.pt \
        --dataset-yaml /workspace/data/training/dataset.yaml \
        --output-dir /workspace/output/evaluation_v1 \
        --save-visualizations

Author: Maximilian Drexler
License: MIT
"""

import os
import csv
import json
import argparse
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

PRIVACY_CRITICAL = {"face", "person", "vehicle", "text_or_logo"}
PRIVACY_WEIGHTS = {
    "face": 0.40,
    "person": 0.30,
    "vehicle": 0.10,
    "text_or_logo": 0.10,
}
PRIVACY_TARGET = 0.80


def compute_privacy_score(per_class):
    """Compute weighted privacy compliance score.

    Face recall dominates because a missed face = GDPR violation.

    Args:
        per_class: Dict of class_name -> {recall, precision, ...}.

    Returns:
        Float between 0 and 1.
    """
    score = 0.0
    other_recalls = []

    for cls_name, m in per_class.items():
        r = m.get("recall", 0.0)
        if cls_name in PRIVACY_WEIGHTS:
            score += PRIVACY_WEIGHTS[cls_name] * r
        else:
            other_recalls.append(r)

    other_weight = 1.0 - sum(PRIVACY_WEIGHTS.values())
    if other_recalls:
        score += other_weight * (sum(other_recalls) / len(other_recalls))

    return score


def save_csv(per_class, overall, output_path):
    """Save evaluation results as CSV for thesis tables.

    Args:
        per_class: Dict of class_name -> metrics dict.
        overall: Dict with overall metrics.
        output_path: Path for CSV file.
    """
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Class", "Precision", "Recall", "F1", "mAP@0.5", "mAP@0.5:0.95"
        ])
        for cls_name, m in per_class.items():
            p, r = m["precision"], m["recall"]
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            writer.writerow([
                cls_name,
                f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}",
                f"{m['mAP50']:.4f}", f"{m['mAP50-95']:.4f}",
            ])
        writer.writerow([])
        writer.writerow(["Overall",
                         f"{overall['precision']:.4f}",
                         f"{overall['recall']:.4f}",
                         "",
                         f"{overall['mAP50']:.4f}",
                         f"{overall['mAP50-95']:.4f}"])


def main():
    parser = argparse.ArgumentParser(
        description="Stage 6: Evaluate trained detection model."
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained YOLO11 weights (.pt).")
    parser.add_argument("--dataset-yaml", type=str, required=True,
                        help="Path to dataset.yaml with val split.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory for evaluation outputs.")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size (default: 640).")
    parser.add_argument("--confidence", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25).")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold for mAP (default: 0.5).")
    parser.add_argument("--save-visualizations", action="store_true",
                        help="Save prediction visualisations.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from ultralytics import YOLO

    log.info("Loading model: %s", args.model)
    model = YOLO(args.model)

    log.info("Running validation...")
    metrics = model.val(
        data=args.dataset_yaml,
        imgsz=args.imgsz,
        conf=args.confidence,
        iou=args.iou_threshold,
        save_json=True,
        plots=True,
        project=args.output_dir,
        name="val_results",
    )

    overall = {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }

    per_class = {}
    class_names = model.names
    for i, (p, r, ap50, ap) in enumerate(zip(
        metrics.box.p, metrics.box.r,
        metrics.box.ap50, metrics.box.ap,
    )):
        cls_name = class_names.get(i, f"class_{i}")
        pr, rc = float(p), float(r)
        f1 = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0
        per_class[cls_name] = {
            "precision": pr, "recall": rc, "f1": f1,
            "mAP50": float(ap50), "mAP50-95": float(ap),
        }

    privacy_score = compute_privacy_score(per_class)

    results = {
        "model": args.model,
        "dataset": args.dataset_yaml,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "imgsz": args.imgsz,
            "confidence": args.confidence,
            "iou_threshold": args.iou_threshold,
        },
        "overall": overall,
        "privacy_score": privacy_score,
        "privacy_target": PRIVACY_TARGET,
        "privacy_target_met": privacy_score >= PRIVACY_TARGET,
        "privacy_weights": {k: v for k, v in PRIVACY_WEIGHTS.items()},
        "per_class": per_class,
    }

    json_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    csv_path = os.path.join(args.output_dir, "evaluation_results.csv")
    save_csv(per_class, overall, csv_path)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  mAP@0.5:        {overall['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95:   {overall['mAP50-95']:.4f}")
    print(f"  Precision:       {overall['precision']:.4f}")
    print(f"  Recall:          {overall['recall']:.4f}")
    print(f"  Privacy Score:   {privacy_score:.4f}  (target: {PRIVACY_TARGET})")
    target_met = privacy_score >= PRIVACY_TARGET
    print(f"  Target met:      {'YES' if target_met else 'NO'}")
    print()
    print(f"  {'Class':20s} {'P':>7s} {'R':>7s} {'F1':>7s} {'mAP50':>7s} {'mAP95':>7s}")
    print(f"  {'-'*20} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for cls_name, m in per_class.items():
        marker = " *" if cls_name in PRIVACY_CRITICAL else ""
        print(f"  {cls_name:20s} {m['precision']:7.4f} {m['recall']:7.4f} "
              f"{m['f1']:7.4f} {m['mAP50']:7.4f} {m['mAP50-95']:7.4f}{marker}")
    print()
    print("  (* = privacy-critical class)")

    face_recall = per_class.get("face", {}).get("recall", 0.0)
    person_recall = per_class.get("person", {}).get("recall", 0.0)

    if face_recall < 0.5:
        print()
        print("  WARNING: Face recall is below 0.50!")
        print("    Every missed face is a potential GDPR violation.")
        print("    Consider: more face annotations, v2 auto-annotator,")
        print("    or a dedicated face detector (SCRFD/RetinaFace).")

    if person_recall < 0.7:
        print()
        print("  WARNING: Person recall is below 0.70!")
        print("    Missed persons reduce tier-3 safety blur coverage.")

    print()
    print(f"  JSON:  {json_path}")
    print(f"  CSV:   {csv_path}")
    print(f"  Plots: {args.output_dir}/val_results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
