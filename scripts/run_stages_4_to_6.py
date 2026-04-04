#!/usr/bin/env python3
"""
Run stages 4-6 of the privacy pipeline end-to-end.

Assumes manual annotation correction in CVAT has been completed and
the corrected dataset is exported in YOLO format.

Stages:
  4. Train: Fine-tune YOLO11 on the corrected annotations.
  5. Anonymise: Apply three-tier confidence-based Gaussian blur.
  6. Evaluate: Compute mAP, per-class recall, privacy score.

Usage:
    python run_stages_4_to_6.py \
        --dataset-yaml /workspace/data/training/dataset.yaml \
        --test-images /workspace/data/construction_images_selected/selected \
        --output-dir /workspace/output

    # Skip training, use existing model:
    python run_stages_4_to_6.py \
        --dataset-yaml /workspace/data/training/dataset.yaml \
        --test-images /workspace/data/construction_images_selected/selected \
        --output-dir /workspace/output \
        --model /workspace/output/construction_v3/weights/best.pt \
        --skip-training

Author: Maximilian Drexler
License: MIT
"""

import os
import sys
import time
import argparse
import logging
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))


def run_command(cmd, stage_name):
    """Execute a command and handle errors.

    Args:
        cmd: List of command arguments.
        stage_name: Human-readable stage name for logging.

    Raises:
        SystemExit: If the command fails.
    """
    log.info("=" * 60)
    log.info("STARTING: %s", stage_name)
    log.info("Command: %s", " ".join(cmd))
    log.info("=" * 60)

    start = time.time()
    result = subprocess.run(cmd, cwd=SCRIPTS_DIR)
    elapsed = time.time() - start

    if result.returncode != 0:
        log.error("%s FAILED (exit code %d) after %.1fs",
                  stage_name, result.returncode, elapsed)
        sys.exit(result.returncode)

    log.info("%s COMPLETED in %.1fs (%.1f min)",
             stage_name, elapsed, elapsed / 60)
    return elapsed


def find_best_weights(output_dir, run_name):
    """Locate best.pt from the most recent training run.

    Args:
        output_dir: Base output directory.
        run_name: Training run name.

    Returns:
        Path to best.pt or None.
    """
    candidates = [
        os.path.join(output_dir, run_name, "weights", "best.pt"),
        os.path.join(output_dir, f"{run_name}2", "weights", "best.pt"),
        os.path.join(output_dir, f"{run_name}3", "weights", "best.pt"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    for d in sorted(os.listdir(output_dir), reverse=True):
        p = os.path.join(output_dir, d, "weights", "best.pt")
        if os.path.exists(p):
            return p

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Run stages 4-6 of the privacy pipeline end-to-end."
    )
    parser.add_argument("--dataset-yaml", type=str, required=True,
                        help="Path to YOLO dataset.yaml with train/val/test splits.")
    parser.add_argument("--test-images", type=str, default=None,
                        help="Directory with images to anonymise (stage 5).")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Base output directory for training and evaluation.")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to existing model weights (skips search).")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip stage 4, use --model or find existing weights.")
    parser.add_argument("--skip-anonymise", action="store_true",
                        help="Skip stage 5 (anonymisation).")
    parser.add_argument("--run-name", type=str, default="construction",
                        help="Training run name (default: construction).")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs (default: 100).")
    parser.add_argument("--batch", type=int, default=16,
                        help="Training batch size (default: 16).")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Training image size (default: 640).")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (default: 20).")
    parser.add_argument("--max-anonymise", type=int, default=None,
                        help="Limit anonymisation to N images (for testing).")
    parser.add_argument("--zones-config", type=str, default=None,
                        help="Path to JSON config with static zone masks per camera setup.")
    args = parser.parse_args()

    total_start = time.time()
    timings = {}
    model_path = args.model

    if not args.skip_training:
        stage4_cmd = [
            sys.executable, os.path.join(SCRIPTS_DIR, "04_train_detector.py"),
            "--dataset-yaml", args.dataset_yaml,
            "--model", "yolo11n.pt",
            "--epochs", str(args.epochs),
            "--imgsz", str(args.imgsz),
            "--batch", str(args.batch),
            "--project", args.output_dir,
            "--name", args.run_name,
            "--patience", str(args.patience),
        ]
        timings["stage4"] = run_command(stage4_cmd, "Stage 4: Train YOLO11")

    if not model_path:
        model_path = find_best_weights(args.output_dir, args.run_name)
        if not model_path:
            log.error("No trained model found. Run training first or use --model.")
            sys.exit(1)

    log.info("Using model: %s", model_path)

    if not args.skip_anonymise and args.test_images:
        anon_dir = os.path.join(args.output_dir, "anonymized")
        stage5_cmd = [
            sys.executable, os.path.join(SCRIPTS_DIR, "05_anonymize.py"),
            "--input-dir", args.test_images,
            "--output-dir", anon_dir,
            "--model", model_path,
            "--blur-strength", "51",
            "--body-blur-strength", "31",
            "--face-high-thresh", "0.5",
        ]
        if args.max_anonymise:
            stage5_cmd.extend(["--max-images", str(args.max_anonymise)])
        if args.zones_config:
            stage5_cmd.extend(["--zones-config", args.zones_config])

        timings["stage5"] = run_command(stage5_cmd, "Stage 5: Anonymise")

    eval_dir = os.path.join(args.output_dir, "evaluation")
    stage6_cmd = [
        sys.executable, os.path.join(SCRIPTS_DIR, "06_evaluate.py"),
        "--model", model_path,
        "--dataset-yaml", args.dataset_yaml,
        "--output-dir", eval_dir,
        "--save-visualizations",
    ]
    timings["stage6"] = run_command(stage6_cmd, "Stage 6: Evaluate")

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 60)
    print("STAGES 4-6 COMPLETE")
    print("=" * 60)
    if "stage4" in timings:
        print(f"  Stage 4 (Training):        {timings['stage4']:.0f}s ({timings['stage4']/60:.1f} min)")
    if "stage5" in timings:
        print(f"  Stage 5 (Anonymise):       {timings['stage5']:.0f}s")
    print(f"  Stage 6 (Evaluate):        {timings['stage6']:.0f}s")
    print(f"  Total:                     {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print()
    print("  Outputs:")
    print(f"    Model:       {model_path}")
    print(f"    Evaluation:  {eval_dir}/evaluation_results.json")
    print(f"    CSV:         {eval_dir}/evaluation_results.csv")
    if "stage5" in timings:
        print(f"    Anonymised:  {os.path.join(args.output_dir, 'anonymized')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
