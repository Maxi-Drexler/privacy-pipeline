#!/usr/bin/env python3
"""
Run stages 1-3 of the privacy pipeline end-to-end.

Takes a raw image directory as input and produces auto-annotated
output ready for manual correction in CVAT.

Stages:
  1. Pre-Filter: Quality-based filtering (no ML model).
     Three dimensions: exposure, sharpness, edge density.
  2. Intelligent Selection: YOLO11 COCO proxy scoring with
     temporal diversity constraints. Selects ~2500 images.
  3. Auto-Annotation: Four-model ensemble (YOLO11 COCO, YOLO-Face,
     YOLO-World, Grounding DINO) + WBF + CLIP enrichment + OCR.

Usage:
    python run_stages_1_to_3.py \
        --raw-images /workspace/data/TUM_KITA_imgs \
        --work-dir /workspace/data \
        --num-samples 2500

    # Quick test with 100 images:
    python run_stages_1_to_3.py \
        --raw-images /workspace/data/TUM_KITA_imgs \
        --work-dir /workspace/data \
        --num-samples 50 \
        --max-prefilter 200

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
GDINO_CONFIG = "/workspace/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GDINO_WEIGHTS = "/workspace/GroundingDINO/weights/groundingdino_swint_ogc.pth"


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


def main():
    parser = argparse.ArgumentParser(
        description="Run stages 1-3 of the privacy pipeline end-to-end."
    )
    parser.add_argument("--raw-images", type=str, required=True,
                        help="Directory with raw camera images.")
    parser.add_argument("--work-dir", type=str, required=True,
                        help="Working directory for intermediate outputs.")
    parser.add_argument("--num-samples", type=int, default=2500,
                        help="Number of images to select (default: 2500).")
    parser.add_argument("--max-prefilter", type=int, default=None,
                        help="Limit pre-filter to N images (for testing).")
    parser.add_argument("--wbf-iou", type=float, default=0.35,
                        help="WBF IoU threshold for stage 3 (default: 0.35).")
    parser.add_argument("--skip-gdino", action="store_true",
                        help="Skip Grounding DINO in stage 3.")
    parser.add_argument("--use-sahi", action="store_true",
                        help="Enable SAHI tile-based inference in stage 3.")
    parser.add_argument("--sahi-slice-size", type=int, default=640,
                        help="SAHI tile size (default: 640).")
    parser.add_argument("--sahi-overlap", type=float, default=0.25,
                        help="SAHI tile overlap ratio (default: 0.25).")
    parser.add_argument("--zones-config", type=str, default=None,
                        help="Path to zones.json — used in Stage 1 (crop) and Stage 3.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers for Stage 1 (default: cpu_count - 1).")
    parser.add_argument("--min-brightness", type=float, default=30.0,
                        help="Pre-filter min brightness (default: 30).")
    parser.add_argument("--max-brightness", type=float, default=250.0,
                        help="Pre-filter max brightness (default: 250).")
    parser.add_argument("--min-contrast", type=float, default=15.0,
                        help="Pre-filter min contrast (default: 15).")
    parser.add_argument("--min-sharpness", type=float, default=50.0,
                        help="Pre-filter min sharpness (default: 50).")
    parser.add_argument("--min-edge-density", type=float, default=0.01,
                        help="Pre-filter min edge density (default: 0.01).")
    args = parser.parse_args()

    prefiltered_dir = os.path.join(args.work_dir, "prefiltered")
    selected_dir = os.path.join(args.work_dir, "selected")
    annotations_dir = os.path.join(args.work_dir, "auto_annotations")

    total_start = time.time()
    timings = {}

    stage1_cmd = [
        sys.executable, os.path.join(SCRIPTS_DIR, "01_prefilter.py"),
        "--input-dir", args.raw_images,
        "--output-dir", prefiltered_dir,
        "--min-brightness", str(args.min_brightness),
        "--max-brightness", str(args.max_brightness),
        "--min-contrast", str(args.min_contrast),
        "--min-sharpness", str(args.min_sharpness),
        "--min-edge-density", str(args.min_edge_density),
    ]
    if args.max_prefilter:
        stage1_cmd.extend(["--max-images", str(args.max_prefilter)])
    if args.zones_config:
        stage1_cmd.extend(["--zones-config", args.zones_config])
    if args.workers:
        stage1_cmd.extend(["--workers", str(args.workers)])

    timings["stage1"] = run_command(stage1_cmd, "Stage 1: Pre-Filter")

    stage2_cmd = [
        sys.executable, os.path.join(SCRIPTS_DIR, "02_intelligent_selection.py"),
        "--input-dir", prefiltered_dir,
        "--output-dir", selected_dir,
        "--num-samples", str(args.num_samples),
        "--index-file", os.path.join(prefiltered_dir, "accepted_images.csv"),
    ]

    timings["stage2"] = run_command(stage2_cmd, "Stage 2: Intelligent Selection")

    selected_images_dir = selected_dir
    if os.path.isdir(os.path.join(selected_dir, "images")):
        selected_images_dir = os.path.join(selected_dir, "images")

    stage3_cmd = [
        sys.executable, os.path.join(SCRIPTS_DIR, "03_auto_annotate.py"),
        "--input-dir", selected_images_dir,
        "--output-dir", annotations_dir,
        "--wbf-iou", str(args.wbf_iou),
        "--no-resume",
    ]
    if not args.skip_gdino and os.path.exists(GDINO_WEIGHTS):
        stage3_cmd.extend([
            "--grounding-dino-config", GDINO_CONFIG,
            "--grounding-dino-weights", GDINO_WEIGHTS,
        ])
    else:
        if not args.skip_gdino:
            log.warning("Grounding DINO weights not found, skipping.")

    if args.use_sahi:
        stage3_cmd.extend([
            "--use-sahi",
            "--sahi-slice-size", str(args.sahi_slice_size),
            "--sahi-overlap", str(args.sahi_overlap),
        ])

    timings["stage3"] = run_command(stage3_cmd, "Stage 3: Auto-Annotation")

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 60)
    print("STAGES 1-3 COMPLETE")
    print("=" * 60)
    print(f"  Stage 1 (Pre-Filter):          {timings['stage1']:.0f}s")
    print(f"  Stage 2 (Selection):           {timings['stage2']:.0f}s")
    print(f"  Stage 3 (Auto-Annotation):     {timings['stage3']:.0f}s")
    print(f"  Total:                         {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print()
    print("  Outputs:")
    print(f"    Pre-filtered index:  {prefiltered_dir}/accepted_images.csv")
    print(f"    Selected images:     {selected_images_dir}")
    print(f"    Auto-annotations:    {annotations_dir}")
    print()
    print("  Next step: Import annotations into CVAT for manual correction.")
    print(f"    CVAT import file:    {annotations_dir}/cvat_import.zip")
    print("=" * 60)


if __name__ == "__main__":
    main()
