#!/usr/bin/env python3
"""
Stage 2: Intelligent dataset curation.

Scores pre-filtered images based on detection relevance, image quality,
and temporal diversity, then selects the optimal subset for annotation.

Usage:
    python scripts/02_intelligent_selection.py \
        --input-dir /workspace/data/prefiltered/relevant \
        --output-dir /workspace/data/selected \
        --num-samples 2500

Author: Maximilian Drexler
License: MIT
Models: YOLO11 — AGPL-3.0
"""

import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.selector import IntelligentImageSelector


def main():
    parser = argparse.ArgumentParser(
        description="Select optimal training images from pre-filtered dataset."
    )
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory with pre-filtered images.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory for selected images.")
    parser.add_argument("--num-samples", type=int, default=2500,
                        help="Number of images to select (default: 2500).")
    parser.add_argument("--min-gap-minutes", type=int, default=3,
                        help="Minimum minutes between selected images (default: 3).")
    parser.add_argument("--max-per-hour", type=int, default=5,
                        help="Maximum selections per hour (default: 5).")
    parser.add_argument("--person-weight", type=float, default=10.0,
                        help="Scoring weight for person detections (default: 10).")
    parser.add_argument("--vehicle-weight", type=float, default=5.0,
                        help="Scoring weight for vehicle detections (default: 5).")
    parser.add_argument("--max-score-images", type=int, default=None,
                        help="Score at most N images (for testing).")
    parser.add_argument("--index-file", type=str, default=None,
                        help="Path to accepted_images.csv from Stage 1 prefilter.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    selector = IntelligentImageSelector(
        source_dir=args.input_dir,
        target_dir=args.output_dir,
        num_samples=args.num_samples,
        min_gap_minutes=args.min_gap_minutes,
        max_per_hour=args.max_per_hour,
        person_weight=args.person_weight,
        vehicle_weight=args.vehicle_weight
    )

    stats = selector.run(max_score_images=args.max_score_images, index_file=args.index_file)

    print("\n" + "=" * 60)
    print("SELECTION RESULTS")
    print("=" * 60)
    print(f"Images scored:     {stats['total_scored']}")
    print(f"Images selected:   {stats['total_selected']}")
    print(f"Average score:     {stats['avg_score']:.2f}")
    print(f"Avg persons/img:   {stats['avg_persons']:.1f}")
    print(f"Avg vehicles/img:  {stats['avg_vehicles']:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
