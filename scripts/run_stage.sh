#!/bin/bash
# Run individual pipeline stages.
# Usage: ./run_stage.sh <stage_number> [--test] [--sahi]

set -e

DATA_DIR="/workspace/data"
OUTPUT_DIR="/workspace/output"
PIPELINE_DIR="/workspace/privacy_pipeline"
SCRIPTS_DIR="$PIPELINE_DIR/scripts"

TEST_FLAG=""
for arg in "$@"; do
    if [[ "$arg" == "--test" ]]; then
        TEST_FLAG="--max-images 50"
        echo ">>> TEST MODE: Processing max 50 images"
    fi
done

case $1 in
    1)
        echo "=== Stage 1: Pre-Filter (quality metrics only, no ML) ==="
        ZONES_FLAG=""
        ZONES_FILE="$PIPELINE_DIR/config/zones.json"
        if [ -f "$ZONES_FILE" ]; then
            ZONES_FLAG="--zones-config $ZONES_FILE"
            echo ">>> Zone crops: $ZONES_FILE"
        fi
        python $SCRIPTS_DIR/01_prefilter.py \
            --input-dir $DATA_DIR/TUM_KITA_imgs \
            --output-dir $DATA_DIR/prefiltered \
            --min-brightness 30 \
            --max-brightness 250 \
            --min-contrast 15 \
            --min-sharpness 50 \
            --min-edge-density 0.01 \
            $ZONES_FLAG $TEST_FLAG
        ;;
    2)
        echo "=== Stage 2: Intelligent Selection ==="
        python $SCRIPTS_DIR/02_intelligent_selection.py \
            --input-dir $DATA_DIR/prefiltered \
            --output-dir $DATA_DIR/selected \
            --num-samples 2500 $TEST_FLAG
        ;;
    3)
        echo "=== Stage 3: Auto-Annotation ==="
        SAHI_FLAG=""
        for arg in "$@"; do
            if [[ "$arg" == "--sahi" ]]; then
                SAHI_FLAG="--use-sahi --sahi-slice-size 640 --sahi-overlap 0.25"
                echo ">>> SAHI enabled (640x640 tiles, 0.25 overlap)"
            fi
        done
        python $SCRIPTS_DIR/03_auto_annotate.py \
            --input-dir $DATA_DIR/construction_images_selected/selected \
            --output-dir $DATA_DIR/auto_annotations \
            --wbf-iou 0.35 \
            --grounding-dino-config /workspace/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
            --grounding-dino-weights /workspace/GroundingDINO/weights/groundingdino_swint_ogc.pth \
            $SAHI_FLAG $TEST_FLAG
        ;;
    4)
        echo "=== Stage 4: Train YOLO11 ==="
        python $SCRIPTS_DIR/04_train_detector.py \
            --dataset-yaml $DATA_DIR/training/dataset.yaml \
            --model yolo11n.pt \
            --epochs 100 \
            --imgsz 640 \
            --batch 16 \
            --project $OUTPUT_DIR \
            --name construction \
            --patience 20
        ;;
    5)
        echo "=== Stage 5: Anonymise ==="
        ZONES_FLAG=""
        ZONES_FILE="$PIPELINE_DIR/config/zones.json"
        if [ -f "$ZONES_FILE" ]; then
            ZONES_FLAG="--zones-config $ZONES_FILE"
            echo ">>> Zone masks: $ZONES_FILE"
        fi
        python $SCRIPTS_DIR/05_anonymize.py \
            --input-dir $DATA_DIR/TUM_KITA_imgs \
            --output-dir $DATA_DIR/anonymized \
            --model $OUTPUT_DIR/construction/weights/best.pt \
            --blur-strength 51 \
            --body-blur-strength 31 \
            --face-high-thresh 0.5 \
            $ZONES_FLAG $TEST_FLAG
        ;;
    6)
        echo "=== Stage 6: Evaluate ==="
        python $SCRIPTS_DIR/06_evaluate.py \
            --model $OUTPUT_DIR/construction/weights/best.pt \
            --dataset-yaml $DATA_DIR/training/dataset.yaml \
            --output-dir $OUTPUT_DIR/evaluation \
            --save-visualizations
        ;;
    *)
        echo "Usage: ./run_stage.sh <1-6> [--test] [--sahi]"
        echo "  1: Pre-Filter (quality metrics)"
        echo "  2: Intelligent Selection"
        echo "  3: Auto-Annotation (add --sahi for tile-based inference)"
        echo "  4: Train YOLO11"
        echo "  5: Anonymise"
        echo "  6: Evaluate"
        exit 1
        ;;
esac

echo "=== Done ==="
