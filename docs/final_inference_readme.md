# Multi-Model Final Inference (Advanced)

> **This script is optional.** If your single model from Stage 4 achieves satisfactory results across all classes, use Stage 5 (`05_anonymize.py`) directly. Multi-model inference is designed for cases where individual classes — particularly face detection — benefit from a dedicated specialist model.

Combines 2-5 specialised YOLO models at inference time via a YAML configuration file. Each model is assigned specific classes; overlapping assignments are resolved via Weighted Boxes Fusion (WBF). The anonymisation logic is identical to Stage 5, including dynamic kernel sizing, confidence-based tier escalation, zone masking, and timestamp overlay.

## When to use this

- Face detection recall from a single model is insufficient for GDPR compliance
- Specific classes (e.g. vehicles, text) benefit from a dedicated detector
- You want to combine models trained at different iterations or on different data subsets

## When you do NOT need this

If your single model from Stage 4 achieves satisfactory results across all classes, skip this script entirely:

```bash
python scripts/05_anonymize.py \
    --model /path/to/model/weights/best.pt \
    --input-dir /path/to/images \
    --output-dir /path/to/output
```

## Quick Start

```bash
# Generate a default config
python scripts/final_inference.py --generate-config config/inference.yaml

# Run with default 3-model setup
python scripts/final_inference.py \
    --input-dir /path/to/images \
    --output-dir output/ \
    --anonymise

# Run with custom config, visualisation, and zone masks
python scripts/final_inference.py \
    --config config/inference.yaml \
    --input-dir images/ \
    --output-dir output/ \
    --anonymise \
    --vis \
    --zones-config config/zones.json
```

## Command-line arguments

| Argument          | Default | Description                              |
|-------------------|---------|------------------------------------------|
| `--config`        | None    | Path to YAML config (uses defaults if omitted) |
| `--input-dir`     | —       | Directory with images to process         |
| `--output-dir`    | output  | Output directory                         |
| `--anonymise`     | False   | Enable three-tier anonymisation          |
| `--vis`           | False   | Generate bounding box visualisations     |
| `--max-images`    | None    | Limit number of images (for testing)     |
| `--no-metadata`   | False   | Disable timestamp overlay on output      |
| `--zones-config`  | None    | JSON config for static zone masks        |

## Configuration

The config file controls model paths, class assignments, fusion, and anonymisation parameters. Generate a default with `--generate-config`:

```yaml
class_names:
  - face
  - person
  - vehicle
  - text_or_logo
  - crane
  - container
  - scaffolding
  - material_stack

models:
  - path: weights/v12_general.pt
    classes: [person, vehicle, text_or_logo, crane, container, scaffolding, material_stack]
    confidence: 0.15
  - path: weights/face_only.pt
    classes: [face]
    confidence: 0.10
  - path: weights/v11_person_supplement.pt
    classes: [person]
    confidence: 0.20

wbf:
  enabled: true
  iou_threshold: 0.30

face_clipping:
  enabled: true
  max_height_ratio: 0.40
  clip_to_ratio: 0.30

anonymisation:
  base_kernel: 51
  body_kernel: 31
  face_high_conf: 0.50
  padding_pct: 10
  person_fallback_ratio: 0.33
  anonymise_classes: [face, person, text_or_logo]
```

### Adding a model

Add an entry to `models`:

```yaml
models:
  - path: weights/v12_general.pt
    classes: [person, vehicle, text_or_logo, crane, container, scaffolding, material_stack]
    confidence: 0.15
  - path: weights/face_only.pt
    classes: [face]
    confidence: 0.10
  - path: weights/my_vehicle_specialist.pt
    classes: [vehicle]
    confidence: 0.20
```

When multiple models detect the same class, WBF automatically fuses overlapping boxes into a single averaged box.

### Removing a model

Delete or comment out the entry. For a single model:

```yaml
models:
  - path: weights/my_single_model.pt
    classes: [face, person, vehicle, text_or_logo, crane, container, scaffolding, material_stack]
    confidence: 0.15
```

### Changing classes

Modify `class_names` and update `classes` in each model accordingly. The script maps model outputs to the global class list by name.

## Architecture

### Merging strategy

1. Each model runs independently on the image
2. Detections are filtered to only the assigned classes
3. Class names from each model are mapped to the global `class_names` list
4. Per-class WBF merges overlapping boxes from different models (if enabled)
5. Face boxes exceeding `max_height_ratio` of a containing person are clipped to `clip_to_ratio`

### Three-tier anonymisation

Identical to Stage 5 (`05_anonymize.py`):

- **Tier 1:** High-confidence face (>= `face_high_conf`) — face region blurred with dynamic kernel + padding
- **Tier 1.5:** Small person reclassified as face — blurred with padding
- **Tier 2:** Low-confidence face — face region + doubled padding
- **Tier 3:** Person without detected face — top `person_fallback_ratio` of person box blurred
- All classes in `anonymise_classes` (excluding face/person) are fully blurred with dynamic kernel + padding

Additional features shared with Stage 5:
- Dynamic kernel sizing (proportional to bounding box dimensions)
- Safety padding around all blur regions
- Static zone masking (via `--zones-config`)
- Timestamp overlay from filename (disable with `--no-metadata`)

## Output

```
output_dir/
  labels/       # YOLO format labels per image
  anonymised/   # Anonymised images (with --anonymise)
  vis/          # Bounding box visualisations (with --vis)
```

## Training your own specialised model

### Step 1: Run the full pipeline (Stages 1-6)

```bash
python scripts/run_stages_1_to_3.py --raw-images /path/to/images --work-dir /path/to/workdir
# Review in CVAT, then:
python scripts/run_stages_4_to_6.py --dataset-yaml /path/to/dataset.yaml --test-images /path/to/images --output-dir /path/to/output
```

### Step 2: Train a class-specific model

Extract labels for your target class and train:

```python
from ultralytics import YOLO

model = YOLO("yolo11s.pt")
model.train(
    data="/path/to/class_specific/dataset.yaml",
    epochs=250,
    imgsz=640,
    batch=8,
    patience=30,
)
```

### Step 3: Add to config

```yaml
models:
  - path: weights/existing_model.pt
    classes: [person, vehicle, crane, container, scaffolding, material_stack]
    confidence: 0.15
  - path: weights/my_new_face_model.pt
    classes: [face]
    confidence: 0.10
```

### Tips

- **Face**: Train a dedicated face-only model using SAHI tiled inference for annotation generation. Filter labels to keep only faces near detected persons.
- **Small objects**: Use SAHI during both annotation and inference.
- **Site adaptation**: Fine-tune on 10-30 manually annotated images from a new site.

## Outlook

The multi-model architecture is designed to be extensible. Potential directions beyond this thesis:

- **Automated model selection**: Evaluate per-class performance on a validation set and automatically select the best-performing model per class, removing the need for manual configuration.
- **Shared anonymisation module**: Extract the anonymisation logic (currently duplicated between `05_anonymize.py` and `final_inference.py`) into a shared module to ensure consistency and reduce maintenance overhead.
- **Streaming / real-time mode**: Extend the inference loop to process frames from a live camera feed rather than a static directory, enabling continuous on-premises monitoring.
- **Additional specialist models**: Integrate non-YOLO detectors (e.g. SCRFD or RetinaFace for faces, PaddleOCR for text) as additional model entries without changing the fusion or anonymisation logic.

## Pre-trained weights

Pre-trained weights for this thesis are available as a GitHub Release:

| Model | File | Classes |
|-------|------|---------|
| General detector | `v12_general.pt` | person, vehicle, text_or_logo, crane, container, scaffolding, material_stack |
| Face detector | `face_only.pt` | face |
| Person supplement | `v11_person_supplement.pt` | person (additional detections) |
