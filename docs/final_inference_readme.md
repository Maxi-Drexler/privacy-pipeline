# Multi-Model Final Inference

Combines 2-5 specialised YOLO models at inference time via a YAML configuration file. Each model is assigned specific classes; overlapping assignments are resolved via Weighted Boxes Fusion (WBF).

## Quick Start

```bash
# Generate a default config
python scripts/final_inference.py --generate-config config/inference.yaml

# Run with default 3-model setup
python scripts/final_inference.py --input-dir /path/to/images --output-dir output/ --anonymise

# Run with custom config
python scripts/final_inference.py --config config/inference.yaml --input-dir images/ --output-dir output/ --anonymise --vis
```

## Configuration

The config file controls everything. Generate a default with `--generate-config`:

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
  blur_kernel: 51
  face_high_conf: 0.50
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
  - path: weights/my_vehicle_specialist.pt    # your new model
    classes: [vehicle]                         # only uses vehicle detections
    confidence: 0.20
```

When multiple models detect the same class (e.g. both `v12_general` and `my_vehicle_specialist` detect `vehicle`), WBF automatically fuses the overlapping boxes into a single averaged box.

### Removing a model

Delete or comment out the entry. With a single model:

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

- Tier 1: High-confidence face (>= `face_high_conf`) -- face region blurred
- Tier 2: Low-confidence face -- face region + 20% buffer blurred
- Tier 3: Person without detected face -- top `person_fallback_ratio` of person box blurred
- All classes in `anonymise_classes` (excluding face/person) are fully blurred

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
# Review in CVAT
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

## When you do NOT need this

If your single model from Stage 4 achieves satisfactory results across all classes, skip this script and use Stage 5 directly:

```bash
python scripts/05_anonymise.py \
    --model /path/to/model/weights/best.pt \
    --input-dir /path/to/images \
    --output-dir /path/to/output
```
