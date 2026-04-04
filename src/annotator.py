"""
Multi-model auto-annotation engine for construction site images.

Combines outputs from multiple detection models to generate comprehensive
pseudo-labels for training data. Model predictions serve as initial labels
for subsequent human refinement in CVAT.

Detection fusion uses Weighted Boxes Fusion (WBF) instead of standard NMS,
producing superior bounding box coordinates by averaging predictions across
models rather than discarding lower-confidence overlapping detections.

Models used:
    - Grounding DINO: High-accuracy open-set detection for faces, persons,
      vehicles, and construction objects. Apache-2.0 licence.
    - YOLO-World: Fast open-vocabulary detection for text/logo elements.
      GPL-3.0 licence.
    - YOLO-Face: Specialised face detection. GPL-3.0 licence.
    - YOLO11 COCO: General object detection for persons and vehicles.
      AGPL-3.0 licence.

Note: This module is the class-based version. The self-contained
03_auto_annotate.py script is the primary implementation used in the
pipeline. This module is retained for backwards compatibility.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import cv2
import numpy as np
from tqdm import tqdm

from src.utils.nms import weighted_boxes_fusion, cross_model_nms, link_faces_to_persons
from src.utils.converters import detections_to_yolo_txt, detections_to_coco
from src.utils.image_quality import get_image_dimensions

logger = logging.getLogger(__name__)

CLASS_NAMES = [
    "face", "person", "vehicle", "text_or_logo",
    "crane", "container", "scaffolding", "material_stack"
]

COCO_TO_TAXONOMY = {
    0: ("person", 1),
    2: ("vehicle", 2),
    3: ("vehicle", 2),
    5: ("vehicle", 2),
    7: ("vehicle", 2),
}


class MultiModelAnnotator:
    """
    Generates pseudo-labels using multiple complementary detection models.

    The annotator runs each model independently on the input image, maps
    detections to the unified 8-class taxonomy, applies cross-model NMS
    to remove duplicates, and links face detections to person detections.

    Attributes:
        input_dir: Directory containing images to annotate.
        output_dir: Directory for annotation outputs.
        use_grounding_dino: Whether to use Grounding DINO.
        use_yolo_face: Whether to use YOLO-Face model.
        use_yolo_world: Whether to use YOLO-World for text/logos.
        checkpoint_interval: Save progress every N images.
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        use_grounding_dino: bool = True,
        use_yolo_face: bool = True,
        use_yolo_world: bool = True,
        grounding_dino_config: Optional[str] = None,
        grounding_dino_weights: Optional[str] = None,
        checkpoint_interval: int = 100,
        cross_model_iou: float = 0.5
    ):
        """
        Initialise the multi-model annotator.

        Args:
            input_dir: Path to images.
            output_dir: Path for outputs.
            use_grounding_dino: Enable Grounding DINO.
            use_yolo_face: Enable YOLO-Face.
            use_yolo_world: Enable YOLO-World.
            grounding_dino_config: Path to Grounding DINO config.
            grounding_dino_weights: Path to Grounding DINO weights.
            checkpoint_interval: Checkpoint frequency.
            cross_model_iou: IoU threshold for cross-model NMS.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.use_grounding_dino = use_grounding_dino
        self.use_yolo_face = use_yolo_face
        self.use_yolo_world = use_yolo_world
        self.gd_config = grounding_dino_config
        self.gd_weights = grounding_dino_weights
        self.checkpoint_interval = checkpoint_interval
        self.cross_model_iou = cross_model_iou

        self.models = {}
        self.all_detections = {}
        self.image_info_list = []

        self._setup_directories()

    def _setup_directories(self) -> None:
        """Create output directory structure."""
        for subdir in ["annotations", "yolo_labels", "checkpoints", "visualizations"]:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)

    def load_models(self) -> None:
        """Load all configured detection models."""
        from ultralytics import YOLO

        logger.info("Loading YOLO11 COCO pretrained...")
        self.models["coco"] = YOLO("yolo11n.pt")

        if self.use_yolo_face:
            logger.info("Loading YOLO-Face...")
            face_model_paths = [
                Path(__file__).parent.parent / "models" / "yolov8n-face.pt",
                Path("/workspace/privacy_pipeline/models/yolov8n-face.pt"),
                Path("yolov8n-face.pt"),
            ]
            loaded = False
            for face_path in face_model_paths:
                if face_path.exists():
                    self.models["face"] = YOLO(str(face_path))
                    logger.info(f"YOLO-Face loaded from: {face_path}")
                    loaded = True
                    break
            if not loaded:
                logger.warning(
                    "YOLO-Face model not found. Looked in: "
                    f"{[str(p) for p in face_model_paths]}. "
                    "Download from https://github.com/akanametov/yolo-face "
                    "and place in models/yolov8n-face.pt. "
                    "Continuing without face specialist model."
                )
                self.use_yolo_face = False

        if self.use_yolo_world:
            logger.info("Loading YOLO-World...")
            self.models["yolo_world"] = YOLO("yolov8s-worldv2")
            self.models["yolo_world"].set_classes([
                "text", "sign", "logo", "company name", "license plate"
            ])

        if self.use_grounding_dino and self.gd_config and self.gd_weights:
            logger.info("Loading Grounding DINO...")
            try:
                from groundingdino.util.inference import load_model
                self.models["grounding_dino"] = load_model(
                    self.gd_config, self.gd_weights
                )
                logger.info("Grounding DINO loaded successfully")
            except ImportError:
                logger.warning(
                    "Grounding DINO not installed. Install from: "
                    "https://github.com/IDEA-Research/GroundingDINO"
                )
                self.use_grounding_dino = False
            except Exception as e:
                logger.warning(f"Failed to load Grounding DINO: {e}")
                self.use_grounding_dino = False

        logger.info(f"Models loaded: {list(self.models.keys())}")

    def _detect_coco(self, image_path: str, confidence: float = 0.35) -> List[Dict]:
        """
        Run COCO pretrained YOLO11 and map to taxonomy classes.

        Args:
            image_path: Path to the image.
            confidence: Minimum detection confidence.

        Returns:
            List of detection dicts mapped to the project taxonomy.
        """
        results = self.models["coco"](image_path, conf=confidence, verbose=False)
        detections = []

        for r in results:
            for box in r.boxes:
                coco_cls = int(box.cls[0])
                if coco_cls in COCO_TO_TAXONOMY:
                    class_name, class_id = COCO_TO_TAXONOMY[coco_cls]
                    detections.append({
                        "bbox": box.xyxy[0].tolist(),
                        "confidence": float(box.conf[0]),
                        "class_id": class_id,
                        "class_name": class_name,
                        "model": "coco_yolo11",
                        "attributes": {}
                    })

        return detections

    def _detect_faces(self, image_path: str, confidence: float = 0.25) -> List[Dict]:
        """
        Run YOLO-Face model for face detection.

        Args:
            image_path: Path to the image.
            confidence: Minimum detection confidence.

        Returns:
            List of face detection dicts.
        """
        if "face" not in self.models:
            return []

        results = self.models["face"](image_path, conf=confidence, verbose=False)
        detections = []

        for r in results:
            for box in r.boxes:
                detections.append({
                    "bbox": box.xyxy[0].tolist(),
                    "confidence": float(box.conf[0]),
                    "class_id": 0,
                    "class_name": "face",
                    "model": "yolo_face",
                    "attributes": {"visibility": "unspecified", "has_helmet": "unspecified"}
                })

        return detections

    def _detect_yolo_world(self, image_path: str, confidence: float = 0.15) -> List[Dict]:
        """
        Run YOLO-World for text and logo detection.

        Args:
            image_path: Path to the image.
            confidence: Minimum detection confidence.

        Returns:
            List of text/logo detection dicts.
        """
        if "yolo_world" not in self.models:
            return []

        results = self.models["yolo_world"](image_path, conf=confidence, verbose=False)
        detections = []

        for r in results:
            for box in r.boxes:
                detections.append({
                    "bbox": box.xyxy[0].tolist(),
                    "confidence": float(box.conf[0]),
                    "class_id": 3,
                    "class_name": "text_or_logo",
                    "model": "yolo_world",
                    "attributes": {"is_logo": "unspecified", "is_text": "unspecified"}
                })

        return detections

    def _detect_grounding_dino(
        self,
        image_path: str,
        box_threshold: float = 0.30,
        text_threshold: float = 0.25
    ) -> List[Dict]:
        """Run Grounding DINO for open-set object detection.

        Provides high-quality detections particularly for faces and
        persons in challenging conditions.

        Args:
            image_path: Path to the image.
            box_threshold: Minimum box confidence.
            text_threshold: Minimum text-box similarity.

        Returns:
            List of detection dicts from Grounding DINO.
        """
        if "grounding_dino" not in self.models:
            return []

        from groundingdino.util.inference import load_image, predict

        prompt_map = {
            "human face": ("face", 0),
            "head with helmet": ("face", 0),
            "face with safety helmet": ("face", 0),
            "person": ("person", 1),
            "worker": ("person", 1),
            "construction worker": ("person", 1),
            "pedestrian": ("person", 1),
            "vehicle": ("vehicle", 2),
            "car": ("vehicle", 2),
            "truck": ("vehicle", 2),
            "excavator": ("vehicle", 2),
            "forklift": ("vehicle", 2),
            "concrete mixer": ("vehicle", 2),
            "text": ("text_or_logo", 3),
            "sign": ("text_or_logo", 3),
            "logo": ("text_or_logo", 3),
            "license plate": ("text_or_logo", 3),
            "crane": ("crane", 4),
            "tower crane": ("crane", 4),
            "shipping container": ("container", 5),
            "site container": ("container", 5),
            "scaffolding": ("scaffolding", 6),
            "scaffold": ("scaffolding", 6),
            "material stack": ("material_stack", 7),
            "building materials": ("material_stack", 7),
        }

        text_prompt = " . ".join(prompt_map.keys()) + " ."

        image_source, image_tensor = load_image(image_path)
        boxes, logits, phrases = predict(
            model=self.models["grounding_dino"],
            image=image_tensor,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )

        h, w = image_source.shape[:2]
        detections = []

        for box, logit, phrase in zip(boxes, logits, phrases):
            cx, cy, bw, bh = box.tolist()
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h

            class_name, class_id = "person", 1
            phrase_lower = phrase.lower().strip()
            for key, (cname, cid) in prompt_map.items():
                if key in phrase_lower or phrase_lower in key:
                    class_name, class_id = cname, cid
                    break

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": float(logit),
                "class_id": class_id,
                "class_name": class_name,
                "model": "grounding_dino",
                "phrase": phrase,
                "attributes": {}
            })

        return detections

    def annotate_image(self, image_path: str) -> List[Dict]:
        """
        Run all detection models on a single image and merge results.

        Args:
            image_path: Path to the image.

        Returns:
            Merged and deduplicated list of detections.
        """
        all_dets = []

        all_dets.extend(self._detect_coco(image_path))
        all_dets.extend(self._detect_faces(image_path))
        all_dets.extend(self._detect_yolo_world(image_path))

        if self.use_grounding_dino:
            all_dets.extend(self._detect_grounding_dino(image_path))

        merged = weighted_boxes_fusion(all_dets, self.cross_model_iou)
        merged = link_faces_to_persons(merged)

        return merged

    def _save_checkpoint(self, processed: int) -> None:
        """Save annotation progress checkpoint."""
        checkpoint = {
            "processed": processed,
            "timestamp": datetime.now().isoformat(),
            "total_images": len(self.image_info_list),
            "total_detections": sum(len(d) for d in self.all_detections.values())
        }
        path = self.output_dir / "checkpoints" / f"checkpoint_{processed}.json"
        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2)

    def _load_last_checkpoint(self) -> int:
        """Load the most recent checkpoint and return processed count."""
        cp_dir = self.output_dir / "checkpoints"
        checkpoints = sorted(cp_dir.glob("checkpoint_*.json"))
        if not checkpoints:
            return 0
        with open(checkpoints[-1]) as f:
            data = json.load(f)
        return data.get("processed", 0)

    def _export_cvat_labels(self) -> None:
        """
        Generate CVAT-compatible label configuration JSON.

        This file can be pasted into CVAT's Raw Label Editor when
        creating a new project, avoiding manual label setup.
        """
        import json as json_mod
        taxonomy_path = Path(__file__).parent.parent / "config" / "taxonomy.json"

        if taxonomy_path.exists():
            with open(taxonomy_path) as f:
                taxonomy = json_mod.load(f)

            cvat_labels = []
            for cls in taxonomy["classes"]:
                label = {"name": cls["name"], "color": cls["color"], "attributes": []}
                for attr_name, attr_def in cls.get("attributes", {}).items():
                    label["attributes"].append({
                        "name": attr_name,
                        "mutable": False,
                        "input_type": "select",
                        "default_value": attr_def.get("default", "unspecified"),
                        "values": attr_def.get("values", [])
                    })
                cvat_labels.append(label)

            cvat_path = self.output_dir / "cvat_labels.json"
            with open(cvat_path, "w") as f:
                json_mod.dump(cvat_labels, f, indent=2)
            logger.info(f"CVAT labels exported to {cvat_path}")

    def _create_cvat_import_zip(self, coco_json_path: Path) -> None:
        """
        Create a ZIP file for CVAT annotation import.

        CVAT expects a ZIP containing annotations/instances_default.json
        in COCO 1.0 format.

        Args:
            coco_json_path: Path to the COCO JSON annotations file.
        """
        import zipfile
        zip_path = self.output_dir / "cvat_import.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(str(coco_json_path), "annotations/instances_default.json")
        logger.info(f"CVAT import ZIP created: {zip_path}")

    def run(self, resume: bool = True) -> Dict:
        """
        Execute auto-annotation on all images.

        Args:
            resume: If True, continue from last checkpoint.

        Returns:
            Statistics dictionary.
        """
        self.load_models()

        extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        images = []
        for ext in extensions:
            images.extend(self.input_dir.rglob(f"*{ext}"))
        images = sorted(set(images))

        start_idx = 0
        if resume:
            start_idx = self._load_last_checkpoint()
            if start_idx > 0:
                logger.info(f"Resuming from image {start_idx}")

        logger.info(f"Annotating {len(images) - start_idx} images...")

        for idx, img_path in enumerate(tqdm(images[start_idx:], desc="Annotating")):
            dims = get_image_dimensions(str(img_path))
            if dims is None:
                continue
            w, h = dims

            img_id = start_idx + idx + 1
            self.image_info_list.append({
                "id": img_id,
                "file_name": img_path.name,
                "width": w,
                "height": h
            })

            detections = self.annotate_image(str(img_path))
            self.all_detections[img_path.name] = detections

            yolo_txt = detections_to_yolo_txt(detections, w, h)
            label_path = self.output_dir / "yolo_labels" / f"{img_path.stem}.txt"
            with open(label_path, "w") as f:
                f.write(yolo_txt)

            if (start_idx + idx + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(start_idx + idx + 1)

        coco_data = detections_to_coco(
            self.all_detections, self.image_info_list, CLASS_NAMES
        )
        coco_path = self.output_dir / "annotations" / "annotations.json"
        with open(coco_path, "w") as f:
            json.dump(coco_data, f, indent=2)

        classes_path = self.output_dir / "yolo_labels" / "classes.txt"
        with open(classes_path, "w") as f:
            f.write("\n".join(CLASS_NAMES))

        self._export_cvat_labels()
        self._create_cvat_import_zip(coco_path)

        stats = {
            "total_images": len(self.image_info_list),
            "total_detections": sum(len(d) for d in self.all_detections.values()),
            "detections_per_class": {},
            "models_used": list(self.models.keys()),
            "timestamp": datetime.now().isoformat()
        }

        for dets in self.all_detections.values():
            for d in dets:
                cn = d["class_name"]
                stats["detections_per_class"][cn] = \
                    stats["detections_per_class"].get(cn, 0) + 1

        stats_path = self.output_dir / "annotation_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Annotation complete: {stats}")
        return stats
