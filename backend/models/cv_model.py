"""
models/cv_model.py — YOLOv8 visual dark-pattern detection pipeline
===================================================================
Loads a fine-tuned YOLOv8n model (or falls back to YOLOv8n pretrained
on COCO) to detect visual signals of dark patterns in page screenshots.
"""

from __future__ import annotations
import base64
import io
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from loguru import logger

# ─── Constants ────────────────────────────────────────────────────────────────
CHECKPOINT_DIR   = Path(__file__).parent.parent / "checkpoints" / "cv"
FINE_TUNED_MODEL = CHECKPOINT_DIR / "best.pt"
YOLO_BASE_MODEL  = "yolov8n.pt"   # downloaded automatically by ultralytics

# YOLO class index → dark pattern category mapping
# These are the classes expected in the fine-tuned model.
# The pretrained COCO model uses them as proxy classes (see COCO_PROXY).
YOLO_CLASSES: Dict[int, Dict] = {
    0: {"category": "urgency",         "label": "countdown_timer",   "severity": "high"},
    1: {"category": "optin",           "label": "prechecked_box",    "severity": "high"},
    2: {"category": "confirm_shaming", "label": "shame_button",      "severity": "high"},
    3: {"category": "hidden_flows",    "label": "hidden_link",       "severity": "medium"},
    4: {"category": "disguised_ads",   "label": "ad_container",      "severity": "medium"},
    5: {"category": "urgency",         "label": "scarcity_badge",    "severity": "medium"},
    6: {"category": "optin",           "label": "consent_banner",    "severity": "low"},
}

# COCO class ID → dark pattern proxy (used when no fine-tuned model exists)
COCO_PROXY: Dict[int, Dict] = {
    # clock → urgency timer proxy
    74: {"category": "urgency",       "label": "clock_element",    "severity": "low",    "conf_floor": 0.50},
    # person → social proof / FOMO signal
    0:  {"category": "urgency",       "label": "social_proof",     "severity": "low",    "conf_floor": 0.60},
    # laptop / tv / cell phone → ad container proxy
    63: {"category": "disguised_ads", "label": "media_container",  "severity": "low",    "conf_floor": 0.55},
    67: {"category": "disguised_ads", "label": "phone_ad",         "severity": "low",    "conf_floor": 0.55},
}

CONFIDENCE_THRESHOLD = 0.40   # minimum YOLO confidence to report a detection


class CVModel:
    """
    YOLOv8-based visual dark-pattern detector.
    Decodes a base64 JPEG/PNG screenshot and runs inference.
    """

    def __init__(self):
        self._model = None
        self._model_type: str = "none"
        self._class_map: Dict = {}
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO

            if FINE_TUNED_MODEL.exists():
                self._model = YOLO(str(FINE_TUNED_MODEL))
                self._model_type = "fine_tuned"
                self._class_map = YOLO_CLASSES
                logger.success(f"[CV] Fine-tuned YOLOv8 loaded from {FINE_TUNED_MODEL}")
            else:
                logger.info("[CV] No fine-tuned checkpoint — using YOLOv8n-COCO with proxy classes")
                self._model = YOLO(YOLO_BASE_MODEL)
                self._model_type = "pretrained_coco"
                self._class_map = {}
                logger.success("[CV] YOLOv8n-COCO loaded (proxy detection mode)")

        except ImportError:
            logger.error("[CV] ultralytics not installed — CV detection disabled")
        except Exception as e:
            logger.error(f"[CV] Model load failed: {e}")

    # ─── Public API ───────────────────────────────────────────────
    def analyze(self, screenshot_b64: Optional[str]) -> List[Dict]:
        """
        Decode base64 screenshot and run YOLO detection.
        Returns list of detection dicts.
        """
        if not screenshot_b64 or not self._model:
            return []

        t0 = time.time()
        try:
            image = self._decode_image(screenshot_b64)
            if image is None:
                return []

            results = self._model.predict(
                source=image,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False,
                imgsz=640,
            )

            detections = self._parse_results(results)
            elapsed = int((time.time() - t0) * 1000)
            logger.debug(f"[CV] Inference complete: {len(detections)} detections in {elapsed}ms")
            return detections

        except Exception as e:
            logger.error(f"[CV] Inference error: {e}")
            return []

    @property
    def model_type(self) -> str:
        return self._model_type

    # ─── Image decoding ───────────────────────────────────────────
    def _decode_image(self, b64_str: str):
        """Decode a data URI or raw base64 string into a PIL Image."""
        try:
            from PIL import Image

            # Strip data URI prefix if present
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]

            raw = base64.b64decode(b64_str)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:
            logger.warning(f"[CV] Image decode error: {e}")
            return None

    # ─── Result parsing ───────────────────────────────────────────
    def _parse_results(self, results) -> List[Dict]:
        detections = []

        for result in results:
            if result.boxes is None:
                continue

            img_h, img_w = result.orig_shape

            for box in result.boxes:
                cls_id     = int(box.cls[0])
                confidence = float(box.conf[0])
                xyxy       = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

                if self._model_type == "fine_tuned":
                    info = YOLO_CLASSES.get(cls_id)
                    if not info:
                        continue
                    if confidence < CONFIDENCE_THRESHOLD:
                        continue
                else:
                    # COCO proxy mode
                    info = COCO_PROXY.get(cls_id)
                    if not info:
                        continue
                    if confidence < info.get("conf_floor", CONFIDENCE_THRESHOLD):
                        continue

                x1, y1, x2, y2 = [int(v) for v in xyxy]

                detections.append({
                    "type":        info["category"],
                    "severity":    info["severity"],
                    "confidence":  round(confidence, 3),
                    "selector":    None,  # visual detection — no DOM selector
                    "rect": {
                        "x": x1, "y": y1,
                        "w": x2 - x1, "h": y2 - y1
                    },
                    "description": f"[CV/{info['label']}] Visual dark pattern detected ({confidence:.0%} confidence)",
                    "source":      "cv",
                    "text":        None,
                })

        return detections
