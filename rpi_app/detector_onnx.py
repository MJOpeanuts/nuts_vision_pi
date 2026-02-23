#!/usr/bin/env python3
"""
detector_onnx.py – ONNX-based IC detector for nuts_vision_pi.

Loads best.onnx (YOLOv8 export) and runs inference on a BGR numpy
image.  Returns detections in the same dict format used by the rest of
the pipeline so that crop.py / storage helpers work unchanged.

Detection dict keys:
    class_name  : str
    confidence  : float
    bbox        : [x1, y1, x2, y2]  (pixel coordinates, floats)
"""

import ast
from pathlib import Path
from typing import Optional
import numpy as np
import cv2

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:  # pragma: no cover
    ORT_AVAILABLE = False


# YOLOv8 default input size
DEFAULT_INPUT_SIZE = 640

# Class names exported with the model.  If the model carries embedded
# metadata these are overridden at load time.
FALLBACK_CLASS_NAMES = ["IC"]


def _letterbox(
    image: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Resize image to new_shape with letterboxing (grey padding).

    Returns:
        (padded_image, scale, (pad_left, pad_top))
    """
    h, w = image.shape[:2]
    target_h, target_w = new_shape

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    pad_w = target_w - new_w
    pad_h = target_h - new_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized,
        pad_top, pad_h - pad_top,
        pad_left, pad_w - pad_left,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return padded, scale, (pad_left, pad_top)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Simple NMS returning indices of kept boxes."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_threshold]
    return keep


class OnnxDetector:
    """Runs YOLOv8 inference using ONNX Runtime."""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        input_size: int = DEFAULT_INPUT_SIZE,
    ):
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self._session: Optional["ort.InferenceSession"] = None  # type: ignore[name-defined]
        self.class_names: list[str] = FALLBACK_CLASS_NAMES[:]

        if ORT_AVAILABLE:
            self._load()

    def _load(self) -> None:
        providers = ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(
            str(self.model_path), providers=providers
        )
        # Try to read class names from model metadata
        meta = self._session.get_modelmeta().custom_metadata_map
        if "names" in meta:
            try:
                names_raw = ast.literal_eval(meta["names"])
                if isinstance(names_raw, dict):
                    self.class_names = [
                        names_raw[i] for i in sorted(names_raw.keys())
                    ]
                elif isinstance(names_raw, list):
                    self.class_names = names_raw
            except Exception:
                pass

    def detect(self, image_bgr: np.ndarray) -> list[dict]:
        """
        Run detection on a BGR image.

        Args:
            image_bgr: OpenCV BGR frame (H×W×3, uint8).

        Returns:
            List of detection dicts with keys: class_name, confidence, bbox.
        """
        if not ORT_AVAILABLE or self._session is None:
            return []

        orig_h, orig_w = image_bgr.shape[:2]
        input_shape = (self.input_size, self.input_size)

        # Pre-processing
        padded, scale, (pad_left, pad_top) = _letterbox(image_bgr, input_shape)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        blob = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis]

        # Inference
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: blob})
        # YOLOv8 ONNX output shape: [1, num_classes+4, num_anchors]
        raw = outputs[0][0]  # (num_classes+4, num_anchors)

        # Transpose to (num_anchors, num_classes+4)
        if raw.shape[0] < raw.shape[1]:
            raw = raw.T  # already (anchors, 4+classes)

        num_classes = len(self.class_names)
        boxes_xywh = raw[:, :4]
        class_scores = raw[:, 4: 4 + num_classes]

        confidences = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1)

        mask = confidences >= self.conf_threshold
        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes_xywh) == 0:
            return []

        # xywh → x1y1x2y2 (still in letterboxed space)
        cx, cy, bw, bh = (
            boxes_xywh[:, 0],
            boxes_xywh[:, 1],
            boxes_xywh[:, 2],
            boxes_xywh[:, 3],
        )
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # NMS
        kept = _nms(boxes_xyxy, confidences, self.iou_threshold)

        detections = []
        for idx in kept:
            bx1, by1, bx2, by2 = boxes_xyxy[idx]
            # Remove padding and rescale to original image space
            bx1 = (bx1 - pad_left) / scale
            bx2 = (bx2 - pad_left) / scale
            by1 = (by1 - pad_top) / scale
            by2 = (by2 - pad_top) / scale
            # Clip
            bx1 = float(max(0, min(orig_w, bx1)))
            bx2 = float(max(0, min(orig_w, bx2)))
            by1 = float(max(0, min(orig_h, by1)))
            by2 = float(max(0, min(orig_h, by2)))

            cid = int(class_ids[idx])
            name = (
                self.class_names[cid]
                if cid < len(self.class_names)
                else str(cid)
            )
            detections.append(
                {
                    "class_name": name,
                    "confidence": float(confidences[idx]),
                    "bbox": [bx1, by1, bx2, by2],
                }
            )
        return detections

    def draw_detections(self, image_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
        """Draw bounding boxes + labels on a copy of *image_bgr*."""
        annotated = image_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text_y = max(y1 - 8, 12)
            cv2.putText(
                annotated,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        return annotated
