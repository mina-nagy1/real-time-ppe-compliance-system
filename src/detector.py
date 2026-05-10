"""
detector.py
-----------
Wraps YOLO11n-seg.pt (or any YOLO model) for PPE detection.

Single forward pass returns:
  • Bounding boxes   [N × 4]  (x1, y1, x2, y2)
  • Confidence       [N]
  • Class IDs        [N]
  • Masks            [N × H × W]  (bool) — None if non-seg model

Falls back silently to detection-only if a non-seg model is provided.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.config_loader import AppConfig, get_config
from src.logger import get_logger, log_exception

log = get_logger(__name__)


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    """
    All detections for a single frame.

    Attributes
    ----------
    boxes       : float32 array [N × 4]  (x1, y1, x2, y2) in pixel coords
    confidences : float32 array [N]
    class_ids   : int32   array [N]
    class_names : list[str]  length N
    masks       : bool array  [N × H × W]  or None (detection-only fallback)
    frame_shape : (H, W) of the original frame
    is_seg      : True when masks are available
    """
    boxes:       np.ndarray                  # [N, 4]
    confidences: np.ndarray                  # [N]
    class_ids:   np.ndarray                  # [N]
    class_names: list[str]
    masks:       Optional[np.ndarray] = None  # [N, H, W] bool or None
    frame_shape: tuple[int, int] = (0, 0)
    is_seg:      bool = False

    @property
    def n(self) -> int:
        return len(self.boxes)

    def __len__(self) -> int:
        return self.n


# ── Detector ──────────────────────────────────────────────────────────────────

class Detector:
    """
    Loads a YOLO model and runs inference.

    Parameters
    ----------
    config : AppConfig  (loaded from configs/)

    Usage
    -----
    detector = Detector()
    result   = detector.detect(frame)   # frame is BGR np.ndarray H×W×3
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self._cfg = config or get_config()
        mc = self._cfg.model

        self._weights      = mc.weights
        self._device       = mc.device
        self._imgsz        = mc.imgsz
        self._conf         = mc.conf_threshold
        self._iou          = mc.iou_threshold
        self._half         = mc.half_precision
        self._seg_enabled  = mc.segmentation.enabled

        # id → name lookup built from config
        self._class_map: dict[int, str] = {int(k): v for k, v in mc.classes.items()}

        self._model = None
        self._is_seg = False
        self._load_model()

    # ── Model loading ─────────────────────────────────────────────────────

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError:
            raise ImportError(
                "ultralytics is not installed. Run: pip install ultralytics"
            )

        weights_path = Path(self._weights)
        if not weights_path.exists():
            log.warning(
                "Weights file not found — downloading default yolo11n-seg.pt",
                extra={"path": str(weights_path)},
            )
            # Ultralytics auto-downloads when just the name is passed
            weights_path = Path("yolo11n-seg.pt")

        log.info("Loading YOLO model", extra={"weights": str(weights_path), "device": self._device})

        self._model = YOLO(str(weights_path))
        self._model.to(self._device)

        # Detect whether this is a segmentation model
        task = getattr(self._model, "task", "")
        self._is_seg = (task == "segment") and self._seg_enabled

        if self._seg_enabled and not self._is_seg:
            log.warning(
                "Segmentation requested but model task is '%s'. "
                "Falling back to detection-only mode.", task
            )

        # Build class map from model metadata if config map is empty
        if not self._class_map and hasattr(self._model, "names"):
            self._class_map = {int(k): str(v) for k, v in self._model.names.items()}

        log.info(
            "Detector ready",
            extra={
                "seg": self._is_seg,
                "classes": len(self._class_map),
                "device": self._device,
            },
        )

    # ── Inference ─────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Run inference on a single BGR frame.

        Returns DetectionResult with arrays of length N (number of detections).
        Empty arrays are returned when nothing is detected.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        h, w = frame.shape[:2]
        empty = DetectionResult(
            boxes=np.empty((0, 4), dtype=np.float32),
            confidences=np.empty(0, dtype=np.float32),
            class_ids=np.empty(0, dtype=np.int32),
            class_names=[],
            masks=None,
            frame_shape=(h, w),
            is_seg=self._is_seg,
        )

        try:
            results = self._model.predict(
                source=frame,
                imgsz=self._imgsz,
                conf=self._conf,
                iou=self._iou,
                device=self._device,
                half=self._half,
                verbose=False,
            )
        except Exception:
            log_exception(log, "YOLO inference failed")
            return empty

        if not results or results[0].boxes is None:
            return empty

        r = results[0]

        boxes_tensor = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        confs_tensor = r.boxes.conf.cpu().numpy().astype(np.float32)
        ids_tensor   = r.boxes.cls.cpu().numpy().astype(np.int32)

        if len(boxes_tensor) == 0:
            return empty

        class_names = [self._class_map.get(int(cid), str(cid)) for cid in ids_tensor]

        # ── Masks (seg model only) ────────────────────────────────────────
        masks: Optional[np.ndarray] = None
        if self._is_seg and r.masks is not None:
            try:
                # r.masks.data is [N, H, W] float tensor (0–1)
                raw_masks = r.masks.data.cpu().numpy()  # [N, mH, mW]
                import cv2
                resized = np.stack([
                    cv2.resize(raw_masks[i], (w, h), interpolation=cv2.INTER_NEAREST)
                    for i in range(raw_masks.shape[0])
                ])
                masks = resized > 0.5  # boolean [N, H, W]
            except Exception:
                log_exception(log, "Mask post-processing failed")
                masks = None

        return DetectionResult(
            boxes=boxes_tensor,
            confidences=confs_tensor,
            class_ids=ids_tensor,
            class_names=class_names,
            masks=masks,
            frame_shape=(h, w),
            is_seg=(masks is not None),
        )

    # ── Accessors ─────────────────────────────────────────────────────────

    @property
    def class_map(self) -> dict[int, str]:
        return self._class_map

    @property
    def is_seg(self) -> bool:
        return self._is_seg

    def warmup(self, iterations: int = 3) -> None:
        """Run dummy inference to warm up the model before the live stream."""
        log.info("Warming up detector", extra={"iterations": iterations})
        dummy = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
        for _ in range(iterations):
            self.detect(dummy)
        log.info("Warmup complete")
