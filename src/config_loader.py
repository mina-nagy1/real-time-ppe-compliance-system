"""
config_loader.py
----------------
Loads and validates all four YAML config files using Pydantic.
Every other module calls get_config() — never reads YAML directly.
Fails LOUD at startup if any value is wrong, not silently mid-stream.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# ── Pydantic models ───────────────────────────────────────────────────────────


class SegmentationConfig(BaseModel):
    enabled: bool = True
    mask_alpha: float = Field(0.40, ge=0.0, le=1.0)


class ComplianceConfig(BaseModel):
    required: list[str] = ["helmet", "vest"]
    violation_classes: list[str] = ["no_helmet", "no_vest"]


class ModelConfig(BaseModel):
    weights: str = "models/best.pt"
    device: str = "cpu"
    imgsz: int = Field(640, ge=32, le=1920)
    conf_threshold: float = Field(0.40, ge=0.0, le=1.0)
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0)
    half_precision: bool = False
    classes: dict[int, str] = {}
    compliance: ComplianceConfig = ComplianceConfig()
    segmentation: SegmentationConfig = SegmentationConfig()

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        allowed = {"cpu", "cuda", "mps"}
        if v not in allowed and not v.startswith("cuda:"):
            raise ValueError(f"device must be one of {allowed}, got '{v}'")
        return v


# ── Tracker ───────────────────────────────────────────────────────────────────

class ByteTrackConfig(BaseModel):
    track_high_thresh: float = 0.50
    track_low_thresh: float = 0.10
    new_track_thresh: float = 0.60
    track_buffer: int = 30
    match_thresh: float = 0.80
    frame_rate: int = 30


class DeepOCsortConfig(BaseModel):
    det_thresh: float = 0.45
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.30
    delta_t: int = 3
    asso_func: str = "iou"
    inertia: float = 0.20
    use_byte: bool = False


class BotSortConfig(BaseModel):
    track_high_thresh: float = 0.50
    track_low_thresh: float = 0.10
    new_track_thresh: float = 0.60
    track_buffer: int = 30
    match_thresh: float = 0.80
    proximity_thresh: float = 0.50
    appearance_thresh: float = 0.25
    frame_rate: int = 30


class TrackerVisualizationConfig(BaseModel):
    trail_length: int = 40
    trail_thickness: int = 2
    mask_alpha: float = Field(0.35, ge=0.0, le=1.0)
    bbox_thickness: int = 2
    font_scale: float = 0.55


class TrackerConfig(BaseModel):
    name: str = "bytetrack"
    reid_weights: str = "osnet_x0_25_msmt17.pt"
    device: str = "cpu"
    half: bool = False
    bytetrack: ByteTrackConfig = ByteTrackConfig()
    deepocsort: DeepOCsortConfig = DeepOCsortConfig()
    botsort: BotSortConfig = BotSortConfig()
    visualization: TrackerVisualizationConfig = TrackerVisualizationConfig()

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        allowed = {"bytetrack", "deepocsort", "botsort"}
        if v not in allowed:
            raise ValueError(f"tracker.name must be one of {allowed}, got '{v}'")
        return v


# ── Stream ────────────────────────────────────────────────────────────────────

class StreamConfig(BaseModel):
    source: Any = 0           # int (webcam index) or str (path / RTSP URL)
    width: int = 1280
    height: int = 720
    fps: int = Field(30, ge=1, le=120)
    buffer_size: int = 2
    save_video: bool = False
    output_path: str = "outputs/videos/output.mp4"
    fourcc: str = "mp4v"
    jpeg_quality: int = Field(85, ge=1, le=100)


# ── Analytics ─────────────────────────────────────────────────────────────────

class CountingLine(BaseModel):
    id: str
    p1: list[int] = Field(..., min_length=2, max_length=2)
    p2: list[int] = Field(..., min_length=2, max_length=2)
    label: str = ""


class HeatmapConfig(BaseModel):
    enabled: bool = True
    sigma: int = 25
    alpha: float = Field(0.50, ge=0.0, le=1.0)
    colormap: str = "JET"
    decay: float = Field(0.98, ge=0.0, le=1.0)
    save_interval_seconds: int = 60


class CrowdConfig(BaseModel):
    max_persons: int = 10
    alert_cooldown_seconds: float = 5.0


class ComplianceAlertConfig(BaseModel):
    alert_on_violation: bool = True
    violation_cooldown_seconds: float = 3.0


class MLflowAnalyticsConfig(BaseModel):
    flush_interval_seconds: float = 5.0


class AnalyticsConfig(BaseModel):
    enabled: bool = True
    counting_lines: list[CountingLine] = []
    heatmap: HeatmapConfig = HeatmapConfig()
    crowd: CrowdConfig = CrowdConfig()
    compliance: ComplianceAlertConfig = ComplianceAlertConfig()
    mlflow: MLflowAnalyticsConfig = MLflowAnalyticsConfig()


# ── Root config container ─────────────────────────────────────────────────────

class AppConfig(BaseModel):
    model: ModelConfig
    tracker: TrackerConfig
    stream: StreamConfig
    analytics: AnalyticsConfig


# ── Loader ────────────────────────────────────────────────────────────────────

CONFIG_DIR = Path("configs")


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=1)
def get_config(config_dir: str = "configs") -> AppConfig:
    """
    Load and validate all config files.
    Result is cached — call get_config() anywhere without I/O overhead.

    Usage
    -----
    from src.config_loader import get_config
    cfg = get_config()
    print(cfg.model.weights)
    """
    base = Path(config_dir)

    raw_model     = _load_yaml(base / "model_config.yaml")
    raw_tracker   = _load_yaml(base / "tracker_config.yaml")
    raw_stream    = _load_yaml(base / "stream_config.yaml")
    raw_analytics = _load_yaml(base / "analytics_config.yaml")

    return AppConfig(
        model     = ModelConfig(**raw_model.get("model", {}),
                                classes=raw_model.get("classes", {}),
                                compliance=raw_model.get("compliance", {}),
                                segmentation=raw_model.get("segmentation", {})),
        tracker   = TrackerConfig(**raw_tracker.get("tracker", {}),
                                  bytetrack=raw_tracker.get("bytetrack", {}),
                                  deepocsort=raw_tracker.get("deepocsort", {}),
                                  botsort=raw_tracker.get("botsort", {}),
                                  visualization=raw_tracker.get("visualization", {})),
        stream    = StreamConfig(**raw_stream.get("stream", {})),
        analytics = AnalyticsConfig(**raw_analytics.get("analytics", {})),
    )


def reload_config(config_dir: str = "configs") -> AppConfig:
    """Force-reload configs (clears cache). Useful for hot-reload scenarios."""
    get_config.cache_clear()
    return get_config(config_dir)
