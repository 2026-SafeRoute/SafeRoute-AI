from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Optional


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.x <= 1.0 and 0.0 <= self.y <= 1.0):
            raise ValueError("ROI coordinates must be normalized to [0, 1]")


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = 0
    class_name: str = "person"

    def bottom_center(self, frame_width: int, frame_height: int) -> Point:
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame dimensions must be positive")
        return Point(
            max(0.0, min(1.0, ((self.x1 + self.x2) / 2.0) / frame_width)),
            max(0.0, min(1.0, self.y2 / frame_height)),
        )


@dataclass(frozen=True)
class WindowSummary:
    window_start_ms: int
    window_end_ms: int
    captured_at_ms: int
    sample_count: int
    avg_headcount: float
    peak_headcount: int


@dataclass(frozen=True)
class CongestionObservation:
    event_id: str
    training_session_id: str
    cctv_code: str
    avg_headcount: float
    peak_headcount: int
    frame_headcount: int
    sample_count: int
    window_start: int
    window_end: int
    captured_at: int
    config_version: int
    monitoring_image_key: Optional[str] = None

    @classmethod
    def from_summary(cls, event_id: str, training_session_id: str, cctv_code: str,
                     config_version: int, summary: WindowSummary,
                     frame_headcount: int,
                     monitoring_image_key: Optional[str] = None) -> "CongestionObservation":
        return cls(
            event_id=event_id,
            training_session_id=training_session_id,
            cctv_code=cctv_code,
            avg_headcount=summary.avg_headcount,
            peak_headcount=summary.peak_headcount,
            frame_headcount=frame_headcount,
            sample_count=summary.sample_count,
            window_start=summary.window_start_ms,
            window_end=summary.window_end_ms,
            captured_at=summary.captured_at_ms,
            config_version=config_version,
            monitoring_image_key=monitoring_image_key,
        )

    def to_json(self) -> dict:
        return {
            "eventId": self.event_id,
            "trainingSessionId": self.training_session_id,
            "cctvCode": self.cctv_code,
            "avgHeadcount": self.avg_headcount,
            "peakHeadcount": self.peak_headcount,
            "frameHeadcount": self.frame_headcount,
            "sampleCount": self.sample_count,
            "windowStart": self.window_start,
            "windowEnd": self.window_end,
            "capturedAt": self.captured_at,
            "monitoringImageKey": self.monitoring_image_key,
            "configVersion": self.config_version,
        }


class CongestionLevel(str, Enum):
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    CROWDED = "CROWDED"
    VERY_CROWDED = "VERY_CROWDED"

    @property
    def rank(self) -> int:
        return list(CongestionLevel).index(self)

    @property
    def is_bottleneck(self) -> bool:
        return self in {CongestionLevel.CROWDED, CongestionLevel.VERY_CROWDED}


@dataclass(frozen=True)
class CongestionThresholds:
    caution_from: float
    crowded_from: float
    very_crowded_from: float

    def __post_init__(self) -> None:
        if not (0 <= self.caution_from < self.crowded_from < self.very_crowded_from):
            raise ValueError("congestion thresholds must be non-negative and increasing")

    def classify(self, density: float) -> CongestionLevel:
        if density >= self.very_crowded_from:
            return CongestionLevel.VERY_CROWDED
        if density >= self.crowded_from:
            return CongestionLevel.CROWDED
        if density >= self.caution_from:
            return CongestionLevel.CAUTION
        return CongestionLevel.NORMAL


@dataclass(frozen=True)
class EventDetectionSettings:
    required_consecutive_frames: int
    recovery_consecutive_frames: int
    cooldown_sec: float


@dataclass(frozen=True)
class DeviceCongestionConfig:
    training_active: bool
    training_session_id: Optional[str]
    cctv_code: str
    config_version: int
    monitored_area_m2: Optional[float] = None
    snapshot_interval_sec: float = 5.0
    target_inference_fps: float = 5.0
    thresholds: Optional[CongestionThresholds] = None
    event_detection: Optional[EventDetectionSettings] = None

    @classmethod
    def from_json(cls, payload: dict) -> "DeviceCongestionConfig":
        active = bool(payload["trainingActive"])
        session_id = payload.get("trainingSessionId")
        if active and (not isinstance(session_id, str) or not session_id):
            raise ValueError("active training requires a UUID string trainingSessionId")
        if active:
            try:
                uuid.UUID(session_id)
            except (ValueError, AttributeError):
                raise ValueError("trainingSessionId must be a valid UUID string")
        raw_thresholds = payload.get("congestionThresholds")
        raw_event = payload.get("eventDetection")
        config = cls(
            training_active=active,
            training_session_id=session_id,
            cctv_code=str(payload["cctvCode"]),
            config_version=int(payload["configVersion"]),
            monitored_area_m2=float(payload["monitoredAreaM2"]) if payload.get("monitoredAreaM2") is not None else None,
            snapshot_interval_sec=float(payload.get("snapshotIntervalSec", 5)),
            target_inference_fps=float(payload.get("targetInferenceFps", 5)),
            thresholds=CongestionThresholds(float(raw_thresholds["CAUTION_FROM"]), float(raw_thresholds["CROWDED_FROM"]), float(raw_thresholds["VERY_CROWDED_FROM"])) if raw_thresholds else None,
            event_detection=EventDetectionSettings(int(raw_event["requiredConsecutiveFrames"]), int(raw_event["recoveryConsecutiveFrames"]), float(raw_event["cooldownSec"])) if raw_event else None,
        )
        if config.config_version <= 0:
            raise ValueError("configVersion must be positive")
        if active and (config.monitored_area_m2 is None or config.monitored_area_m2 <= 0 or config.thresholds is None or config.event_detection is None):
            raise ValueError("active training response is missing required congestion settings")
        if (not math.isfinite(config.snapshot_interval_sec) or config.snapshot_interval_sec <= 0
                or not math.isfinite(config.target_inference_fps) or config.target_inference_fps <= 0):
            raise ValueError("snapshot interval and target FPS must be positive")
        if config.event_detection is not None and (
                config.event_detection.required_consecutive_frames <= 0
                or config.event_detection.recovery_consecutive_frames <= 0
                or not math.isfinite(config.event_detection.cooldown_sec)
                or config.event_detection.cooldown_sec < 0):
            raise ValueError("event detection settings are invalid")
        return config


@dataclass(frozen=True)
class CongestionEvent:
    event_id: str
    training_session_id: str
    cctv_code: str
    event_type: str
    detected_at: int
    headcount: int
    local_density: float
    local_congestion_level: CongestionLevel
    config_version: int

    def to_json(self) -> dict:
        return {
            "eventId": self.event_id,
            "trainingSessionId": self.training_session_id,
            "cctvCode": self.cctv_code,
            "eventType": self.event_type,
            "detectedAt": self.detected_at,
            "headcount": self.headcount,
            "localDensity": self.local_density,
            "localCongestionLevel": self.local_congestion_level.value,
            "configVersion": self.config_version,
        }
