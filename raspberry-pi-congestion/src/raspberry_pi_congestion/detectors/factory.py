from __future__ import annotations

from .base import PersonDetector
from .fake_detector import FakePersonDetector


class DetectorConfigError(RuntimeError):
    pass


def create_detector(config) -> PersonDetector:
    backend = config.detector_backend.lower()
    if backend == "fake":
        return FakePersonDetector()
    if backend == "ultralytics":
        from .ultralytics_detector import UltralyticsPersonDetector
        return UltralyticsPersonDetector(config.model_path or "./models/yolov8n.pt", config.detector_conf_threshold)
    if backend == "onnx":
        from .onnx_detector import OnnxPersonDetector
        if not config.model_path:
            raise DetectorConfigError("MODEL_PATH is required for onnx")
        return OnnxPersonDetector(config.model_path, conf_threshold=config.detector_conf_threshold)
    if backend == "hailo":
        from .hailo_detector import HailoPersonDetector
        if not config.model_path:
            raise DetectorConfigError("MODEL_PATH is required for hailo")
        return HailoPersonDetector(config.model_path, config.detector_conf_threshold)
    raise DetectorConfigError(f"Unknown DETECTOR_BACKEND: {backend}")
