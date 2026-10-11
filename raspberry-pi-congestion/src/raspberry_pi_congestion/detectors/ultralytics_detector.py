from __future__ import annotations

from ..models import Detection
from .base import PersonDetector


class UltralyticsPersonDetector(PersonDetector):
    def __init__(self, model_path: str = "./models/yolov8n.pt", conf_threshold: float = 0.4) -> None:
        from ultralytics import YOLO
        self._model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame) -> list[Detection]:
        results = self._model(frame, classes=[0], conf=self.conf_threshold, verbose=False)
        detections: list[Detection] = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            detections.append(Detection(x1, y1, x2, y2, confidence))
        return detections

    def close(self) -> None:
        self._model = None
