"""ONNX Runtime(CPU) 기반 YOLOv8 person detector.

개발 PC(Windows/PyCharm)에서 실제 Hailo 장치 없이 파이프라인 전체를 검증하기
위한 backend. `yolov8n.onnx` 같은 표준 Ultralytics export를 가정한다
(export 명령: `yolo export model=yolov8n.pt format=onnx`).

무거운 의존성(onnxruntime, numpy 전처리)은 이 모듈 내부에서만 사용하고,
import 자체는 지연시키지 않는다 — ONNX는 개발 PC에 항상 설치돼 있다는 전제이므로
Hailo처럼 "설치 안 된 환경에서 import 실패해도 나머지가 동작해야 한다"는 제약이 없다.
다만 실제 세션 생성은 생성자에서만 일어나므로, onnxruntime이 없는 환경에서도
이 모듈을 import하는 것 자체는 실패하지 않게 import를 함수 내부로 늦춘다.
"""
from __future__ import annotations

import logging
from typing import List, Sequence

from ..models import Detection
from .base import PersonDetector

logger = logging.getLogger(__name__)

# COCO 데이터셋 기준 person class id
COCO_PERSON_CLASS_ID = 0


class OnnxPersonDetector(PersonDetector):
    def __init__(
        self,
        model_path: str,
        input_size: int = 640,
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        person_class_id: int = COCO_PERSON_CLASS_ID,
    ) -> None:
        try:
            import onnxruntime as ort  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "onnxruntime이 설치돼 있지 않습니다. "
                "`pip install onnxruntime`으로 설치하세요."
            ) from e

        import numpy as np  # noqa: F401

        self._ort = ort
        self._np = np
        self.model_path = model_path
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.person_class_id = person_class_id

        self._session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        logger.info("OnnxPersonDetector loaded model=%s", model_path)

    def detect(self, frame) -> List[Detection]:
        np = self._np
        h0, w0 = frame.shape[:2]
        blob, scale, pad_x, pad_y = self._preprocess(frame)

        outputs = self._session.run(None, {self._input_name: blob})
        raw = outputs[0]  # (1, 84, N) for yolov8 export, or (1, N, 84)
        raw = np.squeeze(raw, axis=0)
        if raw.shape[0] < raw.shape[1]:
            raw = raw.transpose(1, 0)  # -> (N, 84)

        boxes_xywh = raw[:, :4]
        class_scores = raw[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)

        keep = (confidences >= self.conf_threshold) & (
            class_ids == self.person_class_id
        )
        boxes_xywh = boxes_xywh[keep]
        confidences = confidences[keep]
        class_ids = class_ids[keep]

        if len(boxes_xywh) == 0:
            return []

        boxes_xyxy = self._xywh_to_xyxy(boxes_xywh)
        # letterbox 역변환: 패딩 제거 후 원본 스케일로 복원
        boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_x) / scale
        boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_y) / scale
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, w0)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, h0)

        keep_idx = self._nms(boxes_xyxy, confidences, self.iou_threshold)

        detections: List[Detection] = []
        for i in keep_idx:
            x1, y1, x2, y2 = boxes_xyxy[i]
            detections.append(
                Detection(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    confidence=float(confidences[i]),
                    class_id=int(class_ids[i]),
                    class_name="person",
                )
            )
        return detections

    def _preprocess(self, frame):
        """letterbox resize + BGR->RGB + CHW + normalize."""
        np = self._np
        h0, w0 = frame.shape[:2]
        scale = min(self.input_size / h0, self.input_size / w0)
        new_h, new_w = int(round(h0 * scale)), int(round(w0 * scale))

        try:
            import cv2

            resized = cv2.resize(frame, (new_w, new_h))
        except ImportError as e:
            raise RuntimeError("opencv-python이 필요합니다 (pip install opencv-python)") from e

        pad_h = self.input_size - new_h
        pad_w = self.input_size - new_w
        pad_top, pad_left = pad_h // 2, pad_w // 2

        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized

        img = canvas[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB, normalize
        img = img.transpose(2, 0, 1)[None, ...]  # NCHW
        return np.ascontiguousarray(img), scale, pad_left, pad_top

    def _xywh_to_xyxy(self, boxes):
        np = self._np
        out = np.empty_like(boxes)
        out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return out

    def _nms(self, boxes, scores, iou_threshold) -> Sequence[int]:
        np = self._np
        if len(boxes) == 0:
            return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            remaining = np.where(iou <= iou_threshold)[0]
            order = order[remaining + 1]
        return keep

    def close(self) -> None:
        self._session = None
