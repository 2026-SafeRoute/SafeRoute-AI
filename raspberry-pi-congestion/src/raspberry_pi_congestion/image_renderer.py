from __future__ import annotations

from typing import Sequence

from .models import Detection, Point


class OpenCvDetectionRenderer:
    """원본 프레임을 변경하지 않고 검출 결과가 그려진 복사본을 만든다.

    ROI가 없으면(화면 전체 집계) ROI 선과 ROI 밖 표시 없이 모든 사람을 집계 대상으로 그린다.
    """

    def __init__(self, roi: Sequence[Point] = ()) -> None:
        self.roi = list(roi)

    def render(self, frame, detections: Sequence[Detection],
               inside_detections: Sequence[Detection]):
        import cv2
        import numpy as np

        rendered = frame.copy()
        height, width = rendered.shape[:2]
        inside_ids = {id(detection) for detection in inside_detections}
        if self.roi:
            roi_points = [
                [
                    min(width - 1, max(0, round(point.x * width))),
                    min(height - 1, max(0, round(point.y * height))),
                ]
                for point in self.roi
            ]
            cv2.polylines(
                rendered,
                [np.asarray(roi_points, dtype=np.int32)],
                True,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        for detection in detections:
            is_inside = not self.roi or id(detection) in inside_ids
            color = (0, 255, 0) if is_inside else (0, 128, 255)
            x1 = min(width - 1, max(0, round(detection.x1)))
            y1 = min(height - 1, max(0, round(detection.y1)))
            x2 = min(width - 1, max(0, round(detection.x2)))
            y2 = min(height - 1, max(0, round(detection.y2)))
            cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                rendered,
                f"person {detection.confidence:.2f}",
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        headcount_label = (
            f"headcount: {len(inside_detections)} / detected: {len(detections)}"
            if self.roi else f"headcount: {len(inside_detections)}"
        )
        cv2.putText(
            rendered,
            headcount_label,
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return rendered
