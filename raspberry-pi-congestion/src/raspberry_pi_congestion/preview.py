from __future__ import annotations

from typing import Sequence

from .image_renderer import OpenCvDetectionRenderer
from .models import Detection, Point


class OpenCvPreview:
    """개발 PC에서만 사용하는 추론 미리보기 창."""

    def __init__(self, roi: Sequence[Point] = (), window_name: str = "SafeRoute Congestion Preview") -> None:
        self.renderer = OpenCvDetectionRenderer(roi)
        self.window_name = window_name
        self._cv2 = None
        self._window_opened = False

    def show(
            self,
            frame,
            detections: Sequence[Detection],
            inside_detections: Sequence[Detection],
    ) -> bool:
        cv2 = self._get_cv2()
        preview = self.renderer.render(frame, detections, inside_detections)
        cv2.putText(
            preview,
            "Q or ESC: quit",
            (16, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(self.window_name, preview)
        self._window_opened = True
        key = cv2.waitKey(1) & 0xFF
        return key not in (ord("q"), ord("Q"), 27)

    def close(self) -> None:
        if self._cv2 is None or not self._window_opened:
            return
        try:
            self._cv2.destroyWindow(self.window_name)
        except Exception:
            self._cv2.destroyAllWindows()
        finally:
            self._window_opened = False

    def _get_cv2(self):
        if self._cv2 is None:
            import cv2
            self._cv2 = cv2
        return self._cv2
