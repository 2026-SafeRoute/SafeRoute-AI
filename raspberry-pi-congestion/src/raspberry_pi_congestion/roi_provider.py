from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from .models import Point


class JsonRoiProvider:
    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def load(self) -> list[Point]:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        points = [Point(float(item["x"]), float(item["y"])) for item in data["points"]]
        if len(points) != 4:
            raise ValueError("ROI configuration must contain exactly four points")
        return points

    def save(self, points: Sequence[Point]) -> None:
        if len(points) != 4:
            raise ValueError("ROI configuration must contain exactly four points")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = {"version": 1, "points": [{"x": p.x, "y": p.y} for p in points]}
        self.path.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class InteractiveRoiSelector:
    def select(self, frame) -> list[Point]:
        import cv2

        selected: list[tuple[int, int]] = []
        window = "Select ROI: click four points (Esc cancels)"

        def callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(selected) < 4:
                selected.append((x, y))

        cv2.namedWindow(window)
        cv2.setMouseCallback(window, callback)
        try:
            while len(selected) < 4:
                preview = frame.copy()
                for point in selected:
                    cv2.circle(preview, point, 5, (0, 0, 255), -1)
                cv2.imshow(window, preview)
                if cv2.waitKey(20) & 0xFF == 27:
                    raise RuntimeError("ROI selection cancelled")
        finally:
            cv2.destroyWindow(window)
        height, width = frame.shape[:2]
        return [Point(x / width, y / height) for x, y in selected]
