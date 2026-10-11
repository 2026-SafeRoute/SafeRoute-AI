from __future__ import annotations

from typing import Callable, List, Optional

from ..models import Detection
from .base import PersonDetector


class FakePersonDetector(PersonDetector):
    """실제 모델 없이 파이프라인을 검증하기 위한 detector.

    - fixed_detections: 매 프레임 동일한 결과를 반환
    - detections_fn: 프레임을 받아 그때그때 Detection 리스트를 계산 (pytest에서
      프레임 인덱스별 시나리오를 만들 때 유용)
    둘 다 없으면 빈 리스트를 반환한다 (사람 없음).
    """

    def __init__(
        self,
        fixed_detections: Optional[List[Detection]] = None,
        detections_fn: Optional[Callable[[object], List[Detection]]] = None,
    ) -> None:
        self._fixed = list(fixed_detections) if fixed_detections else []
        self._fn = detections_fn
        self.call_count = 0

    def detect(self, frame) -> List[Detection]:
        self.call_count += 1
        if self._fn is not None:
            return list(self._fn(frame))
        return list(self._fixed)
