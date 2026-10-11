from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..models import Detection


class PersonDetector(ABC):
    """모든 backend(Hailo/ONNX/Fake)가 구현하는 공통 인터페이스.

    나머지 파이프라인(RoiCounter 등)은 이 인터페이스와
    Detection 객체에만 의존한다. HailoRT 객체나 Ultralytics Results 객체가
    이 경계를 넘어가면 안 된다.
    """

    @abstractmethod
    def detect(self, frame) -> List[Detection]:
        """BGR ndarray(H,W,3) 프레임을 받아 사람 class만 필터링한 Detection 리스트 반환."""
        raise NotImplementedError

    def close(self) -> None:
        """가속기/세션 자원 정리. 기본은 no-op."""
        return None

    def __enter__(self) -> "PersonDetector":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
