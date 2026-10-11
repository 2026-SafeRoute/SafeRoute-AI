from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from typing import Callable, Iterator, Optional

logger = logging.getLogger(__name__)


class VideoSource(ABC):
    @abstractmethod
    def frames(self) -> Iterator[object]: ...

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# OpenCV VideoCaptureProperties 숫자 값. 테스트 대역에서도 cv2 import 없이 조회한다.
_CAP_PROP_POS_FRAMES = 1
_CAP_PROP_FPS = 5
_CAP_PROP_FRAME_COUNT = 7


class FileVideoSource(VideoSource):
    """녹화 영상을 원본 재생 속도에 맞춰 내보낸다.

    realtime 모드에서 디코딩/추론이 재생 시각보다 뒤처지면, 밀린 프레임은 디코딩
    결과(BGR 변환)를 만들지 않고 건너뛰거나 탐색해서 항상 가장 최신 프레임부터
    처리한다. 그래서 처리 속도가 느려도 지연이 누적되지 않는다.
    """

    # 이만큼 이상 뒤처지면 프레임을 하나씩 grab하는 대신 목표 프레임으로 바로 탐색한다.
    seek_threshold_sec = 1.0
    # 훈련이 비활성인 동안 파이프라인이 프레임을 소비하지 않고 재생을 멈춰 둔다.
    # 그래서 녹화 영상은 훈련이 시작될 때 멈춘 위치(처음 시작이면 첫 프레임)부터 재생된다.
    pause_when_training_inactive = True

    def __init__(self, path: str, loop: bool = False, realtime: bool = True,
                 fallback_fps: float = 30.0, capture_factory: Optional[Callable] = None,
                 monotonic: Callable[[], float] = time.monotonic,
                 sleeper: Callable[[float], None] = time.sleep) -> None:
        if not math.isfinite(fallback_fps) or fallback_fps <= 0:
            raise ValueError("fallback_fps must be finite and positive")
        self.path = path
        self.loop = loop
        self.realtime = realtime
        self.fallback_fps = fallback_fps
        self._capture_factory = capture_factory or _opencv_capture
        self._monotonic = monotonic
        self._sleep = sleeper
        self._cap = self._capture_factory(path)
        if not self._cap.isOpened():
            self.close()
            raise RuntimeError(f"Cannot open video file: {path}")
        self._frame_interval_sec = self._resolve_frame_interval()
        self.current_position_ms: Optional[float] = None
        self._resume_requested = False

    def resume_playback(self) -> None:
        """멈춰 있던 재생을 이어 갈 때, 멈춘 시간만큼의 프레임을 밀린 것으로 보고 건너뛰지 않게 한다."""
        self._resume_requested = True

    def frames(self) -> Iterator[object]:
        playback_started_at = self._monotonic()
        segment_start_ms = 0.0
        timeline_offset_ms = 0.0
        frame_index = 0
        while self._cap is not None:
            if self._resume_requested:
                self._resume_requested = False
                # 다음 프레임이 지금 재생되도록 재생 시계를 다시 맞춘다.
                playback_started_at = self._monotonic() - frame_index * self._frame_interval_sec
            if self.realtime:
                frame_index = self._skip_late_frames(playback_started_at, frame_index)
            ok, frame = self._cap.read()
            if ok:
                position_ms = timeline_offset_ms + frame_index * self._frame_interval_sec * 1000.0
                frame_index += 1
                if self.realtime:
                    due_at = playback_started_at + (position_ms - segment_start_ms) / 1000.0
                    delay = due_at - self._monotonic()
                    if delay > 0:
                        self._sleep(delay)
                self.current_position_ms = position_ms
                yield frame
            elif self.loop:
                completed_segment_sec = frame_index * self._frame_interval_sec
                timeline_offset_ms += completed_segment_sec * 1000.0
                self._cap.release()
                self._cap = self._capture_factory(self.path)
                if not self._cap.isOpened():
                    return
                self._frame_interval_sec = self._resolve_frame_interval()
                frame_index = 0
                segment_start_ms = timeline_offset_ms
                playback_started_at += completed_segment_sec
            else:
                return

    def _skip_late_frames(self, playback_started_at: float, frame_index: int) -> int:
        """현재 재생 시각보다 뒤처진 프레임을 디코딩 결과 없이 건너뛰고 다음에 읽을 위치를 반환한다."""
        elapsed_sec = self._monotonic() - playback_started_at
        due_index = math.floor(elapsed_sec / self._frame_interval_sec + 1e-9)
        late_frames = due_index - frame_index
        if late_frames <= 0:
            return frame_index
        if late_frames * self._frame_interval_sec >= self.seek_threshold_sec and self._seek(due_index):
            return due_index
        grab = getattr(self._cap, "grab", None)
        for _ in range(late_frames):
            ok = grab() if grab is not None else self._cap.read()[0]
            if not ok:
                break
            frame_index += 1
        return frame_index

    def _seek(self, frame_index: int) -> bool:
        try:
            frame_count = float(self._cap.get(_CAP_PROP_FRAME_COUNT))
        except (AttributeError, TypeError, ValueError):
            return False
        # 끝을 넘는 탐색은 반복 재생 경계 계산을 흐트러뜨리므로 grab으로 처리한다.
        if not math.isfinite(frame_count) or frame_index >= frame_count:
            return False
        set_property = getattr(self._cap, "set", None)
        return bool(set_property is not None and set_property(_CAP_PROP_POS_FRAMES, frame_index))

    def _resolve_frame_interval(self) -> float:
        try:
            source_fps = float(self._cap.get(_CAP_PROP_FPS))
        except (AttributeError, TypeError, ValueError):
            source_fps = 0.0
        if not math.isfinite(source_fps) or source_fps <= 0:
            logger.warning(
                "영상 FPS를 읽지 못해 fallback FPS %.2f를 사용합니다: %s",
                self.fallback_fps,
                self.path,
            )
            source_fps = self.fallback_fps
        return 1.0 / source_fps

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class RtspStreamLostError(RuntimeError):
    """재연결 한도를 넘겨 RTSP 스트림을 더 이상 받을 수 없을 때 발생한다."""


class RtspVideoSource(VideoSource):
    def __init__(self, url: str, max_reconnects: int = 5, base_delay_sec: float = 1.0,
                 max_delay_sec: float = 30.0, capture_factory: Optional[Callable] = None,
                 sleeper: Callable[[float], None] = time.sleep) -> None:
        if max_reconnects < 0:
            raise ValueError("max_reconnects must not be negative")
        self.url = url
        self.max_reconnects = max_reconnects
        self.base_delay_sec = base_delay_sec
        self.max_delay_sec = max_delay_sec
        self._capture_factory = capture_factory or _opencv_capture
        self._sleep = sleeper
        self._cap = self._capture_factory(url)

    def frames(self) -> Iterator[object]:
        reconnects = 0
        while self._cap is not None:
            ok, frame = self._cap.read() if self._cap.isOpened() else (False, None)
            if ok:
                reconnects = 0
                yield frame
                continue
            if reconnects >= self.max_reconnects:
                # 정상 종료와 구분해야 systemd 같은 감독자가 프로세스를 다시 띄운다.
                raise RtspStreamLostError(f"RTSP reconnect limit reached ({self.max_reconnects})")
            delay = min(self.base_delay_sec * (2 ** reconnects), self.max_delay_sec)
            reconnects += 1
            logger.warning("RTSP read failed; reconnect %d/%d in %.1fs", reconnects, self.max_reconnects, delay)
            self._sleep(delay)
            self._cap.release()
            self._cap = self._capture_factory(self.url)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def create_video_source(source: str, file_loop: bool = False, **rtsp_options) -> VideoSource:
    if source.lower().startswith("rtsp://"):
        return RtspVideoSource(source, **rtsp_options)
    return FileVideoSource(source, loop=file_loop)


def _opencv_capture(source: str):
    import cv2
    return cv2.VideoCapture(source)
