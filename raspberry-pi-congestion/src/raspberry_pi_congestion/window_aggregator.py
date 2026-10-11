from __future__ import annotations

import math
from typing import Optional

from .models import WindowSummary


class WindowAggregator:
    """UTC epoch 밀리초 고정 경계로 프레임별 인원을 집계한다."""

    def __init__(self, window_sec: float = 5.0) -> None:
        self.window_sec = self._validate_window_sec(window_sec)
        self._window_ms = round(self.window_sec * 1000)
        self._window_start_ms: Optional[int] = None
        self._counts: list[int] = []
        self._last_sample_ms: Optional[int] = None

    def add_sample(self, person_count: int, captured_at_ms: int) -> Optional[WindowSummary]:
        if person_count < 0:
            raise ValueError("person_count must not be negative")
        if captured_at_ms < 0:
            raise ValueError("captured_at_ms must not be negative")

        sample_window_start = captured_at_ms - (captured_at_ms % self._window_ms)
        if self._window_start_ms is None:
            self._window_start_ms = sample_window_start
        elif sample_window_start < self._window_start_ms:
            raise ValueError("sample timestamp moved backwards")
        elif sample_window_start > self._window_start_ms:
            completed = self._summary()
            self._counts = [person_count]
            self._window_start_ms = sample_window_start
            self._last_sample_ms = captured_at_ms
            return completed

        self._counts.append(person_count)
        self._last_sample_ms = captured_at_ms
        return None

    def _summary(self) -> WindowSummary:
        if not self._counts or self._window_start_ms is None or self._last_sample_ms is None:
            raise RuntimeError("cannot summarize an empty window")
        peak = max(self._counts)
        average = sum(self._counts) / len(self._counts)
        return WindowSummary(
            window_start_ms=self._window_start_ms,
            window_end_ms=self._window_start_ms + self._window_ms,
            captured_at_ms=self._last_sample_ms,
            sample_count=len(self._counts),
            avg_headcount=min(float(average), peak),
            peak_headcount=peak,
        )

    def _reset(self) -> None:
        self._counts.clear()
        self._window_start_ms = None
        self._last_sample_ms = None

    def reconfigure(self, window_sec: float) -> None:
        self.window_sec = self._validate_window_sec(window_sec)
        self._window_ms = round(self.window_sec * 1000)
        self._reset()

    @staticmethod
    def _validate_window_sec(window_sec: float) -> float:
        value = float(window_sec)
        if not math.isfinite(value) or value < 0.001:
            raise ValueError("window_sec must represent at least one millisecond")
        return value
