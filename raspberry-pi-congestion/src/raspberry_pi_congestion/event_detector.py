from __future__ import annotations

from typing import Optional

from .models import CongestionLevel, EventDetectionSettings


class CongestionEventDetector:
    """Turns per-frame local levels into the three BE event types."""

    def __init__(self) -> None:
        self.current_level = CongestionLevel.NORMAL
        self._candidate: Optional[CongestionLevel] = None
        self._candidate_count = 0
        self._recovery_count = 0
        self._last_started_at_ms: Optional[int] = None

    def reset(self) -> None:
        self.__init__()

    def observe(self, level: CongestionLevel, now_ms: int,
                settings: EventDetectionSettings) -> Optional[str]:
        if not level.is_bottleneck:
            self._candidate = None
            self._candidate_count = 0
            if not self.current_level.is_bottleneck:
                self.current_level = level
                self._recovery_count = 0
                return None
            self._recovery_count += 1
            if self._recovery_count >= settings.recovery_consecutive_frames:
                self.current_level = level
                self._recovery_count = 0
                return "CONGESTION_ENDED"
            return None

        self._recovery_count = 0
        if not self.current_level.is_bottleneck:
            return self._observe_start(level, now_ms, settings)

        if self.current_level == CongestionLevel.VERY_CROWDED:
            self._candidate = None
            self._candidate_count = 0
            if level == CongestionLevel.CROWDED:
                self.current_level = CongestionLevel.CROWDED
            return None

        if level == CongestionLevel.CROWDED:
            self._candidate = None
            self._candidate_count = 0
            return None

        if not self._candidate_reached(level, settings.required_consecutive_frames):
            return None

        self.current_level = CongestionLevel.VERY_CROWDED
        self._clear_candidate()
        return "CONGESTION_LEVEL_UP"

    def _observe_start(self, level: CongestionLevel, now_ms: int,
                       settings: EventDetectionSettings) -> Optional[str]:
        if not self._bottleneck_candidate_reached(
                level, settings.required_consecutive_frames):
            return None

        cooldown_ms = int(settings.cooldown_sec * 1000)
        if self._last_started_at_ms is not None and now_ms - self._last_started_at_ms < cooldown_ms:
            return None

        self.current_level = level
        self._last_started_at_ms = now_ms
        self._clear_candidate()
        return "CONGESTION_STARTED"

    def _bottleneck_candidate_reached(self, level: CongestionLevel,
                                      required_frames: int) -> bool:
        if self._candidate is None or not self._candidate.is_bottleneck:
            self._candidate_count = 1
        else:
            self._candidate_count += 1
        self._candidate = level
        return self._candidate_count >= required_frames

    def _candidate_reached(self, level: CongestionLevel, required_frames: int) -> bool:
        if self._candidate != level:
            self._candidate = level
            self._candidate_count = 1
        else:
            self._candidate_count += 1
        return self._candidate_count >= required_frames

    def _clear_candidate(self) -> None:
        self._candidate = None
        self._candidate_count = 0
