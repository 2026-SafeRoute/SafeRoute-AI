from __future__ import annotations

import logging
import math
import time
import uuid
from typing import Callable, Optional

from .delivery import DeliveryQueue, EventDelivery, MonitoringDelivery, Snapshot
from .event_detector import CongestionEventDetector
from .image_renderer import OpenCvDetectionRenderer
from .models import (
    CongestionEvent, CongestionObservation, CongestionThresholds,
    DeviceCongestionConfig, EventDetectionSettings, WindowSummary,
)
from .offline_queue import OfflineQueue
from .roi_counter import RoiCounter
from .window_aggregator import WindowAggregator

logger = logging.getLogger(__name__)


class CongestionPipeline:
    def __init__(self, video_source, detector, roi_counter: RoiCounter,
                 aggregator: WindowAggregator, reporter,
                 cctv_code: str, offline_queue: Optional[OfflineQueue] = None,
                 target_fps: float = 5.0, flush_interval_sec: float = 30.0,
                 monotonic: Callable[[], float] = time.monotonic,
                 epoch_ms: Callable[[], int] = lambda: int(time.time() * 1000),
                 config_provider=None, config_poll_active_sec: float = 5.0,
                 config_poll_inactive_sec: float = 1.0,
                 preview=None, image_renderer=None,
                 max_presigned_refreshes: int = 1,
                 delivery_queue_max_items: int = 32,
                 shutdown_drain_timeout_sec: float = 5.0,
                 delivery_queue=None,
                 monitoring_jpeg_quality: int = 70,
                 monitoring_image_max_width: int = 960,
                 delivery_monitoring_workers: int = 2,
                 sleeper: Callable[[float], None] = time.sleep) -> None:
        if max_presigned_refreshes < 0:
            raise ValueError("max_presigned_refreshes must not be negative")
        self.video_source = video_source
        self.detector = detector
        self.roi_counter = roi_counter
        self.aggregator = aggregator
        self.reporter = reporter
        self.config_provider = config_provider
        self.cctv_code = cctv_code
        self.offline_queue = offline_queue
        self.target_fps = target_fps
        self.flush_interval_sec = flush_interval_sec
        self.config_poll_active_sec = config_poll_active_sec
        self.config_poll_inactive_sec = config_poll_inactive_sec
        self.preview = preview
        self.image_renderer = image_renderer or OpenCvDetectionRenderer(roi_counter.roi)
        self.max_presigned_refreshes = max_presigned_refreshes
        self._monotonic = monotonic
        self._sleep = sleeper
        self._epoch_ms = epoch_ms
        self._last_inference = float("-inf")
        self._last_queue_flush = monotonic()
        self._last_config_poll = float("-inf")
        self._last_inference_error_log = float("-inf")
        self._next_source_inference_ms: Optional[float] = None
        self._source_epoch_anchor_ms: Optional[int] = None
        self._last_source_position_ms: Optional[float] = None
        self._closed = False
        self._preview_stop_requested = False
        self._event_detector = CongestionEventDetector()
        self._monitoring_snapshot: Optional[Snapshot] = None
        self._delivery_session_id: Optional[str] = None
        self._initial_snapshot_pending = False
        self.delivery_queue = delivery_queue or DeliveryQueue(
            reporter, self.image_renderer, offline_queue,
            max_items=delivery_queue_max_items,
            shutdown_timeout_sec=shutdown_drain_timeout_sec,
            max_presigned_refreshes=max_presigned_refreshes,
            epoch_ms=epoch_ms,
            monitoring_jpeg_quality=monitoring_jpeg_quality,
            monitoring_max_width=monitoring_image_max_width,
            monitoring_workers=delivery_monitoring_workers,
        )
        self._config = None if config_provider else DeviceCongestionConfig(
            True, "local-dry-run", cctv_code, 1, 1.0, aggregator.window_sec,
            target_fps, CongestionThresholds(2.0, 3.0, 5.0), EventDetectionSettings(3, 5, 30),
        )

    def run(self) -> None:
        try:
            frames = iter(self.video_source.frames())
            paused = False
            while True:
                now = self._monotonic()
                self._maybe_refresh_config(now)
                config = self._config
                training_active = config is not None and config.training_active
                if not training_active and getattr(self.video_source, "pause_when_training_inactive", False):
                    # 녹화 영상은 훈련이 활성화되기 전에는 프레임을 소비하지 않고 멈춰 둔다.
                    paused = True
                    self._sleep(self._inactive_config_wait(now))
                    continue
                if paused:
                    paused = False
                    resume = getattr(self.video_source, "resume_playback", None)
                    if resume is not None:
                        resume()
                try:
                    frame = next(frames)
                except StopIteration:
                    break
                if not training_active:
                    continue
                source_position_ms = self._source_position_ms()
                if not self._should_infer(now, source_position_ms):
                    continue
                self.process_frame(frame, self._captured_at_ms(source_position_ms))
                if self._preview_stop_requested:
                    break
                self._maybe_flush_queue(now)
        finally:
            self.close()

    def _inactive_config_wait(self, now: float) -> float:
        next_poll_at = self._last_config_poll + self.config_poll_inactive_sec
        return max(0.01, min(0.25, next_poll_at - now))

    def process_frame(self, frame, captured_at_ms: Optional[int] = None) -> Optional[CongestionObservation]:
        config = self._config
        if config is None or not config.training_active or not config.training_session_id:
            return None
        self._ensure_delivery_session(config.training_session_id)
        captured_at_ms = self._epoch_ms() if captured_at_ms is None else captured_at_ms
        try:
            detections = self.detector.detect(frame)
        except Exception as exc:
            now = self._monotonic()
            if now - self._last_inference_error_log >= 10.0:
                self._last_inference_error_log = now
                logger.error("Person inference failed (further errors suppressed for 10s): %s", type(exc).__name__)
            return None
        height, width = frame.shape[:2]
        inside_detections = self.roi_counter.filter_inside(detections, width, height)
        count = len(inside_detections)
        if self.preview is not None:
            self._preview_stop_requested = not self.preview.show(frame, detections, inside_detections)
        snapshot = Snapshot(frame.copy(), tuple(detections), tuple(inside_detections))
        self._process_local_event(snapshot, count, captured_at_ms, config)
        if self._initial_snapshot_pending:
            # 훈련 활성화 직후 첫 화면은 집계 구간이 끝날 때까지 기다리지 않고 단일 샘플로 바로 보낸다.
            # 이 샘플은 아래 정기 집계에도 그대로 포함된다.
            self._initial_snapshot_pending = False
            self._submit_monitoring(config, WindowSummary(
                captured_at_ms, captured_at_ms, captured_at_ms, 1, float(count), count,
            ), snapshot)
        summary = self.aggregator.add_sample(count, captured_at_ms)
        if summary is None:
            self._monitoring_snapshot = snapshot
            return None
        monitoring_snapshot = self._monitoring_snapshot
        self._monitoring_snapshot = snapshot
        if monitoring_snapshot is None:
            logger.error("Completed observation window has no monitoring frame")
            return None
        return self._submit_monitoring(config, summary, monitoring_snapshot)

    def _submit_monitoring(self, config: DeviceCongestionConfig, summary: WindowSummary,
                           snapshot: Snapshot) -> CongestionObservation:
        event_id = str(uuid.uuid4())
        observation = CongestionObservation.from_summary(
            event_id, config.training_session_id, self.cctv_code,
            config.config_version, summary,
            len(snapshot.inside_detections),
        )
        self.delivery_queue.submit_monitoring(MonitoringDelivery(
            event_id, config.training_session_id, self.cctv_code,
            config.config_version, summary, snapshot,
        ))
        return observation

    def _process_local_event(self, snapshot: Snapshot, count: int, now_ms: int,
                             config: DeviceCongestionConfig) -> None:
        if not isinstance(snapshot, Snapshot):
            snapshot = Snapshot(snapshot.copy(), (), ())
        if config.monitored_area_m2 is None or config.thresholds is None or config.event_detection is None:
            return
        density = count / config.monitored_area_m2
        level = config.thresholds.classify(density)
        event_type = self._event_detector.observe(level, now_ms, config.event_detection)
        if event_type is None or not config.training_session_id:
            return
        event = CongestionEvent(
            str(uuid.uuid4()), config.training_session_id, self.cctv_code,
            event_type, now_ms, count, density, level, config.config_version,
        )
        client = self.config_provider
        if client is None or not hasattr(client, "report_event"):
            logger.info("congestion event: %s", event.to_json())
            return
        self.delivery_queue.submit_event(EventDelivery(event, snapshot))

    def _upload_snapshot(self, frame, image_type: str, reference_id: str,
                         captured_at: int, config: DeviceCongestionConfig) -> Optional[str]:
        """Compatibility helper for explicit uploads; the inference loop never calls it."""
        if not config.training_session_id:
            return None
        self._ensure_delivery_session(config.training_session_id)
        jpeg = self.delivery_queue._encode(frame)
        if jpeg is None:
            return None
        return self.delivery_queue._upload(
            jpeg, config.training_session_id, self.cctv_code,
            image_type, reference_id, captured_at,
        )

    def _ensure_delivery_session(self, session_id: str) -> None:
        if session_id == self._delivery_session_id:
            return
        self._switch_delivery_session(session_id)

    def _switch_delivery_session(self, session_id: Optional[str]) -> None:
        discarded = self.delivery_queue.set_session(session_id)
        if session_id != self._delivery_session_id:
            self._initial_snapshot_pending = session_id is not None
        self._delivery_session_id = session_id
        if discarded:
            logger.info("Discarded %d pending deliveries from an inactive training session", discarded)

    def _source_position_ms(self) -> Optional[float]:
        value = getattr(self.video_source, "current_position_ms", None)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) and value >= 0 else None

    def _captured_at_ms(self, source_position_ms: Optional[float]) -> int:
        if source_position_ms is None:
            return self._epoch_ms()
        if self._source_epoch_anchor_ms is None:
            self._source_epoch_anchor_ms = self._epoch_ms() - round(source_position_ms)
        self._last_source_position_ms = source_position_ms
        return self._source_epoch_anchor_ms + round(source_position_ms)

    def _should_infer(self, now: float, source_position_ms: Optional[float]) -> bool:
        if self.target_fps <= 0:
            return True
        if source_position_ms is None:
            if now - self._last_inference < 1.0 / self.target_fps:
                return False
            self._last_inference = now
            return True
        interval_ms = 1000.0 / self.target_fps
        if self._next_source_inference_ms is None:
            self._next_source_inference_ms = source_position_ms
        if source_position_ms + 0.001 < self._next_source_inference_ms:
            return False
        skipped_slots = math.floor(max(0.0, source_position_ms - self._next_source_inference_ms) / interval_ms)
        self._next_source_inference_ms += (skipped_slots + 1) * interval_ms
        return True

    def _maybe_refresh_config(self, now: float) -> None:
        if self.config_provider is None:
            return
        forced = (hasattr(self.config_provider, "consume_config_refresh_request")
                  and self.config_provider.consume_config_refresh_request())
        interval = self.config_poll_active_sec if self._config and self._config.training_active else self.config_poll_inactive_sec
        if not forced and now - self._last_config_poll < interval:
            return
        self._last_config_poll = now
        new_config = self.config_provider.fetch_config(self.cctv_code)
        if new_config is None:
            return
        previous = self._config
        self._config = new_config
        session_changed = previous is None or previous.training_session_id != new_config.training_session_id
        if (previous is None or previous.config_version != new_config.config_version
                or previous.training_active != new_config.training_active or session_changed):
            self.aggregator.reconfigure(new_config.snapshot_interval_sec)
            self._monitoring_snapshot = None
            self.target_fps = new_config.target_inference_fps
            self._next_source_inference_ms = None
            logger.info("Applied congestion config version %d", new_config.config_version)
        if session_changed or not new_config.training_active:
            self._event_detector.reset()
            session_id = new_config.training_session_id if new_config.training_active else None
            self._switch_delivery_session(session_id)
        if self.offline_queue is not None:
            if not new_config.training_active or not new_config.training_session_id:
                discarded = self.offline_queue.clear()
            elif session_changed:
                discarded = self.offline_queue.discard_except_session(new_config.training_session_id)
            else:
                discarded = 0
            if discarded:
                logger.info("Discarded %d queued operations from an inactive training session", discarded)

    def _maybe_flush_queue(self, now: float) -> None:
        if self.offline_queue is None or now - self._last_queue_flush < self.flush_interval_sec:
            return
        config = self._config
        if config is None or not config.training_active or not config.training_session_id:
            return
        self._last_queue_flush = now
        self.delivery_queue.request_offline_flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.video_source.close()
        self.detector.close()
        if self.preview is not None:
            self.preview.close()
        self.delivery_queue.close()
        if self.offline_queue is not None and not self.delivery_queue.is_alive:
            self.offline_queue.close()
