from __future__ import annotations

import base64
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

from .api_client import ImageUploadResult
from .models import CongestionEvent, CongestionObservation, WindowSummary

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Snapshot:
    frame: object
    detections: tuple
    inside_detections: tuple


@dataclass(frozen=True)
class MonitoringDelivery:
    event_id: str
    training_session_id: str
    cctv_code: str
    config_version: int
    summary: WindowSummary
    snapshot: Snapshot


@dataclass(frozen=True)
class EventDelivery:
    event: CongestionEvent
    snapshot: Snapshot


class DeliveryQueue:
    """Bounded network delivery workers split into an event lane and a monitoring lane.

    Congestion events and offline replay run on one dedicated worker so their
    order is preserved and slow monitoring uploads never delay them. Monitoring
    snapshots run on a small worker pool because the backend tolerates
    out-of-order observations. Each lane holds at most ``max_items`` jobs; when
    the monitoring lane is full, its oldest snapshot is discarded. Frames remain
    unencoded until a worker handles them.

    Monitoring snapshots are only for live congestion monitoring, so they are
    downscaled and encoded at a lower JPEG quality to shorten the S3 PUT.
    Congestion event images are evidence and keep the original resolution.
    """

    def __init__(self, client, renderer, offline_queue=None, max_items: int = 32,
                 shutdown_timeout_sec: float = 5.0,
                 max_presigned_refreshes: int = 1,
                 epoch_ms: Callable[[], int] = lambda: int(time.time() * 1000),
                 monitoring_jpeg_quality: int = 70,
                 monitoring_max_width: int = 960,
                 monitoring_workers: int = 2) -> None:
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        if monitoring_workers <= 0:
            raise ValueError("monitoring_workers must be positive")
        if shutdown_timeout_sec < 0:
            raise ValueError("shutdown_timeout_sec must not be negative")
        if not 1 <= monitoring_jpeg_quality <= 100:
            raise ValueError("monitoring_jpeg_quality must be between 1 and 100")
        if monitoring_max_width <= 0:
            raise ValueError("monitoring_max_width must be positive")
        self.client = client
        self.renderer = renderer
        self.offline_queue = offline_queue
        self.max_items = max_items
        self.shutdown_timeout_sec = shutdown_timeout_sec
        self.max_presigned_refreshes = max_presigned_refreshes
        self.monitoring_jpeg_quality = monitoring_jpeg_quality
        self.monitoring_max_width = monitoring_max_width
        self._epoch_ms = epoch_ms
        self._events: deque[EventDelivery] = deque()
        self._monitoring: deque[MonitoringDelivery] = deque()
        self._condition = threading.Condition()
        self._active_session_id: Optional[str] = None
        self._accepting = True
        self._stop = False
        # 작업 중인 워커 스레드 ident -> 그 작업의 훈련 세션 ID
        self._inflight: dict[int, Optional[str]] = {}
        self._flush_requested = False
        self._threads = [threading.Thread(
            target=self._run_events, name="congestion-event-delivery", daemon=True
        )]
        self._threads += [
            threading.Thread(
                target=self._run_monitoring,
                name=f"congestion-monitoring-delivery-{index + 1}", daemon=True,
            )
            for index in range(monitoring_workers)
        ]
        for thread in self._threads:
            thread.start()

    def set_session(self, session_id: Optional[str]) -> int:
        with self._condition:
            if session_id == self._active_session_id:
                return 0
            self._active_session_id = session_id
            discarded = len(self._events) + len(self._monitoring)
            self._events.clear()
            self._monitoring.clear()
            self._condition.notify_all()
            while any(inflight != session_id for inflight in self._inflight.values()):
                self._condition.wait()
            return discarded

    def submit_monitoring(self, job: MonitoringDelivery) -> bool:
        with self._condition:
            if not self._can_accept(job.training_session_id):
                return False
            if len(self._monitoring) >= self.max_items:
                self._monitoring.popleft()
                logger.warning("Dropped oldest monitoring snapshot from full delivery queue")
            self._monitoring.append(job)
            # 두 레인의 워커가 Condition 하나를 공유하므로 다른 레인 워커만 깨우지 않도록 모두 깨운다.
            self._condition.notify_all()
            return True

    def submit_event(self, job: EventDelivery) -> bool:
        session_id = job.event.training_session_id
        with self._condition:
            if not self._can_accept(session_id):
                return False
            if len(self._events) >= self.max_items:
                logger.error("Delivery queue is saturated with congestion events")
                return False
            self._events.append(job)
            self._condition.notify_all()
            return True

    def request_offline_flush(self) -> None:
        with self._condition:
            self._flush_requested = True
            self._condition.notify_all()

    def wait_idle(self, timeout_sec: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout_sec
        with self._condition:
            while (self._size() or self._inflight or self._flush_requested) and time.monotonic() < deadline:
                self._condition.wait(max(0.0, deadline - time.monotonic()))
            return not self._size() and not self._inflight

    def close(self) -> None:
        with self._condition:
            if not self._accepting:
                return
            self._accepting = False
            self._condition.notify_all()
        self.wait_idle(self.shutdown_timeout_sec)
        with self._condition:
            pending = [*self._events, *self._monitoring]
            self._events.clear()
            self._monitoring.clear()
            self._stop = True
            self._condition.notify_all()
        for job in pending:
            self._persist_pending(job)
        for thread in self._threads:
            thread.join(timeout=0.1)

    @property
    def is_alive(self) -> bool:
        return any(thread.is_alive() for thread in self._threads)

    @property
    def pending_count(self) -> int:
        with self._condition:
            return self._size()

    def _can_accept(self, session_id: str) -> bool:
        return (self._accepting and bool(session_id)
                and session_id == self._active_session_id)

    def _size(self) -> int:
        return len(self._events) + len(self._monitoring)

    def _run_events(self) -> None:
        """순서 보장이 필요한 혼잡 이벤트와 오프라인 재전송을 한 스레드에서 처리한다."""
        while True:
            job = None
            with self._condition:
                while not self._stop and not self._events and not self._flush_requested:
                    self._condition.wait()
                if self._stop:
                    return
                if self._events:
                    job = self._events.popleft()
                else:
                    self._flush_requested = False
                self._begin(job)
            self._execute(job)

    def _run_monitoring(self) -> None:
        while True:
            with self._condition:
                while not self._stop and not self._monitoring:
                    self._condition.wait()
                if self._stop:
                    return
                job = self._monitoring.popleft()
                self._begin(job)
            self._execute(job)

    def _begin(self, job) -> None:
        self._inflight[threading.get_ident()] = (
            self._job_session(job) if job is not None else self._active_session_id
        )

    def _execute(self, job) -> None:
        try:
            if job is None:
                self._flush_offline()
            elif self._session_active(self._job_session(job)):
                if isinstance(job, EventDelivery):
                    self._deliver_event(job)
                else:
                    self._deliver_monitoring(job)
        except Exception as exc:
            logger.exception("Background delivery failed: %s", type(exc).__name__)
            if job is not None:
                self._persist_pending(job)
        finally:
            with self._condition:
                self._inflight.pop(threading.get_ident(), None)
                self._condition.notify_all()

    def _flush_offline(self) -> None:
        if self.offline_queue is None or not self._active_session_id:
            return
        for item in self.offline_queue.peek_oldest(
                limit=5, training_session_id=self._active_session_id):
            payload = item.payload
            success = False
            if item.operation == "event" and hasattr(self.client, "report_event_json"):
                event_payload = payload.get("eventPayload", payload)
                success = self.client.report_event_json(event_payload)
                if success and self._session_active(item.training_session_id):
                    image_key = payload.get("eventImageKey")
                    jpeg = self._decode_jpeg(payload)
                    if not image_key and jpeg is not None:
                        image_key = self._upload(
                            jpeg, item.training_session_id, event_payload["cctvCode"],
                            "CONGESTION_EVENT", item.event_id, event_payload["detectedAt"],
                        )
                        if image_key is None:
                            self.offline_queue.mark_failed_attempt(item.id)
                            break
                    if image_key:
                        attached = self.client.attach_event_image(
                            item.event_id, image_key,
                            int(payload.get("uploadedAt", self._epoch_ms())),
                        )
                        if not attached and self._retryable(f"image:{item.event_id}"):
                            self._enqueue_offline(
                                f"image:{item.event_id}",
                                {"eventId": item.event_id, "eventImageKey": image_key,
                                 "uploadedAt": int(payload.get("uploadedAt", self._epoch_ms()))},
                                "event_image", item.training_session_id,
                            )
                        # POST completed. A failed PATCH now has its own retry item.
                        success = True
            elif item.operation == "event_image" and hasattr(self.client, "attach_event_image"):
                success = self.client.attach_event_image(
                    payload["eventId"], payload["eventImageKey"], payload["uploadedAt"]
                )
            elif item.operation == "pending_observation":
                observation_payload = dict(payload["observationPayload"])
                jpeg = self._decode_jpeg(payload)
                if jpeg is not None:
                    observation_payload["monitoringImageKey"] = self._upload(
                        jpeg, item.training_session_id, observation_payload["cctvCode"],
                        "MONITORING", item.event_id, observation_payload["capturedAt"],
                    )
                success = self.client.report_json(observation_payload)
            else:
                success = self.client.report_json(payload)
            if success:
                self.offline_queue.mark_success(item.id)
                continue
            if not self._retryable(item.event_id):
                logger.error("Dropping terminally rejected queued %s %s", item.operation, item.event_id)
                self.offline_queue.mark_success(item.id)
                continue
            self.offline_queue.mark_failed_attempt(item.id)
            break

    def _deliver_monitoring(self, job: MonitoringDelivery) -> None:
        jpeg = self._encode_monitoring(job.snapshot)
        image_key = self._upload(
            jpeg, job.training_session_id, job.cctv_code, "MONITORING",
            job.event_id, job.summary.captured_at_ms,
        ) if jpeg is not None else None
        if not self._session_active(job.training_session_id):
            return
        observation = CongestionObservation.from_summary(
            job.event_id, job.training_session_id, job.cctv_code,
            job.config_version, job.summary,
            len(job.snapshot.inside_detections), image_key,
        )
        reported = self.client.report(observation)
        if not self._session_active(job.training_session_id):
            return
        if not reported and self._retryable(job.event_id):
            self._enqueue_offline(
                job.event_id, observation.to_json(), "observation", job.training_session_id
            )

    def _deliver_event(self, job: EventDelivery) -> None:
        event = job.event
        if not self.client.report_event(event):
            if self._retryable(event.event_id):
                self._persist_pending(job)
            return
        if not self._session_active(event.training_session_id):
            return
        rendered = self.renderer.render(
            job.snapshot.frame, job.snapshot.detections, job.snapshot.inside_detections
        )
        jpeg = self._encode(rendered)
        image_key = self._upload(
            jpeg, event.training_session_id, event.cctv_code, "CONGESTION_EVENT",
            event.event_id, event.detected_at,
        ) if jpeg is not None else None
        if image_key is None and jpeg is not None and self._session_active(event.training_session_id):
            self._persist_pending(job)
            return
        if image_key and self._session_active(event.training_session_id):
            uploaded_at = self._epoch_ms()
            if (not self.client.attach_event_image(event.event_id, image_key, uploaded_at)
                    and self._retryable(f"image:{event.event_id}")):
                self._enqueue_offline(
                    f"image:{event.event_id}",
                    {"eventId": event.event_id, "eventImageKey": image_key,
                     "uploadedAt": uploaded_at},
                    "event_image", event.training_session_id,
                )

    def _upload(self, jpeg: bytes, session_id: str, cctv_code: str,
                image_type: str, reference_id: str, captured_at: int) -> Optional[str]:
        if not hasattr(self.client, "request_image_upload"):
            return None
        for _ in range(self.max_presigned_refreshes + 1):
            if not self._session_active(session_id):
                return None
            target = self.client.request_image_upload(
                training_session_id=session_id, cctv_code=cctv_code,
                image_type=image_type, reference_id=reference_id,
                captured_at=captured_at,
            )
            if target is None:
                return None
            if target["expiresAt"] <= self._epoch_ms():
                continue
            if not self._session_active(session_id):
                return None
            result = self.client.upload_jpeg(target["uploadUrl"], jpeg)
            if result is True or result == ImageUploadResult.SUCCESS:
                return target["objectKey"]
            if result != ImageUploadResult.EXPIRED:
                return None
        return None

    def _persist_pending(self, job) -> None:
        if self.offline_queue is None:
            return
        if isinstance(job, EventDelivery):
            rendered = self.renderer.render(
                job.snapshot.frame, job.snapshot.detections, job.snapshot.inside_detections
            )
            jpeg = self._encode(rendered)
            payload = {"eventPayload": job.event.to_json()}
            operation = "event"
            event_id = job.event.event_id
            session_id = job.event.training_session_id
        else:
            # Keep the stable observation payload and its encoded snapshot together.
            observation = CongestionObservation.from_summary(
                job.event_id, job.training_session_id, job.cctv_code,
                job.config_version, job.summary,
                len(job.snapshot.inside_detections),
            )
            jpeg = self._encode_monitoring(job.snapshot)
            payload = {"observationPayload": observation.to_json()}
            operation = "pending_observation"
            event_id = job.event_id
            session_id = job.training_session_id
        if jpeg is not None:
            payload["jpegBase64"] = base64.b64encode(jpeg).decode("ascii")
        self._enqueue_offline(event_id, payload, operation, session_id)

    def _enqueue_offline(self, event_id: str, payload: dict, operation: str,
                         session_id: str) -> None:
        if self.offline_queue is not None:
            self.offline_queue.enqueue(event_id, payload, operation, session_id)

    def _retryable(self, event_id: str) -> bool:
        return (not hasattr(self.client, "should_queue_failure")
                or self.client.should_queue_failure(event_id))

    def _session_active(self, session_id: str) -> bool:
        with self._condition:
            return session_id == self._active_session_id

    @staticmethod
    def _job_session(job) -> str:
        return (job.event.training_session_id if isinstance(job, EventDelivery)
                else job.training_session_id)

    def _encode_monitoring(self, snapshot: Snapshot) -> Optional[bytes]:
        import cv2

        rendered = self.renderer.render(
            snapshot.frame, snapshot.detections, snapshot.inside_detections
        )
        height, width = rendered.shape[:2]
        if width > self.monitoring_max_width:
            scaled_height = max(1, round(height * self.monitoring_max_width / width))
            rendered = cv2.resize(
                rendered, (self.monitoring_max_width, scaled_height), interpolation=cv2.INTER_AREA
            )
        return self._encode(rendered, self.monitoring_jpeg_quality)

    @staticmethod
    def _encode(frame, quality: Optional[int] = None) -> Optional[bytes]:
        import cv2
        params = [] if quality is None else [cv2.IMWRITE_JPEG_QUALITY, quality]
        ok, encoded = cv2.imencode(".jpg", frame, params)
        return encoded.tobytes() if ok else None

    @staticmethod
    def _decode_jpeg(payload: dict) -> Optional[bytes]:
        encoded = payload.get("jpegBase64")
        return base64.b64decode(encoded) if encoded else None
