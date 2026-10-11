import numpy as np

from raspberry_pi_congestion.app import CongestionPipeline
from raspberry_pi_congestion.models import (
    CongestionThresholds,
    DeviceCongestionConfig,
    EventDetectionSettings,
    Point,
)
from raspberry_pi_congestion.offline_queue import OfflineQueue
from raspberry_pi_congestion.roi_counter import RoiCounter
from raspberry_pi_congestion.window_aggregator import WindowAggregator


class Source:
    def close(self):
        pass


class Detector:
    def close(self):
        pass


class Client:
    def __init__(self):
        self.replayed_events = []
        self.attached_images = []

    def report_event(self, event):
        return False

    def should_queue_failure(self, event_id):
        return True

    def request_image_upload(self, **kwargs):
        return {
            "objectKey": "training/session/events/CCTV_001/event.jpg",
            "uploadUrl": "https://upload.example.com",
            "expiresAt": 10_000,
        }

    def upload_jpeg(self, upload_url, jpeg):
        return True

    def report_event_json(self, payload):
        self.replayed_events.append(payload)
        return True

    def attach_event_image(self, event_id, image_key, uploaded_at):
        self.attached_images.append((event_id, image_key, uploaded_at))
        return True


def test_failed_event_keeps_uploaded_image_until_replay_succeeds(tmp_path):
    client = Client()
    queue = OfflineQueue(str(tmp_path / "queue.db"))
    pipeline = CongestionPipeline(
        Source(),
        Detector(),
        RoiCounter([Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)]),
        WindowAggregator(),
        client,
        "CCTV_001",
        offline_queue=queue,
        flush_interval_sec=0,
        monotonic=lambda: 0,
        epoch_ms=lambda: 2_000,
        config_provider=client,
    )
    config = DeviceCongestionConfig(
        True,
        "550e8400-e29b-41d4-a716-446655440000",
        "CCTV_001",
        1,
        monitored_area_m2=1,
        thresholds=CongestionThresholds(1, 2, 3),
        event_detection=EventDetectionSettings(1, 1, 30),
    )

    pipeline._ensure_delivery_session(config.training_session_id)
    pipeline._process_local_event(np.zeros((10, 10, 3), dtype=np.uint8), 2, 1_000, config)
    # The event is queued after POST fails; its image is not uploaded out of order.
    assert pipeline.delivery_queue.wait_idle()

    queued_event = queue.peek_oldest()[0]
    assert queued_event.operation == "event"
    assert "eventImageKey" not in queued_event.payload
    assert "jpegBase64" in queued_event.payload
    original_event = queued_event.payload["eventPayload"]

    pipeline._config = config
    pipeline._maybe_flush_queue(1)
    assert pipeline.delivery_queue.wait_idle()

    remaining = queue.peek_oldest()
    assert client.replayed_events == [original_event]
    assert remaining == []
    assert client.attached_images == [
        (queued_event.event_id, "training/session/events/CCTV_001/event.jpg", 2_000)
    ]
    pipeline.close()
