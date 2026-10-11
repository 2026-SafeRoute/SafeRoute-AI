import cv2
import numpy as np

from raspberry_pi_congestion.api_client import ImageUploadResult
from raspberry_pi_congestion.app import CongestionPipeline
from raspberry_pi_congestion.detectors.fake_detector import FakePersonDetector
from raspberry_pi_congestion.models import (
    CongestionThresholds,
    Detection,
    DeviceCongestionConfig,
    EventDetectionSettings,
    Point,
)
from raspberry_pi_congestion.roi_counter import RoiCounter
from raspberry_pi_congestion.window_aggregator import WindowAggregator


SESSION_ID = "550e8400-e29b-41d4-a716-446655440000"
ROI = [Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)]


class Source:
    def close(self):
        pass


class ImageClient:
    def __init__(self, targets, upload_results):
        self.targets = iter(targets)
        self.upload_results = iter(upload_results)
        self.presigned_requests = []
        self.uploads = []
        self.observations = []

    def request_image_upload(self, **kwargs):
        self.presigned_requests.append(kwargs)
        return next(self.targets)

    def upload_jpeg(self, upload_url, jpeg):
        self.uploads.append((upload_url, jpeg))
        return next(self.upload_results)

    def report(self, observation):
        self.observations.append(observation)
        return True

    def should_queue_failure(self, event_id):
        return False


def config():
    return DeviceCongestionConfig(
        True,
        SESSION_ID,
        "CCTV_001",
        1,
        monitored_area_m2=10,
        thresholds=CongestionThresholds(1, 2, 3),
        event_detection=EventDetectionSettings(1, 1, 0),
    )


def target(key, url, expires_at=100_000):
    return {"objectKey": key, "uploadUrl": url, "expiresAt": expires_at}


def pipeline(client, timestamps=None):
    detector = FakePersonDetector([Detection(40, 40, 100, 100, 0.91)])
    instance = CongestionPipeline(
        Source(),
        detector,
        RoiCounter(ROI),
        WindowAggregator(5),
        client,
        "CCTV_001",
        target_fps=0,
        epoch_ms=(lambda: 6_000) if timestamps is None else (lambda: next(timestamps)),
        config_provider=client,
    )
    instance._config = config()
    return instance


def test_uploaded_monitoring_jpeg_contains_bbox_and_matches_observation_reference():
    client = ImageClient(
        [
            target("monitoring/initial.jpg", "https://s3.example.com/initial"),
            target("monitoring/object.jpg", "https://s3.example.com/signed"),
        ],
        [ImageUploadResult.SUCCESS, ImageUploadResult.SUCCESS],
    )
    instance = pipeline(client)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    instance.process_frame(frame, 1_000)
    assert instance.delivery_queue.wait_idle()
    instance.process_frame(frame, 2_000)
    observation = instance.process_frame(frame, 5_000)
    assert instance.delivery_queue.wait_idle()

    uploaded = cv2.imdecode(np.frombuffer(client.uploads[1][1], dtype=np.uint8), cv2.IMREAD_COLOR)
    assert uploaded is not None
    assert uploaded[40, 40, 1] > uploaded[40, 40, 0]
    delivered = client.observations[1]
    assert delivered.monitoring_image_key == "monitoring/object.jpg"
    assert client.presigned_requests[1]["reference_id"] == observation.event_id
    assert client.presigned_requests[1]["captured_at"] == observation.captured_at == 2_000
    assert client.presigned_requests[1]["image_type"] == "MONITORING"
    assert np.array_equal(frame, np.zeros_like(frame))
    instance.close()


def test_first_frame_after_training_starts_is_sent_without_waiting_for_window():
    client = ImageClient(
        [target("monitoring/initial.jpg", "https://s3.example.com/initial")],
        [ImageUploadResult.SUCCESS],
    )
    instance = pipeline(client)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    assert instance.process_frame(frame, 1_000) is None
    assert instance.delivery_queue.wait_idle()

    initial = client.observations[0]
    assert (initial.window_start, initial.window_end, initial.captured_at) == (1_000, 1_000, 1_000)
    assert (initial.sample_count, initial.avg_headcount, initial.peak_headcount) == (1, 1, 1)
    assert initial.frame_headcount == 1
    assert initial.monitoring_image_key == "monitoring/initial.jpg"
    assert client.presigned_requests[0]["captured_at"] == 1_000
    # 같은 세션 안에서는 초기 스냅샷을 다시 보내지 않고 정기 집계만 이어진다.
    instance.process_frame(frame, 2_000)
    assert instance.delivery_queue.wait_idle()
    assert len(client.observations) == 1
    instance.close()


def test_expired_upload_url_is_reissued_with_same_reference_and_timestamp():
    client = ImageClient(
        [
            target("old.jpg", "https://s3.example.com/old"),
            target("new.jpg", "https://s3.example.com/new"),
        ],
        [ImageUploadResult.EXPIRED, ImageUploadResult.SUCCESS],
    )
    instance = pipeline(client, iter([10_000, 10_000]))

    image_key = instance._upload_snapshot(
        np.zeros((20, 20, 3), dtype=np.uint8),
        "MONITORING",
        "observation-id",
        5_000,
        config(),
    )

    assert image_key == "new.jpg"
    assert len(client.presigned_requests) == 2
    assert {request["reference_id"] for request in client.presigned_requests} == {"observation-id"}
    assert {request["captured_at"] for request in client.presigned_requests} == {5_000}
    instance.close()


def test_repeated_image_failure_falls_back_to_observation_without_image():
    client = ImageClient(
        [
            target("monitoring/initial.jpg", "https://s3.example.com/initial"),
            target("monitoring/object.jpg", "https://s3.example.com/signed"),
        ],
        [ImageUploadResult.SUCCESS, ImageUploadResult.FAILED],
    )
    instance = pipeline(client)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    instance.process_frame(frame, 1_000)
    assert instance.delivery_queue.wait_idle()
    instance.process_frame(frame, 2_000)
    observation = instance.process_frame(frame, 5_000)
    assert instance.delivery_queue.wait_idle()

    assert observation.monitoring_image_key is None
    assert client.observations[1:] == [observation]
    instance.close()
