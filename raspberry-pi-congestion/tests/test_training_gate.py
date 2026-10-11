import numpy as np

from raspberry_pi_congestion.api_client import LoggingCongestionReporter
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
    def __init__(self):
        self.closed = False

    def frames(self):
        yield np.zeros((10, 10, 3), dtype=np.uint8)

    def close(self):
        self.closed = True


class PausableFileSource(Source):
    pause_when_training_inactive = True

    def __init__(self):
        super().__init__()
        self.frames_read = 0
        self.resumed = 0

    def frames(self):
        self.frames_read += 1
        yield np.zeros((10, 10, 3), dtype=np.uint8)

    def resume_playback(self):
        self.resumed += 1


class Detector:
    def __init__(self):
        self.calls = 0
        self.closed = False

    def detect(self, frame):
        self.calls += 1
        return []

    def close(self):
        self.closed = True


class InactiveProvider:
    def __init__(self):
        self.calls = 0
    def fetch_config(self, code):
        self.calls += 1
        return DeviceCongestionConfig(
            False,
            None,
            code,
            1,
            snapshot_interval_sec=7,
            target_inference_fps=2,
        )


def test_inactive_training_stops_inference_and_upload_work():
    source, detector, provider = Source(), Detector(), InactiveProvider()
    aggregator = WindowAggregator()
    pipeline = CongestionPipeline(
        source, detector, RoiCounter([Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)]),
        aggregator, LoggingCongestionReporter(), "CCTV_001", config_provider=provider,
    )
    pipeline.run()
    assert provider.calls == 1
    assert detector.calls == 0
    assert aggregator.window_sec == 7
    assert pipeline.target_fps == 2
    assert source.closed and detector.closed


def test_file_frame_is_not_consumed_until_training_becomes_active():
    class Provider:
        def __init__(self):
            self.calls = 0

        def fetch_config(self, code):
            self.calls += 1
            if self.calls == 1:
                return DeviceCongestionConfig(False, None, code, 1)
            return DeviceCongestionConfig(
                True, "550e8400-e29b-41d4-a716-446655440000", code, 2,
                monitored_area_m2=1,
                thresholds=CongestionThresholds(1, 2, 3),
                event_detection=EventDetectionSettings(1, 1, 0),
            )

    class Clock:
        def __init__(self):
            self.now = 0.0

        def __call__(self):
            return self.now

        def sleep(self, seconds):
            self.now += seconds

    source, detector, provider, clock = PausableFileSource(), Detector(), Provider(), Clock()
    pipeline = CongestionPipeline(
        source, detector, RoiCounter([Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)]),
        WindowAggregator(), LoggingCongestionReporter(), "CCTV_001",
        config_provider=provider, config_poll_inactive_sec=1,
        monotonic=clock, sleeper=clock.sleep,
    )

    pipeline.run()

    assert provider.calls == 2
    assert source.frames_read == 1
    assert source.resumed == 1
    assert detector.calls == 1


def test_inactive_training_discards_previous_session_queue(tmp_path):
    source, detector, provider = Source(), Detector(), InactiveProvider()
    queue = OfflineQueue(str(tmp_path / "queue.db"))
    queue.enqueue(
        "old-observation",
        {"eventId": "old-observation", "trainingSessionId": "old-session"},
    )
    pipeline = CongestionPipeline(
        source,
        detector,
        RoiCounter([Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)]),
        WindowAggregator(),
        LoggingCongestionReporter(),
        "CCTV_001",
        offline_queue=queue,
        config_provider=provider,
    )

    pipeline._maybe_refresh_config(0)

    assert queue.size() == 0
    pipeline.close()


class ActiveProvider:
    def fetch_config(self, code):
        return DeviceCongestionConfig(
            True,
            "550e8400-e29b-41d4-a716-446655440000",
            code,
            1,
            monitored_area_m2=1,
            thresholds=CongestionThresholds(1, 2, 3),
            event_detection=EventDetectionSettings(1, 1, 0),
        )


def test_new_training_session_discards_other_session_queue(tmp_path):
    source, detector, provider = Source(), Detector(), ActiveProvider()
    queue = OfflineQueue(str(tmp_path / "queue.db"))
    queue.enqueue("old", {"eventId": "old", "trainingSessionId": "old-session"})
    queue.enqueue(
        "current",
        {
            "eventId": "current",
            "trainingSessionId": "550e8400-e29b-41d4-a716-446655440000",
        },
    )
    pipeline = CongestionPipeline(
        source,
        detector,
        RoiCounter([Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)]),
        WindowAggregator(),
        LoggingCongestionReporter(),
        "CCTV_001",
        offline_queue=queue,
        config_provider=provider,
    )

    pipeline._maybe_refresh_config(0)

    assert [item.event_id for item in queue.peek_oldest()] == ["current"]
    pipeline.close()


class TogglingProvider(ActiveProvider):
    def __init__(self):
        self.active = True

    def fetch_config(self, code):
        if self.active:
            return super().fetch_config(code)
        return InactiveProvider().fetch_config(code)


class RecordingReporter(LoggingCongestionReporter):
    def __init__(self):
        self.items = []

    def report(self, observation):
        self.items.append(observation)
        return True


def test_each_training_activation_sends_initial_snapshot_immediately():
    provider, reporter = TogglingProvider(), RecordingReporter()
    pipeline = CongestionPipeline(
        Source(), Detector(),
        RoiCounter([Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)]),
        WindowAggregator(), reporter, "CCTV_001", config_provider=provider,
    )
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    pipeline._maybe_refresh_config(0)
    pipeline.process_frame(frame, 1_000)
    pipeline.process_frame(frame, 2_000)
    assert pipeline.delivery_queue.wait_idle()
    provider.active = False
    pipeline._maybe_refresh_config(10)
    provider.active = True
    pipeline._maybe_refresh_config(20)
    pipeline.process_frame(frame, 3_000)
    assert pipeline.delivery_queue.wait_idle()

    assert [(item.window_start, item.window_end) for item in reporter.items] == [
        (1_000, 1_000), (3_000, 3_000),
    ]
    pipeline.close()
