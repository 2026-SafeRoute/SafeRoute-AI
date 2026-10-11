import numpy as np

from raspberry_pi_congestion.api_client import CongestionReporter
from raspberry_pi_congestion.app import CongestionPipeline
from raspberry_pi_congestion.detectors.fake_detector import FakePersonDetector
from raspberry_pi_congestion.models import Detection, Point
from raspberry_pi_congestion.roi_counter import RoiCounter
from raspberry_pi_congestion.video_source import FileVideoSource
from raspberry_pi_congestion.window_aggregator import WindowAggregator


class Clock:
    value = 0.0

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class Source:
    def __init__(self, frame, count=1):
        self.frame = frame
        self.count = count
        self.closed = False

    def frames(self):
        for _ in range(self.count):
            yield self.frame

    def close(self):
        self.closed = True


class Reporter(CongestionReporter):
    def __init__(self):
        self.items = []

    def report(self, item):
        self.items.append(item)
        return True


class Preview:
    def __init__(self, keep_running=True):
        self.keep_running = keep_running
        self.calls = []
        self.closed = False

    def show(self, frame, detections, inside_detections):
        self.calls.append((frame, list(detections), list(inside_detections)))
        return self.keep_running

    def close(self):
        self.closed = True


class Capture:
    def __init__(self, frames, fps):
        self.frames = list(frames)
        self.fps = fps
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        return (True, self.frames.pop(0)) if self.frames else (False, None)

    def get(self, _):
        return self.fps

    def release(self):
        self.released = True


def test_fake_detector_pipeline_and_resource_cleanup():
    clock = Clock()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    source = Source(frame)
    detector = FakePersonDetector([Detection(30, 20, 50, 60, .9)])
    detector.closed = False
    detector.close = lambda: setattr(detector, "closed", True)
    reporter = Reporter()
    aggregator = WindowAggregator(5)
    pipeline = CongestionPipeline(source, detector, RoiCounter([Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)]),
                                  aggregator, reporter, "CCTV_TEST", target_fps=0,
                                  monotonic=clock, epoch_ms=lambda: 6000)
    pipeline.process_frame(frame)
    pipeline._epoch_ms = lambda: 10_000
    pipeline.run()
    assert reporter.items[0].sample_count == 1
    assert reporter.items[0].cctv_code == "CCTV_TEST"
    assert source.closed and detector.closed


def test_realtime_file_pipeline_emits_epoch_aligned_window():
    clock = Clock()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    capture = Capture([frame.copy() for _ in range(26)], fps=5)
    source = FileVideoSource(
        "video.mp4", capture_factory=lambda _: capture,
        monotonic=clock, sleeper=clock.sleep,
    )
    detector = FakePersonDetector([Detection(30, 20, 50, 60, .9)])
    reporter = Reporter()
    pipeline = CongestionPipeline(
        source,
        detector,
        RoiCounter([Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)]),
        WindowAggregator(5),
        reporter,
        "CCTV_TEST",
        target_fps=0,
        monotonic=clock,
        epoch_ms=lambda: 1_000 + int(clock() * 1_000),
    )

    pipeline.run()

    # 모니터링 워커가 병렬이라 도착 순서는 보장되지 않는다.
    initial, regular = sorted(reporter.items, key=lambda item: item.sample_count)
    assert (initial.sample_count, initial.window_start, initial.window_end) == (1, 1_000, 1_000)
    assert regular.sample_count == 20
    assert (regular.window_start, regular.window_end) == (0, 5_000)
    assert regular.captured_at == 4_800
    assert capture.released


def test_zero_detection_frames_are_reported_as_zero_headcount():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    reporter = Reporter()
    timestamps = iter([1_000, 2_000, 5_000])
    pipeline = CongestionPipeline(
        Source(frame),
        FakePersonDetector([]),
        RoiCounter([Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)]),
        WindowAggregator(5),
        reporter,
        "CCTV_TEST",
        target_fps=0,
        epoch_ms=lambda: next(timestamps),
    )

    pipeline.process_frame(frame)
    pipeline.process_frame(frame)
    observation = pipeline.process_frame(frame)

    assert observation.avg_headcount == 0
    assert observation.peak_headcount == 0
    assert observation.frame_headcount == 0
    assert observation.sample_count == 2


def test_preview_receives_detections_and_can_stop_pipeline():
    clock = Clock()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    source = Source(frame, count=3)
    detector = FakePersonDetector([Detection(30, 20, 50, 60, .9)])
    preview = Preview(keep_running=False)
    pipeline = CongestionPipeline(
        source,
        detector,
        RoiCounter([Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)]),
        WindowAggregator(5),
        Reporter(),
        "CCTV_TEST",
        target_fps=0,
        monotonic=clock,
        epoch_ms=lambda: 1_000,
        preview=preview,
    )

    pipeline.run()

    assert detector.call_count == 1
    assert len(preview.calls) == 1
    assert len(preview.calls[0][1]) == 1
    assert len(preview.calls[0][2]) == 1
    assert preview.closed


def test_file_timeline_selects_frames_evenly_at_target_inference_fps():
    class TimedSource(Source):
        def __init__(self, frame):
            super().__init__(frame)
            self.current_position_ms = None

        def frames(self):
            for position_ms in range(0, 1_000, 100):
                self.current_position_ms = position_ms
                yield self.frame

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    source = TimedSource(frame)
    detector = FakePersonDetector([])
    pipeline = CongestionPipeline(
        source, detector,
        RoiCounter([Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)]),
        WindowAggregator(5), Reporter(), "CCTV_TEST",
        target_fps=2, epoch_ms=lambda: 1_000_000,
    )

    pipeline.run()

    assert detector.call_count == 2
