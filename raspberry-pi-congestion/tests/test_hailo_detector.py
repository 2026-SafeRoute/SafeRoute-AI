from __future__ import annotations

import numpy as np
import pytest

from raspberry_pi_congestion.detectors.hailo_detector import (
    HailoPersonDetector,
    HailoRuntimeError,
)


class FakeRuntime:
    input_shape = (100, 100, 3)

    def __init__(self, output=None):
        self.output = output if output is not None else [np.empty((0, 5)) for _ in range(80)]
        self.received = None
        self.closed = False

    def infer(self, frame):
        self.received = frame
        return self.output

    def close(self):
        self.closed = True


def make_detector(tmp_path, runtime, threshold=0.4):
    model = tmp_path / "person.hef"
    model.write_bytes(b"fake-hef")
    return HailoPersonDetector(
        str(model), threshold, runtime_factory=lambda _: runtime
    )


def test_missing_hef_is_rejected_before_runtime_initialization(tmp_path):
    with pytest.raises(HailoRuntimeError, match="HEF 모델 파일"):
        HailoPersonDetector(str(tmp_path / "missing.hef"))


def test_preprocesses_bgr_with_letterbox_and_restores_coordinates(tmp_path):
    classes = [np.empty((0, 5), dtype=np.float32) for _ in range(80)]
    # Original 200x100 -> model 100x100: scale=.5, vertical padding=25.
    # This normalized box maps back to (50, 25, 150, 75).
    classes[0] = np.array(
        [
            [0.375, 0.25, 0.625, 0.75, 0.9],
            [0.375, 0.25, 0.625, 0.75, 0.2],
        ],
        dtype=np.float32,
    )
    classes[1] = np.array([[0.0, 0.0, 1.0, 1.0, 0.99]], dtype=np.float32)
    runtime = FakeRuntime(classes)
    detector = make_detector(tmp_path, runtime)
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[:, :, 0] = 10
    frame[:, :, 1] = 20
    frame[:, :, 2] = 30

    detections = detector.detect(frame)

    assert runtime.received.shape == (100, 100, 3)
    assert runtime.received.flags.c_contiguous
    assert runtime.received[25, 0].tolist() == [30, 20, 10]
    assert runtime.received[0, 0].tolist() == [114, 114, 114]
    assert len(detections) == 1
    detection = detections[0]
    assert detection.x1 == pytest.approx(50)
    assert detection.y1 == pytest.approx(25)
    assert detection.x2 == pytest.approx(150)
    assert detection.y2 == pytest.approx(75)
    assert detection.confidence == pytest.approx(0.9)
    assert detection.class_id == 0
    assert detection.class_name == "person"


def test_empty_nms_output_returns_no_detections(tmp_path):
    runtime = FakeRuntime()
    detector = make_detector(tmp_path, runtime)

    assert detector.detect(np.zeros((20, 20, 3), dtype=np.uint8)) == []


@pytest.mark.parametrize(
    "frame",
    [
        np.empty((0, 20, 3), dtype=np.uint8),
        np.empty((20, 0, 3), dtype=np.uint8),
        np.zeros((20, 20, 3), dtype=np.float32),
        np.zeros((20, 20, 3), dtype=np.int16),
    ],
)
def test_rejects_empty_or_non_uint8_frames(tmp_path, frame):
    runtime = FakeRuntime()
    detector = make_detector(tmp_path, runtime)

    with pytest.raises(ValueError, match="non-empty uint8 BGR ndarray"):
        detector.detect(frame)

    assert runtime.received is None


def test_non_nms_output_is_rejected(tmp_path):
    runtime = FakeRuntime(np.zeros((1, 5), dtype=np.float32))
    detector = make_detector(tmp_path, runtime)

    with pytest.raises(HailoRuntimeError, match="NMS class 목록"):
        detector.detect(np.zeros((20, 20, 3), dtype=np.uint8))


def test_close_releases_runtime_once_and_prevents_reuse(tmp_path):
    runtime = FakeRuntime()
    detector = make_detector(tmp_path, runtime)

    detector.close()
    detector.close()

    assert runtime.closed is True
    with pytest.raises(HailoRuntimeError, match="이미 종료"):
        detector.detect(np.zeros((20, 20, 3), dtype=np.uint8))


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_confidence_threshold_must_be_probability(tmp_path, threshold):
    runtime = FakeRuntime()
    with pytest.raises(ValueError, match="between 0 and 1"):
        make_detector(tmp_path, runtime, threshold)
