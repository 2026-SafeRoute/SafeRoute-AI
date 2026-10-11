from raspberry_pi_congestion.detectors import FakePersonDetector, create_detector


class Config:
    detector_backend = "fake"
    model_path = None
    detector_conf_threshold = .4


def test_fake_detector_factory():
    assert isinstance(create_detector(Config()), FakePersonDetector)
