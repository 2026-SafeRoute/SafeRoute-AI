from .base import PersonDetector
from .factory import create_detector
from .fake_detector import FakePersonDetector

__all__ = ["PersonDetector", "create_detector", "FakePersonDetector"]
