from raspberry_pi_congestion.models import Point
from raspberry_pi_congestion.roi_provider import JsonRoiProvider


def test_roi_json_round_trip(tmp_path):
    path = tmp_path / "roi.json"
    points = [Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)]
    provider = JsonRoiProvider(str(path))
    provider.save(points)
    assert provider.load() == points
