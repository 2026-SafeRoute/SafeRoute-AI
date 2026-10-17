import pytest

from raspberry_pi_congestion.models import Detection, Point
from raspberry_pi_congestion.roi_counter import RoiCounter


ROI = [Point(.2, .2), Point(.8, .2), Point(.8, .8), Point(.2, .8)]


def det(center_x, center_y):
    return Detection(center_x - 10, center_y - 25, center_x + 10, center_y + 25, .9)


@pytest.mark.parametrize("detection,expected", [
    (det(500, 500), 1), (det(500, 900), 0), (det(200, 500), 1), (det(800, 800), 1),
])
def test_box_center_inside_outside_and_boundary(detection, expected):
    assert RoiCounter(ROI).count_inside([detection], 1000, 1000) == expected


def test_box_center_inside_but_bottom_edge_outside_is_counted():
    detection = Detection(400, 500, 600, 900, .9)
    assert RoiCounter(ROI).count_inside([detection], 1000, 1000) == 1


def test_bottom_edge_inside_but_box_center_outside_is_not_counted():
    detection = Detection(400, 50, 600, 300, .9)
    assert RoiCounter(ROI).count_inside([detection], 1000, 1000) == 0


def test_without_roi_every_detection_in_frame_is_counted():
    detections = [det(100, 100), det(900, 900)]
    assert RoiCounter().filter_inside(detections, 1000, 1000) == detections
