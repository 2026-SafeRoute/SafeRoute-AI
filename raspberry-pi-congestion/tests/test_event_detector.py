from raspberry_pi_congestion.event_detector import CongestionEventDetector
from raspberry_pi_congestion.models import CongestionLevel, EventDetectionSettings


def test_caution_never_starts_bottleneck():
    detector = CongestionEventDetector()
    settings = EventDetectionSettings(2, 2, 30)

    assert detector.observe(CongestionLevel.CAUTION, 1_000, settings) is None
    assert detector.observe(CongestionLevel.CAUTION, 2_000, settings) is None
    assert detector.current_level == CongestionLevel.CAUTION


def test_started_level_up_and_caution_recovery_consecutive_frames():
    detector = CongestionEventDetector()
    settings = EventDetectionSettings(3, 3, 30)

    assert [detector.observe(CongestionLevel.CROWDED, i, settings) for i in range(3)] == [
        None,
        None,
        "CONGESTION_STARTED",
    ]
    assert [detector.observe(CongestionLevel.VERY_CROWDED, i, settings) for i in range(3, 6)] == [
        None,
        None,
        "CONGESTION_LEVEL_UP",
    ]
    assert [
        detector.observe(CongestionLevel.NORMAL, 6, settings),
        detector.observe(CongestionLevel.CAUTION, 7, settings),
        detector.observe(CongestionLevel.CAUTION, 8, settings),
    ] == [None, None, "CONGESTION_ENDED"]
    assert detector.current_level == CongestionLevel.CAUTION


def test_very_crowded_can_start_bottleneck_directly():
    detector = CongestionEventDetector()
    settings = EventDetectionSettings(2, 2, 0)

    assert detector.observe(CongestionLevel.VERY_CROWDED, 1_000, settings) is None
    assert detector.observe(CongestionLevel.VERY_CROWDED, 2_000, settings) == "CONGESTION_STARTED"
    assert detector.current_level == CongestionLevel.VERY_CROWDED


def test_start_candidate_counts_mixed_bottleneck_levels_as_consecutive():
    detector = CongestionEventDetector()
    settings = EventDetectionSettings(2, 2, 0)

    assert detector.observe(CongestionLevel.CROWDED, 1_000, settings) is None
    assert detector.observe(CongestionLevel.CAUTION, 2_000, settings) is None
    assert detector.observe(CongestionLevel.CROWDED, 3_000, settings) is None
    assert detector.observe(CongestionLevel.VERY_CROWDED, 4_000, settings) == "CONGESTION_STARTED"
    assert detector.current_level == CongestionLevel.VERY_CROWDED


def test_bottleneck_downgrade_is_silent_and_allows_real_level_up_again():
    detector = CongestionEventDetector()
    settings = EventDetectionSettings(2, 2, 0)

    detector.observe(CongestionLevel.CROWDED, 1_000, settings)
    assert detector.observe(CongestionLevel.CROWDED, 2_000, settings) == "CONGESTION_STARTED"
    detector.observe(CongestionLevel.VERY_CROWDED, 3_000, settings)
    assert detector.observe(CongestionLevel.VERY_CROWDED, 4_000, settings) == "CONGESTION_LEVEL_UP"

    assert detector.observe(CongestionLevel.CROWDED, 5_000, settings) is None
    assert detector.current_level == CongestionLevel.CROWDED
    assert detector.observe(CongestionLevel.VERY_CROWDED, 6_000, settings) is None
    assert detector.observe(CongestionLevel.VERY_CROWDED, 7_000, settings) == "CONGESTION_LEVEL_UP"


def test_same_bottleneck_level_does_not_emit_duplicates():
    detector = CongestionEventDetector()
    settings = EventDetectionSettings(1, 1, 0)

    assert detector.observe(CongestionLevel.CROWDED, 1_000, settings) == "CONGESTION_STARTED"
    assert [detector.observe(CongestionLevel.CROWDED, i, settings) for i in range(2_000, 2_010)] == [
        None
    ] * 10


def test_recovery_candidate_is_cancelled_by_bottleneck_frame():
    detector = CongestionEventDetector()
    settings = EventDetectionSettings(1, 2, 0)

    assert detector.observe(CongestionLevel.CROWDED, 1_000, settings) == "CONGESTION_STARTED"
    assert detector.observe(CongestionLevel.CAUTION, 2_000, settings) is None
    assert detector.observe(CongestionLevel.CROWDED, 3_000, settings) is None
    assert detector.observe(CongestionLevel.NORMAL, 4_000, settings) is None
    assert detector.observe(CongestionLevel.NORMAL, 5_000, settings) == "CONGESTION_ENDED"


def test_level_up_bypasses_started_cooldown():
    detector = CongestionEventDetector()
    settings = EventDetectionSettings(1, 1, 30)

    assert detector.observe(CongestionLevel.CROWDED, 1_000, settings) == "CONGESTION_STARTED"
    assert detector.observe(CongestionLevel.VERY_CROWDED, 1_001, settings) == "CONGESTION_LEVEL_UP"


def test_suppressed_restart_fires_when_cooldown_expires():
    detector = CongestionEventDetector()
    settings = EventDetectionSettings(1, 1, 30)

    assert detector.observe(CongestionLevel.CROWDED, 1_000, settings) == "CONGESTION_STARTED"
    assert detector.observe(CongestionLevel.CAUTION, 2_000, settings) == "CONGESTION_ENDED"
    assert detector.observe(CongestionLevel.CROWDED, 3_000, settings) is None
    assert detector.observe(CongestionLevel.CROWDED, 30_999, settings) is None
    assert detector.observe(CongestionLevel.CROWDED, 31_000, settings) == "CONGESTION_STARTED"


def test_reset_clears_bottleneck_and_cooldown_state():
    detector = CongestionEventDetector()
    settings = EventDetectionSettings(1, 1, 30)
    detector.observe(CongestionLevel.CROWDED, 1_000, settings)

    detector.reset()

    assert detector.current_level == CongestionLevel.NORMAL
    assert detector.observe(CongestionLevel.CROWDED, 2_000, settings) == "CONGESTION_STARTED"
