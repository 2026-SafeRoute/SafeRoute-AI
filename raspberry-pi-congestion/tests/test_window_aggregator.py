import pytest

from raspberry_pi_congestion.window_aggregator import WindowAggregator


def test_five_second_window_average_peak_and_sample_count():
    agg = WindowAggregator(5)
    for count, now in [(2, 1000), (4, 2000), (6, 3000)]:
        assert agg.add_sample(count, now) is None
    result = agg.add_sample(1, 5000)
    assert (result.window_start_ms, result.window_end_ms, result.sample_count) == (0, 5000, 3)
    assert result.captured_at_ms == 3000
    assert (result.avg_headcount, result.peak_headcount) == (4, 6)


def test_fractional_average_is_preserved():
    agg = WindowAggregator()
    agg.add_sample(2, 1000)
    agg.add_sample(3, 2000)
    assert agg.add_sample(0, 5000).avg_headcount == 2.5


def test_zero_headcount_window_is_emitted_and_average_never_exceeds_peak():
    agg = WindowAggregator()
    agg.add_sample(0, 6000)
    summary = agg.add_sample(0, 10_000)
    assert summary.sample_count == 1
    assert summary.avg_headcount == 0
    assert summary.peak_headcount == 0
    assert summary.avg_headcount <= summary.peak_headcount


def test_devices_with_different_start_times_use_same_epoch_boundaries():
    left = WindowAggregator(5)
    right = WindowAggregator(5)
    left.add_sample(1, 10_100)
    right.add_sample(2, 12_900)

    left_summary = left.add_sample(1, 15_050)
    right_summary = right.add_sample(2, 15_100)

    assert (left_summary.window_start_ms, left_summary.window_end_ms) == (10_000, 15_000)
    assert (right_summary.window_start_ms, right_summary.window_end_ms) == (10_000, 15_000)


def test_gap_does_not_fabricate_empty_inference_windows():
    agg = WindowAggregator(5)
    agg.add_sample(3, 1_000)

    summary = agg.add_sample(4, 21_000)

    assert (summary.window_start_ms, summary.window_end_ms, summary.sample_count) == (0, 5_000, 1)
    assert agg.add_sample(5, 25_000).window_start_ms == 20_000


def test_rejects_sub_millisecond_or_non_finite_window():
    for value in (0, 0.0009, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="at least one millisecond"):
            WindowAggregator(value)
