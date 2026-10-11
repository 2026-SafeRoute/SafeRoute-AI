import math

import pytest

from raspberry_pi_congestion.video_source import FileVideoSource, RtspStreamLostError, RtspVideoSource


class Capture:
    def __init__(self, frames=(), opened=True, fps=30.0):
        self.frames = list(frames)
        self.opened = opened
        self.released = False
        self.fps = fps

    def isOpened(self):
        return self.opened

    def read(self):
        return (True, self.frames.pop(0)) if self.frames else (False, None)

    def get(self, _):
        return self.fps

    def release(self):
        self.released = True


class Clock:
    def __init__(self):
        self.value = 0.0
        self.sleeps = []

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.value += seconds


def test_file_eof_stops_and_releases():
    cap = Capture(["frame"])
    source = FileVideoSource("video.mp4", realtime=False, capture_factory=lambda _: cap)
    assert list(source.frames()) == ["frame"]
    source.close()
    assert cap.released


def test_file_frames_follow_source_fps_in_realtime_mode():
    cap = Capture(["first", "second", "third"], fps=2.0)
    clock = Clock()
    source = FileVideoSource(
        "video.mp4", capture_factory=lambda _: cap,
        monotonic=clock, sleeper=clock.sleep,
    )

    assert list(source.frames()) == ["first", "second", "third"]
    assert clock.sleeps == [0.5, 0.5]
    source.close()


def test_file_uses_fallback_when_source_fps_is_invalid():
    cap = Capture(["first", "second"], fps=0)
    clock = Clock()
    source = FileVideoSource(
        "video.mp4", fallback_fps=10, capture_factory=lambda _: cap,
        monotonic=clock, sleeper=clock.sleep,
    )

    assert list(source.frames()) == ["first", "second"]
    assert clock.sleeps == [0.1]
    source.close()


@pytest.mark.parametrize("fallback_fps", [math.nan, math.inf, -math.inf, 0, -1])
def test_file_rejects_non_finite_or_non_positive_fallback_fps(fallback_fps):
    with pytest.raises(ValueError, match="finite and positive"):
        FileVideoSource("video.mp4", fallback_fps=fallback_fps)


def test_file_loop_reopens_capture_and_resets_pacing():
    captures = iter([Capture(["first"], fps=2), Capture(["second"], fps=2)])
    clock = Clock()
    source = FileVideoSource(
        "video.mp4", loop=True, capture_factory=lambda _: next(captures),
        monotonic=clock, sleeper=clock.sleep,
    )
    frames = source.frames()

    assert next(frames) == "first"
    assert next(frames) == "second"
    source.close()
    assert clock.sleeps == [0.5]


def test_file_skips_frames_that_are_older_than_playback_clock():
    cap = Capture(list(range(10)), fps=10)
    clock = Clock()
    source = FileVideoSource(
        "video.mp4", capture_factory=lambda _: cap,
        monotonic=clock, sleeper=clock.sleep,
    )
    frames = source.frames()

    assert next(frames) == 0
    clock.value = 0.35
    assert next(frames) == 3
    assert source.current_position_ms == pytest.approx(300)
    source.close()


class SeekableCapture:
    """인덱스로 위치를 관리해 grab/탐색 호출과 실제 디코딩(read)을 구분해 기록한다."""

    def __init__(self, frame_count, fps, clock=None, read_cost_sec=0.0):
        self.frame_count = frame_count
        self.fps = fps
        self.position = 0
        self.clock = clock
        self.read_cost_sec = read_cost_sec
        self.decoded = []
        self.grabbed = []
        self.seeks = []

    def isOpened(self):
        return True

    def read(self):
        if self.position >= self.frame_count:
            return False, None
        if self.clock is not None:
            self.clock.value += self.read_cost_sec
        frame = self.position
        self.decoded.append(frame)
        self.position += 1
        return True, frame

    def grab(self):
        if self.position >= self.frame_count:
            return False
        self.grabbed.append(self.position)
        self.position += 1
        return True

    def set(self, prop, value):
        assert prop == 1
        self.seeks.append(value)
        self.position = int(value)
        return True

    def get(self, prop):
        return {5: self.fps, 7: self.frame_count}.get(prop, 0)

    def release(self):
        pass


def test_file_skips_late_frames_without_decoding_them():
    clock = Clock()
    cap = SeekableCapture(10, fps=10)
    source = FileVideoSource(
        "video.mp4", capture_factory=lambda _: cap,
        monotonic=clock, sleeper=clock.sleep,
    )
    frames = source.frames()

    assert next(frames) == 0
    clock.value = 0.35
    assert next(frames) == 3
    assert cap.grabbed == [1, 2]
    assert cap.decoded == [0, 3]
    assert cap.seeks == []
    source.close()


def test_file_seeks_directly_to_latest_frame_after_long_stall():
    clock = Clock()
    cap = SeekableCapture(100, fps=10)
    source = FileVideoSource(
        "video.mp4", capture_factory=lambda _: cap,
        monotonic=clock, sleeper=clock.sleep,
    )
    frames = source.frames()

    assert next(frames) == 0
    clock.value = 2.53
    assert next(frames) == 25
    assert cap.seeks == [25]
    assert cap.grabbed == []
    assert source.current_position_ms == pytest.approx(2_500)
    source.close()


def test_file_keeps_emitting_latest_frames_when_decoding_is_slower_than_playback():
    clock = Clock()
    # 프레임 간격(0.1초)보다 디코딩(0.15초)이 느려도 멈추지 않고 최신 프레임을 계속 내보낸다.
    cap = SeekableCapture(10, fps=10, clock=clock, read_cost_sec=0.15)
    source = FileVideoSource(
        "video.mp4", capture_factory=lambda _: cap,
        monotonic=clock, sleeper=clock.sleep,
    )

    emitted = list(source.frames())

    assert emitted == [0, 1, 3, 4, 6, 7, 9]
    assert cap.grabbed == [2, 5, 8]
    source.close()


def test_resumed_file_continues_from_paused_frame_instead_of_skipping_pause_time():
    clock = Clock()
    cap = SeekableCapture(100, fps=10)
    source = FileVideoSource(
        "video.mp4", capture_factory=lambda _: cap,
        monotonic=clock, sleeper=clock.sleep,
    )
    frames = source.frames()

    assert next(frames) == 0
    clock.value = 30.0
    source.resume_playback()

    assert next(frames) == 1
    assert cap.seeks == [] and cap.grabbed == []
    assert next(frames) == 2
    assert clock.value == pytest.approx(30.1)
    source.close()


def test_rtsp_reconnect_limit_and_backoff():
    captures = []
    def factory(_):
        cap = Capture()
        captures.append(cap)
        return cap
    sleeps = []
    source = RtspVideoSource("rtsp://user:password@camera/stream", max_reconnects=2,
                             base_delay_sec=.5, capture_factory=factory, sleeper=sleeps.append)
    with pytest.raises(RtspStreamLostError):
        list(source.frames())
    assert sleeps == [.5, 1.0]
    assert len(captures) == 3
    source.close()
