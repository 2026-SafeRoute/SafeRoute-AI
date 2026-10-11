from __future__ import annotations

from pathlib import Path

import pytest

from raspberry_pi_congestion.multi_main import (
    MultiConfigError,
    build_launches,
    discover_profiles,
    run_devices,
)


BASE_ENV = {"SAFEROUTE_SERVER_BASE_URL": "https://example.test"}


def write_profile(path: Path, code: str, queue_path: str | None = None) -> None:
    lines = [
        f"CCTV_CODE={code}",
        f"DEVICE_AUTH_TOKEN=token-{code}",
        f"VIDEO_SOURCE={code}.mp4",
    ]
    if queue_path:
        lines.append(f"OFFLINE_QUEUE_DB_PATH={queue_path}")
    path.write_text("\n".join(lines), encoding="utf-8")


def test_discovers_env_profiles_in_name_order(tmp_path: Path):
    write_profile(tmp_path / "CCTV_002.env", "CCTV_002")
    write_profile(tmp_path / "CCTV_001.env", "CCTV_001")
    (tmp_path / "CCTV.env.example").write_text("example", encoding="utf-8")

    assert [path.name for path in discover_profiles(tmp_path)] == [
        "CCTV_001.env",
        "CCTV_002.env",
    ]


def test_builds_isolated_child_environments(tmp_path: Path):
    first = tmp_path / "first.env"
    second = tmp_path / "second.env"
    write_profile(first, "CCTV_001")
    write_profile(second, "CCTV_002")

    launches = build_launches("file", [first, second], BASE_ENV)

    assert [launch.cctv_code for launch in launches] == ["CCTV_001", "CCTV_002"]
    assert launches[0].env["DEVICE_AUTH_TOKEN"] == "token-CCTV_001"
    assert launches[1].env["DEVICE_AUTH_TOKEN"] == "token-CCTV_002"
    assert launches[0].env["OFFLINE_QUEUE_DB_PATH"] != launches[1].env["OFFLINE_QUEUE_DB_PATH"]
    assert launches[0].env["SAFEROUTE_SERVER_BASE_URL"] == "https://example.test"


def test_rejects_duplicate_cctv_code(tmp_path: Path):
    first = tmp_path / "first.env"
    second = tmp_path / "second.env"
    write_profile(first, "CCTV_001")
    write_profile(second, "CCTV_001")

    with pytest.raises(MultiConfigError, match="중복 CCTV_CODE"):
        build_launches("file", [first, second], BASE_ENV)


def test_rejects_shared_queue_path(tmp_path: Path):
    first = tmp_path / "first.env"
    second = tmp_path / "second.env"
    write_profile(first, "CCTV_001", "shared.sqlite3")
    write_profile(second, "CCTV_002", "shared.sqlite3")

    with pytest.raises(MultiConfigError, match="OFFLINE_QUEUE_DB_PATH"):
        build_launches("file", [first, second], BASE_ENV)


def test_requires_device_specific_token(tmp_path: Path):
    profile = tmp_path / "missing-token.env"
    profile.write_text("CCTV_CODE=CCTV_001\nVIDEO_SOURCE=one.mp4", encoding="utf-8")

    with pytest.raises(MultiConfigError, match="DEVICE_AUTH_TOKEN"):
        build_launches(
            "file",
            [profile],
            {**BASE_ENV, "DEVICE_AUTH_TOKEN": "must-not-be-inherited"},
        )


def test_starts_all_device_processes(tmp_path: Path):
    first = tmp_path / "first.env"
    second = tmp_path / "second.env"
    write_profile(first, "CCTV_001")
    write_profile(second, "CCTV_002")
    launches = build_launches("file", [first, second], BASE_ENV)
    calls = []

    class CompletedProcess:
        def poll(self):
            return 0

    def fake_popen(command, env):
        calls.append((command, env))
        return CompletedProcess()

    assert run_devices("file", launches, popen=fake_popen, poll_interval_sec=0) == 0
    assert len(calls) == 2
    assert all(call[0][-1] == "file" for call in calls)
    assert {call[1]["CCTV_CODE"] for call in calls} == {"CCTV_001", "CCTV_002"}
