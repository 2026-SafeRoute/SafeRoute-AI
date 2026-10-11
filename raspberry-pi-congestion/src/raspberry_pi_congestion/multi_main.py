from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from dotenv import dotenv_values, load_dotenv

from .config import AppConfig, ConfigError


class MultiConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class DeviceLaunch:
    cctv_code: str
    profile_path: Path
    env: dict[str, str]


def discover_profiles(device_dir: Path) -> list[Path]:
    return sorted(path for path in device_dir.glob("*.env") if path.is_file())


def _required_profile_value(values: Mapping[str, object], key: str, path: Path) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise MultiConfigError(f"{path}: {key} 값이 필요합니다")
    return value.strip()


def _queue_path_for(cctv_code: str) -> str:
    safe_code = re.sub(r"[^A-Za-z0-9_.-]", "_", cctv_code)
    return f"./offline_queue_{safe_code}.sqlite3"


def build_launches(
    mode: str,
    profile_paths: Sequence[Path],
    base_env: Mapping[str, str] | None = None,
) -> list[DeviceLaunch]:
    if not profile_paths:
        raise MultiConfigError("실행할 CCTV 프로필이 없습니다")

    inherited = dict(os.environ if base_env is None else base_env)
    launches: list[DeviceLaunch] = []
    seen_codes: set[str] = set()
    seen_queues: set[str] = set()

    for raw_path in profile_paths:
        path = Path(raw_path)
        values = dotenv_values(path)
        cctv_code = _required_profile_value(values, "CCTV_CODE", path)
        _required_profile_value(values, "DEVICE_AUTH_TOKEN", path)
        _required_profile_value(values, "VIDEO_SOURCE", path)

        if cctv_code in seen_codes:
            raise MultiConfigError(f"중복 CCTV_CODE: {cctv_code}")

        profile_env = {
            key: value
            for key, value in values.items()
            if isinstance(value, str)
        }
        child_env = {**inherited, **profile_env}
        child_env["RUN_MODE"] = mode
        child_env["OFFLINE_QUEUE_DB_PATH"] = profile_env.get(
            "OFFLINE_QUEUE_DB_PATH", _queue_path_for(cctv_code)
        )
        child_env["PYTHONUNBUFFERED"] = "1"

        try:
            AppConfig.from_env(child_env, mode=mode)
        except ConfigError as exc:
            raise MultiConfigError(f"{path}: {exc}") from exc

        queue_path = os.path.normcase(os.path.abspath(child_env["OFFLINE_QUEUE_DB_PATH"]))
        if queue_path in seen_queues:
            raise MultiConfigError(
                f"CCTV별 OFFLINE_QUEUE_DB_PATH가 겹칩니다: {child_env['OFFLINE_QUEUE_DB_PATH']}"
            )

        seen_codes.add(cctv_code)
        seen_queues.add(queue_path)
        launches.append(DeviceLaunch(cctv_code, path, child_env))

    return launches


def run_devices(
    mode: str,
    launches: Sequence[DeviceLaunch],
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    poll_interval_sec: float = 0.2,
) -> int:
    processes: list[tuple[DeviceLaunch, subprocess.Popen]] = []
    try:
        for launch in launches:
            process = popen(
                [sys.executable, "-m", "raspberry_pi_congestion.main", mode],
                env=launch.env,
            )
            processes.append((launch, process))
            print(f"[multi] 시작: {launch.cctv_code} ({launch.profile_path})")

        while processes:
            for launch, process in list(processes):
                return_code = process.poll()
                if return_code is None:
                    continue
                processes.remove((launch, process))
                if return_code != 0:
                    print(
                        f"[multi] 실패: {launch.cctv_code} (exit={return_code})",
                        file=sys.stderr,
                    )
                    return return_code
                print(f"[multi] 종료: {launch.cctv_code}")
            if processes:
                time.sleep(poll_interval_sec)
        return 0
    except KeyboardInterrupt:
        print("\n[multi] 종료 요청을 받았습니다")
        return 130
    finally:
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
        for _, process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="여러 CCTV 감지 프로세스를 동시에 실행합니다")
    parser.add_argument("mode", nargs="?", default="file", choices=["file", "rtsp"])
    parser.add_argument(
        "--device-env",
        action="append",
        type=Path,
        help="실행할 CCTV env 파일. 여러 번 지정할 수 있습니다.",
    )
    parser.add_argument(
        "--device-dir",
        type=Path,
        default=Path("device-configs"),
        help="--device-env 생략 시 *.env 파일을 찾을 디렉터리",
    )
    args = parser.parse_args(argv)

    profile_paths = args.device_env or discover_profiles(args.device_dir)
    try:
        launches = build_launches(args.mode, profile_paths)
    except (MultiConfigError, OSError) as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2
    return run_devices(args.mode, launches)


if __name__ == "__main__":
    raise SystemExit(main())
