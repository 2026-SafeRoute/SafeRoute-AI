from __future__ import annotations

import argparse
import logging
import sys
import threading

from .api_client import AuthHeaderProvider, LoggingCongestionReporter, SafeRouteDeviceClient
from .app import CongestionPipeline
from .config import AppConfig, ConfigError
from .detectors import create_detector
from .offline_queue import OfflineQueue
from .preview import OpenCvPreview
from .relay import LightCommandExecutor, RelayController, RelayControllerError
from .roi_counter import RoiCounter
from .roi_provider import InteractiveRoiSelector, JsonRoiProvider
from .video_source import FileVideoSource, RtspStreamLostError, create_video_source
from .window_aggregator import WindowAggregator

logger = logging.getLogger(__name__)


def _setup_roi(config: AppConfig) -> None:
    path = config.roi_config_path or f"./config/roi/{config.cctv_code}.json"
    source = FileVideoSource(config.video_source)
    try:
        frame = next(source.frames())
        JsonRoiProvider(path).save(InteractiveRoiSelector().select(frame))
    finally:
        source.close()
    print(f"ROI saved to {path}. Set ROI_CONFIG_PATH={path} to count only inside this ROI.")


def _start_light_command_executor(config: AppConfig, device_client: SafeRouteDeviceClient) -> None:
    """유도등 명령 폴링을 이 프로세스 안에서 백그라운드 스레드로 돌린다.

    혼잡도 감지(pipeline.run())와 완전히 독립된 관심사라, 릴레이 초기화가
    실패해도 혼잡도 감지 자체는 계속 돼야 한다 - 여기서 예외를 흡수한다.
    """
    try:
        relay = RelayController(host=config.relay_host, port=config.relay_port)
        relay.refresh_status()
    except RelayControllerError as exc:
        logger.warning("릴레이 초기화 실패 - 유도등 명령 폴링 없이 계속 진행: %s", exc)
        return
    executor = LightCommandExecutor(device_client, relay, config.cctv_code)
    thread = threading.Thread(
        target=executor.run_forever,
        kwargs={"interval_sec": config.relay_poll_interval_sec},
        name="light-command-executor",
        daemon=True,
    )
    thread.start()
    logger.info("유도등 명령 폴링 시작: host=%s, port=%s, interval=%ss",
                config.relay_host, config.relay_port, config.relay_poll_interval_sec)


def main(argv=None) -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", choices=["setup-roi", "dry-run", "file", "rtsp", "test"])
    args = parser.parse_args(argv)
    try:
        config = AppConfig.from_env(mode=args.mode)
    except ConfigError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format=f"%(asctime)s %(levelname)s [{config.cctv_code}] %(name)s: %(message)s",
    )
    if config.mode == "setup-roi":
        _setup_roi(config)
        return 0
    device_client = None if config.mode in {"dry-run", "test"} else SafeRouteDeviceClient(
        config.server_base_url or "",
        AuthHeaderProvider(config.device_auth_token, config.auth_header_name, config.auth_header_prefix),
        config.request_timeout_sec, config.max_http_retries,
    )
    reporter = LoggingCongestionReporter() if device_client is None else device_client
    source = create_video_source(config.video_source, file_loop=config.video_loop,
                                 max_reconnects=config.rtsp_max_reconnects,
                                 base_delay_sec=config.rtsp_reconnect_base_delay_sec) if config.mode == "rtsp" else FileVideoSource(
        config.video_source,
        config.video_loop,
        realtime=config.file_realtime,
        fallback_fps=config.file_fallback_fps,
    )
    if config.mode == "test":
        from .detectors import FakePersonDetector
        detector = FakePersonDetector()
    else:
        detector = create_detector(config)
    queue = None if config.mode in {"dry-run", "test"} else OfflineQueue(config.offline_queue_db_path, config.offline_queue_max_age_sec, config.offline_queue_max_items)
    roi = JsonRoiProvider(config.roi_config_path).load() if config.roi_config_path else None
    if roi is None:
        logger.info("ROI_CONFIG_PATH가 없어 화면 전체의 사람을 집계합니다")
    preview = OpenCvPreview(roi or ()) if config.show_preview else None
    pipeline = CongestionPipeline(source, detector, RoiCounter(roi),
                                  WindowAggregator(config.window_sec), reporter, config.cctv_code, queue,
                                  config.target_inference_fps, config.offline_flush_interval_sec,
                                  config_provider=device_client,
                                  config_poll_active_sec=config.config_poll_active_sec,
                                  config_poll_inactive_sec=config.config_poll_inactive_sec,
                                  preview=preview,
                                  monitoring_jpeg_quality=config.monitoring_jpeg_quality,
                                  monitoring_image_max_width=config.monitoring_image_max_width,
                                  delivery_monitoring_workers=config.delivery_monitoring_workers)
    if device_client is not None and config.relay_host:
        _start_light_command_executor(config, device_client)
    try:
        pipeline.run()
    except RtspStreamLostError as exc:
        logger.error("%s; 재시작이 필요해 실패 코드로 종료합니다", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
