from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Mapping, Optional


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class AppConfig:
    mode: str
    video_source: str
    roi_config_path: Optional[str]
    cctv_code: str
    detector_backend: str
    model_path: Optional[str]
    detector_conf_threshold: float
    target_inference_fps: float
    window_sec: float
    server_base_url: Optional[str]
    observation_path: str
    config_poll_active_sec: float
    config_poll_inactive_sec: float
    device_auth_token: Optional[str]
    auth_header_name: str
    auth_header_prefix: str
    request_timeout_sec: float
    max_http_retries: int
    offline_queue_db_path: str
    offline_queue_max_age_sec: float
    offline_queue_max_items: int
    offline_flush_interval_sec: float
    rtsp_max_reconnects: int
    rtsp_reconnect_base_delay_sec: float
    video_loop: bool
    file_realtime: bool
    file_fallback_fps: float
    show_preview: bool
    log_level: str
    relay_host: Optional[str]
    relay_port: Optional[int]
    relay_poll_interval_sec: float
    monitoring_jpeg_quality: int
    monitoring_image_max_width: int
    delivery_monitoring_workers: int

    @staticmethod
    def from_env(env: Optional[Mapping[str, str]] = None, mode: Optional[str] = None) -> "AppConfig":
        e = env if env is not None else os.environ
        selected_mode = mode or e.get("RUN_MODE", "dry-run")
        def required(key: str) -> str:
            value = e.get(key)
            if not value:
                raise ConfigError(f"Missing required environment variable: {key}")
            return value
        def boolean(key: str, default: bool) -> bool:
            return e.get(key, str(default)).lower() in {"1", "true", "yes", "on"}
        def positive_float(key: str, default: str) -> float:
            try:
                value = float(e.get(key, default))
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"{key} must be a positive number") from exc
            if not math.isfinite(value) or value <= 0:
                raise ConfigError(f"{key} must be a positive number")
            return value
        server = e.get("SAFEROUTE_SERVER_BASE_URL")
        cctv_code = required("CCTV_CODE")
        config_poll_active_sec = positive_float("CONFIG_POLL_ACTIVE_SEC", "5")
        config_poll_inactive_sec = positive_float("CONFIG_POLL_INACTIVE_SEC", "1")
        def positive_int(key: str, default: str) -> int:
            try:
                value = int(e.get(key, default))
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"{key} must be a positive integer") from exc
            if value <= 0:
                raise ConfigError(f"{key} must be a positive integer")
            return value
        file_fallback_fps = positive_float("FILE_FALLBACK_FPS", "30")
        # 모니터링 스냅샷은 실시간 화면용이라 화질을 낮춰 업로드 시간을 줄인다. 혼잡 이벤트 이미지는 원본 유지.
        monitoring_jpeg_quality = positive_int("MONITORING_JPEG_QUALITY", "70")
        if monitoring_jpeg_quality > 100:
            raise ConfigError("MONITORING_JPEG_QUALITY must be between 1 and 100")
        if selected_mode in {"file", "rtsp"} and not server:
            raise ConfigError("SAFEROUTE_SERVER_BASE_URL is required for server reporting modes")
        if selected_mode in {"file", "rtsp"} and not e.get("DEVICE_AUTH_TOKEN"):
            raise ConfigError("DEVICE_AUTH_TOKEN is required for server reporting modes")
        # 릴레이(유도등)는 옵션이다 - 이 CCTV(Pi)에 릴레이 보드가 실제로 붙어있을 때만
        # RELAY_HOST를 설정한다. 포트는 기기마다 다르므로(USR-M0/ZLVirCom 설정 프로그램으로
        # 실기기에서 확인) 기본값을 두지 않는다.
        relay_host = e.get("RELAY_HOST")
        relay_port_raw = e.get("RELAY_PORT")
        if relay_host and not relay_port_raw:
            raise ConfigError("RELAY_PORT is required when RELAY_HOST is set")
        try:
            relay_port = int(relay_port_raw) if relay_port_raw else None
        except ValueError as exc:
            raise ConfigError("RELAY_PORT must be an integer") from exc
        return AppConfig(
            mode=selected_mode, video_source=required("VIDEO_SOURCE"),
            # ROI_CONFIG_PATH가 없으면 ROI 없이 화면 전체의 사람을 집계한다.
            roi_config_path=e.get("ROI_CONFIG_PATH") or None, cctv_code=cctv_code,
            detector_backend=e.get("DETECTOR_BACKEND", "ultralytics"), model_path=e.get("MODEL_PATH"),
            detector_conf_threshold=float(e.get("DETECTOR_CONF_THRESHOLD", "0.4")),
            target_inference_fps=float(e.get("TARGET_INFERENCE_FPS", "5")), window_sec=float(e.get("WINDOW_SEC", "5")),
            server_base_url=server, observation_path=e.get("CONGESTION_OBSERVATION_PATH", "/api/v1/device/congestion-observations"),
            config_poll_active_sec=config_poll_active_sec,
            config_poll_inactive_sec=config_poll_inactive_sec,
            device_auth_token=e.get("DEVICE_AUTH_TOKEN"), auth_header_name=e.get("AUTH_HEADER_NAME", "Authorization"),
            auth_header_prefix=e.get("AUTH_HEADER_PREFIX", "Bearer"), request_timeout_sec=float(e.get("REQUEST_TIMEOUT_SEC", "5")),
            max_http_retries=int(e.get("MAX_HTTP_RETRIES", "2")), offline_queue_db_path=e.get("OFFLINE_QUEUE_DB_PATH", "./offline_queue.sqlite3"),
            offline_queue_max_age_sec=float(e.get("OFFLINE_QUEUE_MAX_AGE_SEC", "86400")), offline_queue_max_items=int(e.get("OFFLINE_QUEUE_MAX_ITEMS", "1000")),
            offline_flush_interval_sec=float(e.get("OFFLINE_FLUSH_INTERVAL_SEC", "30")), rtsp_max_reconnects=int(e.get("RTSP_MAX_RECONNECTS", "5")),
            rtsp_reconnect_base_delay_sec=float(e.get("RTSP_RECONNECT_BASE_DELAY_SEC", "1")), video_loop=boolean("VIDEO_LOOP", False),
            file_realtime=boolean("FILE_REALTIME", True), file_fallback_fps=file_fallback_fps,
            show_preview=boolean("SHOW_PREVIEW", False),
            log_level=e.get("LOG_LEVEL", "INFO"),
            relay_host=relay_host, relay_port=relay_port,
            relay_poll_interval_sec=positive_float("RELAY_POLL_INTERVAL_SEC", "2"),
            monitoring_jpeg_quality=monitoring_jpeg_quality,
            monitoring_image_max_width=positive_int("MONITORING_IMAGE_MAX_WIDTH", "960"),
            delivery_monitoring_workers=positive_int("DELIVERY_MONITORING_WORKERS", "2"),
        )
