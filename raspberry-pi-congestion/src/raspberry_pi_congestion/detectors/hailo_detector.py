from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

from ..models import Detection
from .base import PersonDetector

logger = logging.getLogger(__name__)


class HailoRuntimeError(RuntimeError):
    """HailoRT 설치, 장치 초기화 또는 HEF 계약이 올바르지 않을 때 발생한다."""


class _Runtime(Protocol):
    input_shape: tuple[int, int, int]

    def infer(self, frame: np.ndarray): ...

    def close(self) -> None: ...


class _HailoVStreamsRuntime:
    """HailoRT 4.x의 blocking InferVStreams를 감싼 batch=1 런타임."""

    def __init__(self, hef_path: str, multi_process_service: bool = True) -> None:
        try:
            from hailo_platform import (
                ConfigureParams,
                FormatType,
                HailoSchedulingAlgorithm,
                HailoStreamInterface,
                HEF,
                InferVStreams,
                InputVStreamParams,
                OutputVStreamParams,
                VDevice,
            )
            from hailo_platform.pyhailort.pyhailort import FormatOrder
        except ImportError as exc:
            raise HailoRuntimeError(
                "hailo_platform을 불러올 수 없습니다. HailoRT와 현재 Python 버전에 맞는 "
                "PyHailoRT를 설치하고 가상환경에서 import되는지 확인하세요."
            ) from exc

        self._vdevice = None
        self._activation_context = None
        self._infer_context = None
        self._infer_pipeline = None
        try:
            self._hef = HEF(hef_path)
            input_infos = self._hef.get_input_vstream_infos()
            output_infos = self._hef.get_output_vstream_infos()
            if len(input_infos) != 1:
                raise HailoRuntimeError(
                    f"단일 입력 HEF만 지원합니다(현재 입력 {len(input_infos)}개)."
                )
            if len(output_infos) != 1:
                raise HailoRuntimeError(
                    f"Hailo NMS 단일 출력 HEF만 지원합니다(현재 출력 {len(output_infos)}개)."
                )
            if output_infos[0].format.order != FormatOrder.HAILO_NMS_BY_CLASS:
                raise HailoRuntimeError(
                    "HAILO_NMS_BY_CLASS 출력 HEF만 지원합니다"
                    f"(현재 order={output_infos[0].format.order})."
                )

            shape = tuple(int(value) for value in input_infos[0].shape)
            if len(shape) != 3 or shape[2] != 3:
                raise HailoRuntimeError(
                    f"NHWC 3채널 입력 HEF만 지원합니다(현재 입력 shape={shape})."
                )
            self.input_shape = shape
            self._input_name = input_infos[0].name
            self._output_name = output_infos[0].name

            device_params = VDevice.create_params()
            device_params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            device_params.group_id = "SHARED"
            # multi_main은 CCTV마다 별도 프로세스를 띄우므로, 여러 프로세스가 장치 하나를
            # 나눠 쓰려면 hailort.service(multi-process service)를 거쳐야 한다.
            device_params.multi_process_service = multi_process_service
            self._vdevice = VDevice(device_params)

            configure_params = ConfigureParams.create_from_hef(
                hef=self._hef,
                interface=HailoStreamInterface.PCIe,
            )
            network_groups = self._vdevice.configure(self._hef, configure_params)
            if len(network_groups) != 1:
                raise HailoRuntimeError(
                    f"단일 network group HEF만 지원합니다(현재 {len(network_groups)}개)."
                )
            self._network_group = network_groups[0]
            network_params = self._network_group.create_params()
            input_params = InputVStreamParams.make(
                self._network_group, format_type=FormatType.UINT8
            )
            output_params = OutputVStreamParams.make(
                self._network_group, format_type=FormatType.FLOAT32
            )

            self._activation_context = self._network_group.activate(network_params)
            self._activation_context.__enter__()
            self._infer_context = InferVStreams(
                self._network_group,
                input_params,
                output_params,
                tf_nms_format=False,
            )
            self._infer_pipeline = self._infer_context.__enter__()
        except HailoRuntimeError:
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise HailoRuntimeError(
                f"Hailo 장치 또는 HEF 초기화에 실패했습니다: {type(exc).__name__}: {exc}"
            ) from exc

    def infer(self, frame: np.ndarray):
        if self._infer_pipeline is None:
            raise HailoRuntimeError("Hailo 런타임이 이미 종료되었습니다.")
        output = self._infer_pipeline.infer(
            {self._input_name: np.expand_dims(frame, axis=0)}
        )
        if self._output_name not in output:
            raise HailoRuntimeError(
                f"HEF 출력 {self._output_name!r}을 추론 결과에서 찾을 수 없습니다."
            )
        return output[self._output_name]

    def close(self) -> None:
        if self._infer_context is not None:
            self._infer_context.__exit__(None, None, None)
            self._infer_context = None
            self._infer_pipeline = None
        if self._activation_context is not None:
            self._activation_context.__exit__(None, None, None)
            self._activation_context = None
        if self._vdevice is not None:
            self._vdevice.release()
            self._vdevice = None


class HailoPersonDetector(PersonDetector):
    """Hailo NMS 출력 HEF를 사용하는 COCO person detector.

    HEF 출력은 class별 ``[ymin, xmin, ymax, xmax, score]`` 배열이고 좌표는
    0~1로 정규화되어 있다는 Hailo NMS 계약을 사용한다.
    """

    def __init__(
        self,
        hef_path: str,
        conf_threshold: float = 0.4,
        *,
        runtime_factory: Callable[[str], _Runtime] = _HailoVStreamsRuntime,
    ) -> None:
        model_path = Path(hef_path)
        if not model_path.is_file():
            raise HailoRuntimeError(f"HEF 모델 파일을 찾을 수 없습니다: {hef_path}")
        if not 0.0 <= conf_threshold <= 1.0:
            raise ValueError("conf_threshold must be between 0 and 1")

        self.conf_threshold = conf_threshold
        self._runtime = runtime_factory(str(model_path))
        self._closed = False
        logger.info(
            "Hailo detector loaded model=%s input_shape=%s",
            model_path,
            self._runtime.input_shape,
        )

    def detect(self, frame) -> list[Detection]:
        if self._closed:
            raise HailoRuntimeError("Hailo detector가 이미 종료되었습니다.")
        if (
            not isinstance(frame, np.ndarray)
            or frame.ndim != 3
            or frame.shape[2] != 3
            or frame.shape[0] <= 0
            or frame.shape[1] <= 0
            or frame.dtype != np.uint8
        ):
            raise ValueError(
                "frame must be a non-empty uint8 BGR ndarray with shape (H, W, 3)"
            )

        original_height, original_width = frame.shape[:2]
        prepared, scale, pad_x, pad_y = self._preprocess(frame)
        class_outputs = self._runtime.infer(prepared)
        return self._person_detections(
            class_outputs,
            original_width,
            original_height,
            scale,
            pad_x,
            pad_y,
        )

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        import cv2

        input_height, input_width, _ = self._runtime.input_shape
        height, width = frame.shape[:2]
        scale = min(input_width / width, input_height / height)
        resized_width = max(1, min(input_width, int(round(width * scale))))
        resized_height = max(1, min(input_height, int(round(height * scale))))
        resized = cv2.resize(frame, (resized_width, resized_height))
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        pad_x = (input_width - resized_width) // 2
        pad_y = (input_height - resized_height) // 2
        canvas = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
        return np.ascontiguousarray(canvas), scale, pad_x, pad_y

    def _person_detections(
        self,
        class_outputs,
        original_width: int,
        original_height: int,
        scale: float,
        pad_x: int,
        pad_y: int,
    ) -> list[Detection]:
        classes = self._normalize_nms_classes(class_outputs)
        if not classes:
            return []

        rows = np.asarray(classes[0], dtype=np.float32)
        if rows.size == 0:
            return []
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        if rows.ndim != 2 or rows.shape[1] < 5:
            raise HailoRuntimeError(
                f"지원하지 않는 Hailo NMS person 출력 shape입니다: {rows.shape}"
            )

        input_height, input_width, _ = self._runtime.input_shape
        detections: list[Detection] = []
        for row in rows:
            y1, x1, y2, x2, score = (float(value) for value in row[:5])
            if score < self.conf_threshold:
                continue
            x1 = self._unletterbox(x1 * input_width, pad_x, scale, original_width)
            x2 = self._unletterbox(x2 * input_width, pad_x, scale, original_width)
            y1 = self._unletterbox(y1 * input_height, pad_y, scale, original_height)
            y2 = self._unletterbox(y2 * input_height, pad_y, scale, original_height)
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(Detection(x1, y1, x2, y2, score))
        return detections

    @staticmethod
    def _normalize_nms_classes(class_outputs) -> Sequence:
        # InferVStreams batch=1은 class별 ndarray 목록을 반환한다. 일부 HailoRT
        # 버전은 그 목록을 batch 목록 한 겹으로 감싸므로 둘 다 허용한다.
        if isinstance(class_outputs, np.ndarray) and class_outputs.dtype == object:
            class_outputs = class_outputs.tolist()
        if not isinstance(class_outputs, (list, tuple)):
            raise HailoRuntimeError(
                "HEF 출력이 Hailo NMS class 목록이 아닙니다. NMS 포함 HEF를 사용하세요."
            )
        if (
            len(class_outputs) == 1
            and isinstance(class_outputs[0], (list, tuple))
            and class_outputs[0]
            and isinstance(class_outputs[0][0], np.ndarray)
        ):
            return class_outputs[0]
        return class_outputs

    @staticmethod
    def _unletterbox(value: float, padding: int, scale: float, limit: int) -> float:
        return max(0.0, min(float(limit), (value - padding) / scale))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._runtime.close()
