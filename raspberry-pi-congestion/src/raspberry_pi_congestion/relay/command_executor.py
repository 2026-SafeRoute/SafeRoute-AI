"""BE의 유도등 명령 큐를 폴링해서 RelayController로 실행하고 ACK를 보고하는 루프.

BE(EC2)가 사설망의 Pi를 직접 호출할 수 없어서, Pi가 먼저 물어봐서
가져가는 구조다. BE 쪽 API 계약은 SafeRoute-BE 레포의
LightCommandController/LightCommandService를 참고.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

from ..api_client import SafeRouteDeviceClient
from .controller import RelayChannel, RelayController, RelayControllerError

logger = logging.getLogger(__name__)

# BE의 IoTLightDirection(LEFT/RIGHT)과 물리 채널 매핑. 어느 채널이 "왼쪽"인지는
# 실치 배선에 따라 다를 수 있으므로, 실기기 설치 후 확인해서 필요하면 이
# 매핑만 바꾸면 된다 (RelayCoilMap의 코일 주소 매핑과는 별개의 논리 매핑).
LEFT_CHANNEL = RelayChannel.CH1
RIGHT_CHANNEL = RelayChannel.CH2


class LightCommandExecutor:
    """cctv_code 하나(=이 Pi)가 담당하는 유도등 명령을 폴링/실행/ACK한다."""

    def __init__(
        self,
        client: SafeRouteDeviceClient,
        relay: RelayController,
        cctv_code: str,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self._relay = relay
        self._cctv_code = cctv_code
        self._sleep = sleeper

    def poll_once(self) -> int:
        """한 번 폴링해서 받은 명령들을 실행하고 ACK한다. 처리한 명령 수를 반환한다."""
        commands = self._client.fetch_light_commands(self._cctv_code)
        for command in commands:
            self._execute_and_ack(command)
        return len(commands)

    def run_forever(self, interval_sec: float = 2.0) -> None:
        """짧은 주기로 계속 폴링한다. 폴링 한 번의 실패가 루프 전체를 죽이지 않는다."""
        while True:
            try:
                self.poll_once()
            except Exception:  # noqa: BLE001 - 폴링 루프는 예외로 죽으면 안 된다
                logger.exception("유도등 명령 폴링 중 오류")
            self._sleep(interval_sec)

    def _execute_and_ack(self, command: dict) -> None:
        command_id = command.get("commandId")
        direction = command.get("direction")
        try:
            self._apply(direction)
        except (RelayControllerError, ValueError) as exc:
            logger.warning(
                "유도등 명령 실행 실패: commandId=%s, direction=%s, error=%s",
                command_id, direction, exc,
            )
            self._client.ack_light_command(command_id, success=False, fail_reason=str(exc))
            return
        self._client.ack_light_command(command_id, success=True)

    def _apply(self, direction: Optional[str]) -> None:
        if direction == "LEFT":
            self._relay.switch_to(LEFT_CHANNEL)
        elif direction == "RIGHT":
            self._relay.switch_to(RIGHT_CHANNEL)
        elif direction == "OFF":
            self._relay.all_off()
        elif direction == "BOTH":
            self._relay.both_on()
        else:
            raise ValueError(f"알 수 없는 유도등 방향: {direction}")
