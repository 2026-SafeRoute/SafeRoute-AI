"""
JK-MTCP-2 (2채널 Modbus TCP/RTU 이더넷 릴레이) 제어 모듈.

Safe Route 프로젝트 - 라즈베리파이 엣지에서 좌/우 유도등(스마트 비상유도등)을
제어하는 릴레이 보드용 드라이버.

✅ 프로토콜 - 공식 매뉴얼(네이버 카페 "이더넷 MODBUS TCP RTU 릴레이 JK-MCTP-X
사용방법및 소프트웨어 다운로드", https://cafe.naver.com/codingblock/508 — 팀에서
보유한 실기기의 최신 매뉴얼로 확인됨)로 확인 완료.

    - 프로토콜: **Modbus RTU 프레임을 TCP 소켓으로 그대로 전송** (raw
      RTU-over-TCP). 표준 Modbus TCP(MBAP 헤더)가 아니라 CRC16이 붙은
      RTU 프레임이다. 그래서 pymodbus는 반드시
      `ModbusTcpClient(..., framer=FramerType.RTU)` 로 만들어야 한다
      (기본값 FramerType.SOCKET을 쓰면 기기가 응답하지 않는다).
    - 포트: **기기마다 다르다. 502 아님.** 매뉴얼 예시에는 4000(ZLVirCom
      설정툴 기준)과 5000(USR-M0 설정툴 기준)이 등장한다 — 보드에 어떤
      칩이 들어있는지에 따라 설정 프로그램과 기본 포트가 다른 것으로
      보인다. 반드시 USR-M0 또는 ZLVirCom 설정 프로그램으로 기기를
      검색해 실제 설정된 포트를 확인해서 넣을 것 (그래서 이 클래스는
      port에 기본값을 두지 않는다).
    - 출고 시 기본 IP: 192.168.0.81 (Static, DHCP 미지원, Subnet
      255.255.255.0, G/W 192.168.0.1) - 매뉴얼 댓글 기준.
    - Device/Slave(모듈) 주소 기본값: 0x01 (1)
    - 채널(릴레이) 0 = 코일 주소 0x0000, 채널(릴레이) 1 = 코일 주소 0x0001
      → 이 코드의 RelayChannel.CH1 = 코일 0, CH2 = 코일 1로 매핑.
    - 채널 제어: Write Single Coil (Function Code 0x05), ON=0xFF00 /
      OFF=0x0000 → pymodbus의 write_coil(address, True/False)와 동일.
    - 채널 상태 조회: Read Coils (Function Code 0x01) → read_coils(address).
    - 매뉴얼 원문 예시 (모듈 주소 0x01 기준, 이 레포 tests/에서 실제
      pymodbus RTU-over-TCP 클라이언트로 바이트 단위까지 재현/검증함):
        릴레이 0(CH1) ON  : 01 05 00 00 FF 00 8C 3A
        릴레이 0(CH1) OFF : 01 05 00 00 00 00 CD CA
        릴레이 1(CH2) ON  : 01 05 00 01 FF 00 DD FA
        릴레이 1(CH2) OFF : 01 05 00 01 00 00 9C 0A
        릴레이 0(CH1) 상태 읽기 : 01 01 00 00 00 01 FD CA

    (매뉴얼 댓글에 2025-12-23부터 설정 프로그램이 바뀐 후속 버전이 있다는
    언급이 있었으나, 이 문서 자체가 사용자가 보유한 기기의 최신 매뉴얼로
    확인됨 — 별도 대조 불필요.)

안전 설계
--------------------------------------------
1. 훈련 중 방향을 안내할 때는 두 채널이 절대 동시에 ON 상태가 될 수
   없다. (좌/우 유도등이 동시에 켜지면 대피자에게 상반된 방향 신호를
   줄 수 있어 소방법 취지에 반한다.)
   -> turn_on()은 실제 기기에서 반대 채널 상태를 먼저 읽어 확인한 뒤에만
      ON 명령을 보낸다. 반대 채널이 ON이면 BothChannelsOnError를 던지고
      아무 명령도 보내지 않는다.
   -> 단, 훈련이 없는 평상시에는 두 채널 모두 켜진 상태가 정상이다(일반
      출구 표시등과 동일 - 특정 방향을 지시하는 게 아니라 출구 위치만
      알리므로 좌/우 동시 점등이 안전 문제가 아니다). 이 경우에는
      turn_on()의 인터록을 거치지 않는 both_on()을 사용한다.
2. 통신 장애(연결 실패/타임아웃/에러 응답) 발생 시 즉시 fail-safe로 전환한다.
   - 상태 캐시를 UNKNOWN(None)으로 내려 "모르면 ON이라고 말하지 않는다"를
     보장하고,
   - 두 채널 모두에 OFF 명령을 best-effort로 재전송을 시도한다.
   - 이 재전송까지 실패하면 상태는 UNKNOWN으로 남지만, 절대 ON으로
     보고하지 않는다.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import ModbusException
from pymodbus.framer import FramerType

logger = logging.getLogger(__name__)


class RelayChannel(IntEnum):
    """JK-MTCP-2의 물리 채널. 좌/우 유도등에 매핑된다."""

    CH1 = 1
    CH2 = 2


class RelayControllerError(Exception):
    """RelayController 관련 예외의 베이스 클래스."""


class RelayCommunicationError(RelayControllerError):
    """JK-MTCP-2와의 통신(연결/읽기/쓰기)이 실패했을 때 발생한다."""


class BothChannelsOnError(RelayControllerError):
    """두 채널을 동시에 ON 하려는 시도를 막기 위한 안전 인터록 예외."""


@dataclass(frozen=True)
class RelayCoilMap:
    """채널 -> Modbus 코일 주소 매핑. 실기기 검증 후 필요하면 값만 바꾸면 된다."""

    ch1_coil: int = 0
    ch2_coil: int = 1

    def coil_for(self, channel: RelayChannel) -> int:
        return self.ch1_coil if channel is RelayChannel.CH1 else self.ch2_coil


class RelayController:
    """JK-MTCP-2 2채널 릴레이를 Modbus TCP로 제어한다.

    사용 예시
    ---------
    >>> # port는 USR-M0/ZLVirCom 설정 프로그램으로 실기기에서 직접 확인한 값을 넣을 것
    >>> controller = RelayController(host="192.168.0.81", port=5000)
    >>> controller.refresh_status()          # 시작 시 실기기 상태와 캐시 동기화
    >>> controller.switch_to(RelayChannel.CH1)   # 좌측 유도등 방향으로 전환
    >>> controller.both_on()                 # 훈련 종료 -> 평상시(양쪽 점등) 복귀
    >>> controller.all_off()                 # fail-safe 복구 등 강제 OFF
    """

    def __init__(
        self,
        host: str,
        port: int,
        device_id: int = 1,
        coil_map: Optional[RelayCoilMap] = None,
        timeout: float = 3.0,
        retries: int = 1,
        client: Optional[ModbusTcpClient] = None,
    ) -> None:
        self._host = host
        self._port = port
        self._device_id = device_id
        self._coil_map = coil_map or RelayCoilMap()
        self._lock = threading.RLock()
        # 테스트에서는 pymodbus 대신 fake/mock 클라이언트를 주입할 수 있다 (DI).
        # framer=FramerType.RTU가 핵심: 기본값(SOCKET/MBAP)으로는 이 기기가 응답하지 않는다.
        self._client = client or ModbusTcpClient(
            host, port=port, framer=FramerType.RTU, timeout=timeout, retries=retries
        )
        # 마지막으로 실기기에서 확인된 채널 상태. 통신 실패 시 None(UNKNOWN)으로 리셋.
        self._state: dict[RelayChannel, Optional[bool]] = {
            RelayChannel.CH1: None,
            RelayChannel.CH2: None,
        }
        self._healthy = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def healthy(self) -> bool:
        """마지막 통신이 정상이었는지 여부. False면 상위 계층이 유도등을
        오프라인으로 취급하고 BE에 알려야 한다."""
        with self._lock:
            return self._healthy

    def is_on(self, channel: RelayChannel) -> Optional[bool]:
        """마지막으로 확인된 채널 상태. 확인 불가 상태면 None을 반환한다.
        (fail-safe 원칙: 모르면 ON이라고 답하지 않는다.)"""
        with self._lock:
            return self._state[channel]

    def turn_on(self, channel: RelayChannel) -> None:
        """지정한 채널만 켠다.

        반대 채널이 실제로 ON 상태로 확인되면 BothChannelsOnError를 던지고
        아무 명령도 보내지 않는다. 방향을 전환하려면 switch_to()를 사용할 것.
        """
        with self._lock:
            other = self._other_channel(channel)
            if self._read_channel(other):
                raise BothChannelsOnError(
                    f"{other.name}가 이미 ON 상태입니다. "
                    f"{channel.name}을 켜려면 먼저 {other.name}을 끄거나 switch_to()를 사용하세요."
                )
            self._write_channel(channel, True)

    def turn_off(self, channel: RelayChannel) -> None:
        """지정한 채널만 끈다. (OFF는 인터록 대상이 아니므로 항상 허용)"""
        with self._lock:
            self._write_channel(channel, False)

    def switch_to(self, channel: RelayChannel) -> None:
        """반대 채널을 끄고 지정한 채널을 켠다. 좌/우 방향 전환에 사용."""
        with self._lock:
            other = self._other_channel(channel)
            self._write_channel(other, False)
            self._write_channel(channel, True)

    def both_on(self) -> None:
        """두 채널을 모두 켠다 (평상시/훈련 미진행 상태 전용).

        turn_on()과 달리 반대 채널 상태를 확인하지 않고 인터록을 거치지
        않는다 - 훈련이 없을 때는 좌/우 동시 점등이 정상 상태이기 때문.
        훈련 중 방향 안내에는 절대 쓰지 말고 turn_on()/switch_to()를
        사용할 것.
        """
        with self._lock:
            first_error: Optional[RelayCommunicationError] = None
            for ch in RelayChannel:
                try:
                    self._write_channel(ch, True)
                except RelayCommunicationError as exc:
                    first_error = first_error or exc
            if first_error is not None:
                raise first_error

    def all_off(self) -> None:
        """두 채널 모두 OFF. fail-safe 복구나 훈련 종료 시 호출한다."""
        with self._lock:
            first_error: Optional[RelayCommunicationError] = None
            for ch in RelayChannel:
                try:
                    self._write_channel(ch, False)
                except RelayCommunicationError as exc:
                    first_error = first_error or exc
            if first_error is not None:
                raise first_error

    def refresh_status(self) -> dict[RelayChannel, Optional[bool]]:
        """실기기에서 두 채널 상태를 읽어와 내부 캐시를 동기화한다.
        컨트롤러 생성 직후, 혹은 통신 장애 복구 후 반드시 한 번 호출할 것."""
        with self._lock:
            for ch in RelayChannel:
                self._read_channel(ch)
            return dict(self._state)

    def close(self) -> None:
        with self._lock:
            self._client.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _other_channel(channel: RelayChannel) -> RelayChannel:
        return RelayChannel.CH2 if channel is RelayChannel.CH1 else RelayChannel.CH1

    def _read_channel(self, channel: RelayChannel) -> bool:
        """실기기에서 채널 상태를 읽는다. 실패 시 fail-safe로 전환 후 예외를 던진다."""
        address = self._coil_map.coil_for(channel)
        try:
            if not self._client.connect():
                raise RelayCommunicationError(f"{self._host}:{self._port} 연결 실패")
            result = self._client.read_coils(address, count=1, device_id=self._device_id)
            if result.isError():
                raise RelayCommunicationError(f"read_coils({address}) 실패: {result}")
        except RelayCommunicationError:
            self._enter_fail_safe()
            raise
        except (ModbusException, OSError) as exc:
            self._enter_fail_safe()
            raise RelayCommunicationError(str(exc)) from exc
        else:
            state = bool(result.bits[0])
            self._state[channel] = state
            self._healthy = True
            return state

    def _write_channel(self, channel: RelayChannel, on: bool) -> None:
        """실기기에 채널 ON/OFF 명령을 보낸다. 실패 시 fail-safe로 전환 후 예외를 던진다."""
        address = self._coil_map.coil_for(channel)
        try:
            if not self._client.connect():
                raise RelayCommunicationError(f"{self._host}:{self._port} 연결 실패")
            result = self._client.write_coil(address, on, device_id=self._device_id)
            if result.isError():
                raise RelayCommunicationError(f"write_coil({address}, {on}) 실패: {result}")
        except RelayCommunicationError:
            self._enter_fail_safe()
            raise
        except (ModbusException, OSError) as exc:
            self._enter_fail_safe()
            raise RelayCommunicationError(str(exc)) from exc
        else:
            self._state[channel] = on
            self._healthy = True

    def _enter_fail_safe(self) -> None:
        """통신 실패 시 안전 상태로 전환한다.

        1) 상태를 UNKNOWN(None)으로 내려, 실패 이후 확인되지 않은 상태를
           절대 ON으로 잘못 보고하지 않게 한다.
        2) 두 채널 모두에 OFF 명령을 best-effort로 재시도한다. 이 재시도
           자체가 실패해도 (1)에서 이미 UNKNOWN으로 내렸으므로 안전하다.
        재귀적으로 다시 실패 처리에 빠지지 않도록 _write_channel/_read_channel을
        거치지 않고 클라이언트를 직접 호출한다.
        """
        logger.error("JK-MTCP-2(%s:%s) 통신 실패 - fail-safe 전환", self._host, self._port)
        self._healthy = False
        self._state[RelayChannel.CH1] = None
        self._state[RelayChannel.CH2] = None

        for ch in RelayChannel:
            address = self._coil_map.coil_for(ch)
            try:
                if not self._client.connect():
                    continue
                result = self._client.write_coil(address, False, device_id=self._device_id)
                if not result.isError():
                    self._state[ch] = False
            except Exception:  # noqa: BLE001 - fail-safe 경로는 어떤 예외에도 죽으면 안 된다
                logger.exception("fail-safe OFF 재시도 중 추가 오류 (channel=%s)", ch)
