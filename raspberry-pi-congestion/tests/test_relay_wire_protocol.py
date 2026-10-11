"""RelayController가 실제로 매뉴얼과 '바이트 단위로' 동일한 명령을 보내는지 검증.

tests/test_relay_controller.py는 FakeModbusClient로 비즈니스 로직(인터록,
fail-safe)만 검증한다. 이 파일은 그와 별개로, RelayController가 진짜
pymodbus RTU-over-TCP 클라이언트를 통해 실제로 내보내는 바이트가 공식
매뉴얼(네이버 카페 https://cafe.naver.com/codingblock/508)의 예시와
정확히 일치하는지를 로컬 TCP 소켓으로 캡처해서 확인한다.

이 테스트가 통과한다는 것은 "이 코드가 진짜 JK-MTCP-2에 연결됐을 때
매뉴얼이 말하는 그 명령을 보낸다"는 것을 보장한다는 뜻이다.
"""

from __future__ import annotations

import socket
import threading

from raspberry_pi_congestion.relay import RelayChannel, RelayController

# 매뉴얼 원문 예시 (모듈 주소 0x01 기준)
MANUAL_CH1_ON = bytes.fromhex("01 05 00 00 FF 00 8C 3A".replace(" ", ""))
MANUAL_CH1_OFF = bytes.fromhex("01 05 00 00 00 00 CD CA".replace(" ", ""))
MANUAL_CH2_ON = bytes.fromhex("01 05 00 01 FF 00 DD FA".replace(" ", ""))
MANUAL_CH2_OFF = bytes.fromhex("01 05 00 01 00 00 9C 0A".replace(" ", ""))
MANUAL_READ_CH1_STATUS = bytes.fromhex("01 01 00 00 00 01 FD CA".replace(" ", ""))

# Read Coils(func 0x01) 정상 응답 예시(코일 OFF, 1개): slave=1, func=1,
# byte_count=1, data=0x00 + CRC16. write_coil 요청은 응답이 요청과 동일한
# 바이트열이므로(Modbus 표준) 그 경우엔 그대로 에코한다.
_READ_COILS_OFF_RESPONSE = bytes.fromhex("01 01 01 00 51 88".replace(" ", ""))


class _RecordingServer:
    """RelayController가 보낸 요청을 그대로 캡처하는 더미 TCP 서버.

    Write Single Coil(func 0x05) 요청은 표준상 응답이 요청과 동일한 바이트열이라
    그대로 에코하고, Read Coils(func 0x01) 요청에는 유효한 형식의 더미 응답을
    돌려준다. 어느 쪽이든 "실제로 보낸 요청 바이트"는 self.received에 그대로
    남으므로, 이 테스트가 검증하려는 건 응답이 아니라 RelayController가 보낸
    요청이 매뉴얼과 일치하는지다.
    """

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))
        self.port = self.sock.getsockname()[1]
        self.sock.listen(1)
        self.received: list[bytes] = []
        self._thread = threading.Thread(target=self._serve_once, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _serve_once(self) -> None:
        conn, _ = self.sock.accept()
        with conn:
            conn.settimeout(2)
            try:
                while True:
                    data = conn.recv(256)
                    if not data:
                        break
                    self.received.append(data)
                    if len(data) >= 2 and data[1] == 0x01:  # Read Coils 요청
                        conn.sendall(_READ_COILS_OFF_RESPONSE)
                    else:  # Write Single Coil 요청 -> 표준상 요청 그대로 에코
                        conn.sendall(data)
            except (socket.timeout, OSError):
                pass

    def stop(self) -> None:
        self._thread.join(timeout=2)
        self.sock.close()


def test_turn_on_CH1이_매뉴얼의_릴레이0_ON_명령과_바이트단위로_일치한다():
    server = _RecordingServer()
    server.start()
    controller = RelayController(host="127.0.0.1", port=server.port, timeout=2)

    try:
        controller.turn_on(RelayChannel.CH1)
    finally:
        controller.close()
        server.stop()

    assert server.received[-1] == MANUAL_CH1_ON


def test_turn_off_CH1이_매뉴얼의_릴레이0_OFF_명령과_바이트단위로_일치한다():
    server = _RecordingServer()
    server.start()
    controller = RelayController(host="127.0.0.1", port=server.port, timeout=2)

    try:
        controller.turn_off(RelayChannel.CH1)
    finally:
        controller.close()
        server.stop()

    assert server.received[-1] == MANUAL_CH1_OFF


def test_switch_to_CH2가_매뉴얼의_릴레이0_OFF_릴레이1_ON_명령과_일치한다():
    server = _RecordingServer()
    server.start()
    controller = RelayController(host="127.0.0.1", port=server.port, timeout=2)

    try:
        controller.switch_to(RelayChannel.CH2)
    finally:
        controller.close()
        server.stop()

    # switch_to(CH2)는 반대 채널(CH1) OFF -> CH2 ON 순서로 두 개의 명령을 보낸다
    assert server.received == [MANUAL_CH1_OFF, MANUAL_CH2_ON]


def test_both_on이_매뉴얼의_릴레이0_ON_릴레이1_ON_명령과_일치한다():
    server = _RecordingServer()
    server.start()
    controller = RelayController(host="127.0.0.1", port=server.port, timeout=2)

    try:
        controller.both_on()
    finally:
        controller.close()
        server.stop()

    # both_on()은 CH1 ON -> CH2 ON 순서로 두 개의 명령을 보낸다
    assert server.received == [MANUAL_CH1_ON, MANUAL_CH2_ON]


def test_상태_조회가_매뉴얼의_릴레이0_상태_읽기_명령과_일치한다():
    server = _RecordingServer()
    server.start()
    controller = RelayController(host="127.0.0.1", port=server.port, timeout=2)

    try:
        controller.refresh_status()
    finally:
        controller.close()
        server.stop()

    assert server.received[0] == MANUAL_READ_CH1_STATUS
