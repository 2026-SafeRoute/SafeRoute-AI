"""RelayController(JK-MTCP-2) 단위 테스트.

실기기 없이도 돌아가도록 FakeModbusClient를 주입해서 테스트한다.
요구된 테스트 범위: 채널 1/2 ON/OFF, 두 채널 동시 ON 방지, 통신 실패 시
OFF(fail-safe) 세 가지 + 관련 안전/상태 케이스.
"""

from __future__ import annotations

import pytest
from pymodbus.exceptions import ConnectionException

from raspberry_pi_congestion.relay import (
    BothChannelsOnError,
    RelayChannel,
    RelayCommunicationError,
    RelayController,
)


class _FakeCoilResult:
    """pymodbus 응답 객체(.isError(), .bits)를 흉내내는 더미."""

    def __init__(self, bits=None, error: bool = False):
        self.bits = bits
        self._error = error

    def isError(self) -> bool:
        return self._error


class FakeModbusClient:
    """RelayController가 사용하는 만큼만 구현한 fake Modbus TCP 클라이언트.

    Parameters
    ----------
    coil_states:
        코일 주소 -> 현재 ON/OFF 상태. "실기기의 실제 상태"를 흉내낸다.
    connect_ok:
        connect()가 성공(True)/실패(False)할지 여부.
    fail_at:
        몇 번째 read/write 호출에서 통신 실패(ConnectionException)를
        일으킬지 지정하는 1부터 시작하는 인덱스 집합. 예) {1} = 첫 호출만 실패.
    always_fail:
        True면 모든 read/write 호출이 실패한다 (기기에 아예 연결할 수 없는 상황).
    """

    def __init__(
        self,
        coil_states: dict[int, bool] | None = None,
        connect_ok: bool = True,
        fail_at: set[int] | None = None,
        always_fail: bool = False,
    ):
        self.coil_states = dict(coil_states or {})
        self.connect_ok = connect_ok
        self.fail_at = fail_at or set()
        self.always_fail = always_fail
        self.write_calls: list[tuple[int, bool]] = []
        self.read_calls: list[int] = []
        self.closed = False
        self._call_count = 0

    def connect(self) -> bool:
        return self.connect_ok

    def close(self) -> None:
        self.closed = True

    def _maybe_fail(self) -> None:
        self._call_count += 1
        if self.always_fail or self._call_count in self.fail_at:
            raise ConnectionException("simulated JK-MTCP-2 communication failure")

    def write_coil(self, address: int, value: bool, device_id: int = 1) -> _FakeCoilResult:
        self.write_calls.append((address, value))
        self._maybe_fail()
        self.coil_states[address] = value
        return _FakeCoilResult()

    def read_coils(self, address: int, count: int = 1, device_id: int = 1) -> _FakeCoilResult:
        self.read_calls.append(address)
        self._maybe_fail()
        bits = [self.coil_states.get(address + i, False) for i in range(count)]
        return _FakeCoilResult(bits=bits)


def make_controller(client: FakeModbusClient) -> RelayController:
    return RelayController(host="127.0.0.1", port=15020, client=client)


# === 채널 1/2 ON/OFF ===


def test_채널1을_켤_수_있다():
    # given: 두 채널 모두 꺼져 있는 상태
    client = FakeModbusClient(coil_states={0: False, 1: False})
    controller = make_controller(client)

    # when
    controller.turn_on(RelayChannel.CH1)

    # then
    assert controller.is_on(RelayChannel.CH1) is True
    assert (0, True) in client.write_calls


def test_채널1을_끌_수_있다():
    # given: CH1이 켜져 있는 상태
    client = FakeModbusClient(coil_states={0: True, 1: False})
    controller = make_controller(client)

    # when
    controller.turn_off(RelayChannel.CH1)

    # then
    assert controller.is_on(RelayChannel.CH1) is False
    assert (0, False) in client.write_calls


def test_채널2를_켤_수_있다():
    # given: 두 채널 모두 꺼져 있는 상태
    client = FakeModbusClient(coil_states={0: False, 1: False})
    controller = make_controller(client)

    # when
    controller.turn_on(RelayChannel.CH2)

    # then
    assert controller.is_on(RelayChannel.CH2) is True
    assert (1, True) in client.write_calls


def test_채널2를_끌_수_있다():
    # given: CH2가 켜져 있는 상태
    client = FakeModbusClient(coil_states={0: False, 1: True})
    controller = make_controller(client)

    # when
    controller.turn_off(RelayChannel.CH2)

    # then
    assert controller.is_on(RelayChannel.CH2) is False
    assert (1, False) in client.write_calls


# === 두 채널 동시 ON 방지 ===


def test_반대_채널이_켜져있으면_ON을_거부한다():
    # given: CH1이 실제로 켜져 있는 상태
    client = FakeModbusClient(coil_states={0: True, 1: False})
    controller = make_controller(client)

    # when / then: CH2를 켜려고 하면 예외가 발생하고
    with pytest.raises(BothChannelsOnError):
        controller.turn_on(RelayChannel.CH2)

    # 실제로 CH2에 ON 명령이 전송되지 않았어야 한다
    assert (1, True) not in client.write_calls
    assert controller.is_on(RelayChannel.CH1) is True
    # CH2는 애초에 조회/제어를 시도하지도 않았으므로 여전히 UNKNOWN이다
    assert controller.is_on(RelayChannel.CH2) is None


def test_반대_채널이_꺼져있으면_ON이_허용된다():
    # given: CH1이 꺼져 있는 상태
    client = FakeModbusClient(coil_states={0: False, 1: False})
    controller = make_controller(client)

    # when
    controller.turn_on(RelayChannel.CH2)

    # then
    assert controller.is_on(RelayChannel.CH2) is True


def test_switch_to는_반대_채널을_끄고_대상_채널을_켠다():
    # given: CH1이 켜져 있는 상태 (좌측 유도등 점등 중)
    client = FakeModbusClient(coil_states={0: True, 1: False})
    controller = make_controller(client)

    # when: 우측(CH2)으로 방향 전환
    controller.switch_to(RelayChannel.CH2)

    # then: CH1은 꺼지고 CH2만 켜진다 (두 채널이 동시에 ON인 순간이 API상 노출되지 않음)
    assert controller.is_on(RelayChannel.CH1) is False
    assert controller.is_on(RelayChannel.CH2) is True


# === both_on (평상시 양쪽 점등) ===


def test_both_on은_인터록_없이_두_채널을_모두_켠다():
    # given: 두 채널 모두 꺼져 있는 상태
    client = FakeModbusClient(coil_states={0: False, 1: False})
    controller = make_controller(client)

    # when
    controller.both_on()

    # then: turn_on()과 달리 반대 채널이 켜져 있어도 예외 없이 둘 다 켜진다
    assert controller.is_on(RelayChannel.CH1) is True
    assert controller.is_on(RelayChannel.CH2) is True


def test_both_on은_한_채널만_켜진_상태에서도_예외_없이_나머지를_켠다():
    # given: CH1만 켜져 있는 상태 (turn_on(CH2)였다면 BothChannelsOnError가 났을 상황)
    client = FakeModbusClient(coil_states={0: True, 1: False})
    controller = make_controller(client)

    # when / then: 예외 없이 CH2도 켜진다
    controller.both_on()
    assert controller.is_on(RelayChannel.CH1) is True
    assert controller.is_on(RelayChannel.CH2) is True


# === 통신 실패 시 OFF / fail-safe ===


def test_통신_실패_시_RelayCommunicationError를_던지고_unhealthy가_된다():
    # given: 모든 Modbus 호출이 실패하는 상황 (기기 응답 없음/타임아웃)
    client = FakeModbusClient(coil_states={0: False, 1: False}, always_fail=True)
    controller = make_controller(client)

    # when / then
    with pytest.raises(RelayCommunicationError):
        controller.turn_on(RelayChannel.CH1)

    assert controller.healthy is False


def test_완전히_연결할_수_없으면_상태를_UNKNOWN으로_유지하고_ON으로_보고하지_않는다():
    # given: fail-safe의 OFF 재시도조차 실패하는 최악의 상황
    client = FakeModbusClient(coil_states={0: True, 1: False}, always_fail=True)
    controller = make_controller(client)

    # when
    with pytest.raises(RelayCommunicationError):
        controller.refresh_status()

    # then: 실제 상태를 확인/보장할 수 없으므로 ON이라고 보고하지 않는다 (None = UNKNOWN)
    assert controller.is_on(RelayChannel.CH1) is None
    assert controller.is_on(RelayChannel.CH2) is None
    assert controller.healthy is False


def test_통신_실패_시_fail_safe가_실기기에_OFF를_재전송한다():
    # given: CH1이 켜져 있는 상태에서, 최초 명령 1회만 실패하고 그 이후(재시도)는 성공
    client = FakeModbusClient(coil_states={0: True, 1: False}, fail_at={1})
    controller = make_controller(client)

    # when: 통신 장애가 발생하는 시점에 CH1을 끄려고 시도
    with pytest.raises(RelayCommunicationError):
        controller.turn_off(RelayChannel.CH1)

    # then: 최초 명령은 실패했지만, fail-safe 재시도로 실기기의 두 채널이 모두 OFF된다
    assert client.coil_states[0] is False
    assert client.coil_states[1] is False
    # 컨트롤러 상태도 재시도 성공분만큼 OFF로 갱신된다
    assert controller.is_on(RelayChannel.CH1) is False
    assert controller.is_on(RelayChannel.CH2) is False
    # 다만 원래 명령 자체는 실패했으므로 unhealthy로 표시해 상위 계층이 알 수 있게 한다
    assert controller.healthy is False


def test_all_off는_두_채널_모두_끈다():
    # given: 두 채널 모두 켜져 있다고 가정할 수 없는 상황(둘 다 ON은 불가)이므로 CH1만 ON
    client = FakeModbusClient(coil_states={0: True, 1: False})
    controller = make_controller(client)

    # when
    controller.all_off()

    # then
    assert controller.is_on(RelayChannel.CH1) is False
    assert controller.is_on(RelayChannel.CH2) is False
    assert client.coil_states[0] is False
    assert client.coil_states[1] is False
