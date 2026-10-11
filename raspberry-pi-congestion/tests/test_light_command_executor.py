"""LightCommandExecutor(폴링 -> 릴레이 실행 -> ACK 보고) 단위 테스트.

실기기 없이도 돌아가도록 fake device client와 FakeModbusClient를 주입해서
테스트한다.
"""

from __future__ import annotations

from pymodbus.exceptions import ConnectionException

from raspberry_pi_congestion.relay import LightCommandExecutor, RelayChannel, RelayController


class _FakeCoilResult:
    def __init__(self, bits=None, error: bool = False):
        self.bits = bits
        self._error = error

    def isError(self) -> bool:
        return self._error


class FakeModbusClient:
    """test_relay_controller.py와 동일한 최소 fake Modbus TCP 클라이언트."""

    def __init__(self, coil_states=None, always_fail: bool = False):
        self.coil_states = dict(coil_states or {})
        self.always_fail = always_fail
        self.write_calls: list[tuple[int, bool]] = []

    def connect(self) -> bool:
        return True

    def close(self) -> None:
        pass

    def write_coil(self, address: int, value: bool, device_id: int = 1) -> _FakeCoilResult:
        self.write_calls.append((address, value))
        if self.always_fail:
            raise ConnectionException("simulated relay failure")
        self.coil_states[address] = value
        return _FakeCoilResult()

    def read_coils(self, address: int, count: int = 1, device_id: int = 1) -> _FakeCoilResult:
        if self.always_fail:
            raise ConnectionException("simulated relay failure")
        bits = [self.coil_states.get(address + i, False) for i in range(count)]
        return _FakeCoilResult(bits=bits)


class FakeDeviceClient:
    """SafeRouteDeviceClient의 유도등 명령 관련 인터페이스만 흉내낸 fake."""

    def __init__(self, commands: list[dict]):
        self._commands = commands
        self.acks: list[tuple[str, bool, str | None]] = []

    def fetch_light_commands(self, cctv_code: str) -> list[dict]:
        return self._commands

    def ack_light_command(self, command_id: str, success: bool, fail_reason: str | None = None) -> bool:
        self.acks.append((command_id, success, fail_reason))
        return True


def make_relay(coil_states=None, always_fail: bool = False) -> RelayController:
    client = FakeModbusClient(coil_states=coil_states or {0: False, 1: False}, always_fail=always_fail)
    return RelayController(host="127.0.0.1", port=15020, client=client)


def test_LEFT_명령을_받으면_CH1으로_전환하고_성공_ACK를_보낸다():
    relay = make_relay()
    client = FakeDeviceClient([{"commandId": "cmd-1", "lightCode": "LIGHT_001", "direction": "LEFT"}])
    executor = LightCommandExecutor(client, relay, "CCTV_001")

    processed = executor.poll_once()

    assert processed == 1
    assert relay.is_on(RelayChannel.CH1) is True
    assert relay.is_on(RelayChannel.CH2) is False
    assert client.acks == [("cmd-1", True, None)]


def test_RIGHT_명령을_받으면_CH2로_전환한다():
    relay = make_relay()
    client = FakeDeviceClient([{"commandId": "cmd-1", "lightCode": "LIGHT_001", "direction": "RIGHT"}])
    executor = LightCommandExecutor(client, relay, "CCTV_001")

    executor.poll_once()

    assert relay.is_on(RelayChannel.CH2) is True


def test_OFF_명령을_받으면_두_채널_모두_끈다():
    relay = make_relay(coil_states={0: True, 1: False})
    client = FakeDeviceClient([{"commandId": "cmd-1", "lightCode": "LIGHT_001", "direction": "OFF"}])
    executor = LightCommandExecutor(client, relay, "CCTV_001")

    executor.poll_once()

    assert relay.is_on(RelayChannel.CH1) is False
    assert relay.is_on(RelayChannel.CH2) is False


def test_BOTH_명령을_받으면_인터록_없이_두_채널을_모두_켠다():
    relay = make_relay(coil_states={0: True, 1: False})
    client = FakeDeviceClient([{"commandId": "cmd-1", "lightCode": "LIGHT_001", "direction": "BOTH"}])
    executor = LightCommandExecutor(client, relay, "CCTV_001")

    executor.poll_once()

    assert relay.is_on(RelayChannel.CH1) is True
    assert relay.is_on(RelayChannel.CH2) is True
    assert client.acks == [("cmd-1", True, None)]


def test_알수없는_방향이면_실패_ACK를_보내고_다른_명령_처리를_막지_않는다():
    relay = make_relay()
    client = FakeDeviceClient([
        {"commandId": "cmd-1", "lightCode": "LIGHT_001", "direction": "UNKNOWN"},
        {"commandId": "cmd-2", "lightCode": "LIGHT_002", "direction": "LEFT"},
    ])
    executor = LightCommandExecutor(client, relay, "CCTV_001")

    processed = executor.poll_once()

    assert processed == 2
    assert client.acks[0] == ("cmd-1", False, "알 수 없는 유도등 방향: UNKNOWN")
    assert client.acks[1] == ("cmd-2", True, None)


def test_릴레이_통신_실패_시_실패_ACK를_보낸다():
    relay = make_relay(always_fail=True)
    client = FakeDeviceClient([{"commandId": "cmd-1", "lightCode": "LIGHT_001", "direction": "LEFT"}])
    executor = LightCommandExecutor(client, relay, "CCTV_001")

    executor.poll_once()

    assert client.acks[0][0] == "cmd-1"
    assert client.acks[0][1] is False


def test_명령이_없으면_아무_것도_실행하지_않는다():
    relay = make_relay()
    client = FakeDeviceClient([])
    executor = LightCommandExecutor(client, relay, "CCTV_001")

    processed = executor.poll_once()

    assert processed == 0
    assert client.acks == []
