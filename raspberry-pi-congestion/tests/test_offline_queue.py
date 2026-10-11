from raspberry_pi_congestion.offline_queue import OfflineQueue


def test_event_id_survives_restart(tmp_path):
    path = str(tmp_path / "queue.db")
    payload = {
        "eventId": "same-event",
        "trainingSessionId": "session-1",
        "windowStart": 1,
        "windowEnd": 2,
    }
    first = OfflineQueue(path)
    first.enqueue("same-event", payload)
    first.close()
    second = OfflineQueue(path)
    item = second.peek_oldest()[0]
    assert item.event_id == "same-event" and item.payload == payload
    assert item.training_session_id == "session-1"
    second.close()


def test_capacity_is_bounded(tmp_path):
    queue = OfflineQueue(str(tmp_path / "queue.db"), max_items=2)
    for i in range(3):
        queue.enqueue(str(i), {"eventId": str(i)})
    assert [item.event_id for item in queue.peek_oldest()] == ["1", "2"]
    queue.close()


def test_only_active_session_items_are_kept_and_selected(tmp_path):
    queue = OfflineQueue(str(tmp_path / "queue.db"))
    queue.enqueue("old", {"eventId": "old", "trainingSessionId": "old-session"})
    queue.enqueue("current", {"eventId": "current", "trainingSessionId": "current-session"})
    queue.enqueue("unknown", {"eventId": "unknown"})

    assert queue.discard_except_session("current-session") == 2
    items = queue.peek_oldest(training_session_id="current-session")

    assert [item.event_id for item in items] == ["current"]
    queue.close()


def test_wrapped_event_payload_provides_session_id(tmp_path):
    queue = OfflineQueue(str(tmp_path / "queue.db"))
    queue.enqueue(
        "event",
        {"eventPayload": {"eventId": "event", "trainingSessionId": "session-1"}},
        "event",
    )

    assert queue.peek_oldest()[0].training_session_id == "session-1"
    queue.close()
