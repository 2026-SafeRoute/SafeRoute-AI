from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class QueuedObservation:
    id: int
    event_id: str
    payload: dict
    created_at: float
    attempts: int
    operation: str = "observation"
    training_session_id: Optional[str] = None


class OfflineQueue:
    def __init__(self, db_path: str, max_age_sec: float = 86400, max_items: int = 1000) -> None:
        self.max_age_sec = max_age_sec
        self.max_items = max_items
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS observations (id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL UNIQUE, payload TEXT NOT NULL, created_at REAL NOT NULL, attempts INTEGER NOT NULL DEFAULT 0)")
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(observations)")}
        if "operation" not in columns:
            self._conn.execute("ALTER TABLE observations ADD COLUMN operation TEXT NOT NULL DEFAULT 'observation'")
        if "training_session_id" not in columns:
            self._conn.execute("ALTER TABLE observations ADD COLUMN training_session_id TEXT")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_observations_created ON observations(created_at)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_observations_session ON observations(training_session_id)")
        self._conn.commit()

    def enqueue(self, event_id: str, payload: dict, operation: str = "observation",
                training_session_id: Optional[str] = None) -> None:
        with self._lock:
            session_id = training_session_id or self._extract_session_id(payload)
            self._conn.execute(
                "INSERT OR IGNORE INTO observations(event_id,payload,created_at,operation,training_session_id) VALUES(?,?,?,?,?)",
                (event_id, json.dumps(payload), time.time(), operation, session_id),
            )
            self._conn.commit()
            overflow = self.size() - self.max_items
            if overflow > 0:
                self._conn.execute("DELETE FROM observations WHERE id IN (SELECT id FROM observations ORDER BY created_at LIMIT ?)", (overflow,))
                self._conn.commit()

    def peek_oldest(self, limit: int = 10,
                    training_session_id: Optional[str] = None) -> list[QueuedObservation]:
        with self._lock:
            cutoff = time.time() - self.max_age_sec
            self._conn.execute("DELETE FROM observations WHERE created_at < ?", (cutoff,))
            self._conn.commit()
            if training_session_id is None:
                rows = self._conn.execute(
                    "SELECT id,event_id,payload,created_at,attempts,operation,training_session_id "
                    "FROM observations ORDER BY created_at LIMIT ?", (limit,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT id,event_id,payload,created_at,attempts,operation,training_session_id "
                    "FROM observations WHERE training_session_id=? ORDER BY created_at LIMIT ?",
                    (training_session_id, limit),
                ).fetchall()
        return [QueuedObservation(r[0], r[1], json.loads(r[2]), r[3], r[4], r[5], r[6]) for r in rows]

    def discard_except_session(self, training_session_id: str) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM observations WHERE training_session_id IS NULL OR training_session_id<>?",
                (training_session_id,),
            )
            self._conn.commit()
            return max(0, cursor.rowcount)

    def clear(self) -> int:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM observations")
            self._conn.commit()
            return max(0, cursor.rowcount)

    def mark_success(self, item_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM observations WHERE id=?", (item_id,))
            self._conn.commit()

    def mark_failed_attempt(self, item_id: int) -> None:
        with self._lock:
            self._conn.execute("UPDATE observations SET attempts=attempts+1 WHERE id=?", (item_id,))
            self._conn.commit()

    def size(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def _extract_session_id(payload: dict) -> Optional[str]:
        session_id = payload.get("trainingSessionId")
        if session_id is None and isinstance(payload.get("eventPayload"), dict):
            session_id = payload["eventPayload"].get("trainingSessionId")
        return str(session_id) if session_id else None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
