"""
Structured event log for lab scheduler observability.

Every state transition in the scheduler — instrument registration, workflow
submission, step execution, failures — is recorded here as a typed event with
a UTC timestamp and a free-form payload dict.

Design rationale:
- Structured events (not plain log strings) are machine-readable: a dashboard,
  alert rule, or post-experiment analysis script can filter by EventType without
  parsing log strings.
- Thread-safe by default. The scheduler dispatches steps concurrently and all
  threads write to the same log without external coordination.
- In-memory for simplicity; in production this would be backed by a time-series
  database (InfluxDB, TimescaleDB) or an event streaming system (Kafka, Kinesis).
  The interface stays identical — swap EventLog for a subclass that writes to the
  backing store.
"""

from __future__ import annotations

import threading
from datetime import datetime
from enum import Enum, auto
from typing import Any


class EventType(Enum):
    """
    Enumeration of all observable events in the lab scheduler lifecycle.

    Using an enum rather than free-form strings enforces consistency across
    callers and makes it impossible to accidentally emit a typo'd event type
    that silently goes undetected until an analysis script breaks.
    """
    INSTRUMENT_REGISTERED = auto()   # A new instrument was added to the scheduler
    SCHEDULER_STARTED = auto()       # Scheduler.run() began processing the queue
    SCHEDULER_STOPPED = auto()       # Scheduler.run() exited (normally or via exception)
    WORKFLOW_SUBMITTED = auto()      # A workflow was accepted and queued
    WORKFLOW_STARTED = auto()        # The scheduler began executing a workflow
    WORKFLOW_COMPLETED = auto()      # All steps finished successfully
    WORKFLOW_FAILED = auto()         # A step failed and the workflow was aborted
    STEP_STARTED = auto()            # A single workflow step began execution
    STEP_COMPLETED = auto()          # A step finished successfully
    STEP_FAILED = auto()             # A step exhausted its retries and was marked failed


class EventLog:
    """
    Thread-safe, in-memory log of typed scheduler events.

    Each event is stored as a dict:
        {
            "timestamp": datetime (UTC),
            "event_type": EventType,
            "payload": dict,          # event-specific context
        }

    Usage:
        log = EventLog()
        log.record(EventType.WORKFLOW_SUBMITTED, {"workflow_id": "abc", ...})
        events = log.get_events(EventType.STEP_FAILED)
    """

    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def record(self, event_type: EventType, payload: dict[str, Any]) -> None:
        """
        Append a new event to the log.

        This is the only write path. All callers — scheduler, instruments, agents —
        go through here. Thread-safe: multiple executor threads can call concurrently.

        Args:
            event_type: The category of event being recorded.
            payload:    Arbitrary key-value context relevant to this event type.
                        Values must be JSON-serializable for downstream consumers.
        """
        event = {
            "timestamp": datetime.utcnow(),
            "event_type": event_type,
            "payload": payload,
        }
        with self._lock:
            self._events.append(event)

    def get_events(self, event_type: EventType | None = None) -> list[dict[str, Any]]:
        """
        Return recorded events, optionally filtered to a single EventType.

        Returns a snapshot copy — callers can iterate freely without holding a lock.
        Events are returned in insertion order (chronological).

        Args:
            event_type: If provided, return only events of this type.
                        If None, return all events.

        Returns:
            List of event dicts, each with "timestamp", "event_type", "payload".
        """
        with self._lock:
            events = list(self._events)

        if event_type is not None:
            events = [e for e in events if e["event_type"] == event_type]

        return events

    def clear(self) -> None:
        """
        Remove all recorded events.

        Primarily useful in tests that want a clean slate between cases.
        Not recommended for production use — prefer a separate EventLog instance.
        """
        with self._lock:
            self._events.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    def __repr__(self) -> str:
        with self._lock:
            count = len(self._events)
        return f"EventLog({count} events)"
