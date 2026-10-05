"""
Base instrument abstraction.

Every physical instrument in the lab — liquid handler, thermocycler, plate reader,
incubator — implements this interface. The scheduler works exclusively through this
abstraction; it never knows which vendor or model it is talking to.

Design rationale:
- Status is always explicit and queryable. The scheduler never assumes an instrument
  is ready; it checks. This prevents silent failures where an instrument appears
  available but is actually mid-recovery from a prior error.
- Operations are synchronous by default but return structured results that the
  scheduler can log, retry, or escalate. Callers should not catch raw exceptions
  from instrument methods — use InstrumentResult and check .success.
- Instruments own their own connection lifecycle. The scheduler calls connect()
  before use and disconnect() when done or on error; it does not manage sockets
  or serial ports directly.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class InstrumentStatus(Enum):
    """Observable states of a laboratory instrument."""
    DISCONNECTED = auto()   # Not connected; no commands can be sent
    IDLE = auto()           # Connected and ready to accept a command
    BUSY = auto()           # Currently executing an operation
    ERROR = auto()          # In an error state; requires intervention or reset
    INITIALIZING = auto()   # Connecting or running startup self-check


class InstrumentError(Exception):
    """
    Raised when an instrument operation fails in a way that the scheduler
    cannot automatically recover from. Includes the instrument id and
    the last known status to help incident analysis.
    """
    def __init__(self, message: str, instrument_id: str, status: InstrumentStatus):
        super().__init__(message)
        self.instrument_id = instrument_id
        self.status = status


@dataclass
class InstrumentResult:
    """
    Structured result returned by every instrument operation.

    The scheduler logs every result regardless of success/failure.
    On failure, `error` contains a human-readable description suitable
    for both automated analysis and incident reports.
    """
    instrument_id: str
    operation: str
    success: bool
    started_at: datetime
    finished_at: datetime
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def duration_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()

    def __str__(self) -> str:
        status = "OK" if self.success else f"FAILED: {self.error}"
        return (
            f"[{self.instrument_id}] {self.operation} "
            f"({self.duration_seconds:.2f}s) -> {status}"
        )


class Instrument(ABC):
    """
    Abstract base class for all laboratory instruments.

    Subclasses implement the vendor-specific communication layer.
    The scheduler and AI planning layer work through this interface only.
    """

    def __init__(self, instrument_id: str, name: str):
        self.instrument_id = instrument_id
        self.name = name
        self._status = InstrumentStatus.DISCONNECTED
        self._logger = logging.getLogger(f"{__name__}.{instrument_id}")

    @property
    def status(self) -> InstrumentStatus:
        return self._status

    @property
    def is_available(self) -> bool:
        """True only when connected and idle — the only state where commands are safe."""
        return self._status == InstrumentStatus.IDLE

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection and run instrument self-check.
        Returns True if the instrument is ready for operation.
        Sets status to IDLE on success, ERROR on failure.
        """
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """
        Gracefully release the connection.
        Safe to call in any state, including ERROR.
        """
        ...

    @abstractmethod
    def reset(self) -> bool:
        """
        Attempt to recover from ERROR state.
        Returns True if the instrument returned to IDLE.
        The scheduler calls this before giving up and escalating.
        """
        ...

    @abstractmethod
    def get_status(self) -> InstrumentStatus:
        """
        Query the instrument directly for its current state.
        More reliable than reading self._status in long-running operations.
        """
        ...

    def _start_result(self, operation: str) -> datetime:
        """Record start time for a result. Call at the top of each operation."""
        self._logger.debug("Starting operation: %s", operation)
        return datetime.utcnow()

    def _make_result(
        self,
        operation: str,
        started_at: datetime,
        success: bool,
        data: dict | None = None,
        error: str | None = None,
    ) -> InstrumentResult:
        result = InstrumentResult(
            instrument_id=self.instrument_id,
            operation=operation,
            success=success,
            started_at=started_at,
            finished_at=datetime.utcnow(),
            data=data or {},
            error=error,
        )
        if success:
            self._logger.info("%s", result)
        else:
            self._logger.error("%s", result)
        return result
