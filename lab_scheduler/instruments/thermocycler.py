"""
Thermocycler instrument.

Models a PCR thermocycler (e.g., Bio-Rad C1000, Applied Biosystems ProFlex).
Executes temperature cycling programs on loaded plates. Unlike the liquid handler,
thermocycler operations are long-running (minutes to hours) and asynchronous —
the instrument runs independently once a program starts.

Key scheduling implication: the scheduler must track in-progress thermocycler
runs and not allocate the instrument to another workflow until run_status()
reports completion. Polling interval is configurable to balance responsiveness
against unnecessary network traffic.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

from .base import Instrument, InstrumentResult, InstrumentStatus

logger = logging.getLogger(__name__)


@dataclass
class ThermalStep:
    """A single step in a thermal cycling program."""

    temperature_c: float  # Target temperature in Celsius
    hold_seconds: int  # How long to hold at this temperature
    label: str = ""  # Human-readable label (e.g., "denaturation", "annealing")

    def validate(self) -> str | None:
        if not (-10 <= self.temperature_c <= 110):
            return f"temperature {self.temperature_c}°C out of range (-10 to 110)"
        if self.hold_seconds <= 0:
            return f"hold_seconds must be positive, got {self.hold_seconds}"
        return None


@dataclass
class ThermalProgram:
    """
    A complete thermal cycling protocol.

    cycles: number of times to repeat the steps list (1 = run once, no cycling)
    steps: ordered list of thermal steps executed each cycle
    lid_temperature_c: heated lid temperature to prevent condensation (typically 105°C)
    """

    name: str
    steps: list[ThermalStep]
    cycles: int = 1
    lid_temperature_c: float = 105.0

    def validate(self) -> str | None:
        if not self.steps:
            return "program must have at least one step"
        if self.cycles < 1:
            return f"cycles must be >= 1, got {self.cycles}"
        for i, step in enumerate(self.steps):
            err = step.validate()
            if err:
                return f"step {i} ({step.label or 'unlabeled'}): {err}"
        return None

    @property
    def estimated_duration_seconds(self) -> int:
        """Rough estimate: sum of all step hold times * cycles. Ignores ramp time."""
        return sum(s.hold_seconds for s in self.steps) * self.cycles


# Standard protocols used in synthetic biology workflows
STANDARD_PROGRAMS = {
    "pcr_standard": ThermalProgram(
        name="pcr_standard",
        steps=[
            ThermalStep(95.0, 180, "initial_denaturation"),
            ThermalStep(95.0, 30, "denaturation"),
            ThermalStep(60.0, 30, "annealing"),
            ThermalStep(72.0, 60, "extension"),
            ThermalStep(72.0, 300, "final_extension"),
            ThermalStep(4.0, 0, "hold"),  # hold_seconds=0 means hold indefinitely
        ],
        cycles=30,
    ),
    "colony_pcr": ThermalProgram(
        name="colony_pcr",
        steps=[
            ThermalStep(95.0, 300, "initial_denaturation"),
            ThermalStep(95.0, 30, "denaturation"),
            ThermalStep(55.0, 30, "annealing"),
            ThermalStep(72.0, 90, "extension"),
            ThermalStep(72.0, 300, "final_extension"),
            ThermalStep(4.0, 0, "hold"),
        ],
        cycles=25,
    ),
}


class Thermocycler(Instrument):
    """
    PCR thermocycler.

    Long-running operation model:
        run_program() starts the instrument and returns immediately.
        The scheduler polls get_run_status() until the run completes or errors.
        wait_for_completion() is provided as a convenience for simple single-instrument
        workflows; concurrent multi-instrument workflows should poll manually.
    """

    def __init__(self, instrument_id: str, name: str, host: str, port: int = 5001):
        super().__init__(instrument_id, name)
        self.host = host
        self.port = port
        self._current_program: ThermalProgram | None = None
        self._run_started_at: datetime | None = None

    def connect(self) -> bool:
        self._status = InstrumentStatus.INITIALIZING
        self._logger.info("Connecting thermocycler %s at %s:%d", self.name, self.host, self.port)
        try:
            time.sleep(0.1)
            self._status = InstrumentStatus.IDLE
            return True
        except Exception as exc:
            self._status = InstrumentStatus.ERROR
            self._logger.error("Thermocycler connection failed: %s", exc)
            return False

    def disconnect(self) -> None:
        self._connection = None
        self._status = InstrumentStatus.DISCONNECTED

    def reset(self) -> bool:
        self._logger.warning("Resetting thermocycler %s", self.name)
        self._current_program = None
        self._run_started_at = None
        self.disconnect()
        return self.connect()

    def get_status(self) -> InstrumentStatus:
        return self._status

    def run_program(self, program: ThermalProgram) -> InstrumentResult:
        """
        Start a thermal program. Returns immediately; instrument runs asynchronously.
        Check get_run_status() or call wait_for_completion() to track progress.
        """
        started_at = self._start_result("run_program")

        validation_error = program.validate()
        if validation_error:
            return self._make_result(
                "run_program",
                started_at,
                success=False,
                error=f"Invalid ThermalProgram: {validation_error}",
            )

        if not self.is_available:
            return self._make_result(
                "run_program",
                started_at,
                success=False,
                error=f"Thermocycler not available (status={self._status.name})",
            )

        self._status = InstrumentStatus.BUSY
        self._current_program = program
        self._run_started_at = datetime.utcnow()

        self._logger.info(
            "Starting program '%s': %d cycles, ~%ds estimated",
            program.name,
            program.cycles,
            program.estimated_duration_seconds,
        )

        # Production: send program definition to instrument, await ACK
        return self._make_result(
            "run_program",
            started_at,
            success=True,
            data={
                "program": program.name,
                "cycles": program.cycles,
                "estimated_duration_seconds": program.estimated_duration_seconds,
            },
        )

    def get_run_status(self) -> dict:
        """
        Query current run state. Called by the scheduler polling loop.
        Returns a dict with 'running', 'complete', 'error', and 'elapsed_seconds'.
        """
        if self._status == InstrumentStatus.BUSY and self._run_started_at:
            elapsed = (datetime.utcnow() - self._run_started_at).total_seconds()
            estimated = (
                self._current_program.estimated_duration_seconds if self._current_program else 0
            )
            # Production: query instrument API for actual step/cycle position
            complete = elapsed >= estimated  # Simulation only
            if complete:
                self._status = InstrumentStatus.IDLE
                self._current_program = None
            return {
                "running": not complete,
                "complete": complete,
                "error": False,
                "elapsed_seconds": elapsed,
                "program": self._current_program.name if self._current_program else None,
            }
        return {
            "running": False,
            "complete": self._status == InstrumentStatus.IDLE,
            "error": self._status == InstrumentStatus.ERROR,
            "elapsed_seconds": 0,
            "program": None,
        }

    def wait_for_completion(
        self, poll_interval_seconds: float = 5.0, timeout_seconds: float = 7200.0
    ) -> InstrumentResult:
        """
        Block until the current run finishes, errors, or times out.

        timeout_seconds defaults to 2 hours — long enough for most PCR programs,
        short enough to avoid hanging indefinitely on a lost connection.
        """
        started_at = self._start_result("wait_for_completion")
        deadline = datetime.utcnow() + timedelta(seconds=timeout_seconds)

        while datetime.utcnow() < deadline:
            status = self.get_run_status()
            if status["error"]:
                return self._make_result(
                    "wait_for_completion",
                    started_at,
                    success=False,
                    error="Thermocycler reported error during run",
                )
            if status["complete"]:
                return self._make_result(
                    "wait_for_completion",
                    started_at,
                    success=True,
                    data={"elapsed_seconds": status["elapsed_seconds"]},
                )
            time.sleep(poll_interval_seconds)

        self._status = InstrumentStatus.ERROR
        return self._make_result(
            "wait_for_completion",
            started_at,
            success=False,
            error=f"Run timed out after {timeout_seconds}s",
        )
