"""
Liquid handler instrument — the workhorse of lab automation.

Models a motorized pipetting workstation (e.g., Tecan Fluent, Beckman Biomek).
Responsible for moving precise volumes of liquid between wells on a deck.

In a real integration, connect() would open a socket or serial connection to the
instrument controller, and each operation would serialize a vendor-specific command
(FluentControl script, SILA2 call, etc.) and parse the response. The structure here
— explicit state transitions, per-operation results, clean error surface — stays
identical regardless of which vendor API sits underneath.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .base import Instrument, InstrumentError, InstrumentResult, InstrumentStatus

logger = logging.getLogger(__name__)


@dataclass
class TransferSpec:
    """
    Defines a single liquid transfer operation.

    volume_ul: volume in microliters
    source: well address, e.g. "A1", "B3", or a named reservoir
    destination: well address on the target plate
    tip_index: which tip position to use (prevents cross-contamination)
    mix_cycles: number of aspirate/dispense cycles after transfer (0 = no mixing)
    """
    volume_ul: float
    source: str
    destination: str
    tip_index: int
    mix_cycles: int = 0

    def validate(self) -> Optional[str]:
        """Returns an error string if the spec is invalid, None if OK."""
        if self.volume_ul <= 0:
            return f"volume_ul must be positive, got {self.volume_ul}"
        if self.volume_ul > 1000:
            return f"volume_ul {self.volume_ul} exceeds max tip capacity (1000 µL)"
        if not self.source:
            return "source well must be specified"
        if not self.destination:
            return "destination well must be specified"
        if self.tip_index < 0:
            return f"tip_index must be non-negative, got {self.tip_index}"
        return None


class LiquidHandler(Instrument):
    """
    Liquid handling workstation.

    Key failure modes this class defends against:
    - Tip pickup failure (tip not seated, tip rack empty)
    - Volume out of range for loaded tip type
    - Deck position conflict (two operations targeting same plate simultaneously)
    - Connection loss mid-transfer (partial transfer — worst case for data integrity)

    On any failure, status transitions to ERROR and the operation result captures
    the failure point. The scheduler is responsible for deciding whether to retry,
    abort the workflow, or escalate to a human.
    """

    def __init__(self, instrument_id: str, name: str, host: str, port: int = 5000):
        super().__init__(instrument_id, name)
        self.host = host
        self.port = port
        self._connection = None  # Vendor SDK connection object in real implementation
        self._tip_count: dict[int, int] = {}  # tip_index -> remaining tips

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        self._status = InstrumentStatus.INITIALIZING
        self._logger.info("Connecting to %s at %s:%d", self.name, self.host, self.port)
        try:
            # In production: self._connection = VendorSDK.connect(self.host, self.port)
            # Followed by a self-check: self._connection.run_diagnostics()
            time.sleep(0.1)  # Simulate connection latency
            self._status = InstrumentStatus.IDLE
            self._logger.info("Connected and ready: %s", self.name)
            return True
        except Exception as exc:
            self._status = InstrumentStatus.ERROR
            self._logger.error("Connection failed: %s", exc)
            return False

    def disconnect(self) -> None:
        self._logger.info("Disconnecting %s", self.name)
        try:
            if self._connection is not None:
                pass  # self._connection.close()
        finally:
            self._connection = None
            self._status = InstrumentStatus.DISCONNECTED

    def reset(self) -> bool:
        self._logger.warning("Resetting %s after error", self.name)
        self.disconnect()
        return self.connect()

    def get_status(self) -> InstrumentStatus:
        # In production: query instrument directly rather than returning cached state
        return self._status

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def transfer(self, spec: TransferSpec) -> InstrumentResult:
        """
        Execute a single liquid transfer.

        Validates the spec before sending any commands — catching bad parameters
        early avoids partial operations that leave the deck in an ambiguous state.
        """
        started_at = self._start_result("transfer")

        # Pre-flight validation
        validation_error = spec.validate()
        if validation_error:
            return self._make_result(
                "transfer", started_at, success=False,
                error=f"Invalid TransferSpec: {validation_error}"
            )

        if not self.is_available:
            return self._make_result(
                "transfer", started_at, success=False,
                error=f"Instrument not available (status={self._status.name})"
            )

        self._status = InstrumentStatus.BUSY
        try:
            self._logger.info(
                "Transferring %.1f µL: %s -> %s (tip %d, mix %d)",
                spec.volume_ul, spec.source, spec.destination,
                spec.tip_index, spec.mix_cycles
            )
            # Production: send command to vendor API, await acknowledgment
            time.sleep(0.05)  # Simulate operation time

            self._status = InstrumentStatus.IDLE
            return self._make_result(
                "transfer", started_at, success=True,
                data={
                    "volume_ul": spec.volume_ul,
                    "source": spec.source,
                    "destination": spec.destination,
                    "tip_index": spec.tip_index,
                    "mix_cycles": spec.mix_cycles,
                }
            )
        except Exception as exc:
            self._status = InstrumentStatus.ERROR
            return self._make_result(
                "transfer", started_at, success=False,
                error=f"Transfer failed at instrument: {exc}"
            )

    def transfer_batch(self, specs: list[TransferSpec]) -> list[InstrumentResult]:
        """
        Execute a sequence of transfers, stopping on first failure.

        Stopping on failure (rather than continuing) is a deliberate choice:
        subsequent transfers may depend on earlier ones, and a partial batch
        with some failed steps produces data that is difficult to interpret.
        The caller receives the full result list and can see exactly where the
        sequence broke.
        """
        results = []
        for i, spec in enumerate(specs):
            result = self.transfer(spec)
            results.append(result)
            if not result.success:
                self._logger.error(
                    "Batch transfer aborted at step %d/%d: %s",
                    i + 1, len(specs), result.error
                )
                break
        return results

    def eject_tips(self) -> InstrumentResult:
        """Eject all loaded tips. Always called between samples to prevent cross-contamination."""
        started_at = self._start_result("eject_tips")
        if not self.is_available:
            return self._make_result(
                "eject_tips", started_at, success=False,
                error=f"Cannot eject tips: status={self._status.name}"
            )
        self._status = InstrumentStatus.BUSY
        try:
            time.sleep(0.02)
            self._status = InstrumentStatus.IDLE
            return self._make_result("eject_tips", started_at, success=True)
        except Exception as exc:
            self._status = InstrumentStatus.ERROR
            return self._make_result(
                "eject_tips", started_at, success=False,
                error=str(exc)
            )
