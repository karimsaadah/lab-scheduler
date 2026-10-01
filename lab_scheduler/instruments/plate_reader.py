"""
Plate reader instrument.

Models an optical plate reader (e.g., BioTek Synergy, Tecan Spark).
Measures optical properties of samples in a 96- or 384-well plate:
fluorescence intensity, absorbance, or luminescence.

Plate reader operations are fast (seconds to a few minutes) and synchronous —
they return measurement data directly. The primary data quality concern is
ensuring the plate is fully settled before reading (no bubbles, correct
temperature) and that the instrument is calibrated for the assay wavelengths.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Optional

from .base import Instrument, InstrumentResult, InstrumentStatus

logger = logging.getLogger(__name__)


class ReadMode(Enum):
    FLUORESCENCE = auto()
    ABSORBANCE = auto()
    LUMINESCENCE = auto()


@dataclass
class ReadSpec:
    """
    Parameters for a plate measurement.

    plate_id: identifier linking this measurement to an experiment record
    mode: measurement type (fluorescence, absorbance, luminescence)
    excitation_nm: excitation wavelength in nanometers (fluorescence only)
    emission_nm: emission wavelength in nanometers (fluorescence only)
    wavelength_nm: measurement wavelength (absorbance only)
    gain: detector gain — 'auto' lets the instrument optimize
    """
    plate_id: str
    mode: ReadMode
    excitation_nm: Optional[int] = None   # fluorescence
    emission_nm: Optional[int] = None     # fluorescence
    wavelength_nm: Optional[int] = None   # absorbance
    gain: str = "auto"

    def validate(self) -> Optional[str]:
        if not self.plate_id:
            return "plate_id is required"
        if self.mode == ReadMode.FLUORESCENCE:
            if self.excitation_nm is None or self.emission_nm is None:
                return "fluorescence mode requires excitation_nm and emission_nm"
            if not (200 <= self.excitation_nm <= 1000):
                return f"excitation_nm {self.excitation_nm} out of range"
            if not (200 <= self.emission_nm <= 1000):
                return f"emission_nm {self.emission_nm} out of range"
        elif self.mode == ReadMode.ABSORBANCE:
            if self.wavelength_nm is None:
                return "absorbance mode requires wavelength_nm"
            if not (200 <= self.wavelength_nm <= 1000):
                return f"wavelength_nm {self.wavelength_nm} out of range"
        return None


class PlateReader(Instrument):
    """
    Optical plate reader.

    Returns raw well-by-well measurement data as a dict keyed by well address
    (e.g., {"A1": 1234.5, "A2": 987.2, ...}). Downstream data pipeline components
    are responsible for normalization, blank subtraction, and statistical analysis.

    The reader does not interpret results — it only measures and reports.
    Keeping measurement and interpretation separate makes it straightforward to
    replay raw data with different analysis parameters.
    """

    def __init__(self, instrument_id: str, name: str, host: str, port: int = 5002):
        super().__init__(instrument_id, name)
        self.host = host
        self.port = port

    def connect(self) -> bool:
        self._status = InstrumentStatus.INITIALIZING
        self._logger.info("Connecting plate reader %s", self.name)
        try:
            time.sleep(0.1)
            self._status = InstrumentStatus.IDLE
            return True
        except Exception as exc:
            self._status = InstrumentStatus.ERROR
            self._logger.error("Plate reader connection failed: %s", exc)
            return False

    def disconnect(self) -> None:
        self._status = InstrumentStatus.DISCONNECTED

    def reset(self) -> bool:
        self.disconnect()
        return self.connect()

    def get_status(self) -> InstrumentStatus:
        return self._status

    def read_plate(self, spec: ReadSpec, well_count: int = 96) -> InstrumentResult:
        """
        Measure all wells of a plate according to the given spec.

        Returns measurement data in result.data["measurements"] as a dict
        mapping well addresses to raw signal values. Missing wells (e.g., empty
        positions) are represented as None rather than 0 to avoid ambiguity
        between "no signal" and "not measured."
        """
        started_at = self._start_result("read_plate")

        validation_error = spec.validate()
        if validation_error:
            return self._make_result(
                "read_plate", started_at, success=False,
                error=f"Invalid ReadSpec: {validation_error}"
            )

        if not self.is_available:
            return self._make_result(
                "read_plate", started_at, success=False,
                error=f"Plate reader not available (status={self._status.name})"
            )

        self._status = InstrumentStatus.BUSY
        try:
            self._logger.info(
                "Reading plate %s: mode=%s, gain=%s",
                spec.plate_id, spec.mode.name, spec.gain
            )
            time.sleep(0.1)  # Simulate read time

            # Production: send ReadSpec to instrument, parse returned well matrix
            # Simulated measurements for demonstration
            rows = "ABCDEFGH"
            cols = range(1, 13)
            measurements = {
                f"{r}{c}": round(1000 + hash(f"{spec.plate_id}{r}{c}") % 500, 2)
                for r in rows for c in cols
            }

            self._status = InstrumentStatus.IDLE
            return self._make_result(
                "read_plate", started_at, success=True,
                data={
                    "plate_id": spec.plate_id,
                    "mode": spec.mode.name,
                    "well_count": well_count,
                    "measurements": measurements,
                }
            )
        except Exception as exc:
            self._status = InstrumentStatus.ERROR
            return self._make_result(
                "read_plate", started_at, success=False,
                error=f"Read failed: {exc}"
            )
