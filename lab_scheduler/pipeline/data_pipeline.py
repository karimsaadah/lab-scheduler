"""
Data pipeline for processing and QC-checking instrument results.

After the scheduler executes a workflow, raw InstrumentResult objects pass
through this pipeline for normalization, quality control, and report generation.

Design rationale:
- Separating measurement from analysis preserves raw data fidelity. The plate
  reader returns numbers; this module decides what those numbers mean.
- Z-score normalization makes multi-plate comparisons valid across runs where
  signal intensity varies due to gain settings or reagent age.
- PipelineResult is a dataclass (not a dict) so that downstream code can rely
  on type annotations rather than string keys — a common source of bugs in
  data-heavy pipelines.
- QC thresholds are per-instance configuration, not module-level constants,
  making it straightforward to run the same pipeline with different stringency
  settings (e.g., tighter controls for clinical vs. research assays).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..instruments.base import InstrumentResult


@dataclass
class PipelineResult:
    """
    The output of processing a single InstrumentResult through the data pipeline.

    Attributes:
        processed_data: Normalized or transformed data, ready for analysis.
                        Structure mirrors the input data but with transformations applied.
        summary:        Human-readable summary of key statistics (mean, std, n).
        qc_passed:      True if the result meets configured quality criteria.
        qc_notes:       List of QC check descriptions and their outcomes.
        instrument_id:  Traceability back to the originating instrument.
        operation:      Which operation produced this result.
        processed_at:   UTC timestamp of pipeline processing.
        raw_result:     Reference to the original InstrumentResult for full traceability.
    """

    processed_data: dict[str, Any]
    summary: dict[str, Any]
    qc_passed: bool
    qc_notes: list[str] = field(default_factory=list)
    instrument_id: str = ""
    operation: str = ""
    processed_at: datetime = field(default_factory=datetime.utcnow)
    raw_result: InstrumentResult | None = field(default=None, repr=False)

    def __str__(self) -> str:
        status = "PASS" if self.qc_passed else "FAIL"
        return (
            f"PipelineResult [{self.instrument_id}/{self.operation}] QC={status} | {self.summary}"
        )


class DataPipeline:
    """
    Processes InstrumentResult objects into analysis-ready PipelineResults.

    Handles:
    - Plate reader data: z-score normalization and per-well QC
    - Thermocycler data: duration tracking, basic completion validation
    - Liquid handler data: transfer volume summary
    - Generic fallback for other instrument types

    Args:
        cv_threshold:      Maximum acceptable coefficient of variation (0-1) for
                           plate reads. Plates exceeding this are flagged as QC fail.
        min_signal:        Minimum acceptable mean signal for plate reads. Plates
                           with mean below this threshold are flagged (may indicate
                           reagent failure or empty plate).
        zscore_outlier_threshold: Wells with |z-score| above this are flagged as
                           outliers in QC notes.
    """

    def __init__(
        self,
        cv_threshold: float = 0.30,
        min_signal: float = 100.0,
        zscore_outlier_threshold: float = 3.0,
    ) -> None:
        self.cv_threshold = cv_threshold
        self.min_signal = min_signal
        self.zscore_outlier_threshold = zscore_outlier_threshold

    def process(self, result: InstrumentResult) -> PipelineResult:
        """
        Route an InstrumentResult to the appropriate processor.

        Args:
            result: Result from any instrument operation.

        Returns:
            PipelineResult with processed data, summary stats, and QC verdict.
        """
        if not result.success:
            return PipelineResult(
                processed_data={},
                summary={"error": result.error, "instrument_id": result.instrument_id},
                qc_passed=False,
                qc_notes=[f"Raw result was not successful: {result.error}"],
                instrument_id=result.instrument_id,
                operation=result.operation,
                raw_result=result,
            )

        # Dispatch by operation type
        if result.operation == "read_plate" and "measurements" in result.data:
            return self._process_plate_read(result)
        elif result.operation in ("run_program", "wait_for_completion"):
            return self._process_thermocycler(result)
        elif result.operation in ("transfer", "eject_tips"):
            return self._process_liquid_handler(result)
        else:
            return self._process_generic(result)

    def normalize_plate_data(self, measurements: dict[str, float | None]) -> dict[str, float]:
        """
        Apply z-score normalization to a plate's measurements.

        Z-score: (value - mean) / std_dev

        Wells with None values (not measured) are excluded from statistics and
        returned as None in the normalized output. Wells with constant signal
        (std_dev == 0) are normalized to 0.0.

        Args:
            measurements: Dict mapping well address to raw signal value.

        Returns:
            Dict mapping well address to z-score, with None for unmeasured wells.
        """
        values = [v for v in measurements.values() if v is not None]

        if len(values) < 2:
            # Cannot normalize fewer than 2 values; return unchanged
            return dict(measurements)

        mean = statistics.mean(values)
        std = statistics.stdev(values)

        if std == 0.0:
            return {well: 0.0 if val is not None else None for well, val in measurements.items()}

        return {
            well: round((val - mean) / std, 6) if val is not None else None
            for well, val in measurements.items()
        }

    def generate_report(self, workflow_id: str, results: list[PipelineResult]) -> dict[str, Any]:
        """
        Aggregate multiple PipelineResults from a workflow into a report dict.

        This is the document an experiment management system would store after
        a workflow completes: traceability, QC summary, and per-step outcomes.

        Args:
            workflow_id: The workflow whose results are being reported.
            results:     List of PipelineResult objects in execution order.

        Returns:
            Report dict suitable for JSON serialization and database storage.
        """
        all_passed = all(r.qc_passed for r in results)
        step_summaries = []

        for i, r in enumerate(results):
            step_summaries.append(
                {
                    "step_index": i,
                    "instrument_id": r.instrument_id,
                    "operation": r.operation,
                    "qc_passed": r.qc_passed,
                    "qc_notes": r.qc_notes,
                    "summary": r.summary,
                    "processed_at": r.processed_at.isoformat(),
                }
            )

        return {
            "report_version": "1.0",
            "workflow_id": workflow_id,
            "generated_at": datetime.utcnow().isoformat(),
            "overall_qc_passed": all_passed,
            "step_count": len(results),
            "steps": step_summaries,
        }

    # ------------------------------------------------------------------
    # Operation-specific processors
    # ------------------------------------------------------------------

    def _process_plate_read(self, result: InstrumentResult) -> PipelineResult:
        measurements: dict[str, float | None] = result.data.get("measurements", {})
        values = [v for v in measurements.values() if v is not None]

        if not values:
            return PipelineResult(
                processed_data={"measurements": {}, "normalized": {}},
                summary={"n": 0, "mean": None, "std": None, "cv": None},
                qc_passed=False,
                qc_notes=["No measurements found in plate read result"],
                instrument_id=result.instrument_id,
                operation=result.operation,
                raw_result=result,
            )

        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        cv = (std / mean) if mean != 0 else float("inf")

        normalized = self.normalize_plate_data(measurements)

        # QC checks
        qc_notes = []
        qc_passed = True

        if mean < self.min_signal:
            qc_notes.append(
                f"FAIL: mean signal {mean:.1f} below minimum threshold {self.min_signal}"
            )
            qc_passed = False
        else:
            qc_notes.append(f"PASS: mean signal {mean:.1f} >= {self.min_signal}")

        if cv > self.cv_threshold:
            qc_notes.append(
                f"FAIL: CV {cv:.3f} ({cv * 100:.1f}%) exceeds threshold {self.cv_threshold * 100:.1f}%"
            )
            qc_passed = False
        else:
            qc_notes.append(
                f"PASS: CV {cv:.3f} ({cv * 100:.1f}%) within threshold {self.cv_threshold * 100:.1f}%"
            )

        # Outlier detection
        if std > 0:
            outliers = [
                well
                for well, val in measurements.items()
                if val is not None and abs((val - mean) / std) > self.zscore_outlier_threshold
            ]
            if outliers:
                qc_notes.append(
                    f"WARNING: {len(outliers)} outlier well(s) detected: {outliers[:5]}"
                    + (" ..." if len(outliers) > 5 else "")
                )

        summary = {
            "plate_id": result.data.get("plate_id"),
            "mode": result.data.get("mode"),
            "n_wells": len(values),
            "mean": round(mean, 4),
            "std": round(std, 4),
            "cv": round(cv, 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }

        return PipelineResult(
            processed_data={
                "measurements": measurements,
                "normalized": normalized,
            },
            summary=summary,
            qc_passed=qc_passed,
            qc_notes=qc_notes,
            instrument_id=result.instrument_id,
            operation=result.operation,
            raw_result=result,
        )

    def _process_thermocycler(self, result: InstrumentResult) -> PipelineResult:
        elapsed = result.data.get("elapsed_seconds", result.duration_seconds)
        program_name = result.data.get("program", "unknown")

        summary = {
            "program": program_name,
            "duration_seconds": round(elapsed, 2),
            "duration_minutes": round(elapsed / 60, 2),
        }

        return PipelineResult(
            processed_data=result.data,
            summary=summary,
            qc_passed=True,
            qc_notes=["Thermocycler operation completed successfully"],
            instrument_id=result.instrument_id,
            operation=result.operation,
            raw_result=result,
        )

    def _process_liquid_handler(self, result: InstrumentResult) -> PipelineResult:
        volume = result.data.get("volume_ul")
        summary = {
            "operation": result.operation,
            "volume_ul": volume,
            "source": result.data.get("source"),
            "destination": result.data.get("destination"),
            "duration_seconds": round(result.duration_seconds, 3),
        }

        return PipelineResult(
            processed_data=result.data,
            summary=summary,
            qc_passed=True,
            qc_notes=["Liquid handler operation completed successfully"],
            instrument_id=result.instrument_id,
            operation=result.operation,
            raw_result=result,
        )

    def _process_generic(self, result: InstrumentResult) -> PipelineResult:
        summary = {
            "instrument_id": result.instrument_id,
            "operation": result.operation,
            "duration_seconds": round(result.duration_seconds, 3),
        }

        return PipelineResult(
            processed_data=result.data,
            summary=summary,
            qc_passed=True,
            qc_notes=["No specialized processor for this operation; raw data passed through"],
            instrument_id=result.instrument_id,
            operation=result.operation,
            raw_result=result,
        )
