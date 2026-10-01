"""
Integration tests for lab_scheduler using real (simulated) instrument objects.

These tests do NOT mock instrument operations. They use the actual instrument
classes (LiquidHandler, PlateReader, Thermocycler) with their simulated
implementations, exercising the full code path:

    Workflow definition
    → LabScheduler.submit()
    → LabScheduler.run()
    → Instrument.connect()
    → step.operation(instrument)
    → InstrumentResult returned
    → StepStatus / WorkflowStatus transitions
    → EventLog entries

This validates that the scheduler, workflow, and instrument layers integrate
correctly end-to-end — something unit tests with mocked instruments cannot verify.
"""

from __future__ import annotations

import pytest

from lab_scheduler.instruments.liquid_handler import LiquidHandler, TransferSpec
from lab_scheduler.instruments.plate_reader import PlateReader, ReadMode, ReadSpec
from lab_scheduler.instruments.thermocycler import (
    Thermocycler,
    ThermalProgram,
    ThermalStep,
    STANDARD_PROGRAMS,
)
from lab_scheduler.observability.event_log import EventLog, EventType
from lab_scheduler.scheduler.scheduler import LabScheduler
from lab_scheduler.scheduler.workflow import (
    StepStatus,
    Workflow,
    WorkflowStatus,
    WorkflowStep,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def liquid_handler() -> LiquidHandler:
    lh = LiquidHandler("lh_01", "BioTek Liquid Handler", host="localhost")
    lh.connect()
    return lh


@pytest.fixture
def plate_reader() -> PlateReader:
    pr = PlateReader("pr_01", "Synergy Neo2", host="localhost")
    pr.connect()
    return pr


@pytest.fixture
def thermocycler() -> Thermocycler:
    tc = Thermocycler("tc_01", "Bio-Rad C1000", host="localhost")
    tc.connect()
    return tc


@pytest.fixture
def event_log() -> EventLog:
    return EventLog()


@pytest.fixture
def scheduler(event_log) -> LabScheduler:
    return LabScheduler(max_workers=4, event_log=event_log)


# ---------------------------------------------------------------------------
# Integration test: LiquidHandler + PlateReader (2-step workflow)
# ---------------------------------------------------------------------------

class TestTransferThenRead:
    """
    Validates a 2-step workflow:
      Step 0: Transfer 50 µL from reservoir to A1 (LiquidHandler)
      Step 1: Read fluorescence from plate (PlateReader) — depends on step 0
    """

    def test_both_steps_complete(self, scheduler, liquid_handler, plate_reader):
        scheduler.register_instrument(liquid_handler)
        scheduler.register_instrument(plate_reader)

        transfer_spec = TransferSpec(
            volume_ul=50.0,
            source="reservoir_1",
            destination="A1",
            tip_index=0,
            mix_cycles=2,
        )
        read_spec = ReadSpec(
            plate_id="integration_test_plate",
            mode=ReadMode.FLUORESCENCE,
            excitation_nm=485,
            emission_nm=528,
        )

        step_transfer = WorkflowStep(
            instrument_id="lh_01",
            name="transfer_samples",
            operation=lambda inst, spec=transfer_spec: inst.transfer(spec),
        )
        step_read = WorkflowStep(
            instrument_id="pr_01",
            name="read_plate",
            operation=lambda inst, spec=read_spec: inst.read_plate(spec),
            depends_on=[0],
        )

        wf = Workflow(
            steps=[step_transfer, step_read],
            name="integration_transfer_then_read",
            submitted_by="integration_test",
        )

        scheduler.submit(wf)
        scheduler.run(poll_interval_seconds=0.05)

        # Both steps must complete successfully
        assert wf.status == WorkflowStatus.COMPLETED
        assert step_transfer.status == StepStatus.COMPLETED
        assert step_read.status == StepStatus.COMPLETED

    def test_measurements_are_non_empty(self, scheduler, liquid_handler, plate_reader):
        """The plate read result must contain well measurements."""
        scheduler.register_instrument(liquid_handler)
        scheduler.register_instrument(plate_reader)

        transfer_spec = TransferSpec(
            volume_ul=100.0,
            source="sample_trough",
            destination="A1",
            tip_index=0,
        )
        read_spec = ReadSpec(
            plate_id="qc_plate_001",
            mode=ReadMode.ABSORBANCE,
            wavelength_nm=600,
        )

        step_transfer = WorkflowStep(
            instrument_id="lh_01",
            name="load_plate",
            operation=lambda inst, s=transfer_spec: inst.transfer(s),
        )
        step_read = WorkflowStep(
            instrument_id="pr_01",
            name="absorbance_read",
            operation=lambda inst, s=read_spec: inst.read_plate(s),
            depends_on=[0],
        )

        wf = Workflow(
            steps=[step_transfer, step_read],
            name="absorbance_workflow",
            submitted_by="integration_test",
        )

        scheduler.submit(wf)
        scheduler.run(poll_interval_seconds=0.05)

        assert wf.status == WorkflowStatus.COMPLETED

        # Verify measurements are populated
        read_result = step_read.result
        assert read_result is not None
        assert read_result.success is True
        measurements = read_result.data.get("measurements", {})
        assert len(measurements) > 0, "Expected non-empty measurements dict from plate reader"
        # Spot-check that values are numeric
        sample_value = next(iter(measurements.values()))
        assert isinstance(sample_value, (int, float))

    def test_event_log_captures_all_transitions(self, scheduler, liquid_handler, plate_reader):
        """Integration test that all expected events are recorded in order."""
        scheduler.register_instrument(liquid_handler)
        scheduler.register_instrument(plate_reader)

        step_a = WorkflowStep(
            instrument_id="lh_01",
            name="transfer",
            operation=lambda inst: inst.transfer(
                TransferSpec(volume_ul=50, source="R1", destination="A1", tip_index=0)
            ),
        )
        step_b = WorkflowStep(
            instrument_id="pr_01",
            name="read",
            operation=lambda inst: inst.read_plate(
                ReadSpec(plate_id="ev_test", mode=ReadMode.FLUORESCENCE,
                         excitation_nm=485, emission_nm=528)
            ),
            depends_on=[0],
        )

        wf = Workflow(steps=[step_a, step_b], name="event_log_test")
        scheduler.submit(wf)
        scheduler.run(poll_interval_seconds=0.05)

        # Use the scheduler's own event log (same instance used throughout)
        log = scheduler._event_log

        # Check event types emitted
        assert len(log.get_events(EventType.INSTRUMENT_REGISTERED)) == 2
        assert len(log.get_events(EventType.WORKFLOW_SUBMITTED)) == 1
        assert len(log.get_events(EventType.WORKFLOW_STARTED)) == 1
        assert len(log.get_events(EventType.WORKFLOW_COMPLETED)) == 1
        assert len(log.get_events(EventType.STEP_STARTED)) == 2
        assert len(log.get_events(EventType.STEP_COMPLETED)) == 2
        assert len(log.get_events(EventType.STEP_FAILED)) == 0
        assert len(log.get_events(EventType.SCHEDULER_STARTED)) == 1
        assert len(log.get_events(EventType.SCHEDULER_STOPPED)) == 1


# ---------------------------------------------------------------------------
# Integration test: Thermocycler quick protocol
# ---------------------------------------------------------------------------

class TestThermocyclerWorkflow:
    def test_short_pcr_completes(self, scheduler, thermocycler):
        """
        Run a very short thermal program (1 step, 1 cycle, minimal hold time)
        to verify the thermocycler + scheduler integration without a long wait.
        """
        scheduler.register_instrument(thermocycler)

        # 1-step, 1-cycle program with minimal hold to keep test fast
        fast_program = ThermalProgram(
            name="fast_test",
            steps=[ThermalStep(temperature_c=72.0, hold_seconds=1, label="brief_hold")],
            cycles=1,
        )

        step_run = WorkflowStep(
            instrument_id="tc_01",
            name="start_program",
            operation=lambda inst, p=fast_program: inst.run_program(p),
        )
        step_wait = WorkflowStep(
            instrument_id="tc_01",
            name="wait_complete",
            operation=lambda inst: inst.wait_for_completion(
                poll_interval_seconds=0.01,
                timeout_seconds=30.0,
            ),
            depends_on=[0],
        )

        wf = Workflow(
            steps=[step_run, step_wait],
            name="fast_pcr_integration",
            submitted_by="integration_test",
        )

        scheduler.submit(wf)
        scheduler.run(poll_interval_seconds=0.05)

        assert wf.status == WorkflowStatus.COMPLETED
        assert step_run.status == StepStatus.COMPLETED
        assert step_wait.status == StepStatus.COMPLETED


# ---------------------------------------------------------------------------
# Integration test: concurrent independent workflows
# ---------------------------------------------------------------------------

class TestConcurrentWorkflows:
    def test_two_workflows_different_instruments_both_complete(
        self, scheduler, liquid_handler, plate_reader
    ):
        """
        Two workflows targeting different instruments can execute concurrently.
        Both should complete independently.
        """
        scheduler.register_instrument(liquid_handler)
        scheduler.register_instrument(plate_reader)

        # Workflow 1: liquid handler
        step_lh = WorkflowStep(
            instrument_id="lh_01",
            name="transfer",
            operation=lambda inst: inst.transfer(
                TransferSpec(volume_ul=25.0, source="R1", destination="B1", tip_index=0)
            ),
        )
        wf1 = Workflow(steps=[step_lh], name="wf_liquid_handler", submitted_by="test")

        # Workflow 2: plate reader
        step_pr = WorkflowStep(
            instrument_id="pr_01",
            name="read",
            operation=lambda inst: inst.read_plate(
                ReadSpec(plate_id="concurrent_test", mode=ReadMode.LUMINESCENCE)
            ),
        )
        wf2 = Workflow(steps=[step_pr], name="wf_plate_reader", submitted_by="test")

        scheduler.submit(wf1)
        scheduler.submit(wf2)
        scheduler.run(poll_interval_seconds=0.05)

        assert wf1.status == WorkflowStatus.COMPLETED
        assert wf2.status == WorkflowStatus.COMPLETED
