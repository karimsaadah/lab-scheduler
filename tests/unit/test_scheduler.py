"""
Unit tests for LabScheduler.

Uses unittest.mock to replace instrument operations with controlled fakes,
allowing test scenarios that would take minutes (real PCR runs) to complete
in milliseconds, and failure scenarios that would require physical instruments.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from lab_scheduler.instruments.base import (
    Instrument,
    InstrumentResult,
    InstrumentStatus,
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
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_result(
    instrument_id: str, operation: str, success: bool, error: str | None = None
) -> InstrumentResult:
    now = datetime.utcnow()
    return InstrumentResult(
        instrument_id=instrument_id,
        operation=operation,
        success=success,
        started_at=now,
        finished_at=now,
        data={},
        error=error,
    )


def _make_instrument(
    instrument_id: str = "inst_1", status: InstrumentStatus = InstrumentStatus.IDLE
) -> MagicMock:
    """Create a mock Instrument that behaves like a connected, idle instrument."""
    inst = MagicMock(spec=Instrument)
    inst.instrument_id = instrument_id
    inst.name = f"Mock {instrument_id}"
    inst.status = status
    inst.get_status.return_value = status
    inst.is_available = status == InstrumentStatus.IDLE
    return inst


def _make_workflow(
    instrument_id: str = "inst_1",
    operation=None,
    name: str = "test_wf",
    retry_count: int = 0,
) -> Workflow:
    if operation is None:

        def operation(inst):
            return _make_result(instrument_id, "op", True)

    step = WorkflowStep(
        instrument_id=instrument_id,
        name="mock_step",
        operation=operation,
        retry_count=retry_count,
    )
    return Workflow(steps=[step], name=name, submitted_by="pytest")


@pytest.fixture
def scheduler() -> LabScheduler:
    log = EventLog()
    return LabScheduler(max_workers=4, event_log=log)


@pytest.fixture
def idle_instrument() -> MagicMock:
    return _make_instrument("inst_1")


# ---------------------------------------------------------------------------
# register_instrument
# ---------------------------------------------------------------------------


class TestRegisterInstrument:
    def test_register_idle_instrument_succeeds(self, scheduler, idle_instrument):
        scheduler.register_instrument(idle_instrument)
        assert scheduler.get_instrument("inst_1") is idle_instrument

    def test_register_non_idle_raises(self, scheduler):
        inst = _make_instrument("inst_err", InstrumentStatus.ERROR)
        with pytest.raises(ValueError, match="expected IDLE"):
            scheduler.register_instrument(inst)

    def test_register_disconnected_raises(self, scheduler):
        inst = _make_instrument("inst_disc", InstrumentStatus.DISCONNECTED)
        with pytest.raises(ValueError, match="expected IDLE"):
            scheduler.register_instrument(inst)

    def test_register_busy_raises(self, scheduler):
        inst = _make_instrument("inst_busy", InstrumentStatus.BUSY)
        with pytest.raises(ValueError, match="expected IDLE"):
            scheduler.register_instrument(inst)

    def test_register_emits_event(self, scheduler, idle_instrument):
        scheduler.register_instrument(idle_instrument)
        events = scheduler._event_log.get_events(EventType.INSTRUMENT_REGISTERED)
        assert len(events) == 1
        assert events[0]["payload"]["instrument_id"] == "inst_1"

    def test_get_unknown_instrument_raises(self, scheduler):
        with pytest.raises(KeyError, match="not registered"):
            scheduler.get_instrument("ghost_instrument")


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------


class TestSubmit:
    def test_submit_returns_workflow_id(self, scheduler, idle_instrument):
        scheduler.register_instrument(idle_instrument)
        wf = _make_workflow("inst_1")
        wf_id = scheduler.submit(wf)
        assert wf_id == wf.workflow_id

    def test_submit_stores_workflow(self, scheduler, idle_instrument):
        scheduler.register_instrument(idle_instrument)
        wf = _make_workflow("inst_1")
        scheduler.submit(wf)
        assert scheduler.get_workflow(wf.workflow_id) is wf

    def test_submit_missing_instrument_raises(self, scheduler):
        wf = _make_workflow("unregistered_instrument")
        with pytest.raises(ValueError, match="unregistered instruments"):
            scheduler.submit(wf)

    def test_submit_emits_workflow_submitted_event(self, scheduler, idle_instrument):
        scheduler.register_instrument(idle_instrument)
        wf = _make_workflow("inst_1")
        scheduler.submit(wf)
        events = scheduler._event_log.get_events(EventType.WORKFLOW_SUBMITTED)
        assert len(events) == 1

    def test_submit_priority_sort(self, scheduler, idle_instrument):
        """Verify higher-priority workflows sort to front of queue."""
        scheduler.register_instrument(idle_instrument)
        wf_low = _make_workflow("inst_1", name="low_priority")
        wf_low.priority = 1
        wf_high = _make_workflow("inst_1", name="high_priority")
        wf_high.priority = 10

        scheduler.submit(wf_low)
        scheduler.submit(wf_high)

        # Queue is sorted descending by priority
        with scheduler._queue_lock:
            assert scheduler._queue[0].priority >= scheduler._queue[1].priority


# ---------------------------------------------------------------------------
# Full workflow execution (mock instruments)
# ---------------------------------------------------------------------------


class TestWorkflowExecution:
    def test_successful_single_step_workflow(self, scheduler, idle_instrument):
        """A workflow with one step that succeeds should complete."""
        scheduler.register_instrument(idle_instrument)
        op = MagicMock(return_value=_make_result("inst_1", "op", True))
        wf = _make_workflow("inst_1", operation=op)
        scheduler.submit(wf)
        scheduler.run(poll_interval_seconds=0.05)

        assert wf.status == WorkflowStatus.COMPLETED
        assert wf.steps[0].status == StepStatus.COMPLETED
        op.assert_called_once_with(idle_instrument)

    def test_failed_step_marks_workflow_failed(self, scheduler, idle_instrument):
        scheduler.register_instrument(idle_instrument)
        failing_op = MagicMock(
            return_value=_make_result("inst_1", "op", False, error="instrument error")
        )
        wf = _make_workflow("inst_1", operation=failing_op)
        scheduler.submit(wf)
        scheduler.run(poll_interval_seconds=0.05)

        assert wf.status == WorkflowStatus.FAILED
        assert wf.steps[0].status == StepStatus.FAILED

    def test_two_steps_different_instruments_both_complete(self, scheduler):
        """Steps on different instruments can run concurrently and both complete."""
        inst_a = _make_instrument("inst_a")
        inst_b = _make_instrument("inst_b")
        scheduler.register_instrument(inst_a)
        scheduler.register_instrument(inst_b)

        op_a = MagicMock(return_value=_make_result("inst_a", "op_a", True))
        op_b = MagicMock(return_value=_make_result("inst_b", "op_b", True))

        step_a = WorkflowStep(instrument_id="inst_a", name="step_a", operation=op_a)
        step_b = WorkflowStep(instrument_id="inst_b", name="step_b", operation=op_b)
        wf = Workflow(steps=[step_a, step_b], name="two_instrument_wf")
        scheduler.submit(wf)
        scheduler.run(poll_interval_seconds=0.05)

        assert wf.status == WorkflowStatus.COMPLETED
        assert step_a.status == StepStatus.COMPLETED
        assert step_b.status == StepStatus.COMPLETED

    def test_dependent_step_runs_after_dependency(self, scheduler, idle_instrument):
        """Step B with depends_on=[0] must not run before step A completes."""
        scheduler.register_instrument(idle_instrument)

        execution_order = []

        def op_a(inst):
            execution_order.append("a")
            return _make_result("inst_1", "op_a", True)

        def op_b(inst):
            execution_order.append("b")
            return _make_result("inst_1", "op_b", True)

        step_a = WorkflowStep(instrument_id="inst_1", name="step_a", operation=op_a)
        step_b = WorkflowStep(instrument_id="inst_1", name="step_b", operation=op_b, depends_on=[0])
        wf = Workflow(steps=[step_a, step_b], name="dep_wf")
        scheduler.submit(wf)
        scheduler.run(poll_interval_seconds=0.02)

        assert wf.status == WorkflowStatus.COMPLETED
        assert execution_order == ["a", "b"], f"Steps ran out of order: {execution_order}"

    def test_workflow_emits_started_and_completed_events(self, scheduler, idle_instrument):
        scheduler.register_instrument(idle_instrument)
        op = MagicMock(return_value=_make_result("inst_1", "op", True))
        wf = _make_workflow("inst_1", operation=op)
        scheduler.submit(wf)
        scheduler.run(poll_interval_seconds=0.05)

        log = scheduler._event_log
        assert len(log.get_events(EventType.WORKFLOW_STARTED)) == 1
        assert len(log.get_events(EventType.WORKFLOW_COMPLETED)) == 1
        assert len(log.get_events(EventType.WORKFLOW_FAILED)) == 0

    def test_failed_workflow_emits_failed_event(self, scheduler, idle_instrument):
        scheduler.register_instrument(idle_instrument)
        failing_op = MagicMock(return_value=_make_result("inst_1", "op", False, "oops"))
        wf = _make_workflow("inst_1", operation=failing_op)
        scheduler.submit(wf)
        scheduler.run(poll_interval_seconds=0.05)

        log = scheduler._event_log
        assert len(log.get_events(EventType.WORKFLOW_FAILED)) == 1
        assert len(log.get_events(EventType.WORKFLOW_COMPLETED)) == 0


# ---------------------------------------------------------------------------
# Retry behavior
# ---------------------------------------------------------------------------


class TestRetryBehavior:
    def test_step_retries_on_failure_then_succeeds(self, scheduler, idle_instrument):
        """An operation that fails twice then succeeds should complete with retry_count=2."""
        scheduler.register_instrument(idle_instrument)

        call_count = {"n": 0}

        def flaky_op(inst):
            call_count["n"] += 1
            if call_count["n"] < 3:
                return _make_result("inst_1", "op", False, "transient error")
            return _make_result("inst_1", "op", True)

        step = WorkflowStep(
            instrument_id="inst_1",
            name="flaky_step",
            operation=flaky_op,
            retry_count=2,
        )
        wf = Workflow(steps=[step], name="retry_wf")
        scheduler.submit(wf)

        # Override retry sleep to speed up test
        with patch("lab_scheduler.scheduler.scheduler.time.sleep"):
            scheduler.run(poll_interval_seconds=0.05)

        assert wf.status == WorkflowStatus.COMPLETED
        assert step.status == StepStatus.COMPLETED
        assert step.attempts == 3  # 2 failures + 1 success

    def test_step_fails_after_exhausting_retries(self, scheduler, idle_instrument):
        """An operation that always fails should exhaust retries and fail the workflow."""
        scheduler.register_instrument(idle_instrument)

        always_fail = MagicMock(return_value=_make_result("inst_1", "op", False, "permanent error"))
        step = WorkflowStep(
            instrument_id="inst_1",
            name="doomed_step",
            operation=always_fail,
            retry_count=2,
        )
        wf = Workflow(steps=[step], name="doomed_wf")
        scheduler.submit(wf)

        with patch("lab_scheduler.scheduler.scheduler.time.sleep"):
            scheduler.run(poll_interval_seconds=0.05)

        assert wf.status == WorkflowStatus.FAILED
        assert step.status == StepStatus.FAILED
        assert step.attempts == 3  # 1 initial + 2 retries
        assert always_fail.call_count == 3

    def test_zero_retry_fails_immediately(self, scheduler, idle_instrument):
        """A step with retry_count=0 should fail after exactly one attempt."""
        scheduler.register_instrument(idle_instrument)

        failing_op = MagicMock(return_value=_make_result("inst_1", "op", False, "error"))
        step = WorkflowStep(
            instrument_id="inst_1",
            name="no_retry_step",
            operation=failing_op,
            retry_count=0,
        )
        wf = Workflow(steps=[step], name="no_retry_wf")
        scheduler.submit(wf)
        scheduler.run(poll_interval_seconds=0.05)

        assert step.status == StepStatus.FAILED
        assert step.attempts == 1
        failing_op.assert_called_once()

    def test_exception_in_operation_treated_as_failure(self, scheduler, idle_instrument):
        """An operation that raises an exception should be retried or marked failed."""
        scheduler.register_instrument(idle_instrument)

        def raising_op(inst):
            raise RuntimeError("unexpected hardware error")

        step = WorkflowStep(
            instrument_id="inst_1",
            name="explosive_step",
            operation=raising_op,
            retry_count=1,
        )
        wf = Workflow(steps=[step], name="exception_wf")
        scheduler.submit(wf)

        with patch("lab_scheduler.scheduler.scheduler.time.sleep"):
            scheduler.run(poll_interval_seconds=0.05)

        assert wf.status == WorkflowStatus.FAILED
        assert step.status == StepStatus.FAILED


# ---------------------------------------------------------------------------
# instrument_status_report
# ---------------------------------------------------------------------------


class TestStatusReport:
    def test_status_report_includes_all_instruments(self, scheduler):
        inst_a = _make_instrument("inst_a")
        inst_b = _make_instrument("inst_b")
        scheduler.register_instrument(inst_a)
        scheduler.register_instrument(inst_b)

        report = scheduler.instrument_status_report()
        assert "inst_a" in report
        assert "inst_b" in report
        assert report["inst_a"] == "IDLE"
