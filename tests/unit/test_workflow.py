"""
Unit tests for Workflow and WorkflowStep.

Tests the data model in isolation — no scheduler, no instruments, no I/O.
"""

from __future__ import annotations

import pytest
from datetime import datetime

from lab_scheduler.scheduler.workflow import (
    Workflow,
    WorkflowStep,
    WorkflowStatus,
    StepStatus,
)
from lab_scheduler.instruments.base import InstrumentResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_step(
    instrument_id: str = "inst_1",
    name: str = "test_step",
    depends_on: list[int] | None = None,
    retry_count: int = 0,
) -> WorkflowStep:
    """Create a WorkflowStep with a no-op operation for testing."""
    return WorkflowStep(
        instrument_id=instrument_id,
        name=name,
        operation=lambda inst: None,
        depends_on=depends_on or [],
        retry_count=retry_count,
    )


def _make_workflow(steps: list[WorkflowStep] | None = None, priority: int = 0) -> Workflow:
    return Workflow(
        steps=steps or [_make_step()],
        name="test_workflow",
        submitted_by="pytest",
        priority=priority,
    )


# ---------------------------------------------------------------------------
# get_ready_steps
# ---------------------------------------------------------------------------

class TestGetReadySteps:
    def test_all_pending_no_deps_are_ready(self):
        """Steps with no dependencies and PENDING status are all immediately ready."""
        steps = [_make_step(name="a"), _make_step(name="b"), _make_step(name="c")]
        wf = _make_workflow(steps)

        ready = wf.get_ready_steps()
        assert len(ready) == 3
        assert {s.name for _, s in ready} == {"a", "b", "c"}

    def test_empty_workflow_has_no_ready_steps(self):
        wf = Workflow(steps=[], name="empty")
        assert wf.get_ready_steps() == []

    def test_deps_satisfied_step_is_ready(self):
        """A step whose dependency is COMPLETED should appear in get_ready_steps."""
        step_a = _make_step(name="a")
        step_b = _make_step(name="b", depends_on=[0])
        wf = _make_workflow([step_a, step_b])

        # Satisfy step_a
        step_a.status = StepStatus.COMPLETED

        ready = wf.get_ready_steps()
        assert len(ready) == 1
        assert ready[0][1].name == "b"

    def test_deps_not_satisfied_step_is_not_ready(self):
        """A step whose dependency is still PENDING must not appear in get_ready_steps."""
        step_a = _make_step(name="a")
        step_b = _make_step(name="b", depends_on=[0])
        wf = _make_workflow([step_a, step_b])

        # step_a remains PENDING
        ready = wf.get_ready_steps()
        # Only step_a (no deps) is ready; step_b is blocked
        ready_names = {s.name for _, s in ready}
        assert "a" in ready_names
        assert "b" not in ready_names

    def test_running_step_not_returned(self):
        """Steps already in RUNNING status must not be re-dispatched."""
        step = _make_step(name="a")
        step.status = StepStatus.RUNNING
        wf = _make_workflow([step])

        assert wf.get_ready_steps() == []

    def test_completed_step_not_returned(self):
        step = _make_step(name="a")
        step.status = StepStatus.COMPLETED
        wf = _make_workflow([step])

        assert wf.get_ready_steps() == []

    def test_failed_dep_does_not_unblock_step(self):
        """A failed dependency means the dependent step stays blocked (will be SKIPPED)."""
        step_a = _make_step(name="a")
        step_b = _make_step(name="b", depends_on=[0])
        wf = _make_workflow([step_a, step_b])

        step_a.status = StepStatus.FAILED

        ready = wf.get_ready_steps()
        ready_names = {s.name for _, s in ready}
        assert "b" not in ready_names

    def test_multi_level_deps(self):
        """A -> B -> C: only A is ready initially; B becomes ready after A completes."""
        step_a = _make_step(name="a")
        step_b = _make_step(name="b", depends_on=[0])
        step_c = _make_step(name="c", depends_on=[1])
        wf = _make_workflow([step_a, step_b, step_c])

        # Initially only A
        ready = wf.get_ready_steps()
        assert {s.name for _, s in ready} == {"a"}

        # After A completes, B becomes ready
        step_a.status = StepStatus.COMPLETED
        ready = wf.get_ready_steps()
        assert {s.name for _, s in ready} == {"b"}

        # After B completes, C becomes ready
        step_b.status = StepStatus.COMPLETED
        ready = wf.get_ready_steps()
        assert {s.name for _, s in ready} == {"c"}


# ---------------------------------------------------------------------------
# is_terminal
# ---------------------------------------------------------------------------

class TestIsTerminal:
    @pytest.mark.parametrize("status,expected", [
        (WorkflowStatus.PENDING, False),
        (WorkflowStatus.RUNNING, False),
        (WorkflowStatus.COMPLETED, True),
        (WorkflowStatus.FAILED, True),
        (WorkflowStatus.CANCELLED, True),
    ])
    def test_terminal_states(self, status, expected):
        wf = _make_workflow()
        wf.status = status
        assert wf.is_terminal() == expected


# ---------------------------------------------------------------------------
# get_required_instruments
# ---------------------------------------------------------------------------

class TestGetRequiredInstruments:
    def test_single_instrument(self):
        wf = _make_workflow([_make_step(instrument_id="lh_01")])
        assert wf.get_required_instruments() == {"lh_01"}

    def test_multiple_same_instrument(self):
        steps = [
            _make_step(instrument_id="lh_01", name="step1"),
            _make_step(instrument_id="lh_01", name="step2"),
        ]
        wf = _make_workflow(steps)
        # Returns a set — duplicates collapsed
        assert wf.get_required_instruments() == {"lh_01"}

    def test_multiple_distinct_instruments(self):
        steps = [
            _make_step(instrument_id="lh_01", name="transfer"),
            _make_step(instrument_id="tc_01", name="pcr"),
            _make_step(instrument_id="pr_01", name="read"),
        ]
        wf = _make_workflow(steps)
        assert wf.get_required_instruments() == {"lh_01", "tc_01", "pr_01"}

    def test_empty_workflow(self):
        wf = Workflow(steps=[], name="empty")
        assert wf.get_required_instruments() == set()


# ---------------------------------------------------------------------------
# Priority sorting
# ---------------------------------------------------------------------------

class TestPrioritySorting:
    def test_priority_field_is_stored(self):
        wf_low = _make_workflow(priority=0)
        wf_high = _make_workflow(priority=10)
        assert wf_low.priority == 0
        assert wf_high.priority == 10

    def test_priority_sort_order(self):
        """Higher priority workflows should sort first (reverse=True behavior)."""
        workflows = [
            _make_workflow(priority=1),
            _make_workflow(priority=10),
            _make_workflow(priority=5),
        ]
        sorted_workflows = sorted(workflows, key=lambda w: w.priority, reverse=True)
        priorities = [w.priority for w in sorted_workflows]
        assert priorities == [10, 5, 1]


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------

class TestWorkflowSummary:
    def test_summary_pending_workflow(self):
        wf = _make_workflow([_make_step(name="a"), _make_step(name="b")])
        s = wf.summary()

        assert s["status"] == "PENDING"
        assert s["steps_total"] == 2
        assert s["steps_completed"] == 0
        assert s["steps_failed"] == 0
        assert s["name"] == "test_workflow"
        assert s["submitted_by"] == "pytest"
        assert "workflow_id" in s
        assert "experiment_id" in s

    def test_summary_partial_completion(self):
        step_a = _make_step(name="a")
        step_b = _make_step(name="b")
        step_c = _make_step(name="c")
        wf = _make_workflow([step_a, step_b, step_c])

        step_a.status = StepStatus.COMPLETED
        step_b.status = StepStatus.FAILED
        # step_c remains PENDING
        wf.status = WorkflowStatus.FAILED

        s = wf.summary()
        assert s["status"] == "FAILED"
        assert s["steps_total"] == 3
        assert s["steps_completed"] == 1
        assert s["steps_failed"] == 1

    def test_summary_completed_workflow(self):
        steps = [_make_step(name=f"step_{i}") for i in range(4)]
        wf = _make_workflow(steps)
        for step in steps:
            step.status = StepStatus.COMPLETED
        wf.status = WorkflowStatus.COMPLETED

        s = wf.summary()
        assert s["status"] == "COMPLETED"
        assert s["steps_completed"] == 4
        assert s["steps_failed"] == 0


# ---------------------------------------------------------------------------
# WorkflowStep fields
# ---------------------------------------------------------------------------

class TestWorkflowStepFields:
    def test_default_status_is_pending(self):
        step = _make_step()
        assert step.status == StepStatus.PENDING

    def test_retry_count_stored(self):
        step = _make_step(retry_count=3)
        assert step.retry_count == 3

    def test_attempts_starts_at_zero(self):
        step = _make_step()
        assert step.attempts == 0

    def test_result_starts_as_none(self):
        step = _make_step()
        assert step.result is None
