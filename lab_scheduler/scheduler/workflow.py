"""
Workflow and WorkflowStep — the data model the scheduler executes.

A Workflow is the unit of work submitted by an AI planning agent or a human researcher.
It describes WHAT needs to happen (which instrument, which operation, with what parameters)
without knowing HOW the scheduler will sequence it against other concurrent workflows.

Design principles:
- Workflows are immutable once submitted. The scheduler does not modify them.
- Each step declares its instrument dependency explicitly. The scheduler uses this
  to detect conflicts and sequence operations safely.
- Steps can declare dependencies on prior steps within the same workflow, enabling
  the scheduler to enforce ordering (e.g., "don't read the plate until the thermocycler
  run that started in step 2 is complete").
- Workflows are serializable (dataclasses + enums) so they can be persisted,
  replayed, and audited. Every submitted workflow should be logged before execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Optional
import uuid


class WorkflowStatus(Enum):
    PENDING = auto()      # Submitted, not yet started
    RUNNING = auto()      # At least one step has started
    COMPLETED = auto()    # All steps finished successfully
    FAILED = auto()       # A step failed and the workflow was aborted
    CANCELLED = auto()    # Cancelled before completion


class StepStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()      # Skipped because a prior step failed


@dataclass
class WorkflowStep:
    """
    A single operation within a workflow.

    instrument_id: which instrument this step requires (must be registered with the scheduler)
    operation: callable that takes the instrument and returns an InstrumentResult
    name: human-readable label for logging and observability
    depends_on: list of step indices within this workflow that must complete before this step runs
    retry_count: how many times the scheduler should retry this step on failure (0 = no retry)
    timeout_seconds: if the step takes longer than this, mark it failed
    """
    instrument_id: str
    operation: Callable
    name: str
    depends_on: list[int] = field(default_factory=list)
    retry_count: int = 0
    timeout_seconds: Optional[float] = None

    # Set by scheduler during execution — not part of the submitted spec
    status: StepStatus = field(default=StepStatus.PENDING, init=False)
    result: Any = field(default=None, init=False)
    started_at: Optional[datetime] = field(default=None, init=False)
    finished_at: Optional[datetime] = field(default=None, init=False)
    attempts: int = field(default=0, init=False)


@dataclass
class Workflow:
    """
    A complete experimental workflow submitted for scheduling.

    experiment_id: links this workflow to a broader experiment record
    steps: ordered list of operations to execute
    submitted_by: identifier of the submitter (AI agent ID, researcher name, etc.)
    priority: higher values run before lower values when the scheduler must choose
    """
    steps: list[WorkflowStep]
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "unnamed_workflow"
    submitted_by: str = "unknown"
    priority: int = 0
    submitted_at: datetime = field(default_factory=datetime.utcnow)

    # Set by scheduler
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: WorkflowStatus = field(default=WorkflowStatus.PENDING, init=False)
    started_at: Optional[datetime] = field(default=None, init=False)
    finished_at: Optional[datetime] = field(default=None, init=False)

    def get_required_instruments(self) -> set[str]:
        """Returns the set of instrument IDs this workflow needs."""
        return {step.instrument_id for step in self.steps}

    def get_ready_steps(self) -> list[tuple[int, WorkflowStep]]:
        """
        Returns steps that are ready to run: PENDING status and all dependencies COMPLETED.
        Used by the scheduler to find work it can dispatch immediately.
        """
        ready = []
        for i, step in enumerate(self.steps):
            if step.status != StepStatus.PENDING:
                continue
            deps_met = all(
                self.steps[dep_idx].status == StepStatus.COMPLETED
                for dep_idx in step.depends_on
            )
            if deps_met:
                ready.append((i, step))
        return ready

    def is_terminal(self) -> bool:
        return self.status in (
            WorkflowStatus.COMPLETED,
            WorkflowStatus.FAILED,
            WorkflowStatus.CANCELLED,
        )

    def summary(self) -> dict:
        total = len(self.steps)
        completed = sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)
        failed = sum(1 for s in self.steps if s.status == StepStatus.FAILED)
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "experiment_id": self.experiment_id,
            "status": self.status.name,
            "steps_total": total,
            "steps_completed": completed,
            "steps_failed": failed,
            "submitted_by": self.submitted_by,
        }
