"""
LabScheduler — the central coordinator for laboratory workflow execution.

Responsibilities:
1. Maintain a registry of connected instruments and their current availability.
2. Accept workflow submissions from AI agents or human operators.
3. Sequence workflow steps respecting instrument availability and intra-workflow
   dependencies, running independent steps concurrently where possible.
4. Handle step failures: retry up to the configured limit, then mark the workflow
   failed and release its instrument locks.
5. Emit structured log events at every state transition for observability.

Concurrency model:
    Steps across different workflows that target different instruments run concurrently
    using a thread pool. Steps within a single workflow that declare no dependencies
    can also run concurrently if they target different instruments.

    Instrument locking is per-instrument-id. A step acquires the lock for its instrument
    before calling the operation and releases it on completion or failure. The scheduler
    never holds locks on behalf of a step longer than the step's execution time.

    This is deliberately simple — one lock per instrument. More sophisticated scheduling
    (e.g., time-window reservations, plate conveyor routing) would be added here as
    the system grows, without changing the instrument or workflow interfaces.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime

from ..instruments.base import Instrument, InstrumentStatus
from ..observability.event_log import EventLog, EventType
from .workflow import StepStatus, Workflow, WorkflowStatus, WorkflowStep

logger = logging.getLogger(__name__)


class LabScheduler:
    """
    Central scheduler for laboratory workflow execution.

    Usage:
        scheduler = LabScheduler(max_workers=8)
        scheduler.register_instrument(liquid_handler)
        scheduler.register_instrument(thermocycler)
        workflow_id = scheduler.submit(workflow)
        scheduler.run()   # blocks until queue is empty
        result = scheduler.get_workflow(workflow_id)
    """

    def __init__(self, max_workers: int = 8, event_log: EventLog | None = None):
        self._instruments: dict[str, Instrument] = {}
        self._instrument_locks: dict[str, threading.Lock] = {}
        self._workflows: dict[str, Workflow] = {}
        self._queue: list[Workflow] = []
        self._queue_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._event_log = event_log or EventLog()
        self._running = False

    # ------------------------------------------------------------------
    # Instrument registry
    # ------------------------------------------------------------------

    def register_instrument(self, instrument: Instrument) -> None:
        """
        Add an instrument to the scheduler's registry.
        The instrument must already be connected (status == IDLE).
        """
        if instrument.status != InstrumentStatus.IDLE:
            raise ValueError(
                f"Cannot register {instrument.instrument_id}: "
                f"status is {instrument.status.name}, expected IDLE. "
                f"Call instrument.connect() before registering."
            )
        self._instruments[instrument.instrument_id] = instrument
        self._instrument_locks[instrument.instrument_id] = threading.Lock()
        logger.info("Registered instrument: %s (%s)", instrument.instrument_id, instrument.name)
        self._event_log.record(
            EventType.INSTRUMENT_REGISTERED,
            {
                "instrument_id": instrument.instrument_id,
                "name": instrument.name,
            },
        )

    def get_instrument(self, instrument_id: str) -> Instrument:
        if instrument_id not in self._instruments:
            raise KeyError(f"Instrument '{instrument_id}' not registered with scheduler")
        return self._instruments[instrument_id]

    def instrument_status_report(self) -> dict[str, str]:
        """Returns a snapshot of all instrument statuses. Useful for health checks."""
        return {iid: inst.get_status().name for iid, inst in self._instruments.items()}

    # ------------------------------------------------------------------
    # Workflow submission
    # ------------------------------------------------------------------

    def submit(self, workflow: Workflow) -> str:
        """
        Submit a workflow for execution. Returns the workflow_id.

        Validates that all required instruments are registered before accepting.
        Rejects workflows referencing unknown instruments immediately rather than
        discovering the problem at execution time.
        """
        missing = workflow.get_required_instruments() - set(self._instruments)
        if missing:
            raise ValueError(
                f"Workflow '{workflow.name}' references unregistered instruments: {missing}"
            )

        with self._queue_lock:
            self._workflows[workflow.workflow_id] = workflow
            self._queue.append(workflow)
            # Higher priority workflows sort to the front
            self._queue.sort(key=lambda w: w.priority, reverse=True)

        logger.info(
            "Workflow submitted: %s (id=%s, steps=%d, priority=%d)",
            workflow.name,
            workflow.workflow_id,
            len(workflow.steps),
            workflow.priority,
        )
        self._event_log.record(EventType.WORKFLOW_SUBMITTED, workflow.summary())
        return workflow.workflow_id

    def get_workflow(self, workflow_id: str) -> Workflow | None:
        return self._workflows.get(workflow_id)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, poll_interval_seconds: float = 0.5) -> None:
        """
        Run the scheduler until all submitted workflows reach a terminal state.
        Blocks the calling thread.
        """
        self._running = True
        logger.info("Scheduler started")
        self._event_log.record(EventType.SCHEDULER_STARTED, {})

        active_futures: dict[str, Future] = {}  # workflow_id -> future

        try:
            while self._running:
                self._dispatch_ready_workflows(active_futures)
                self._collect_completed(active_futures)

                with self._queue_lock:
                    pending = [w for w in self._workflows.values() if not w.is_terminal()]

                if not pending and not active_futures:
                    logger.info("All workflows complete. Scheduler idle.")
                    break

                time.sleep(poll_interval_seconds)
        finally:
            self._running = False
            self._event_log.record(EventType.SCHEDULER_STOPPED, {})
            logger.info("Scheduler stopped")

    def _dispatch_ready_workflows(self, active_futures: dict) -> None:
        with self._queue_lock:
            dispatchable = [
                w
                for w in self._queue
                if w.workflow_id not in active_futures and not w.is_terminal()
            ]

        for workflow in dispatchable:
            future = self._executor.submit(self._execute_workflow, workflow)
            active_futures[workflow.workflow_id] = future

    def _collect_completed(self, active_futures: dict) -> None:
        done = [wid for wid, f in active_futures.items() if f.done()]
        for wid in done:
            future = active_futures.pop(wid)
            exc = future.exception()
            if exc:
                logger.error("Workflow %s raised unhandled exception: %s", wid, exc)

    def _execute_workflow(self, workflow: Workflow) -> None:
        """
        Execute a workflow in a worker thread.

        Runs ready steps concurrently (when they target different instruments),
        waits for all steps to complete or fail, then marks the workflow terminal.
        """
        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = datetime.utcnow()
        self._event_log.record(EventType.WORKFLOW_STARTED, workflow.summary())

        step_futures: dict[int, Future] = {}

        while True:
            ready = workflow.get_ready_steps()

            for step_idx, step in ready:
                if step_idx not in step_futures:
                    f = self._executor.submit(self._execute_step, workflow, step_idx, step)
                    step_futures[step_idx] = f

            # Check for failures
            any_failed = any(s.status == StepStatus.FAILED for s in workflow.steps)
            if any_failed:
                # Cancel steps that haven't started
                for s in workflow.steps:
                    if s.status == StepStatus.PENDING:
                        s.status = StepStatus.SKIPPED
                workflow.status = WorkflowStatus.FAILED
                break

            # Check for completion
            all_done = all(
                s.status in (StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED)
                for s in workflow.steps
            )
            if all_done:
                any_failed = any(s.status == StepStatus.FAILED for s in workflow.steps)
                workflow.status = WorkflowStatus.FAILED if any_failed else WorkflowStatus.COMPLETED
                break

            time.sleep(0.1)

        workflow.finished_at = datetime.utcnow()
        event_type = (
            EventType.WORKFLOW_COMPLETED
            if workflow.status == WorkflowStatus.COMPLETED
            else EventType.WORKFLOW_FAILED
        )
        self._event_log.record(event_type, workflow.summary())
        logger.info("Workflow %s: %s", workflow.name, workflow.status.name)

    def _execute_step(self, workflow: Workflow, step_idx: int, step: WorkflowStep) -> None:
        """
        Execute a single workflow step, holding the instrument lock for the duration.
        Retries up to step.retry_count times on failure.
        """
        instrument = self._instruments[step.instrument_id]
        lock = self._instrument_locks[step.instrument_id]

        step.status = StepStatus.RUNNING
        step.started_at = datetime.utcnow()
        self._event_log.record(
            EventType.STEP_STARTED,
            {
                "workflow_id": workflow.workflow_id,
                "step_index": step_idx,
                "step_name": step.name,
                "instrument_id": step.instrument_id,
            },
        )

        last_result = None
        for attempt in range(step.retry_count + 1):
            step.attempts = attempt + 1
            with lock:
                try:
                    last_result = step.operation(instrument)
                except Exception as exc:
                    logger.error(
                        "Step '%s' raised exception on attempt %d: %s", step.name, attempt + 1, exc
                    )
                    last_result = None

            if last_result and last_result.success:
                break

            if attempt < step.retry_count:
                logger.warning(
                    "Step '%s' failed (attempt %d/%d), retrying...",
                    step.name,
                    attempt + 1,
                    step.retry_count + 1,
                )
                time.sleep(1.0)  # Brief pause before retry

        step.result = last_result
        step.finished_at = datetime.utcnow()

        if last_result and last_result.success:
            step.status = StepStatus.COMPLETED
            self._event_log.record(
                EventType.STEP_COMPLETED,
                {
                    "workflow_id": workflow.workflow_id,
                    "step_name": step.name,
                    "duration_seconds": last_result.duration_seconds,
                },
            )
        else:
            step.status = StepStatus.FAILED
            error_msg = last_result.error if last_result else "unhandled exception"
            logger.error(
                "Step '%s' failed after %d attempt(s): %s", step.name, step.attempts, error_msg
            )
            self._event_log.record(
                EventType.STEP_FAILED,
                {
                    "workflow_id": workflow.workflow_id,
                    "step_name": step.name,
                    "error": error_msg,
                    "attempts": step.attempts,
                },
            )
