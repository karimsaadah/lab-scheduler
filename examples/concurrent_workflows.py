"""
Concurrent Workflows Example
=============================

Demonstrates that the lab scheduler can execute independent workflows
on different instruments simultaneously, maximizing instrument utilization
— a key requirement for high-throughput laboratory automation.

Setup:
  - LiquidHandler "lh_01": performs a batch of sample transfers
  - PlateReader  "pr_01": reads a plate in parallel

Both workflows are submitted before scheduler.run() is called.
The scheduler dispatches them concurrently across its thread pool.
Wall-clock time is compared against serial execution to demonstrate
the throughput advantage.

Run from the project root:
    python examples/concurrent_workflows.py
"""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lab_scheduler.instruments.liquid_handler import LiquidHandler, TransferSpec
from lab_scheduler.instruments.plate_reader import PlateReader, ReadMode, ReadSpec
from lab_scheduler.observability.event_log import EventLog, EventType
from lab_scheduler.scheduler.scheduler import LabScheduler
from lab_scheduler.scheduler.workflow import (
    Workflow,
    WorkflowStatus,
    WorkflowStep,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)7s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("example.concurrent_workflows")


def build_liquid_handler_workflow(instrument_id: str = "lh_01") -> Workflow:
    """
    A 3-step sample transfer workflow targeting a liquid handler.
    Transfers 50 µL from a reservoir into 3 destination wells.
    """
    specs = [
        TransferSpec(volume_ul=50.0, source="reservoir_1", destination=f"A{col}", tip_index=col - 1)
        for col in range(1, 4)
    ]

    steps = []
    for i, spec in enumerate(specs):
        steps.append(
            WorkflowStep(
                instrument_id=instrument_id,
                name=f"transfer_A{i + 1}",
                operation=lambda inst, s=spec: inst.transfer(s),
                depends_on=[i - 1] if i > 0 else [],  # chain sequentially (same instrument)
                retry_count=1,
            )
        )

    # Final step: eject tips for cleanliness
    steps.append(
        WorkflowStep(
            instrument_id=instrument_id,
            name="eject_tips",
            operation=lambda inst: inst.eject_tips(),
            depends_on=[len(specs) - 1],
        )
    )

    return Workflow(
        steps=steps,
        name="sample_transfer_workflow",
        submitted_by="liquid_handling_agent",
        priority=5,
    )


def build_plate_reader_workflow(instrument_id: str = "pr_01") -> Workflow:
    """
    A multi-mode plate reading workflow: fluorescence + absorbance reads.
    Both steps target the same instrument so they run sequentially.
    """
    fluor_spec = ReadSpec(
        plate_id="control_plate_A",
        mode=ReadMode.FLUORESCENCE,
        excitation_nm=485,
        emission_nm=528,
        gain="auto",
    )
    abs_spec = ReadSpec(
        plate_id="control_plate_A",
        mode=ReadMode.ABSORBANCE,
        wavelength_nm=600,
        gain="auto",
    )

    return Workflow(
        steps=[
            WorkflowStep(
                instrument_id=instrument_id,
                name="fluorescence_read",
                operation=lambda inst, s=fluor_spec: inst.read_plate(s),
                retry_count=1,
            ),
            WorkflowStep(
                instrument_id=instrument_id,
                name="absorbance_read",
                operation=lambda inst, s=abs_spec: inst.read_plate(s),
                depends_on=[0],
                retry_count=1,
            ),
        ],
        name="multi_mode_plate_read",
        submitted_by="analysis_agent",
        priority=5,
    )


def print_workflow_result(wf: Workflow) -> None:
    print(f"\n  Workflow: {wf.name}")
    print(f"  Status:  {wf.status.name}")
    for i, step in enumerate(wf.steps):
        duration = f"{step.result.duration_seconds:.3f}s" if step.result else "—"
        print(f"    Step {i}: {step.name} → {step.status.name} ({duration})")


def main():
    logger.info("=== Concurrent Workflows Example ===")

    # ----------------------------------------------------------------
    # Connect instruments
    # ----------------------------------------------------------------
    liquid_handler = LiquidHandler("lh_01", "Tecan Fluent", host="localhost")
    plate_reader = PlateReader("pr_01", "BioTek Synergy H1", host="localhost")

    for inst in (liquid_handler, plate_reader):
        ok = inst.connect()
        if not ok:
            logger.error("Failed to connect: %s", inst.name)
            sys.exit(1)
        logger.info("Connected: %s [%s]", inst.name, inst.instrument_id)

    # ----------------------------------------------------------------
    # Build workflows
    # ----------------------------------------------------------------
    wf_lh = build_liquid_handler_workflow("lh_01")
    wf_pr = build_plate_reader_workflow("pr_01")

    logger.info(
        "Liquid handler workflow: %d steps targeting %s",
        len(wf_lh.steps),
        wf_lh.get_required_instruments(),
    )
    logger.info(
        "Plate reader workflow: %d steps targeting %s",
        len(wf_pr.steps),
        wf_pr.get_required_instruments(),
    )

    # ----------------------------------------------------------------
    # Submit both workflows before calling run()
    # The scheduler will dispatch them concurrently.
    # ----------------------------------------------------------------
    event_log = EventLog()
    scheduler = LabScheduler(max_workers=8, event_log=event_log)
    scheduler.register_instrument(liquid_handler)
    scheduler.register_instrument(plate_reader)

    scheduler.submit(wf_lh)
    scheduler.submit(wf_pr)

    logger.info("Submitted %d workflows. Starting scheduler...", 2)

    wall_start = time.monotonic()
    scheduler.run(poll_interval_seconds=0.05)
    wall_elapsed = time.monotonic() - wall_start

    # ----------------------------------------------------------------
    # Results
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CONCURRENT EXECUTION RESULTS")
    print("=" * 60)
    print(f"Total wall-clock time: {wall_elapsed:.3f}s")

    print_workflow_result(wf_lh)
    print_workflow_result(wf_pr)

    # Collect step durations to estimate serial time
    all_step_durations = [
        step.result.duration_seconds for wf in (wf_lh, wf_pr) for step in wf.steps if step.result
    ]
    serial_estimate = sum(all_step_durations)
    print(f"\n  Serial estimate (sum of all step durations): {serial_estimate:.3f}s")
    print(f"  Concurrent wall time:                        {wall_elapsed:.3f}s")
    if serial_estimate > 0 and wall_elapsed > 0:
        speedup = serial_estimate / wall_elapsed
        print(f"  Concurrency speedup:                         {speedup:.2f}x")

    print("\nEvent summary:")
    for etype in (
        EventType.WORKFLOW_SUBMITTED,
        EventType.WORKFLOW_STARTED,
        EventType.WORKFLOW_COMPLETED,
        EventType.STEP_COMPLETED,
    ):
        count = len(event_log.get_events(etype))
        print(f"  {etype.name}: {count}")

    failed = [wf for wf in (wf_lh, wf_pr) if wf.status != WorkflowStatus.COMPLETED]
    if failed:
        print(f"\n{len(failed)} workflow(s) did not complete!")
        sys.exit(1)
    else:
        print("\nAll workflows completed successfully.")


if __name__ == "__main__":
    main()
