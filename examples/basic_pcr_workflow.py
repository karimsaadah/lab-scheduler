"""
Basic PCR Workflow Example
==========================

Demonstrates the core lab_scheduler API with a real thermocycler simulation:

1. Create and connect a Thermocycler instrument
2. Define a 2-step PCR workflow:
   - Step 0: Submit the PCR program to the thermocycler
   - Step 1: Wait for completion (depends on step 0)
3. Register the instrument with LabScheduler
4. Submit the workflow
5. Run the scheduler
6. Print results and event log summary

Run from the project root:
    python examples/basic_pcr_workflow.py
"""

import logging
import sys
from pathlib import Path

# Allow running from project root without install
sys.path.insert(0, str(Path(__file__).parent.parent))

from lab_scheduler.instruments.thermocycler import (
    Thermocycler,
    ThermalProgram,
    ThermalStep,
    STANDARD_PROGRAMS,
)
from lab_scheduler.observability.event_log import EventLog, EventType
from lab_scheduler.scheduler.scheduler import LabScheduler
from lab_scheduler.scheduler.workflow import Workflow, WorkflowStep, StepStatus

# Configure logging so we can see what's happening
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("example.pcr_workflow")


def main():
    # ----------------------------------------------------------------
    # 1. Create and connect instrument
    # ----------------------------------------------------------------
    logger.info("=== Basic PCR Workflow Example ===")

    thermocycler = Thermocycler(
        instrument_id="tc_01",
        name="Bio-Rad C1000 Touch",
        host="localhost",
        port=5001,
    )
    connected = thermocycler.connect()
    if not connected:
        logger.error("Failed to connect to thermocycler. Exiting.")
        sys.exit(1)

    logger.info("Thermocycler connected: %s", thermocycler.name)

    # ----------------------------------------------------------------
    # 2. Define a short PCR protocol (fast_pcr for demo purposes)
    #    In a real experiment, you'd use STANDARD_PROGRAMS["pcr_standard"]
    #    which runs 30 cycles. This demo program runs in < 1 second.
    # ----------------------------------------------------------------
    demo_program = ThermalProgram(
        name="demo_pcr",
        steps=[
            ThermalStep(95.0, 1, "initial_denaturation"),
            ThermalStep(60.0, 1, "annealing"),
            ThermalStep(72.0, 1, "extension"),
        ],
        cycles=1,
    )

    logger.info(
        "Program '%s': %d steps, %d cycle(s), ~%ds estimated",
        demo_program.name,
        len(demo_program.steps),
        demo_program.cycles,
        demo_program.estimated_duration_seconds,
    )

    # ----------------------------------------------------------------
    # 3. Build the workflow
    # ----------------------------------------------------------------
    step_run = WorkflowStep(
        instrument_id="tc_01",
        name="submit_pcr_program",
        operation=lambda inst, p=demo_program: inst.run_program(p),
        retry_count=1,  # retry once if instrument fails to accept the program
    )

    step_wait = WorkflowStep(
        instrument_id="tc_01",
        name="wait_for_completion",
        operation=lambda inst: inst.wait_for_completion(
            poll_interval_seconds=0.05,
            timeout_seconds=30.0,
        ),
        depends_on=[0],  # must wait for step_run to complete first
    )

    workflow = Workflow(
        steps=[step_run, step_wait],
        name="basic_pcr",
        submitted_by="researcher_karim",
        priority=5,
    )

    logger.info(
        "Workflow '%s': %d steps, instruments: %s",
        workflow.name,
        len(workflow.steps),
        workflow.get_required_instruments(),
    )

    # ----------------------------------------------------------------
    # 4. Register instrument and submit workflow
    # ----------------------------------------------------------------
    event_log = EventLog()
    scheduler = LabScheduler(max_workers=4, event_log=event_log)
    scheduler.register_instrument(thermocycler)

    workflow_id = scheduler.submit(workflow)
    logger.info("Workflow submitted: %s", workflow_id)

    # ----------------------------------------------------------------
    # 5. Run the scheduler (blocks until complete)
    # ----------------------------------------------------------------
    logger.info("Starting scheduler...")
    scheduler.run(poll_interval_seconds=0.1)

    # ----------------------------------------------------------------
    # 6. Print results
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("WORKFLOW RESULTS")
    print("=" * 60)

    summary = workflow.summary()
    print(f"Workflow:    {summary['name']} ({summary['workflow_id'][:8]}...)")
    print(f"Status:      {summary['status']}")
    print(f"Steps total: {summary['steps_total']}")
    print(f"Completed:   {summary['steps_completed']}")
    print(f"Failed:      {summary['steps_failed']}")

    print("\nStep details:")
    for i, step in enumerate(workflow.steps):
        duration = ""
        if step.result:
            duration = f"  ({step.result.duration_seconds:.3f}s)"
        print(f"  [{i}] {step.name}: {step.status.name}{duration}")
        if step.result and step.result.data:
            for key, val in step.result.data.items():
                print(f"      {key}: {val}")

    print("\nEvent log summary:")
    all_events = event_log.get_events()
    for event_type in EventType:
        count = len(event_log.get_events(event_type))
        if count > 0:
            print(f"  {event_type.name}: {count}")

    if workflow.status.name == "COMPLETED":
        print("\nPCR run complete. Plate is ready for the next step.")
    else:
        print("\nWorkflow did not complete successfully.")
        sys.exit(1)


if __name__ == "__main__":
    main()
