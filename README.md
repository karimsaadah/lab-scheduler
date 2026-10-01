# lab-scheduler

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![CI](https://img.shields.io/github/actions/workflow/status/karimsaadah/lab-scheduler/ci.yml?branch=main&label=CI)
![Coverage](https://img.shields.io/codecov/c/github/karimsaadah/lab-scheduler)

A concurrent workflow scheduler for laboratory automation — the execution backbone of a self-driving laboratory.

---

## Overview

Modern biofoundries run dozens of instruments in parallel: liquid handlers, thermocyclers, plate readers, colony pickers, incubators. Coordinating these instruments manually creates bottlenecks, introduces human error, and makes throughput scale with headcount rather than with hardware. `lab-scheduler` addresses this by providing a structured execution layer that sits between an AI planning agent and the physical instruments — accepting high-level experimental workflows, managing per-instrument locks, running independent steps concurrently, and recording every state transition as a structured event for audit and analysis.

The architecture follows the pattern of production self-driving laboratory systems (Emerald Cloud Lab, Strateos, Arctoris): a declarative workflow model that separates *what* needs to happen from *how* and *when* the scheduler decides to run it. An AI agent submits a `Workflow` describing the experiment; the scheduler handles sequencing, concurrency, retry logic, and observability transparently. This makes it straightforward to connect LLM-based planning agents — a natural-language experimental goal becomes a `Workflow` object; the scheduler executes it regardless of whether it came from a researcher, a Bayesian optimizer, or a GPT-4 function call.

---

## Architecture

```
  Experimental Goal (str)
          │
          ▼
  ┌───────────────────┐
  │   PlanningAgent   │  ← AI layer: translates goals into Workflows
  │  (LLM / rules)   │    (LLM integration point)
  └────────┬──────────┘
           │  Workflow (steps + deps)
           ▼
  ┌───────────────────┐
  │   LabScheduler    │  ← Coordinator: queues, dispatches, retries
  │  ThreadPoolExec.  │    priority queue, per-instrument locking
  └────────┬──────────┘
           │ acquires lock per step
           ▼
  ┌────────────────────────────────────────┐
  │            Instrument Locks            │
  │  ┌──────────┐ ┌──────────┐ ┌────────┐ │
  │  │LiquidHdlr│ │Thermocycl│ │Plate   │ │
  │  │  lh_01   │ │  tc_01   │ │Reader  │ │
  │  └──────────┘ └──────────┘ └────────┘ │
  └────────────────────────────────────────┘
           │
           ▼
  ┌───────────────────┐
  │    EventLog       │  ← Observability: typed events, timestamps, payloads
  │  (time-series)    │    all state transitions recorded
  └───────────────────┘
           │
           ▼
  ┌───────────────────┐
  │   DataPipeline    │  ← Analysis: z-score normalization, QC, report gen
  └───────────────────┘
```

---

## Quick Start

```python
from lab_scheduler.instruments.thermocycler import Thermocycler, STANDARD_PROGRAMS
from lab_scheduler.scheduler.scheduler import LabScheduler
from lab_scheduler.scheduler.workflow import Workflow, WorkflowStep

tc = Thermocycler("tc_01", "Bio-Rad C1000", host="localhost")
tc.connect()

scheduler = LabScheduler()
scheduler.register_instrument(tc)

program = STANDARD_PROGRAMS["pcr_standard"]
workflow = Workflow(
    steps=[
        WorkflowStep(
            instrument_id="tc_01",
            name="run_pcr",
            operation=lambda inst: inst.run_program(program),
        ),
        WorkflowStep(
            instrument_id="tc_01",
            name="wait_for_completion",
            operation=lambda inst: inst.wait_for_completion(),
            depends_on=[0],
        ),
    ],
    name="my_pcr_experiment",
)

scheduler.submit(workflow)
scheduler.run()
print(workflow.summary())
```

---

## Key Design Decisions

### 1. Per-instrument locking, not a global lock

Each instrument has its own `threading.Lock`. Steps targeting different instruments acquire different locks and run fully concurrently. A global lock would serialize all step execution and eliminate the concurrency advantage that makes multi-instrument automation worthwhile.

The tradeoff: per-instrument locking requires every step to declare its `instrument_id` upfront, but this is also the right engineering discipline — a step that touches multiple instruments is a bug waiting to happen. Steps should be atomic at the instrument level.

### 2. Abstract base class for instruments

`Instrument` (ABC) defines the interface — `connect`, `disconnect`, `reset`, `get_status` — and every physical driver implements it. The scheduler, AI agent, and data pipeline never import a concrete class. This makes it possible to swap vendors (Tecan → Beckman → Hamilton) by writing a new subclass, run tests with mock instruments, and add new instrument types without touching the scheduling or planning logic.

### 3. Structured events, not plain log strings

Every state transition — workflow submitted, step started, step failed — is recorded as a typed `EventType` enum value with a structured payload dict rather than a log string. Structured events are machine-readable: a dashboard query for all `STEP_FAILED` events with their `workflow_id` and `error` fields is a dict filter, not a regex. In production these events would stream to a time-series database (InfluxDB) or an event bus (Kafka), enabling real-time alerting and post-run analysis without parsing log files.

### 4. Step-level retry, not workflow-level retry

Retries are configured per `WorkflowStep` (`retry_count`) and executed by the scheduler before marking the step failed. Retrying at the workflow level would re-run steps that already succeeded — wasteful and potentially harmful (re-dispensing reagents that were already dispensed). Step-level retry is conservative: only the failing operation is retried; upstream steps that produced good results are preserved. The retry limit and backoff are tunable per step, so a flaky plate reader can have `retry_count=2` while a thermocycler program start (which is rarely transient) has `retry_count=0`.

---

## Project Layout

```
lab-scheduler/
├── lab_scheduler/
│   ├── __init__.py
│   ├── instruments/
│   │   ├── __init__.py
│   │   ├── base.py              # Instrument ABC, InstrumentResult, InstrumentStatus
│   │   ├── liquid_handler.py    # LiquidHandler + TransferSpec
│   │   ├── thermocycler.py      # Thermocycler + ThermalProgram
│   │   └── plate_reader.py      # PlateReader + ReadSpec
│   ├── scheduler/
│   │   ├── __init__.py
│   │   ├── workflow.py          # Workflow, WorkflowStep, WorkflowStatus, StepStatus
│   │   └── scheduler.py         # LabScheduler (ThreadPoolExecutor + per-instrument locks)
│   ├── observability/
│   │   ├── __init__.py
│   │   └── event_log.py         # EventLog + EventType enum
│   ├── agents/
│   │   ├── __init__.py
│   │   └── planning_agent.py    # PlanningAgent (rule-based + LLM integration point)
│   └── pipeline/
│       ├── __init__.py
│       └── data_pipeline.py     # DataPipeline + PipelineResult (normalization, QC, reports)
├── tests/
│   ├── unit/
│   │   ├── test_workflow.py     # Workflow/WorkflowStep unit tests
│   │   └── test_scheduler.py    # LabScheduler unit tests (mocked instruments)
│   └── integration/
│       └── test_full_workflow.py # End-to-end tests with real (simulated) instruments
├── examples/
│   ├── basic_pcr_workflow.py    # Single-instrument PCR example
│   └── concurrent_workflows.py  # Concurrent multi-instrument example
├── pyproject.toml
└── README.md
```

---

## Running Tests

Install test dependencies:

```bash
pip install -e ".[test]"
```

Run the full test suite:

```bash
pytest
```

Run with coverage report:

```bash
pytest --cov=lab_scheduler --cov-report=term-missing
```

Run only unit tests (fast, no I/O):

```bash
pytest tests/unit/
```

Run integration tests:

```bash
pytest tests/integration/
```

---

## Running Examples

```bash
# Basic PCR workflow (single instrument, 2 steps with dependency)
python examples/basic_pcr_workflow.py

# Concurrent workflows (2 workflows on different instruments simultaneously)
python examples/concurrent_workflows.py
```

---

## Extending: Adding a New Instrument in 4 Steps

1. **Subclass `Instrument`** in `lab_scheduler/instruments/your_instrument.py`:
   ```python
   class YourInstrument(Instrument):
       def connect(self) -> bool: ...
       def disconnect(self) -> None: ...
       def reset(self) -> bool: ...
       def get_status(self) -> InstrumentStatus: ...
       def your_operation(self, params) -> InstrumentResult: ...
   ```

2. **Export it** from `lab_scheduler/instruments/__init__.py`.

3. **Connect and register** before submitting workflows:
   ```python
   instrument = YourInstrument("your_id", "Your Instrument", host="...")
   instrument.connect()
   scheduler.register_instrument(instrument)
   ```

4. **Reference it in `WorkflowStep`**:
   ```python
   WorkflowStep(
       instrument_id="your_id",
       operation=lambda inst: inst.your_operation(params),
   )
   ```

The scheduler, event log, data pipeline, and AI planning agent all work with the new instrument without modification.

---

## Notes

This project demonstrates the architectural patterns used in real laboratory automation systems: abstract instrument interfaces, structured event logging, concurrent step dispatch with per-resource locking, and a clean separation between AI planning and mechanical execution. The simulated instrument operations (sleep-based delays, hash-generated plate measurements) are stand-ins for vendor SDK calls; the structure and interfaces are production-grade.

The `PlanningAgent` in `lab_scheduler/agents/planning_agent.py` includes a documented integration point showing how a real LLM call (Anthropic Claude, OpenAI GPT-4) would replace the rule-based planner to accept arbitrary natural-language experimental goals.

---

*Author: Karim Saadah*
