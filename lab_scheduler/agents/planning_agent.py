"""
AI Planning Agent — translates high-level experimental goals into executable Workflows.

This module is the integration point between an AI reasoning layer and the lab
scheduler's execution layer. In a production self-driving laboratory, this class
would wrap an LLM call (GPT-4, Claude, or a fine-tuned domain model) that
receives the experimental goal and instrument inventory as context, reasons about
the optimal protocol, and emits a structured Workflow.

The rule-based planner implemented here serves two purposes:
1. It demonstrates the expected interface and data flow without requiring an
   LLM API key or network access during development and testing.
2. It provides a deterministic fallback for well-known, fixed protocols (PCR,
   standard plate reads) where LLM reasoning adds no value and reproducibility
   is paramount.

To integrate a real LLM:
    1. Replace or augment the _rule_based_plan() dispatcher with an _llm_plan()
       method that constructs a prompt from the goal and available_instruments.
    2. Parse the LLM's structured output (JSON tool calls recommended) into the
       same WorkflowStep objects returned here.
    3. Keep _rule_based_plan() as a validator / fallback for known-good protocols.

Example LLM prompt strategy:
    System: "You are a lab automation planner. Given an experimental goal and a
    list of available instruments, output a JSON array of steps. Each step must
    specify: instrument_id, operation_name, parameters."
    User: f"Goal: {goal}. Available instruments: {available_instruments}"
"""

from __future__ import annotations

import logging
from typing import Any

from ..instruments.liquid_handler import LiquidHandler, TransferSpec
from ..instruments.plate_reader import PlateReader, ReadMode, ReadSpec
from ..instruments.thermocycler import STANDARD_PROGRAMS, Thermocycler
from ..scheduler.workflow import Workflow, WorkflowStep

logger = logging.getLogger(__name__)


class PlanningError(Exception):
    """Raised when the planning agent cannot generate a valid workflow for the goal."""


class PlanningAgent:
    """
    AI planning agent that converts experimental goals into Workflows.

    The agent is stateless between plan() calls — it holds no memory of prior
    experiments. Experiment history and context injection are the responsibility
    of the caller (e.g., a higher-level campaign orchestrator).

    Args:
        agent_id: Identifier for this agent instance, recorded in submitted_by
                  on generated workflows.
    """

    def __init__(self, agent_id: str = "planning_agent_v1") -> None:
        self.agent_id = agent_id
        self._logger = logging.getLogger(f"{__name__}.{agent_id}")

    def plan(self, goal: str, available_instruments: dict[str, Any]) -> Workflow:
        """
        Generate a Workflow for the given experimental goal.

        This is the primary public interface. In a production system, this method
        would dispatch to an LLM reasoning chain. Currently it dispatches to a
        rule-based planner that handles a fixed set of known protocols.

        Args:
            goal: Human-readable description of what the experiment should achieve.
                  Supported values: "run_pcr", "read_plate", "pcr_then_read".
                  An LLM integration would accept arbitrary natural-language goals.
            available_instruments: Dict mapping instrument_id to Instrument objects.
                  The planner uses this to validate that required instruments exist
                  before generating steps that reference them.

        Returns:
            A Workflow ready for submission to LabScheduler.submit().

        Raises:
            PlanningError: If the goal is not recognized, or if required instruments
                           are not present in available_instruments.
        """
        self._logger.info("Planning workflow for goal: '%s'", goal)
        self._logger.debug("Available instruments: %s", list(available_instruments.keys()))

        # ----------------------------------------------------------------
        # LLM integration point:
        # workflow = await self._llm_plan(goal, available_instruments)
        # return workflow
        # ----------------------------------------------------------------

        return self._rule_based_plan(goal, available_instruments)

    # ------------------------------------------------------------------
    # Rule-based planner (deterministic fallback / built-in protocols)
    # ------------------------------------------------------------------

    def _rule_based_plan(self, goal: str, available_instruments: dict[str, Any]) -> Workflow:
        """
        Deterministic planner for a fixed set of known experimental protocols.

        Dispatches on goal string. Each handler validates instrument availability,
        constructs WorkflowSteps with concrete operation callables, and returns
        a Workflow.
        """
        goal_normalized = goal.strip().lower()

        handlers = {
            "run_pcr": self._plan_pcr,
            "read_plate": self._plan_read_plate,
            "pcr_then_read": self._plan_pcr_then_read,
            "transfer_samples": self._plan_transfer_samples,
        }

        handler = handlers.get(goal_normalized)
        if handler is None:
            supported = ", ".join(f"'{k}'" for k in handlers)
            raise PlanningError(
                f"Unrecognized goal: '{goal}'. "
                f"Rule-based planner supports: {supported}. "
                f"For arbitrary goals, integrate an LLM into PlanningAgent.plan()."
            )

        return handler(available_instruments)

    def _require_instrument(
        self,
        available_instruments: dict[str, Any],
        instrument_type: type,
        friendly_name: str,
    ) -> Any:
        """
        Find the first available instrument of the required type.

        Raises PlanningError if none is found.
        """
        for iid, inst in available_instruments.items():
            if isinstance(inst, instrument_type):
                return iid, inst
        raise PlanningError(
            f"Goal requires a {friendly_name} but none is registered. "
            f"Available instruments: {list(available_instruments.keys())}"
        )

    def _plan_pcr(self, available_instruments: dict[str, Any]) -> Workflow:
        """
        Standard PCR protocol: run the pcr_standard program and wait for completion.
        """
        tc_id, tc = self._require_instrument(available_instruments, Thermocycler, "Thermocycler")
        program = STANDARD_PROGRAMS["pcr_standard"]

        steps = [
            WorkflowStep(
                instrument_id=tc_id,
                name="start_pcr_program",
                operation=lambda inst, p=program: inst.run_program(p),  # type: ignore[misc]
                retry_count=1,
            ),
            WorkflowStep(
                instrument_id=tc_id,
                name="wait_for_pcr_completion",
                operation=lambda inst: inst.wait_for_completion(
                    poll_interval_seconds=0.05,
                    timeout_seconds=7200.0,
                ),
                depends_on=[0],
                retry_count=0,
                timeout_seconds=7300.0,
            ),
        ]

        return Workflow(
            steps=steps,
            name="pcr_standard",
            submitted_by=self.agent_id,
            priority=5,
        )

    def _plan_read_plate(self, available_instruments: dict[str, Any]) -> Workflow:
        """
        Plate read protocol: measure fluorescence at GFP wavelengths (485/528 nm).
        """
        pr_id, pr = self._require_instrument(available_instruments, PlateReader, "PlateReader")
        spec = ReadSpec(
            plate_id="plate_001",
            mode=ReadMode.FLUORESCENCE,
            excitation_nm=485,
            emission_nm=528,
            gain="auto",
        )

        steps = [
            WorkflowStep(
                instrument_id=pr_id,
                name="read_fluorescence",
                operation=lambda inst, s=spec: inst.read_plate(s),  # type: ignore[misc]
                retry_count=1,
            ),
        ]

        return Workflow(
            steps=steps,
            name="plate_read_fluorescence",
            submitted_by=self.agent_id,
            priority=3,
        )

    def _plan_pcr_then_read(self, available_instruments: dict[str, Any]) -> Workflow:
        """
        Sequential PCR + plate read: run PCR to completion, then read the plate.

        The read step declares depends_on=[1] (wait_for_completion) to ensure
        the thermocycler has finished before the plate is moved to the reader.

        In a real workflow, a liquid handling step would move the plate between
        instruments; here we assume manual transfer for simplicity.
        """
        tc_id, _ = self._require_instrument(available_instruments, Thermocycler, "Thermocycler")
        pr_id, _ = self._require_instrument(available_instruments, PlateReader, "PlateReader")

        program = STANDARD_PROGRAMS["pcr_standard"]
        read_spec = ReadSpec(
            plate_id="plate_001",
            mode=ReadMode.FLUORESCENCE,
            excitation_nm=485,
            emission_nm=528,
        )

        steps = [
            WorkflowStep(
                instrument_id=tc_id,
                name="start_pcr",
                operation=lambda inst, p=program: inst.run_program(p),  # type: ignore[misc]
                retry_count=1,
            ),
            WorkflowStep(
                instrument_id=tc_id,
                name="wait_for_pcr",
                operation=lambda inst: inst.wait_for_completion(poll_interval_seconds=0.05),
                depends_on=[0],
            ),
            WorkflowStep(
                instrument_id=pr_id,
                name="read_post_pcr",
                operation=lambda inst, s=read_spec: inst.read_plate(s),  # type: ignore[misc]
                depends_on=[1],
                retry_count=1,
            ),
        ]

        return Workflow(
            steps=steps,
            name="pcr_then_read",
            submitted_by=self.agent_id,
            priority=7,
        )

    def _plan_transfer_samples(self, available_instruments: dict[str, Any]) -> Workflow:
        """
        Sample transfer: move 50 µL from reservoir to wells A1-A4.
        """
        lh_id, _ = self._require_instrument(available_instruments, LiquidHandler, "LiquidHandler")

        transfer_specs = [
            TransferSpec(
                volume_ul=50.0,
                source="reservoir_1",
                destination=f"A{col}",
                tip_index=col - 1,
                mix_cycles=3,
            )
            for col in range(1, 5)
        ]

        steps = [
            WorkflowStep(
                instrument_id=lh_id,
                name=f"transfer_to_A{col}",
                operation=lambda inst, spec=spec: inst.transfer(spec),  # type: ignore[misc]
                depends_on=[col - 2] if col > 1 else [],
                retry_count=1,
            )
            for col, spec in enumerate(transfer_specs, start=1)
        ]

        return Workflow(
            steps=steps,
            name="transfer_samples",
            submitted_by=self.agent_id,
            priority=4,
        )
