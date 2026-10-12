"""
AI Planning Agent — translates high-level experimental goals into executable Workflows.

This module is the integration point between an AI reasoning layer and the lab
scheduler's execution layer. The agent supports two planning modes:

1. Rule-based (deterministic): handles a fixed set of known protocols (PCR,
   standard plate reads) where LLM reasoning adds no value and reproducibility
   is paramount. No API key or network access required.

2. LLM-based (generative): for novel, free-text goals not covered by the
   rule-based planner, the agent calls the Anthropic Messages API using only
   Python's stdlib ``urllib`` — no third-party packages — to stay compatible
   with air-gapped instrument control networks. The LLM receives the goal and
   a structured description of available instruments, and returns a JSON plan
   that is validated and translated into WorkflowStep objects.

Dispatch logic (Option A — automatic fallback):
    Known goal string  →  rule-based planner (fast, deterministic, no API call)
    Unknown goal       →  LLM planner       (requires ANTHROPIC_API_KEY env var)

To use the LLM planner, set the environment variable before running:
    export ANTHROPIC_API_KEY=sk-ant-...

LLM output format (the model is instructed to return this JSON schema):
    {
      "workflow_name": "<short slug>",
      "priority": <int 1-10>,
      "steps": [
        {
          "instrument_type": "Thermocycler" | "PlateReader" | "LiquidHandler",
          "name": "<step name>",
          "operation": "run_program" | "wait_for_completion" | "read_plate" | "transfer",
          "params": { <operation-specific params> },
          "depends_on": [<step indices>],
          "retry_count": <int>,
          "timeout_seconds": <float | null>
        }
      ]
    }
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

from ..instruments.liquid_handler import LiquidHandler, TransferSpec
from ..instruments.plate_reader import PlateReader, ReadMode, ReadSpec
from ..instruments.thermocycler import STANDARD_PROGRAMS, Thermocycler
from ..scheduler.workflow import Workflow, WorkflowStep

logger = logging.getLogger(__name__)

# Anthropic API constants — no SDK needed
_ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_API_VERSION = "2023-06-01"
_MODEL = "claude-3-5-haiku-20241022"  # Fast, cheap; sufficient for structured planning

_SYSTEM_PROMPT = """\
You are a laboratory automation planner. Your job is to translate a researcher's
experimental goal into a JSON workflow plan that a lab scheduler can execute.

You will receive:
- A natural-language experimental goal
- A list of available instruments with their types and IDs

You must respond with ONLY valid JSON matching this exact schema (no markdown, no explanation):
{
  "workflow_name": "<short descriptive slug, e.g. 'pcr_then_read'>",
  "priority": <integer 1-10, higher = more urgent>,
  "steps": [
    {
      "instrument_type": "<one of: Thermocycler, PlateReader, LiquidHandler>",
      "name": "<short step name>",
      "operation": "<one of: run_program, wait_for_completion, read_plate, transfer>",
      "params": {<operation-specific parameters, see below>},
      "depends_on": [<list of 0-based step indices this step must wait for>],
      "retry_count": <integer, 0 = no retry>,
      "timeout_seconds": <float or null>
    }
  ]
}

Operation params schemas:
  run_program:          {"program_name": "<key in STANDARD_PROGRAMS, e.g. 'pcr_standard'>"}
  wait_for_completion:  {"poll_interval_seconds": <float>, "timeout_seconds": <float>}
  read_plate:           {"plate_id": "<str>", "mode": "<FLUORESCENCE|ABSORBANCE|LUMINESCENCE>",
                         "excitation_nm": <int|null>, "emission_nm": <int|null>,
                         "wavelength_nm": <int|null>, "gain": "<str, default 'auto'>"}
  transfer:             {"volume_ul": <float>, "source": "<well>", "destination": "<well>",
                         "tip_index": <int>, "mix_cycles": <int>}

Rules:
- Only use instrument_types that appear in the available instruments list.
- For Thermocycler: always pair run_program with a wait_for_completion step.
- wait_for_completion must depend_on the run_program step index.
- For unknown program names, default to 'pcr_standard'.
- Keep workflows minimal: only include steps necessary for the stated goal.
- If the goal is impossible with the available instruments, return:
  {"error": "<explanation>"}
"""


class PlanningError(Exception):
    """Raised when the planning agent cannot generate a valid workflow for the goal."""


class PlanningAgent:
    """
    AI planning agent that converts experimental goals into Workflows.

    The agent is stateless between plan() calls — it holds no memory of prior
    experiments. Experiment history and context injection are the responsibility
    of the caller (e.g., a higher-level campaign orchestrator).

    Dispatch:
        Known goals (run_pcr, read_plate, pcr_then_read, transfer_samples)
            → rule-based planner (deterministic, no API key needed)
        Unknown / free-text goals
            → LLM planner (requires ANTHROPIC_API_KEY environment variable)

    Args:
        agent_id: Identifier for this agent instance, recorded in submitted_by
                  on generated workflows.
        api_key:  Anthropic API key. Defaults to the ANTHROPIC_API_KEY env var.
                  Only needed for free-text goals not handled by the rule-based planner.
    """

    def __init__(
        self,
        agent_id: str = "planning_agent_v1",
        api_key: str | None = None,
    ) -> None:
        self.agent_id = agent_id
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._logger = logging.getLogger(f"{__name__}.{agent_id}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def plan(self, goal: str, available_instruments: dict[str, Any]) -> Workflow:
        """
        Generate a Workflow for the given experimental goal.

        Known goals are handled by the deterministic rule-based planner.
        Novel free-text goals are sent to the LLM planner.

        Args:
            goal: Human-readable description of what the experiment should achieve.
            available_instruments: Dict mapping instrument_id → Instrument objects.

        Returns:
            A Workflow ready for submission to LabScheduler.submit().

        Raises:
            PlanningError: If the goal cannot be planned, or the LLM returns an
                           invalid / impossible plan.
        """
        self._logger.info("Planning workflow for goal: '%s'", goal)
        self._logger.debug("Available instruments: %s", list(available_instruments.keys()))

        goal_normalized = goal.strip().lower()

        _KNOWN_GOALS = {"run_pcr", "read_plate", "pcr_then_read", "transfer_samples"}
        if goal_normalized in _KNOWN_GOALS:
            self._logger.debug("Goal '%s' matched rule-based planner", goal_normalized)
            return self._rule_based_plan(goal_normalized, available_instruments)

        self._logger.info(
            "Goal '%s' not in known protocols — delegating to LLM planner", goal
        )
        return self._llm_plan(goal, available_instruments)

    # ------------------------------------------------------------------
    # LLM planner
    # ------------------------------------------------------------------

    def _llm_plan(self, goal: str, available_instruments: dict[str, Any]) -> Workflow:
        """
        Generate a workflow plan for a novel free-text goal.

        If ANTHROPIC_API_KEY is set, calls the Anthropic Messages API (stdlib urllib,
        no third-party packages). Otherwise falls back to the keyword-based mock planner
        so development and testing work without API credentials.
        """
        if not self._api_key:
            self._logger.warning(
                "ANTHROPIC_API_KEY not set — using mock LLM planner for goal: '%s'", goal
            )
            return self._mock_llm_plan(goal, available_instruments)

        # Build instrument inventory description for the prompt
        instrument_descriptions = self._describe_instruments(available_instruments)
        user_message = (
            f"Goal: {goal}\n\n"
            f"Available instruments:\n{instrument_descriptions}"
        )

        self._logger.debug("Sending goal to LLM: %s", goal)
        raw_json = self._call_anthropic(user_message)
        self._logger.debug("LLM raw response: %s", raw_json)

        try:
            plan = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise PlanningError(
                f"LLM returned invalid JSON for goal '{goal}': {exc}\nRaw: {raw_json}"
            ) from exc

        if "error" in plan:
            raise PlanningError(
                f"LLM cannot plan goal '{goal}': {plan['error']}"
            )

        return self._build_workflow_from_plan(plan, available_instruments, goal)

    # ------------------------------------------------------------------
    # Mock LLM planner (keyword-based, no API key required)
    # ------------------------------------------------------------------

    def _mock_llm_plan(self, goal: str, available_instruments: dict[str, Any]) -> Workflow:
        """
        Keyword-based mock that simulates LLM planning without an API key.

        Inspects the goal string for instrument/operation keywords and assembles
        a realistic JSON plan, then routes it through the same
        _build_workflow_from_plan() path as a real LLM response — so the
        validation and WorkflowStep construction logic is exercised identically.

        Supported keyword patterns (case-insensitive):
          - any of: pcr, thermocycl, amplif, cycl  → run_program + wait_for_completion
          - any of: absorbance, od, optical density  → read_plate (ABSORBANCE, 600 nm)
          - any of: fluores, gfp, rfp, emit          → read_plate (FLUORESCENCE, 485/528 nm)
          - any of: luminesc, luciferase             → read_plate (LUMINESCENCE)
          - any of: transfer, pipet, dispense, liquid → transfer (50 µL reservoir→A1)
          - compound goals combine the above in order

        Falls back to a generic plate-read workflow if no keyword matches.
        """
        goal_lower = goal.lower()

        has_pcr = any(k in goal_lower for k in ("pcr", "thermocycl", "amplif", "cycl"))
        has_absorbance = any(k in goal_lower for k in ("absorbance", " od ", "optical density", "600"))
        has_fluorescence = any(k in goal_lower for k in ("fluores", "gfp", "rfp", "emit", "485", "528"))
        has_luminescence = any(k in goal_lower for k in ("luminesc", "luciferase"))
        has_transfer = any(k in goal_lower for k in ("transfer", "pipet", "dispense", "liquid"))

        # Determine which instrument types are actually available
        has_tc  = any(isinstance(i, Thermocycler)  for i in available_instruments.values())
        has_pr  = any(isinstance(i, PlateReader)   for i in available_instruments.values())
        has_lh  = any(isinstance(i, LiquidHandler) for i in available_instruments.values())

        steps: list[dict[str, Any]] = []
        dep_offset = 0  # tracks current step index for depends_on

        # 1. Optional transfer step
        if has_transfer and has_lh:
            steps.append({
                "instrument_type": "LiquidHandler",
                "name": "transfer_samples",
                "operation": "transfer",
                "params": {
                    "volume_ul": 50.0,
                    "source": "reservoir_1",
                    "destination": "A1",
                    "tip_index": 0,
                    "mix_cycles": 3,
                },
                "depends_on": [],
                "retry_count": 1,
                "timeout_seconds": None,
            })
            dep_offset += 1

        # 2. Optional PCR step (run + wait)
        if has_pcr and has_tc:
            steps.append({
                "instrument_type": "Thermocycler",
                "name": "start_pcr_program",
                "operation": "run_program",
                "params": {"program_name": "pcr_standard"},
                "depends_on": [dep_offset - 1] if dep_offset > 0 else [],
                "retry_count": 1,
                "timeout_seconds": None,
            })
            steps.append({
                "instrument_type": "Thermocycler",
                "name": "wait_for_pcr_completion",
                "operation": "wait_for_completion",
                "params": {"poll_interval_seconds": 0.05, "timeout_seconds": 7200.0},
                "depends_on": [dep_offset],
                "retry_count": 0,
                "timeout_seconds": 7300.0,
            })
            dep_offset += 2

        # 3. Optional plate read step
        if has_pr and (has_absorbance or has_fluorescence or has_luminescence or not has_pcr):
            if has_absorbance:
                read_params: dict[str, Any] = {
                    "plate_id": "plate_001",
                    "mode": "ABSORBANCE",
                    "wavelength_nm": 600,
                    "gain": "auto",
                }
            elif has_luminescence:
                read_params = {
                    "plate_id": "plate_001",
                    "mode": "LUMINESCENCE",
                    "gain": "auto",
                }
            else:  # default: fluorescence (GFP wavelengths)
                read_params = {
                    "plate_id": "plate_001",
                    "mode": "FLUORESCENCE",
                    "excitation_nm": 485,
                    "emission_nm": 528,
                    "gain": "auto",
                }
            steps.append({
                "instrument_type": "PlateReader",
                "name": "read_plate",
                "operation": "read_plate",
                "params": read_params,
                "depends_on": [dep_offset - 1] if dep_offset > 0 else [],
                "retry_count": 1,
                "timeout_seconds": None,
            })

        # Fallback: nothing matched — default to a fluorescence read if possible
        if not steps:
            self._logger.warning(
                "Mock planner: no keyword matched goal '%s'; defaulting to fluorescence read",
                goal,
            )
            if has_pr:
                steps.append({
                    "instrument_type": "PlateReader",
                    "name": "read_plate_default",
                    "operation": "read_plate",
                    "params": {
                        "plate_id": "plate_001",
                        "mode": "FLUORESCENCE",
                        "excitation_nm": 485,
                        "emission_nm": 528,
                        "gain": "auto",
                    },
                    "depends_on": [],
                    "retry_count": 1,
                    "timeout_seconds": None,
                })
            else:
                raise PlanningError(
                    f"Mock planner cannot plan goal '{goal}': no matching keywords and "
                    "no PlateReader available for a default fallback."
                )

        plan: dict[str, Any] = {
            "workflow_name": "mock_llm_" + "_".join(goal_lower.split()[:3]),
            "priority": 5,
            "steps": steps,
        }
        self._logger.info(
            "Mock LLM planner generated %d-step plan for goal: '%s'", len(steps), goal
        )
        return self._build_workflow_from_plan(plan, available_instruments, goal)

    def _describe_instruments(self, available_instruments: dict[str, Any]) -> str:
        """Produce a compact instrument inventory string for the LLM prompt."""
        lines = []
        for iid, inst in available_instruments.items():
            type_name = type(inst).__name__
            lines.append(f"  - id={iid!r}, type={type_name}")
        return "\n".join(lines) if lines else "  (none)"

    def _call_anthropic(self, user_message: str) -> str:
        """
        POST to the Anthropic Messages API using stdlib urllib.

        Returns the text content of the first response block.
        Raises PlanningError on HTTP errors or unexpected response shapes.
        """
        payload = json.dumps({
            "model": _MODEL,
            "max_tokens": 1024,
            "system": _SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_message}],
        }).encode()

        req = urllib.request.Request(
            _ANTHROPIC_API_URL,
            data=payload,
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": _ANTHROPIC_API_VERSION,
                "content-type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode(errors="replace")
            raise PlanningError(
                f"Anthropic API error {exc.code}: {error_body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise PlanningError(f"Network error calling Anthropic API: {exc}") from exc

        try:
            return body["content"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise PlanningError(
                f"Unexpected Anthropic API response shape: {body}"
            ) from exc

    def _build_workflow_from_plan(
        self,
        plan: dict[str, Any],
        available_instruments: dict[str, Any],
        goal: str,
    ) -> Workflow:
        """
        Translate the LLM JSON plan into a Workflow with concrete WorkflowSteps.

        Each step's 'operation' string is resolved to a callable that the
        scheduler will invoke with the actual instrument object.
        """
        raw_steps: list[dict[str, Any]] = plan.get("steps", [])
        if not raw_steps:
            raise PlanningError(f"LLM plan for '{goal}' contains no steps.")

        workflow_steps: list[WorkflowStep] = []

        for i, raw in enumerate(raw_steps):
            instrument_type_name: str = raw.get("instrument_type", "")
            operation_name: str = raw.get("operation", "")
            params: dict[str, Any] = raw.get("params", {})
            name: str = raw.get("name", f"step_{i}")
            depends_on: list[int] = raw.get("depends_on", [])
            retry_count: int = int(raw.get("retry_count", 0))
            timeout_seconds: float | None = raw.get("timeout_seconds")

            # Find a registered instrument of the requested type
            instrument_id, _ = self._find_instrument_by_type(
                available_instruments, instrument_type_name, goal, step_index=i
            )

            # Resolve operation string → callable
            operation = self._resolve_operation(
                operation_name, instrument_type_name, params, goal, step_index=i
            )

            workflow_steps.append(
                WorkflowStep(
                    instrument_id=instrument_id,
                    operation=operation,
                    name=name,
                    depends_on=depends_on,
                    retry_count=retry_count,
                    timeout_seconds=timeout_seconds,
                )
            )

        return Workflow(
            steps=workflow_steps,
            name=plan.get("workflow_name", "llm_planned_workflow"),
            submitted_by=self.agent_id,
            priority=int(plan.get("priority", 5)),
        )

    def _find_instrument_by_type(
        self,
        available_instruments: dict[str, Any],
        type_name: str,
        goal: str,
        step_index: int,
    ) -> tuple[str, Any]:
        """Return (instrument_id, instrument) for the first match of the given type name."""
        type_map: dict[str, type] = {
            "Thermocycler": Thermocycler,
            "PlateReader": PlateReader,
            "LiquidHandler": LiquidHandler,
        }
        target_type = type_map.get(type_name)
        if target_type is None:
            raise PlanningError(
                f"LLM plan step {step_index} references unknown instrument type "
                f"'{type_name}' for goal '{goal}'. Valid types: {list(type_map)}"
            )
        for iid, inst in available_instruments.items():
            if isinstance(inst, target_type):
                return iid, inst
        raise PlanningError(
            f"LLM plan step {step_index} requires a {type_name} but none is registered. "
            f"Available instruments: {list(available_instruments.keys())}"
        )

    def _resolve_operation(
        self,
        operation_name: str,
        instrument_type_name: str,
        params: dict[str, Any],
        goal: str,
        step_index: int,
    ) -> Any:
        """
        Translate an operation name + params dict into a callable (inst) → result.

        Each returned lambda captures its params by value (default-argument trick)
        to avoid the classic loop-closure bug.
        """
        if operation_name == "run_program":
            program_name = params.get("program_name", "pcr_standard")
            if program_name not in STANDARD_PROGRAMS:
                self._logger.warning(
                    "Unknown program '%s' in LLM plan step %d; falling back to 'pcr_standard'",
                    program_name,
                    step_index,
                )
                program_name = "pcr_standard"
            program = STANDARD_PROGRAMS[program_name]
            return lambda inst, p=program: inst.run_program(p)  # type: ignore[misc]

        if operation_name == "wait_for_completion":
            poll = float(params.get("poll_interval_seconds", 5.0))
            timeout = float(params.get("timeout_seconds", 7200.0))
            return lambda inst, pi=poll, to=timeout: inst.wait_for_completion(
                poll_interval_seconds=pi, timeout_seconds=to
            )

        if operation_name == "read_plate":
            mode_str = params.get("mode", "FLUORESCENCE").upper()
            try:
                mode = ReadMode[mode_str]
            except KeyError:
                self._logger.warning(
                    "Unknown ReadMode '%s' in LLM plan step %d; defaulting to FLUORESCENCE",
                    mode_str,
                    step_index,
                )
                mode = ReadMode.FLUORESCENCE
            spec = ReadSpec(
                plate_id=params.get("plate_id", "plate_001"),
                mode=mode,
                excitation_nm=params.get("excitation_nm"),
                emission_nm=params.get("emission_nm"),
                wavelength_nm=params.get("wavelength_nm"),
                gain=params.get("gain", "auto"),
            )
            return lambda inst, s=spec: inst.read_plate(s)  # type: ignore[misc]

        if operation_name == "transfer":
            spec = TransferSpec(
                volume_ul=float(params.get("volume_ul", 50.0)),
                source=params.get("source", "reservoir_1"),
                destination=params.get("destination", "A1"),
                tip_index=int(params.get("tip_index", 0)),
                mix_cycles=int(params.get("mix_cycles", 0)),
            )
            return lambda inst, s=spec: inst.transfer(s)  # type: ignore[misc]

        raise PlanningError(
            f"LLM plan step {step_index} specifies unknown operation '{operation_name}' "
            f"for goal '{goal}'. Valid operations: run_program, wait_for_completion, "
            f"read_plate, transfer."
        )

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
        handlers = {
            "run_pcr": self._plan_pcr,
            "read_plate": self._plan_read_plate,
            "pcr_then_read": self._plan_pcr_then_read,
            "transfer_samples": self._plan_transfer_samples,
        }

        handler = handlers.get(goal)
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
