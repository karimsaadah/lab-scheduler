"""
Unit tests for lab_scheduler.pipeline.planning_agent.

All tests are offline — no network calls, no API key required.
The LLM path (_call_anthropic) is patched where tested.
"""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from lab_scheduler.instruments.liquid_handler import LiquidHandler
from lab_scheduler.instruments.plate_reader import PlateReader
from lab_scheduler.instruments.thermocycler import Thermocycler
from lab_scheduler.pipeline.planning_agent import PlanningAgent, PlanningError
from lab_scheduler.scheduler.workflow import Workflow, WorkflowStatus

# ---------------------------------------------------------------------------
# Helpers — lightweight mock instruments (no hardware needed)
# ---------------------------------------------------------------------------


def _make_tc(iid: str = "tc1") -> tuple[str, MagicMock]:
    inst = MagicMock(spec=Thermocycler)
    return iid, inst


def _make_pr(iid: str = "pr1") -> tuple[str, MagicMock]:
    inst = MagicMock(spec=PlateReader)
    return iid, inst


def _make_lh(iid: str = "lh1") -> tuple[str, MagicMock]:
    inst = MagicMock(spec=LiquidHandler)
    return iid, inst


def _instruments(*pairs: tuple[str, MagicMock]) -> dict:
    return dict(pairs)


# ---------------------------------------------------------------------------
# TestRuleBasedPlanner
# ---------------------------------------------------------------------------


class TestRuleBasedPlanner:
    def setup_method(self):
        self.agent = PlanningAgent(agent_id="test_agent", api_key="")

    def test_run_pcr_returns_workflow(self):
        instruments = _instruments(_make_tc())
        wf = self.agent.plan("run_pcr", instruments)
        assert isinstance(wf, Workflow)
        assert wf.name == "pcr_standard"

    def test_run_pcr_has_two_steps(self):
        instruments = _instruments(_make_tc())
        wf = self.agent.plan("run_pcr", instruments)
        assert len(wf.steps) == 2
        assert wf.steps[0].name == "start_pcr_program"
        assert wf.steps[1].name == "wait_for_pcr_completion"

    def test_run_pcr_step2_depends_on_step0(self):
        instruments = _instruments(_make_tc())
        wf = self.agent.plan("run_pcr", instruments)
        assert wf.steps[1].depends_on == [0]

    def test_run_pcr_missing_thermocycler_raises(self):
        with pytest.raises(PlanningError, match="Thermocycler"):
            self.agent.plan("run_pcr", _instruments(_make_pr()))

    def test_read_plate_returns_workflow(self):
        instruments = _instruments(_make_pr())
        wf = self.agent.plan("read_plate", instruments)
        assert isinstance(wf, Workflow)
        assert wf.name == "plate_read_fluorescence"

    def test_read_plate_has_one_step(self):
        instruments = _instruments(_make_pr())
        wf = self.agent.plan("read_plate", instruments)
        assert len(wf.steps) == 1
        assert wf.steps[0].name == "read_fluorescence"

    def test_read_plate_missing_plate_reader_raises(self):
        with pytest.raises(PlanningError, match="PlateReader"):
            self.agent.plan("read_plate", _instruments(_make_tc()))

    def test_pcr_then_read_returns_workflow(self):
        instruments = _instruments(_make_tc(), _make_pr())
        wf = self.agent.plan("pcr_then_read", instruments)
        assert wf.name == "pcr_then_read"

    def test_pcr_then_read_has_three_steps(self):
        instruments = _instruments(_make_tc(), _make_pr())
        wf = self.agent.plan("pcr_then_read", instruments)
        assert len(wf.steps) == 3

    def test_pcr_then_read_dependencies(self):
        instruments = _instruments(_make_tc(), _make_pr())
        wf = self.agent.plan("pcr_then_read", instruments)
        assert wf.steps[1].depends_on == [0]  # wait depends on start
        assert wf.steps[2].depends_on == [1]  # read depends on wait

    def test_transfer_samples_returns_workflow(self):
        instruments = _instruments(_make_lh())
        wf = self.agent.plan("transfer_samples", instruments)
        assert wf.name == "transfer_samples"

    def test_transfer_samples_has_four_steps(self):
        instruments = _instruments(_make_lh())
        wf = self.agent.plan("transfer_samples", instruments)
        assert len(wf.steps) == 4

    def test_transfer_samples_chained_dependencies(self):
        instruments = _instruments(_make_lh())
        wf = self.agent.plan("transfer_samples", instruments)
        assert wf.steps[0].depends_on == []
        assert wf.steps[1].depends_on == [0]
        assert wf.steps[2].depends_on == [1]
        assert wf.steps[3].depends_on == [2]

    def test_goal_is_case_insensitive(self):
        instruments = _instruments(_make_pr())
        wf = self.agent.plan("  Read_Plate  ", instruments)
        assert wf.name == "plate_read_fluorescence"

    def test_submitted_by_uses_agent_id(self):
        instruments = _instruments(_make_pr())
        wf = self.agent.plan("read_plate", instruments)
        assert wf.submitted_by == "test_agent"

    def test_workflow_starts_pending(self):
        instruments = _instruments(_make_pr())
        wf = self.agent.plan("read_plate", instruments)
        assert wf.status == WorkflowStatus.PENDING


# ---------------------------------------------------------------------------
# TestMockLlmPlanner  (no API key → mock keyword planner)
# ---------------------------------------------------------------------------


class TestMockLlmPlanner:
    def setup_method(self):
        # No api_key → always uses mock planner for unknown goals
        self.agent = PlanningAgent(agent_id="test_agent", api_key="")

    def test_pcr_keyword_triggers_thermocycler_steps(self):
        instruments = _instruments(_make_tc())
        wf = self.agent.plan("amplify my DNA samples", instruments)
        names = [s.name for s in wf.steps]
        assert "start_pcr_program" in names
        assert "wait_for_pcr_completion" in names

    def test_fluorescence_keyword_triggers_plate_read(self):
        instruments = _instruments(_make_pr())
        wf = self.agent.plan("measure GFP fluorescence", instruments)
        assert len(wf.steps) == 1
        assert wf.steps[0].name == "read_plate"

    def test_absorbance_keyword_triggers_absorbance_read(self):
        instruments = _instruments(_make_pr())
        wf = self.agent.plan("measure absorbance at 600 nm", instruments)
        assert len(wf.steps) == 1

    def test_transfer_keyword_triggers_liquid_handler(self):
        instruments = _instruments(_make_lh(), _make_pr())
        wf = self.agent.plan("pipet samples into wells", instruments)
        assert wf.steps[0].name == "transfer_samples"

    def test_no_keyword_fallback_to_fluorescence(self):
        instruments = _instruments(_make_pr())
        wf = self.agent.plan("do something unrecognized", instruments)
        assert len(wf.steps) == 1
        assert wf.steps[0].name == "read_plate"

    def test_no_keyword_no_plate_reader_raises(self):
        instruments = _instruments(_make_tc())
        with pytest.raises(PlanningError, match="no PlateReader"):
            self.agent.plan("do something unrecognized", instruments)

    def test_compound_goal_pcr_then_read(self):
        instruments = _instruments(_make_tc(), _make_pr())
        wf = self.agent.plan("run PCR then measure fluorescence", instruments)
        names = [s.name for s in wf.steps]
        assert "start_pcr_program" in names
        assert "read_plate" in names

    def test_workflow_name_derived_from_goal(self):
        instruments = _instruments(_make_pr())
        wf = self.agent.plan("measure GFP fluorescence", instruments)
        assert wf.name.startswith("mock_llm_")


# ---------------------------------------------------------------------------
# TestLlmPlannerWithApiKey  (patches _call_anthropic)
# ---------------------------------------------------------------------------


class TestLlmPlannerWithApiKey:
    def setup_method(self):
        self.agent = PlanningAgent(agent_id="test_agent", api_key="sk-ant-fake")

    def _fake_response(self, plan: dict) -> str:
        return json.dumps(plan)

    def test_valid_llm_plan_builds_workflow(self):
        plan = {
            "workflow_name": "test_wf",
            "priority": 5,
            "steps": [
                {
                    "instrument_type": "PlateReader",
                    "name": "read_step",
                    "operation": "read_plate",
                    "params": {
                        "plate_id": "p1",
                        "mode": "FLUORESCENCE",
                        "excitation_nm": 485,
                        "emission_nm": 528,
                        "gain": "auto",
                    },
                    "depends_on": [],
                    "retry_count": 1,
                    "timeout_seconds": None,
                }
            ],
        }
        with patch.object(self.agent, "_call_anthropic", return_value=self._fake_response(plan)):
            wf = self.agent.plan("novel goal", _instruments(_make_pr()))
        assert wf.name == "test_wf"
        assert len(wf.steps) == 1

    def test_llm_returns_error_field_raises(self):
        plan = {"error": "No suitable instrument"}
        with (
            patch.object(self.agent, "_call_anthropic", return_value=self._fake_response(plan)),
            pytest.raises(PlanningError, match="No suitable instrument"),
        ):  # noqa: E501
            self.agent.plan("impossible goal", _instruments(_make_pr()))

    def test_llm_returns_invalid_json_raises(self):
        with (
            patch.object(self.agent, "_call_anthropic", return_value="not json {{"),
            pytest.raises(PlanningError, match="invalid JSON"),
        ):  # noqa: E501
            self.agent.plan("novel goal", _instruments(_make_pr()))

    def test_llm_plan_empty_steps_raises(self):
        plan = {"workflow_name": "empty", "priority": 5, "steps": []}
        with (
            patch.object(self.agent, "_call_anthropic", return_value=self._fake_response(plan)),
            pytest.raises(PlanningError, match="no steps"),
        ):  # noqa: E501
            self.agent.plan("novel goal", _instruments(_make_pr()))

    def test_llm_plan_unknown_instrument_type_raises(self):
        plan = {
            "workflow_name": "bad_wf",
            "priority": 5,
            "steps": [
                {
                    "instrument_type": "Centrifuge",
                    "name": "spin",
                    "operation": "read_plate",
                    "params": {},
                    "depends_on": [],
                    "retry_count": 0,
                    "timeout_seconds": None,
                }
            ],
        }
        with (
            patch.object(self.agent, "_call_anthropic", return_value=self._fake_response(plan)),
            pytest.raises(PlanningError, match="unknown instrument type"),
        ):  # noqa: E501
            self.agent.plan("novel goal", _instruments(_make_pr()))

    def test_llm_plan_unknown_operation_raises(self):
        plan = {
            "workflow_name": "bad_op",
            "priority": 5,
            "steps": [
                {
                    "instrument_type": "PlateReader",
                    "name": "s",
                    "operation": "spin_down",
                    "params": {},
                    "depends_on": [],
                    "retry_count": 0,
                    "timeout_seconds": None,
                }
            ],
        }
        with (
            patch.object(self.agent, "_call_anthropic", return_value=self._fake_response(plan)),
            pytest.raises(PlanningError, match="unknown operation"),
        ):  # noqa: E501
            self.agent.plan("novel goal", _instruments(_make_pr()))

    def test_http_error_raises_planning_error(self):
        err = urllib.error.HTTPError(
            url="https://api.anthropic.com",
            code=401,
            msg="Unauthorized",
            hdrs=None,  # type: ignore[arg-type]
            fp=MagicMock(read=lambda: b"bad key"),
        )
        with (
            patch("urllib.request.urlopen", side_effect=err),
            pytest.raises(PlanningError, match="Anthropic API error 401"),
        ):  # noqa: E501
            self.agent.plan("novel goal", _instruments(_make_pr()))

    def test_url_error_raises_planning_error(self):
        err = urllib.error.URLError(reason="Name or service not known")
        with (
            patch("urllib.request.urlopen", side_effect=err),
            pytest.raises(PlanningError, match="Network error"),
        ):  # noqa: E501
            self.agent.plan("novel goal", _instruments(_make_pr()))

    def test_llm_plan_thermocycler_run_program(self):
        plan = {
            "workflow_name": "pcr_wf",
            "priority": 8,
            "steps": [
                {
                    "instrument_type": "Thermocycler",
                    "name": "run",
                    "operation": "run_program",
                    "params": {"program_name": "pcr_standard"},
                    "depends_on": [],
                    "retry_count": 1,
                    "timeout_seconds": None,
                },
                {
                    "instrument_type": "Thermocycler",
                    "name": "wait",
                    "operation": "wait_for_completion",
                    "params": {"poll_interval_seconds": 1.0, "timeout_seconds": 3600.0},
                    "depends_on": [0],
                    "retry_count": 0,
                    "timeout_seconds": 3700.0,
                },
            ],
        }
        with patch.object(self.agent, "_call_anthropic", return_value=self._fake_response(plan)):
            wf = self.agent.plan("novel goal", _instruments(_make_tc()))
        assert len(wf.steps) == 2
        assert wf.priority == 8

    def test_llm_plan_transfer_operation(self):
        plan = {
            "workflow_name": "transfer_wf",
            "priority": 3,
            "steps": [
                {
                    "instrument_type": "LiquidHandler",
                    "name": "transfer",
                    "operation": "transfer",
                    "params": {
                        "volume_ul": 25.0,
                        "source": "src",
                        "destination": "A1",
                        "tip_index": 0,
                        "mix_cycles": 2,
                    },
                    "depends_on": [],
                    "retry_count": 1,
                    "timeout_seconds": None,
                },
            ],
        }
        with patch.object(self.agent, "_call_anthropic", return_value=self._fake_response(plan)):
            wf = self.agent.plan("novel goal", _instruments(_make_lh()))
        assert wf.steps[0].name == "transfer"

    def test_llm_plan_missing_instrument_raises(self):
        plan = {
            "workflow_name": "wf",
            "priority": 5,
            "steps": [
                {
                    "instrument_type": "Thermocycler",
                    "name": "s",
                    "operation": "run_program",
                    "params": {"program_name": "pcr_standard"},
                    "depends_on": [],
                    "retry_count": 0,
                    "timeout_seconds": None,
                }
            ],
        }
        with (
            patch.object(self.agent, "_call_anthropic", return_value=self._fake_response(plan)),
            pytest.raises(PlanningError, match="Thermocycler"),
        ):  # noqa: E501
            self.agent.plan("novel goal", _instruments(_make_pr()))
