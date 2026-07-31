"""The agent state graph.

    question -> plan -> act -> review(=plan) -> ... -> synthesize -> answer

Nodes are pure ``AgentState -> AgentState`` functions with an explicit router,
so the control flow is a real state machine rather than a tangle of conditions.
Everything is deadline-aware: the graph will always reach ``synthesize`` and
return an answer, because a missing answer scores zero.

Model roles are fixed by the challenge rules and enforced here:
  * ``plan``      - Qwen only. Chooses tools, never states facts.
  * ``act``       - application code only. Executes and validates tool calls.
  * ``review``    - Qwen again, judging whether the results answer the question.
  * ``synthesize``- fine-tuned Nemotron only. Writes the graded answer.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .brain import BrainClient, BrainTurn, ToolCallRequest, fallback_plan
from .config import Settings, get_settings
from .prompts import BRAIN_REVIEW, BRAIN_SYSTEM
from .schemas import QueryResponse, ToolResult, ToolTraceEntry
from .synth import Synthesizer
from .tools import run_tool

log = logging.getLogger(__name__)

#: Leave enough runway to still synthesize an answer after the last tool call.
SYNTHESIS_RESERVE_S = 12.0
#: Below this, skip another planning turn and answer with what we have.
PLANNING_RESERVE_S = 18.0


@dataclass
class AgentState:
    question: str
    started: float
    deadline: float
    messages: list[dict[str, Any]] = field(default_factory=list)
    results: list[ToolResult] = field(default_factory=list)
    tool_trace: list[ToolTraceEntry] = field(default_factory=list)
    pending_calls: list[ToolCallRequest] = field(default_factory=list)
    brain_turns: int = 0
    tool_calls_made: int = 0
    used_fallback_planner: bool = False
    brain_error: str | None = None
    answer: str = ""
    synth_mode: str = ""
    path: list[str] = field(default_factory=list)

    @property
    def remaining(self) -> float:
        return self.deadline - time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    @property
    def steps(self) -> int:
        """Reported in the response: tool calls plus the synthesis step."""
        return self.tool_calls_made + 1


class AgentGraph:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.brain = BrainClient(self.settings)
        self.synthesizer = Synthesizer(self.settings)
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool")

    # -- nodes -----------------------------------------------------------
    def plan(self, state: AgentState) -> AgentState:
        """Qwen decides which tools to call, or reviews results and stops."""
        state.path.append("review" if state.brain_turns else "plan")
        budget = min(self.settings.brain_timeout_s, max(state.remaining - SYNTHESIS_RESERVE_S, 1.0))
        turn: BrainTurn = self.brain.plan(state.messages, timeout=budget)
        state.brain_turns += 1

        if turn.used_fallback:
            state.brain_error = turn.error
            if state.results:
                state.pending_calls = []  # already have evidence; stop looping
                return state
            state.used_fallback_planner = True
            state.pending_calls = fallback_plan(state.question)
            state.path.append("fallback_planner")
            return state

        state.messages.append(
            {
                "role": "assistant",
                "content": turn.content or None,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": _json_args(call.args)},
                    }
                    for call in turn.tool_calls
                ]
                or None,
            }
        )
        state.pending_calls = turn.tool_calls
        return state

    def act(self, state: AgentState) -> AgentState:
        """Application code executes the requested calls. Models never do."""
        state.path.append("act")
        calls = state.pending_calls
        state.pending_calls = []

        futures = [self._pool.submit(run_tool, call.name, call.args) for call in calls]
        for call, future in zip(calls, futures):
            try:
                result = future.result(timeout=max(state.remaining - 2.0, 1.0))
            except Exception as exc:
                result = ToolResult.failure(call.name, call.args, f"{type(exc).__name__}: {exc}")
            state.results.append(result)
            state.tool_trace.append(result.trace_entry())
            state.tool_calls_made += 1

            if not state.used_fallback_planner:
                state.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result.brain_view(self.settings.tool_result_char_budget),
                    }
                )

        if not state.used_fallback_planner:
            state.messages.append({"role": "user", "content": BRAIN_REVIEW})
        return state

    def synthesize(self, state: AgentState) -> AgentState:
        """Fine-tuned Nemotron writes the graded answer from verified facts."""
        state.path.append("synthesize")
        budget = min(self.settings.synth_timeout_s, max(state.remaining - 2.0, 3.0))
        state.answer, state.synth_mode = self.synthesizer.write(
            state.question, state.results, timeout=budget
        )
        return state

    # -- router ----------------------------------------------------------
    def _next(self, state: AgentState) -> str:
        if state.pending_calls:
            return "act"
        if state.used_fallback_planner or state.brain_error:
            return "synthesize"
        if state.brain_turns >= self.settings.max_agent_steps:
            return "synthesize"
        if state.remaining < PLANNING_RESERVE_S:
            return "synthesize"
        if state.brain_turns == 0:
            return "plan"
        return "synthesize"  # the review turn returned no further calls

    # -- driver ----------------------------------------------------------
    def run(self, question: str) -> tuple[QueryResponse, AgentState]:
        state = AgentState(
            question=question,
            started=time.monotonic(),
            deadline=time.monotonic() + self.settings.agent_deadline_s,
            messages=[
                {
                    "role": "system",
                    "content": BRAIN_SYSTEM.format(max_steps=self.settings.max_agent_steps),
                },
                {"role": "user", "content": question},
            ],
        )

        node = "plan"
        while node != "synthesize":
            if node == "plan":
                state = self.plan(state)
            elif node == "act":
                state = self.act(state)
            node = self._next(state)
        state = self.synthesize(state)

        return (
            QueryResponse(
                answer=state.answer or "No answer could be produced for this question.",
                steps=state.steps,
                tool_trace=state.tool_trace,
            ),
            state,
        )


def _json_args(args: dict[str, Any]) -> str:
    import json

    return json.dumps(args, ensure_ascii=False)
