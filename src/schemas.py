"""API contract and the internal evidence format.

The evidence format is the load-bearing idea in this agent:

    ToolResult.brain_view()  -> compact text for Qwen (4096-token context)
    ToolResult.evidence()    -> full structured facts for fine-tuned Nemotron

Facts carry a pre-rendered ``text`` string. The synthesis model composes
sentences from those strings rather than reformatting raw numbers, which is what
keeps counts, dates, signs, and units exact in the graded answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# HTTP contract (Participant_Package/validate.json)
# --------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str = Field(min_length=1)


class ToolTraceEntry(BaseModel):
    tool: str
    args: dict[str, Any]
    result: str


class QueryResponse(BaseModel):
    answer: str = Field(min_length=1)
    steps: int = Field(ge=0)
    tool_trace: list[ToolTraceEntry] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    datasets_loaded: bool
    domain_predict_mode: str


# --------------------------------------------------------------------------
# Internal evidence format
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Fact:
    """One graded component of an answer, pre-formatted for the synthesis model."""

    label: str
    value: Any
    text: str

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, "value": self.value, "text": self.text}


@dataclass
class ToolResult:
    tool: str
    args: dict[str, Any]
    summary: str
    facts: list[Fact] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def brain_view(self, char_budget: int = 700) -> str:
        """Compact rendering for the Qwen planning context.

        Never includes article bodies or long listings - those stay in
        :meth:`evidence` and reach the synthesis model directly.
        """
        if self.error:
            return f"ERROR: {self.error}"
        lines = [self.summary]
        for note in self.notes:
            lines.append(f"NOTE: {note}")
        text = "\n".join(lines)
        if len(text) > char_budget:
            text = text[: char_budget - 3].rstrip() + "..."
        return text

    def evidence(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "tool": self.tool,
            "args": self.args,
            "summary": self.summary,
            "facts": [f.as_dict() for f in self.facts],
        }
        if self.notes:
            payload["notes"] = self.notes
        if self.detail:
            payload["detail"] = self.detail
        if self.error:
            payload["error"] = self.error
        return payload

    def trace_entry(self) -> ToolTraceEntry:
        return ToolTraceEntry(
            tool=self.tool,
            args=self.args,
            result=self.brain_view(char_budget=400),
        )

    @classmethod
    def failure(cls, tool: str, args: dict[str, Any], message: str) -> "ToolResult":
        return cls(tool=tool, args=args, summary=f"ERROR: {message}", error=message)


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)
