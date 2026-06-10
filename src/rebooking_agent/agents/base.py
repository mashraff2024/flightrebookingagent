"""Base class shared by the four worker agents."""

from __future__ import annotations

from pydantic import BaseModel

from ..llm import LLMClient
from ..models import State
from ..tools import Toolbox


class Agent:
    name: str = "agent"
    system_prompt: str = ""

    def __init__(self, llm: LLMClient, toolbox: Toolbox):
        self.llm = llm
        self.toolbox = toolbox

    def _reason(self, state: State, *, task: str, user: str,
                context: dict, schema: type[BaseModel]) -> BaseModel:
        """Single funnel for every LLM call -> keeps llm_calls accounting honest."""
        from ..observability import TRACER
        state.llm_calls += 1
        TRACER.dump(f"reasoning input \u2192 {task}", context)
        result = self.llm.structured(
            task=task, system=self.system_prompt, user=user,
            context=context, schema=schema)
        TRACER.dump(f"reasoning output \u2190 {task}", result)
        return result
