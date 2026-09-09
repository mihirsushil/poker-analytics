"""Model clients for the eval harness. Each wraps one provider's native tool-use API and
returns a single ToolCall (or a no-tool-call result) for one scenario prompt.

Provider SDKs are imported lazily inside each client's __init__, not at module load time,
so `import poker_analytics.eval.clients` never requires anthropic/openai to be installed
unless that specific client is actually instantiated - the scoring/aggregation logic and
its tests never need either package.
"""
from __future__ import annotations

from typing import Protocol

from .schema import ToolCall
from .tool_specs import TOOL_SPECS

SYSTEM_PROMPT = (
    "You are the tool-routing layer for a poker hand-history analytics assistant. Given the "
    "user's question, either call exactly one of the available tools, or - if no tool "
    "genuinely helps answer the question - respond with plain text and call no tool at all. "
    "Never call a tool just because one is available."
)


class ModelClient(Protocol):
    model_name: str

    def select_tool(self, prompt: str) -> ToolCall:
        """Send `prompt` to the model with TOOL_SPECS available and return what it did."""
        ...


class AnthropicModelClient:
    def __init__(self, model_name: str) -> None:
        import anthropic  # noqa: PLC0415 - deliberately lazy, see module docstring

        self.model_name = model_name
        self._client = anthropic.Anthropic()
        self._tools = [
            {"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
            for t in TOOL_SPECS
        ]

    def select_tool(self, prompt: str) -> ToolCall:
        response = self._client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=self._tools,
            messages=[{"role": "user", "content": prompt}],
        )
        for block in response.content:
            if block.type == "tool_use":
                return ToolCall(tool_name=block.name, args=dict(block.input))
        return ToolCall(tool_name=None, args={})


class OpenAIModelClient:
    def __init__(self, model_name: str) -> None:
        import openai  # noqa: PLC0415 - deliberately lazy, see module docstring

        self.model_name = model_name
        self._client = openai.OpenAI()
        self._tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in TOOL_SPECS
        ]

    def select_tool(self, prompt: str) -> ToolCall:
        import json  # noqa: PLC0415

        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            tools=self._tools,
        )
        message = response.choices[0].message
        if message.tool_calls:
            call = message.tool_calls[0]
            return ToolCall(tool_name=call.function.name, args=json.loads(call.function.arguments))
        return ToolCall(tool_name=None, args={})


PROVIDER_PREFIXES: dict[str, type] = {
    "claude-": AnthropicModelClient,
    "gpt-": OpenAIModelClient,
    "o1": OpenAIModelClient,
    "o3": OpenAIModelClient,
}


def make_client(model_name: str) -> ModelClient:
    for prefix, client_cls in PROVIDER_PREFIXES.items():
        if model_name.startswith(prefix):
            return client_cls(model_name)
    raise ValueError(
        f"unrecognized model {model_name!r} - add its prefix to PROVIDER_PREFIXES in clients.py"
    )
