"""Prompted-frontier-model comparison arm, gated on an API key being set.

Mirrors eval/clients.py's lazy-import pattern - importing this module never
requires anthropic/openai, only instantiating a client does. No live API
call is ever made unless the corresponding key is actually present in the
environment; benchmark.py checks is_configured() before including this arm
in a run, and this module is never invoked from a fork/session that hasn't
been told a key is present.
"""
from __future__ import annotations

import os

from .dataset import SYSTEM_PROMPT, format_prompt
from .extract import DecisionExample

PROVIDER_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def is_configured(provider: str = "anthropic") -> bool:
    env_var = PROVIDER_ENV_VARS.get(provider)
    return bool(env_var and os.environ.get(env_var))


class FrontierModelClient:
    """Predicts an action by prompting a real frontier model. Raises if the
    relevant API key isn't set rather than silently no-op'ing - callers
    should check is_configured() first (benchmark.py does)."""

    def __init__(self, model_name: str, provider: str = "anthropic") -> None:
        if not is_configured(provider):
            env_var = PROVIDER_ENV_VARS[provider]
            raise RuntimeError(
                f"{env_var} is not set - FrontierModelClient requires a real API key, "
                "see phase4/README.md for how to enable this comparison arm"
            )
        self.model_name = model_name
        self.provider = provider
        if provider == "anthropic":
            import anthropic  # noqa: PLC0415 - deliberately lazy

            self._client = anthropic.Anthropic()
        elif provider == "openai":
            import openai  # noqa: PLC0415

            self._client = openai.OpenAI()
        else:
            raise ValueError(f"unsupported provider {provider!r}")

    def predict(self, ex: DecisionExample) -> str:
        prompt = format_prompt(ex)
        if self.provider == "anthropic":
            response = self._client.messages.create(
                model=self.model_name,
                max_tokens=16,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in response.content if b.type == "text").strip()
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=16,
        )
        return (response.choices[0].message.content or "").strip()
