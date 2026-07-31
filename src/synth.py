"""Fine-tuned Nemotron `domain-ft` client: final grounded answer synthesis.

The synthesis model receives the question plus verified tool facts and writes
the graded ``answer``. It never sees raw data and never chooses tools.

Two safeguards matter here:

* ``detailed thinking off`` - Nemotron-Nano emits a reasoning monologue by
  default, and a leaked ``<think>`` block in the answer field scores zero.
  Any residual block is stripped anyway.
* a deterministic composer - if the model is unreachable, times out, or returns
  something unusable, the facts are stitched into an answer directly. A missing
  answer scores zero, so this path must never be reachable-but-broken.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .config import Settings, get_settings
from .prompts import SYNTH_SENTIMENT_HINT, SYNTH_SYSTEM, SYNTH_USER
from .schemas import ToolResult, dumps

log = logging.getLogger(__name__)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_OPEN_THINK = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)
_SENTIMENT_HINT_WORDS = ("sentiment", "positive", "negative", "mixed", "direction", "impact")

MODE_LLM = "llm"
MODE_MOCK = "mock"
MODE_FALLBACK = "fallback"


def strip_reasoning(text: str) -> str:
    text = _THINK_BLOCK.sub("", text)
    text = _OPEN_THINK.sub("", text)
    return text.strip()


def needs_sentiment(question: str) -> bool:
    lowered = question.lower()
    return any(word in lowered for word in _SENTIMENT_HINT_WORDS)


def render_evidence(results: list[ToolResult], max_article_chars: int = 6000) -> str:
    """Compact, ordered evidence bundle for the synthesis prompt."""
    payload = []
    for result in results:
        item = result.evidence()
        detail = item.get("detail") or {}
        article = detail.get("article_text")
        if article:
            item["detail"] = {
                key: value for key, value in detail.items() if key != "article_text"
            }
            item["article_text"] = article[:max_article_chars]
        else:
            item.pop("detail", None)
        payload.append(item)
    return dumps(payload)


def compose(results: list[ToolResult]) -> str:
    """Deterministic answer built straight from the verified facts."""
    statements: list[str] = []
    seen: set[str] = set()
    for result in results:
        for fact in result.facts:
            text = fact.text.strip().rstrip(".")
            if text and text.lower() not in seen:
                seen.add(text.lower())
                statements.append(text)

    if not statements:
        for result in results:
            text = (result.summary or "").strip().rstrip(".")
            if text and text.lower() not in seen:
                seen.add(text.lower())
                statements.append(text)

    limitations = [note for result in results for note in result.notes]
    if not statements:
        if limitations:
            return " ".join(limitations)
        return (
            "The supplied datasets do not contain enough evidence to answer this "
            "question, so no figure can be stated."
        )

    answer = "; ".join(statements)
    answer = answer[0].upper() + answer[1:]
    return answer + "."


class Synthesizer:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._client = None

    def _get_client(self):
        if self._client is None:
            import httpx
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.settings.litellm_base_url,
                api_key=self.settings.litellm_key,
                timeout=httpx.Timeout(self.settings.synth_timeout_s, connect=2.0),
                max_retries=0,
            )
        return self._client

    def write(
        self, question: str, results: list[ToolResult], timeout: float | None = None
    ) -> tuple[str, str]:
        """Return ``(answer, mode)`` - mode records which path produced it."""
        if not self.settings.uses_fine_tuned_model:
            return compose(results), MODE_MOCK

        user = SYNTH_USER.format(question=question, evidence=render_evidence(results))
        if needs_sentiment(question):
            user += SYNTH_SENTIMENT_HINT

        try:
            response = self._get_client().chat.completions.create(
                model=self.settings.domain_ft_model,
                messages=[
                    {"role": "system", "content": SYNTH_SYSTEM},
                    {"role": "user", "content": user},
                ],
                temperature=0.1,
                top_p=0.95,
                max_tokens=self.settings.synth_max_tokens,
                timeout=timeout or self.settings.synth_timeout_s,
            )
            answer = strip_reasoning(response.choices[0].message.content or "")
        except Exception as exc:
            log.warning("synthesis model unavailable (%s: %s); composing from facts",
                        type(exc).__name__, exc)
            return compose(results), MODE_FALLBACK

        if len(answer) < 8:
            log.warning("synthesis returned an unusable answer (%r); composing from facts", answer)
            return compose(results), MODE_FALLBACK
        return answer, MODE_LLM
