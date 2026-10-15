"""Stop sequences: cut generation at the end of a tool call (subject V.6 tip).

Without a stop sequence the model keeps writing after its tool call and hallucinates the tool's
output. Defaults are per prompt format and can be overridden through ``ModelConfig`` or per call.
This module is a leaf (it must not import ``agent_smith.core.config``).
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Iterable, Sequence

__all__ = [
    "DEFAULT_STOP_SEQUENCES",
    "MAX_PROVIDER_STOP_SEQUENCES",
    "PromptFormat",
    "normalize_for_provider",
    "resolve_stop_sequences",
    "truncate_at_stop",
]

logger = logging.getLogger(__name__)

# OpenAI-compatible APIs reject more than 4 stop sequences.
MAX_PROVIDER_STOP_SEQUENCES = 4


class PromptFormat(str, enum.Enum):
    PYTHON = "python"  # ```python blocks ended by <end_code>
    XML = "xml"  # <tool_call>...</tool_call>
    HERMES = "hermes"  # JSON tool calls inside <tool_call> tags
    REACT = "react"  # Thought / Action / Observation text


DEFAULT_STOP_SEQUENCES: dict[PromptFormat, tuple[str, ...]] = {
    PromptFormat.PYTHON: ("<end_code>",),
    PromptFormat.XML: ("</tool_call>",),
    PromptFormat.HERMES: ("</tool_call>",),
    PromptFormat.REACT: ("Observation:",),  # stop before the model invents the observation
}


def _dedupe(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(s for s in items if s))


def resolve_stop_sequences(
    fmt: PromptFormat | str,
    override: Sequence[str] | None = None,
    extra: Sequence[str] = (),
) -> list[str]:
    """Stop sequences for ``fmt``: ``override`` replaces the defaults, ``extra`` is appended."""
    if override is not None:
        return _dedupe([*override, *extra])
    return _dedupe([*DEFAULT_STOP_SEQUENCES[PromptFormat(fmt)], *extra])


def normalize_for_provider(stop: Sequence[str]) -> list[str]:
    """Dedupe and cap to what providers accept, keeping order and warning about dropped ones."""
    seqs = _dedupe(stop)
    if len(seqs) > MAX_PROVIDER_STOP_SEQUENCES:
        logger.warning(
            "Dropping stop sequences beyond the provider limit of %d: %s",
            MAX_PROVIDER_STOP_SEQUENCES,
            seqs[MAX_PROVIDER_STOP_SEQUENCES:],
        )
        seqs = seqs[:MAX_PROVIDER_STOP_SEQUENCES]
    return seqs


def truncate_at_stop(text: str, stop: Sequence[str]) -> tuple[str, str | None]:
    """Cut ``text`` at the earliest stop sequence (safety net for backends that ignore ``stop``).

    Returns the text before the match and the matched sequence, or ``(text, None)``.
    """
    best: tuple[int, str] | None = None
    for seq in stop:
        if not seq:
            continue
        idx = text.find(seq)
        if idx != -1 and (best is None or idx < best[0]):
            best = (idx, seq)
    if best is None:
        return text, None
    return text[: best[0]], best[1]
