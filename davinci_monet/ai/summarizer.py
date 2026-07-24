"""Build and run the single-prompt Claude summary for an analysis run."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from davinci_monet.ai.images import EncodedImage
from davinci_monet.ai.payload import SummaryPayload

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a climate and atmospheric-composition data-comparison analyst.
You are given the configuration, summary statistics, and figures from a single DAVINCI
analysis run. The run contains one or more comparisons; each lists the exact sections to
write, with a required format and a word budget.

For EACH comparison below, output a markdown block headed by the comparison name that
contains EXACTLY its listed sections, in order. Obey each section's stated format and stay
within its word budget. Be specific and quantitative. Do not invent numbers that are not
present in the provided statistics or visible in the figures."""


class SummaryError(Exception):
    """Raised when the summary cannot be produced (degraded non-fatally)."""


@dataclass
class SummaryResult:
    """Result of a successful summary generation."""

    markdown: str
    model: str
    usage: dict[str, Any]
    plots_used: list[str]
    images_sent: int
    credits_remaining: float | None = None


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


_BULLET_RE = re.compile(r"^\s*[-*•]\s+(.*\S)\s*$")
_SUBHEADING_RE = re.compile(r"^\s*#{2,6}\s+(.*\S)\s*$")
_OVERFLOW_ITEM = "… (full brief in AI_summary.md)"
_CODE_SPAN_RE = re.compile(r"`([^`]+)`")
_STRONG_EMPHASIS_RE = re.compile(r"(\*\*|__)(.+?)\1")
_STAR_EMPHASIS_RE = re.compile(r"(?<!\*)\*(?!\s|\*)(.+?)(?<!\s)(?<!\*)\*(?!\*)")
_UNDERSCORE_EMPHASIS_RE = re.compile(r"(?<!\w)_(?!\s|_)(.+?)(?<!\s)_(?!\w|_)")


def _strip_emphasis(text: str) -> str:
    text = _CODE_SPAN_RE.sub(r"\1", text)
    text = _STRONG_EMPHASIS_RE.sub(r"\2", text)
    text = _STAR_EMPHASIS_RE.sub(r"\1", text)
    text = _UNDERSCORE_EMPHASIS_RE.sub(r"\1", text)
    return text.strip()


def extract_bullets(markdown: str, *, max_items: int = 12) -> list[str]:
    """Condense a brief to an itemized list for terminal display.

    Returns every markdown bullet line (-, *, or bullet char) across all
    sections, marker stripped. If the brief has no bullets, falls back to the
    level-2+ section headings. Returns an empty list if neither is present.
    Caps at ``max_items``; when truncated, the final item points to the file.
    """
    lines = markdown.splitlines()
    items = [_strip_emphasis(m.group(1)) for line in lines if (m := _BULLET_RE.match(line))]
    if not items:
        items = [_strip_emphasis(m.group(1)) for line in lines if (m := _SUBHEADING_RE.match(line))]
    items = [item for item in items if item]
    if not items:
        return []
    if len(items) > max_items:
        items = items[: max_items - 1] + [_OVERFLOW_ITEM]
    return items


def render_text(payload: SummaryPayload) -> str:
    """Render the textual portion of the user message from the payload.

    Emits the run header, then one templated block per ``(pair, variable)``
    comparison: the comparison heading, its template's section instructions
    (heading + format + word budget), and that comparison's statistics. Each row
    carries a resolved ``template``; rows without one fall back to ``generic_eval``.
    """
    from davinci_monet.ai.templates import get_template_registry

    lines: list[str] = ["# Analysis run"]
    period = payload.period
    lines.append(f"Period: {period.get('start')} to {period.get('end')}")
    if payload.sources_summary:
        lines.append("Sources: " + ", ".join(payload.sources_summary))
    if payload.pairs_summary:
        lines.append("Pairs: " + ", ".join(payload.pairs_summary))

    if payload.stats_rows:
        for row in payload.stats_rows:
            template = row.get("template") or get_template_registry().get("generic_eval")
            lines.append("")
            lines.append(f"## {row['pair']} — {row['variable']}")
            lines.append("Write these sections, obeying each format and word budget:")
            for section in template.sections:
                lines.append(f"### {section.heading} — {section.format_instruction()}")
            metric_str = ", ".join(f"{k}={_fmt(v)}" for k, v in row["metrics"].items())
            lines.append(f"Statistics: {metric_str}")
    else:
        lines.append("")
        lines.append("## Statistics")
        lines.append("(no statistics available)")

    if payload.instructions:
        lines.append("")
        lines.append("## Additional instructions")
        lines.append(payload.instructions)

    return "\n".join(lines)


def build_prompt(
    payload: SummaryPayload,
    encoded_images: list[tuple[str, EncodedImage]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the (system, user_content) blocks for messages.create. Pure, no IO."""
    system: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]

    content: list[dict[str, Any]] = [{"type": "text", "text": render_text(payload)}]
    for caption, enc in encoded_images:
        content.append({"type": "text", "text": f"Figure: {caption}"})
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": enc.media_type,
                    "data": enc.data,
                },
            }
        )
    return system, content


def resolve_api_key(cfg: Any) -> str:
    """Resolve the API key from ``api_key_file`` (if set) else ``api_key_env``.

    Raises SummaryError if the file is unreadable/empty or no key is found.
    """
    if cfg.api_key_file:
        path = Path(os.path.expanduser(cfg.api_key_file))
        try:
            key = path.read_text().strip()
        except OSError as exc:
            raise SummaryError(f"could not read api_key_file '{cfg.api_key_file}': {exc}") from exc
        if not key:
            raise SummaryError(f"api_key_file '{cfg.api_key_file}' is empty")
        return key

    key = os.environ.get(cfg.api_key_env, "")
    if not key:
        raise SummaryError(
            f"API key not found: set env '{cfg.api_key_env}' or summary.api_key_file"
        )
    return key


def _build_client(cfg: Any) -> Any:
    """Construct a real Anthropic client (lazy import). Raises SummaryError."""
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - exercised via stage tests
        raise SummaryError(
            "anthropic package not installed; install with: pip install 'davinci[ai]'"
        ) from exc

    key = resolve_api_key(cfg)
    return anthropic.Anthropic(api_key=key)


def generate_summary(
    payload: SummaryPayload,
    *,
    cfg: Any,
    client: Any | None = None,
) -> SummaryResult:
    """Encode images, then dispatch to the configured provider.

    ``client`` is the injectable Anthropic client (used when
    ``cfg.provider == "anthropic"``). The OpenRouter path's injectable seam is
    ``openrouter._send_openrouter_request``.
    """
    from davinci_monet.ai.images import encode_image

    encoded: list[tuple[str, EncodedImage]] = []
    for img in payload.images:
        try:
            encoded.append((img.caption, encode_image(img.path)))
        except Exception as exc:  # noqa: BLE001 - bad figure must not abort summary
            logger.warning("Skipping figure %s: %s", img.path, exc)

    provider = getattr(cfg, "provider", "anthropic")
    if provider == "openrouter":
        from davinci_monet.ai.openrouter import call_openrouter  # lazy: avoid cycle

        return call_openrouter(SYSTEM_PROMPT, render_text(payload), encoded, cfg)

    return _call_anthropic(payload, encoded, cfg, client=client)


def _call_anthropic(
    payload: SummaryPayload,
    encoded: list[tuple[str, EncodedImage]],
    cfg: Any,
    *,
    client: Any | None = None,
) -> SummaryResult:
    """Call the Anthropic Messages API and return a SummaryResult."""
    if client is None:
        client = _build_client(cfg)

    system, content = build_prompt(payload, encoded)

    try:
        response = client.messages.create(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
        )
    except Exception as exc:  # noqa: BLE001 - any API/network failure degrades
        raise SummaryError(f"Claude API request failed: {exc}") from exc

    try:
        markdown = response.content[0].text
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
    except (AttributeError, IndexError) as exc:
        raise SummaryError(f"Unexpected API response shape: {exc}") from exc

    return SummaryResult(
        markdown=markdown,
        model=getattr(response, "model", cfg.model),
        usage=usage,
        plots_used=[caption for caption, _ in encoded],
        images_sent=len(encoded),
    )
