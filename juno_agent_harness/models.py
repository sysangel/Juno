from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from .provider_policy import DEFAULT_MODEL_OPENROUTER

Provider = Literal["openrouter", "openai", "anthropic", "claude-code", "echo"]
Role = Literal["system", "user", "assistant"]
PROVIDERS = {"openrouter", "openai", "anthropic", "claude-code", "echo"}
DEFAULT_HOME = Path(os.getenv("AGENT_HARNESS_HOME", Path.home() / ".agent-harness"))
DEFAULT_OPENROUTER_MODEL = DEFAULT_MODEL_OPENROUTER
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1")
DEFAULT_ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
DEFAULT_CLAUDE_CODE_MODEL = os.getenv("CLAUDE_CODE_MODEL", "sonnet")
DEFAULT_OPENROUTER_DP_MODEL = os.getenv("HARNESS_DP_MODEL", "deepseek/deepseek-v4-pro")
DEFAULT_OPENROUTER_FM3V4_MODEL = os.getenv("HARNESS_FM3V4_MODEL", "deepseek/deepseek-v4-pro")
DEFAULT_OPENROUTER_FBUDGET_MODEL = os.getenv("HARNESS_FBUDGET_MODEL", "openrouter/auto")
DEFAULT_OPENROUTER_FHIGH_MODEL = os.getenv("HARNESS_FHIGH_MODEL", "openrouter/auto")

@dataclass(frozen=True)
class ModelOption:
    key: str
    section: str
    label: str
    provider: Provider
    model: str
    description: str = ""
    tier: str = "standard"      # one of the 5 tier keys in TIERS
    icon: str = ""              # "" -> derive from provider/model via STRENGTH_ICONS


@dataclass(frozen=True)
class SkillSummary:
    """A discovered skill from a local SKILL.md file."""
    name: str
    category: str
    description: str
    path: Path


def discover_local_skills(home: Path, extra_roots: Iterable[Path] = ()) -> list[SkillSummary]:
    """Scan ``**/SKILL.md`` under *home*/skills and any extra roots.

    Parses minimal YAML-ish frontmatter (``name:``, ``description:``) and
    groups by the parent folder name as category.
    """
    roots = [home / "skills", *extra_roots]
    skills: list[SkillSummary] = []
    for root in roots:
        if not root.is_dir():
            continue
        for skill_md in sorted(root.rglob("SKILL.md")):
            summary = _parse_skill_md(skill_md)
            if summary:
                skills.append(summary)
    return skills


def _parse_skill_md(path: Path) -> SkillSummary | None:
    """Extract name, description, and category from a SKILL.md file."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    name = ""
    description = ""
    in_frontmatter = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == "---":
            if not in_frontmatter:
                in_frontmatter = True
                continue
            else:
                break
        if not in_frontmatter:
            # If there is no frontmatter, use the first # heading as name.
            if line.startswith("# ") and not name:
                name = line[2:].strip()
            continue
        if ":" in line:
            key, _, val = line.partition(":")
            key, val = key.strip().lower(), val.strip().strip('"\'')
            if key == "name":
                name = val
            elif key == "description":
                description = val
    if not name:
        return None
    # Category is the parent folder name
    category = path.parent.name.replace("-", " ").replace("_", " ").title()
    return SkillSummary(name=name, category=category, description=description, path=path)
def default_model(provider: Provider) -> str:
    if provider == "openrouter":
        return DEFAULT_OPENROUTER_MODEL
    if provider == "openai":
        return DEFAULT_OPENAI_MODEL
    if provider == "anthropic":
        return DEFAULT_ANTHROPIC_MODEL
    if provider == "claude-code":
        return DEFAULT_CLAUDE_CODE_MODEL
    return "echo"


class HarnessConfig:
    """Persistent Juno configuration stored as ``<home>/config.json``.

    Currently manages the default model/provider so users can set a
    personal favorite that survives across sessions.
    """

    def __init__(self, home: Path) -> None:
        self.path = home / "config.json"

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)

    def default_model(self) -> tuple[Provider, str] | None:
        """Return (provider, model) from the config, or None."""
        data = self.load()
        model_block = data.get("model")
        if not isinstance(model_block, dict):
            return None
        provider = model_block.get("provider")
        model = model_block.get("default")
        if not provider or not model:
            return None
        # Validate the stored provider is still known.
        try:
            p = normalize_provider(provider)
        except ValueError:
            return None
        return (p, model)

    def set_default_model(self, provider: Provider, model: str) -> None:
        data = self.load()
        data["model"] = {"provider": provider, "default": model}
        self.save(data)


def normalize_provider(value: str) -> Provider:
    aliases = {
        "claude": "claude-code",
        "cc": "claude-code",
        "chatgpt": "openai",
        "gpt": "openai",
        "codex": "openai",
        "or": "openrouter",
    }
    provider = aliases.get(value.strip().lower(), value.strip().lower())
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {value!r}; choose one of: {', '.join(sorted(PROVIDERS))}")
    return provider  # type: ignore[return-value]


def _raw_model_catalog() -> list[ModelOption]:
    """Return all selectable model aliases, including hidden duplicate routes.

    `model_catalog()` deduplicates this for display so the picker stays clean,
    but aliases such as `fm3v4` should still resolve even when they point at the
    same provider/model as another visible row.
    """
    return [
        ModelOption(
            key="codex",
            section="OpenAI / ChatGPT",
            label="GPT-5.1 Codex",
            provider="openai",
            model="gpt-5.1-codex",
            description="Coding-focused OpenAI route; use as the explicit Codex shortcut.",
            tier="flagship",
        ),
        ModelOption(
            key="gpt5",
            section="OpenAI / ChatGPT",
            label="GPT-5.1",
            provider="openai",
            model="gpt-5.1",
            description="Current general ChatGPT/OpenAI route.",
            tier="flagship",
        ),
        ModelOption(
            key="gpt5mini",
            section="OpenAI / ChatGPT",
            label="GPT-5.1 Mini",
            provider="openai",
            model="gpt-5.1-mini",
            description="Cheaper/faster GPT-5.1 route for lighter work.",
            tier="strong",
        ),
        ModelOption(
            key="chatgpt",
            section="OpenAI / ChatGPT",
            label="ChatGPT latest",
            provider="openai",
            model="chatgpt-4o-latest",
            description="ChatGPT-flavored OpenAI route for conversational testing.",
            tier="strong",
        ),
        ModelOption(
            key="claude",
            section="Anthropic / Claude",
            label=f"Claude Code default ({DEFAULT_CLAUDE_CODE_MODEL})",
            provider="claude-code",
            model=DEFAULT_CLAUDE_CODE_MODEL,
            description="Local Claude Code OAuth bridge; no ANTHROPIC_API_KEY required.",
            tier="strong",
        ),
        ModelOption(
            key="sonnet",
            section="Anthropic / Claude",
            label="Claude Code Sonnet",
            provider="claude-code",
            model="sonnet",
            description="Fast Claude Code OAuth model alias.",
            tier="strong",
        ),
        ModelOption(
            key="opus",
            section="Anthropic / Claude",
            label="Claude Code Opus",
            provider="claude-code",
            model="opus",
            description="Heavier Claude Code OAuth model alias if your account allows it.",
            tier="flagship",
        ),
        ModelOption(
            key="anthropic",
            section="Anthropic / Claude",
            label=f"Anthropic API default ({DEFAULT_ANTHROPIC_MODEL})",
            provider="anthropic",
            model=DEFAULT_ANTHROPIC_MODEL,
            description="Direct Anthropic Messages API; needs ANTHROPIC_API_KEY.",
            tier="flagship",
        ),
        ModelOption(
            key="or",
            section="OpenRouter",
            label=f"OpenRouter default ({DEFAULT_OPENROUTER_MODEL})",
            provider="openrouter",
            model=DEFAULT_OPENROUTER_MODEL,
            description="Uses the repo privacy-screened OpenRouter provider preferences.",
            tier="standard",
        ),
        ModelOption(
            key="dp",
            section="OpenRouter",
            label=f"DeepSeek Pro ({DEFAULT_OPENROUTER_DP_MODEL})",
            provider="openrouter",
            model=DEFAULT_OPENROUTER_DP_MODEL,
            description="Hermes-style dp shortcut.",
            tier="strong",
        ),
        ModelOption(
            key="fm3v4",
            section="OpenRouter",
            label=f"Fusion M3/V4 ({DEFAULT_OPENROUTER_FM3V4_MODEL})",
            provider="openrouter",
            model=DEFAULT_OPENROUTER_FM3V4_MODEL,
            description="Env override: HARNESS_FM3V4_MODEL.",
            tier="strong",
        ),
        ModelOption(
            key="fbudget",
            section="OpenRouter",
            label=f"Fusion budget ({DEFAULT_OPENROUTER_FBUDGET_MODEL})",
            provider="openrouter",
            model=DEFAULT_OPENROUTER_FBUDGET_MODEL,
            description="Env override: HARNESS_FBUDGET_MODEL.",
            tier="standard",
        ),
        ModelOption(
            key="fhigh",
            section="OpenRouter",
            label=f"Fusion high ({DEFAULT_OPENROUTER_FHIGH_MODEL})",
            provider="openrouter",
            model=DEFAULT_OPENROUTER_FHIGH_MODEL,
            description="Env override: HARNESS_FHIGH_MODEL.",
            tier="strong",
        ),
        ModelOption(
            key="echo",
            section="Offline",
            label="Echo smoke-test provider",
            provider="echo",
            model="echo",
            description="No API key; returns `echo: <input>`.",
            tier="light",
        ),
    ]


def model_catalog() -> list[ModelOption]:
    """Return deduplicated grouped model choices for `/model` and the picker.

    Duplicate provider/model rows make the prompt_toolkit picker look uneven and
    confusing. Keep the first visible option for a route; `model_aliases()` still
    resolves every raw key for power users.
    """
    options: list[ModelOption] = []
    seen: set[tuple[str, Provider, str]] = set()
    for option in _raw_model_catalog():
        route = (option.section, option.provider, option.model)
        if route in seen:
            continue
        seen.add(route)
        options.append(option)
    return options


def model_aliases() -> dict[str, ModelOption]:
    aliases: dict[str, ModelOption] = {}
    for option in _raw_model_catalog():
        aliases[option.key.lower()] = option
    # Friendly synonyms.
    aliases.update({
        "openai": aliases["gpt5"],
        "gpt": aliases["gpt5"],
        "gpt-5": aliases["gpt5"],
        "gpt-5.1": aliases["gpt5"],
        "gpt-5.1-codex": aliases["codex"],
        "gpt-5.1-mini": aliases["gpt5mini"],
        "chatgpt-4o-latest": aliases["chatgpt"],
        "claude-code": aliases["claude"],
        "cc": aliases["claude"],
        "openrouter": aliases["or"],
        "deepseek": aliases["dp"],
    })
    return aliases


def resolve_model_selection(first: str, second: str | None = None) -> tuple[Provider, str, str]:
    """Resolve `/model ...` input.

    Returns (provider, model, note). Supports:
    - `/model <catalog-key>` e.g. `/model sonnet`, `/model dp`, `/model fhigh`
    - `/model <provider> [model]` e.g. `/model openrouter deepseek/...`
    - provider aliases (`claude`, `codex`, `or`) and raw provider/model pairs.
    """
    token = first.strip().lower()
    if second is None:
        option = model_aliases().get(token)
        if option is not None:
            return option.provider, option.model, f"selected {option.section}: {option.label}"
    provider = normalize_provider(first)
    model = second or default_model(provider)
    return provider, model, ""


def model_catalog_by_key() -> dict[str, ModelOption]:
    """Map every catalog key (and friendly synonym) to its ModelOption."""
    return model_aliases()


def resolve_model_ref(ref: str) -> tuple[Provider, str]:
    """Resolve either `PROVIDER/MODEL` or a catalog `KEY` to (provider, model).

    `KeyError` surfaces an unknown catalog key to the caller.
    """
    ref = ref.strip()
    if "/" in ref:
        prov, model = ref.split("/", 1)
        return normalize_provider(prov), model
    opt = model_catalog_by_key()[ref.lower()]
    return opt.provider, opt.model


def format_model_catalog(current_provider: str | None = None, current_model: str | None = None) -> str:
    lines = [
        "model catalog",
        "  Select in Juno with /model, or type one of:",
        "  /model <key>                  e.g. /model sonnet, /model dp, /model fhigh",
        "  /model <provider> [model]     e.g. /model openrouter deepseek/deepseek-v4-pro",
    ]
    current = (current_provider, current_model)
    last_section = None
    for option in model_catalog():
        if option.section != last_section:
            lines.append(f"\n{option.section}")
            last_section = option.section
        marker = "*" if (option.provider, option.model) == current else " "
        desc = f" — {option.description}" if option.description else ""
        lines.append(f" {marker} {option.key:<10} {option.provider:<12} {option.model:<34} {option.label}{desc}")
    lines.append("\n* = current model")
    return "\n".join(lines)
