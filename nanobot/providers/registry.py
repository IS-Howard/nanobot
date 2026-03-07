"""
Provider Registry — single source of truth for LLM provider metadata.

Adding a new provider:
  1. Add a ProviderSpec to PROVIDERS below.
  2. Add a field to ProvidersConfig in config/schema.py.
  Done. Env vars, prefixing, config matching, status display all derive from here.

Order matters — it controls match priority and fallback. Gateways first.
Every entry writes out all fields so you can copy-paste as a template.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProviderSpec:
    """One LLM provider's metadata. See PROVIDERS below for real examples.

    Placeholders in env_extras values:
      {api_key}  — the user's API key
      {api_base} — api_base from config, or this spec's default_api_base
    """

    # identity
    name: str                       # config field name, e.g. "dashscope"
    keywords: tuple[str, ...]       # model-name keywords for matching (lowercase)
    env_key: str                    # LiteLLM env var, e.g. "DASHSCOPE_API_KEY"
    display_name: str = ""          # shown in `nanobot status`

    # model prefixing
    litellm_prefix: str = ""                 # "dashscope" → model becomes "dashscope/{model}"
    skip_prefixes: tuple[str, ...] = ()      # don't prefix if model already starts with these

    # extra env vars, e.g. (("ZHIPUAI_API_KEY", "{api_key}"),)
    env_extras: tuple[tuple[str, str], ...] = ()

    # gateway / local detection
    is_gateway: bool = False                 # routes any model (OpenRouter, AiHubMix)
    is_local: bool = False                   # local deployment (vLLM, Ollama)
    detect_by_key_prefix: str = ""           # match api_key prefix, e.g. "sk-or-"
    detect_by_base_keyword: str = ""         # match substring in api_base URL
    default_api_base: str = ""               # fallback base URL

    # gateway behavior
    strip_model_prefix: bool = False         # strip "provider/" before re-prefixing

    # per-model param overrides, e.g. (("kimi-k2.5", {"temperature": 1.0}),)
    model_overrides: tuple[tuple[str, dict[str, Any]], ...] = ()

    # OAuth-based providers (e.g., OpenAI Codex) don't use API keys
    is_oauth: bool = False                   # if True, uses OAuth flow instead of API key

    # Direct providers bypass LiteLLM entirely (e.g., CustomProvider)
    is_direct: bool = False

    # Provider supports cache_control on content blocks (e.g. Anthropic prompt caching)
    supports_prompt_caching: bool = False

    @property
    def label(self) -> str:
        return self.display_name or self.name.title()


# ---------------------------------------------------------------------------
# PROVIDERS — the registry. Order = priority. Copy any entry as template.
# ---------------------------------------------------------------------------

PROVIDERS: tuple[ProviderSpec, ...] = (

    # === Custom (direct OpenAI-compatible endpoint, bypasses LiteLLM) ======
    ProviderSpec(
        name="custom",
        keywords=(),
        env_key="",
        display_name="Custom",
        litellm_prefix="",
        is_direct=True,
    ),

    # === Gateways (detected by api_key / api_base, not model name) =========
    # Gateways can route any model, so they win in fallback.

    # OpenRouter: global gateway, keys start with "sk-or-"
    ProviderSpec(
        name="openrouter",
        keywords=("openrouter",),
        env_key="OPENROUTER_API_KEY",
        display_name="OpenRouter",
        litellm_prefix="openrouter",        # claude-3 → openrouter/claude-3
        skip_prefixes=(),
        env_extras=(),
        is_gateway=True,
        is_local=False,
        detect_by_key_prefix="sk-or-",
        detect_by_base_keyword="openrouter",
        default_api_base="https://openrouter.ai/api/v1",
        strip_model_prefix=False,
        model_overrides=(),
        supports_prompt_caching=True,
    ),

    # OpenCode Zen: free model gateway, direct OpenAI-compatible API
    ProviderSpec(
        name="opencode",
        keywords=("opencode", "zen"),
        env_key="",
        display_name="OpenCode Zen",
        litellm_prefix="",
        is_gateway=True,
        is_direct=True,
        detect_by_base_keyword="opencode",
        default_api_base="https://opencode.ai/zen/v1",
    ),

    # === Standard providers (matched by model-name keywords) ===============

    # Gemini: needs "gemini/" prefix for LiteLLM.
    ProviderSpec(
        name="gemini",
        keywords=("gemini",),
        env_key="GEMINI_API_KEY",
        display_name="Gemini",
        litellm_prefix="gemini",            # gemini-pro → gemini/gemini-pro
        skip_prefixes=("gemini/",),         # avoid double-prefix
        env_extras=(),
        is_gateway=False,
        is_local=False,
        detect_by_key_prefix="",
        detect_by_base_keyword="",
        default_api_base="",
        strip_model_prefix=False,
        model_overrides=(),
    ),

    # NVIDIA NIM: needs "nvidia_nim/" prefix for LiteLLM routing.
    ProviderSpec(
        name="nvidia",
        keywords=("nvidia", "nvidia_nim", "nemotron"),
        env_key="NVIDIA_NIM_API_KEY",
        display_name="NVIDIA NIM",
        litellm_prefix="nvidia_nim",
        skip_prefixes=("nvidia_nim/",),
        env_extras=(),
        is_gateway=False,
        is_local=False,
        detect_by_key_prefix="",
        detect_by_base_keyword="nvidia",
        default_api_base="https://integrate.api.nvidia.com/v1",
        strip_model_prefix=False,
        model_overrides=(),
    ),

    # === Auxiliary (not a primary LLM provider) ============================

    # Groq: mainly used for Whisper voice transcription, also usable for LLM.
    # Needs "groq/" prefix for LiteLLM routing. Placed last — it rarely wins fallback.
    ProviderSpec(
        name="groq",
        keywords=("groq",),
        env_key="GROQ_API_KEY",
        display_name="Groq",
        litellm_prefix="groq",              # llama3-8b-8192 → groq/llama3-8b-8192
        skip_prefixes=("groq/",),           # avoid double-prefix
        env_extras=(),
        is_gateway=False,
        is_local=False,
        detect_by_key_prefix="",
        detect_by_base_keyword="",
        default_api_base="",
        strip_model_prefix=False,
        model_overrides=(),
    ),
)


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def find_by_model(model: str) -> ProviderSpec | None:
    """Match a standard provider by model-name keyword (case-insensitive).
    Skips gateways/local — those are matched by api_key/api_base instead."""
    model_lower = model.lower()
    model_normalized = model_lower.replace("-", "_")
    model_prefix = model_lower.split("/", 1)[0] if "/" in model_lower else ""
    normalized_prefix = model_prefix.replace("-", "_")
    std_specs = [s for s in PROVIDERS if not s.is_gateway and not s.is_local]

    # Prefer explicit provider prefix — prevents `github-copilot/...codex` matching openai_codex.
    for spec in std_specs:
        if model_prefix and normalized_prefix == spec.name:
            return spec

    for spec in std_specs:
        if any(kw in model_lower or kw.replace("-", "_") in model_normalized for kw in spec.keywords):
            return spec
    return None


def find_gateway(
    provider_name: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
) -> ProviderSpec | None:
    """Detect gateway/local provider.

    Priority:
      1. provider_name — if it maps to a gateway/local spec, use it directly.
      2. api_key prefix — e.g. "sk-or-" → OpenRouter.
      3. api_base keyword — e.g. "aihubmix" in URL → AiHubMix.

    A standard provider with a custom api_base (e.g. DeepSeek behind a proxy)
    will NOT be mistaken for vLLM — the old fallback is gone.
    """
    # 1. Direct match by config key
    if provider_name:
        spec = find_by_name(provider_name)
        if spec and (spec.is_gateway or spec.is_local):
            return spec

    # 2. Auto-detect by api_key prefix / api_base keyword
    for spec in PROVIDERS:
        if spec.detect_by_key_prefix and api_key and api_key.startswith(spec.detect_by_key_prefix):
            return spec
        if spec.detect_by_base_keyword and api_base and spec.detect_by_base_keyword in api_base:
            return spec

    return None


def find_by_name(name: str) -> ProviderSpec | None:
    """Find a provider spec by config field name, e.g. "dashscope"."""
    for spec in PROVIDERS:
        if spec.name == name:
            return spec
    return None
