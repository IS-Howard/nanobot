"""Configuration schema using Pydantic."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel
from pydantic_settings import BaseSettings


class Base(BaseModel):
    """Base model that accepts both camelCase and snake_case keys."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class LineQuickReplyAction(Base):
    """One quick-reply button attached to outbound text messages."""

    label: str  # ≤20 chars per LINE spec
    type: str = "postback"  # postback | message | uri | camera | cameraRoll | location | datetimepicker
    data: str = ""  # for type=postback
    text: str = ""  # for type=message
    display_text: str = ""


class LineConfig(Base):
    """LINE Messaging API channel configuration (webhook-based)."""

    enabled: bool = False
    channel_access_token: str = ""  # Long-lived channel access token from LINE Developers console
    channel_secret: str = ""  # Channel secret for webhook signature verification
    host: str = "0.0.0.0"  # Webhook server bind address
    port: int = 18800  # Webhook server port
    webhook_path: str = "/line/webhook"  # Webhook endpoint path
    allow_from: list[str] = Field(default_factory=list)  # Allowed LINE user IDs
    quick_reply_enabled: bool = True  # Attach quick-reply buttons to outbound text
    quick_reply_in_groups: bool = False  # Also show in group/room chats (1:1 only by default)
    stickers_enabled: bool = True  # Parse [sticker:pkg/id] markers in outbound text
    # "Stop" and "Continue" are added contextually based on agent/queue state;
    # this list provides the additional static navigation buttons.
    quick_reply_actions: list[LineQuickReplyAction] = Field(
        default_factory=lambda: [
            LineQuickReplyAction(label="New", data="action=new"),
            LineQuickReplyAction(label="Attach", data="action=attach"),
            LineQuickReplyAction(label="Help", data="action=help"),
        ]
    )


class ChannelsConfig(Base):
    """Configuration for chat channels."""

    send_progress: bool = True    # stream agent's text progress to the channel
    send_tool_hints: bool = False  # stream tool-call hints (e.g. read_file("…"))
    line: LineConfig = Field(default_factory=LineConfig)


class AgentDefaults(Base):
    """Default agent configuration."""

    workspace: str = "~/.nanobot/workspace"
    model: str = "anthropic/claude-opus-4-5"
    provider: str = "auto"  # Provider name (e.g. "anthropic", "openrouter") or "auto" for auto-detection
    max_tokens: int = 8192
    temperature: float = 0.1
    max_tool_iterations: int = 40
    max_context_items: int = 20  # max history messages sent to LLM
    reasoning_effort: str | None = None  # low / medium / high — enables LLM thinking mode
    parallel: bool = False  # True: per-session parallel processing; False: single global lock
    admin_passphrase: str = ""  # Seeds access.json on first run for RBAC authentication


class AgentsConfig(Base):
    """Agent configuration."""

    defaults: AgentDefaults = Field(default_factory=AgentDefaults)


class ProviderConfig(Base):
    """LLM provider configuration."""

    api_key: str = ""
    api_base: str | None = None
    extra_headers: dict[str, str] | None = None  # Custom headers (e.g. APP-Code for AiHubMix)


class ProvidersConfig(Base):
    """Configuration for LLM providers."""

    custom: ProviderConfig = Field(default_factory=ProviderConfig)  # Any OpenAI-compatible endpoint
    opencode: ProviderConfig = Field(default_factory=ProviderConfig)  # OpenCode Zen free models
    openrouter: ProviderConfig = Field(default_factory=ProviderConfig)
    gemini: ProviderConfig = Field(default_factory=ProviderConfig)
    groq: ProviderConfig = Field(default_factory=ProviderConfig)
    nvidia: ProviderConfig = Field(default_factory=ProviderConfig)  # NVIDIA NIM inference


class HeartbeatConfig(Base):
    """Heartbeat service configuration."""

    enabled: bool = True
    interval_s: int = 30 * 60  # 30 minutes


class GatewayConfig(Base):
    """Gateway/server configuration."""

    host: str = "0.0.0.0"
    port: int = 18790
    heartbeat: HeartbeatConfig = Field(default_factory=HeartbeatConfig)


class WebSearchConfig(Base):
    """Web search tool configuration."""

    api_key: str = ""  # Brave Search API key
    max_results: int = 5


class WebToolsConfig(Base):
    """Web tools configuration."""

    proxy: str | None = None  # HTTP/SOCKS5 proxy URL, e.g. "http://127.0.0.1:7890" or "socks5://127.0.0.1:1080"
    search: WebSearchConfig = Field(default_factory=WebSearchConfig)


class ExecToolConfig(Base):
    """Shell exec tool configuration."""

    timeout: int = 60
    path_append: str = ""
    python_via_uv: bool = True  # Rewrite bare `python`/`pip` to `uv run python`/`uv pip`


class MCPServerConfig(Base):
    """MCP server connection configuration (stdio or HTTP)."""

    command: str = ""  # Stdio: command to run (e.g. "npx")
    args: list[str] = Field(default_factory=list)  # Stdio: command arguments
    env: dict[str, str] = Field(default_factory=dict)  # Stdio: extra env vars
    url: str = ""  # HTTP: streamable HTTP endpoint URL
    headers: dict[str, str] = Field(default_factory=dict)  # HTTP: Custom HTTP Headers
    tool_timeout: int = 30  # Seconds before a tool call is cancelled


class ToolsConfig(Base):
    """Tools configuration."""

    web: WebToolsConfig = Field(default_factory=WebToolsConfig)
    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)
    restrict_to_workspace: bool = False  # If true, restrict all tool access to workspace directory
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)


class DatabaseConfig(Base):
    """PostgreSQL database configuration."""

    url: str = ""  # postgresql://user:pass@host/db
    max_files_per_session: int = 2  # FIFO cleanup


class TranscriptionConfig(Base):
    """Audio/voice transcription configuration.

    ``backend`` selects the transcription engine:
      - ``auto``  : use local faster-whisper if installed, else Groq (default)
      - ``local`` : force local faster-whisper (``pip install nanobot-ai[whisper]``)
      - ``groq``  : force Groq's hosted Whisper API (requires GROQ_API_KEY)
    """

    backend: str = "auto"
    model: str = "base"  # faster-whisper model size: tiny/base/small/medium/large-v3
    device: str = "cpu"  # cpu / cuda / auto
    compute_type: str = "int8"  # int8 (cpu) / float16 (gpu) / int8_float16 ...
    language: str = ""  # ISO code (e.g. "en"); empty = auto-detect
    cpu_threads: int = 0  # ctranslate2 CPU threads (0 = library default; ~8 is a good sweet spot)


class Config(BaseSettings):
    """Root configuration for nanobot."""

    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)

    @property
    def workspace_path(self) -> Path:
        """Get expanded workspace path."""
        return Path(self.agents.defaults.workspace).expanduser()

    def _match_provider(self, model: str | None = None) -> tuple["ProviderConfig | None", str | None]:
        """Match provider config and its registry name. Returns (config, spec_name)."""
        from nanobot.providers.registry import PROVIDERS

        forced = self.agents.defaults.provider
        if forced != "auto":
            p = getattr(self.providers, forced, None)
            return (p, forced) if p else (None, None)

        model_lower = (model or self.agents.defaults.model).lower()
        model_normalized = model_lower.replace("-", "_")
        model_prefix = model_lower.split("/", 1)[0] if "/" in model_lower else ""
        normalized_prefix = model_prefix.replace("-", "_")

        def _kw_matches(kw: str) -> bool:
            kw = kw.lower()
            return kw in model_lower or kw.replace("-", "_") in model_normalized

        # Explicit provider prefix wins — prevents `github-copilot/...codex` matching openai_codex.
        for spec in PROVIDERS:
            p = getattr(self.providers, spec.name, None)
            if p and model_prefix and normalized_prefix == spec.name:
                if spec.is_oauth or p.api_key:
                    return p, spec.name

        # Match by keyword (order follows PROVIDERS registry)
        for spec in PROVIDERS:
            p = getattr(self.providers, spec.name, None)
            if p and any(_kw_matches(kw) for kw in spec.keywords):
                if spec.is_oauth or p.api_key:
                    return p, spec.name

        # Fallback: gateways first, then others (follows registry order)
        # OAuth providers are NOT valid fallbacks — they require explicit model selection
        for spec in PROVIDERS:
            if spec.is_oauth:
                continue
            p = getattr(self.providers, spec.name, None)
            if p and p.api_key:
                return p, spec.name
        return None, None

    def get_provider(self, model: str | None = None) -> ProviderConfig | None:
        """Get matched provider config (api_key, api_base, extra_headers). Falls back to first available."""
        p, _ = self._match_provider(model)
        return p

    def get_provider_name(self, model: str | None = None) -> str | None:
        """Get the registry name of the matched provider (e.g. "deepseek", "openrouter")."""
        _, name = self._match_provider(model)
        return name

    def get_api_key(self, model: str | None = None) -> str | None:
        """Get API key for the given model. Falls back to first available key."""
        p = self.get_provider(model)
        return p.api_key if p else None

    def get_api_base(self, model: str | None = None) -> str | None:
        """Get API base URL for the given model. Applies default URLs for known gateways."""
        from nanobot.providers.registry import find_by_name

        p, name = self._match_provider(model)
        if p and p.api_base:
            return p.api_base
        # Only gateways get a default api_base here. Standard providers
        # (like Moonshot) set their base URL via env vars in _setup_env
        # to avoid polluting the global litellm.api_base.
        if name:
            spec = find_by_name(name)
            if spec and spec.is_gateway and spec.default_api_base:
                return spec.default_api_base
        return None

    model_config = ConfigDict(env_prefix="NANOBOT_", env_nested_delimiter="__")
