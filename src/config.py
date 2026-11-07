"""Configuration: environment variables, provider defaults and the routing policy file.

Nothing here contains a credential. Keys are read from the environment (or a local,
git-ignored .env file) at call time so tests and operators can change them without a restart.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = ROOT / "config" / "policy.json"
SYSTEM_VERSION = "0.1.0"

# provider -> (base url, key env var, default model). All speak the OpenAI chat-completions dialect,
# which is why switching provider is a .env change and not a code change.
PROVIDERS = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", "gemini-2.5-flash"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "meta-llama/llama-3.1-8b-instruct"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.1-8b-instant"),
}

_PLACEHOLDERS = {"", "your_key_here", "changeme", "change_me"}


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env reader (no dependency). Real environment variables always win."""
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_rpm: float
    embedding_backend: str
    embedding_model: str
    retrieval_top_k: int
    storage_dir: Path
    auto_respond_enabled: bool
    data_dir: Path

    @property
    def llm_enabled(self) -> bool:
        return self.llm_provider != "none" and self.llm_api_key.strip().lower() not in _PLACEHOLDERS


def get_settings() -> Settings:
    provider = os.environ.get("LLM_PROVIDER", "gemini").strip().lower()
    base, key_var, default_model = PROVIDERS.get(provider, ("", "LLM_API_KEY", ""))
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get(key_var, "")
    storage = Path(os.environ.get("STORAGE_DIR", ROOT / "storage"))
    if not storage.is_absolute():
        storage = ROOT / storage
    return Settings(
        llm_provider=provider,
        llm_base_url=os.environ.get("LLM_BASE_URL", base).rstrip("/"),
        llm_api_key=api_key,
        llm_model=os.environ.get("MODEL_NAME", default_model),
        llm_rpm=float(os.environ.get("LLM_RPM", "10")),
        embedding_backend=os.environ.get("EMBEDDING_BACKEND", "auto").lower(),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        retrieval_top_k=int(os.environ.get("RETRIEVAL_TOP_K", "3")),
        storage_dir=storage,
        auto_respond_enabled=_env_bool("AUTO_RESPOND_ENABLED", True),
        data_dir=Path(os.environ.get("DATA_DIR", ROOT / "data")),
    )


def load_policy() -> dict:
    """Routing policy: must-escalate intents, thresholds and safety keywords.

    CONFIDENCE_THRESHOLD / RETRIEVAL_MIN_SCORE environment variables override the file so an operator
    can tighten the system without a deployment.
    """
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if os.environ.get("CONFIDENCE_THRESHOLD"):
        policy["confidence_threshold"] = float(os.environ["CONFIDENCE_THRESHOLD"])
    if os.environ.get("RETRIEVAL_MIN_SCORE"):
        policy["retrieval_min_score"] = float(os.environ["RETRIEVAL_MIN_SCORE"])
    return policy


def kill_switch_engaged(settings: Settings | None = None) -> bool:
    """True when automatic answering must stop. Checked on every ticket, so it takes effect immediately."""
    settings = settings or get_settings()
    if not settings.auto_respond_enabled:
        return True
    return (settings.storage_dir / "KILL_SWITCH").exists()
