"""Environment-driven settings.

Every endpoint, credential, and path comes from the environment so nothing
machine-specific is committed. Names match the organizer scaffold documented in
Participant_Package/Setup_Instructions.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value else default


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    # --- data ---
    data_root: Path
    artifacts_dir: Path

    # --- model serving (LiteLLM proxy on the brain/agent node) ---
    litellm_base_url: str
    litellm_key: str
    brain_model: str
    domain_ft_model: str
    domain_predict_mode: str  # "mock" (bootstrap) | "llm" (required for evaluation)

    # --- agent behaviour ---
    max_agent_steps: int
    agent_deadline_s: float
    brain_timeout_s: float
    synth_timeout_s: float
    brain_max_tokens: int
    synth_max_tokens: int
    tool_result_char_budget: int

    # --- serving ---
    agent_host: str
    agent_port: int
    log_dir: Path

    @property
    def rba_csv(self) -> Path:
        return self.data_root / "RBA Rates" / "RBA-rates.csv"

    @property
    def asx_dir(self) -> Path:
        return self.data_root / "ASX"

    @property
    def afr_dir(self) -> Path:
        return self.data_root / "AFR"

    @property
    def uses_fine_tuned_model(self) -> bool:
        return self.domain_predict_mode.lower() == "llm"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    data_root = Path(_str("DATA_ROOT", str(REPO_ROOT / "data set"))).expanduser()
    artifacts = Path(_str("ARTIFACTS_DIR", str(REPO_ROOT / "artifacts"))).expanduser()
    logs = Path(_str("LOG_DIR", str(REPO_ROOT / "logs"))).expanduser()

    return Settings(
        data_root=data_root,
        artifacts_dir=artifacts,
        litellm_base_url=_str("LITELLM_BASE_URL", _str("LITELLM_URL", "http://localhost:4000/v1")),
        litellm_key=_str("LITELLM_KEY", "sk-local-cluster"),
        brain_model=_str("BRAIN_MODEL", "agent-brain"),
        domain_ft_model=_str("DOMAIN_FT_MODEL", "domain-ft"),
        domain_predict_mode=_str("DOMAIN_PREDICT_MODE", "mock"),
        max_agent_steps=_int("MAX_AGENT_STEPS", 3),
        agent_deadline_s=_float("AGENT_DEADLINE_S", 50.0),
        brain_timeout_s=_float("BRAIN_TIMEOUT_S", 20.0),
        synth_timeout_s=_float("SYNTH_TIMEOUT_S", 20.0),
        brain_max_tokens=_int("BRAIN_MAX_TOKENS", 400),
        synth_max_tokens=_int("SYNTH_MAX_TOKENS", 300),
        tool_result_char_budget=_int("TOOL_RESULT_CHAR_BUDGET", 700),
        agent_host=_str("AGENT_HOST", "0.0.0.0"),
        agent_port=_int("AGENT_PORT", 5000),
        log_dir=logs,
    )
