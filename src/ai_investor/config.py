from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9 and 3.10
    import tomli as tomllib


@dataclass(frozen=True)
class Settings:
    mode: str
    live_trading: bool
    openai_model: str
    max_output_tokens: int
    max_llm_calls_per_run: int
    max_tool_calls_per_run: int
    max_estimated_cost_per_run_usd: float
    max_monthly_llm_budget_usd: float

    @classmethod
    def load(cls, path: Path) -> "Settings":
        with path.open("rb") as handle:
            raw = tomllib.load(handle)

        mode = os.getenv("AI_INVESTOR_MODE", str(raw["mode"]))
        live = os.getenv(
            "AI_INVESTOR_LIVE_TRADING", str(raw["live_trading"])
        ).lower() in {"1", "true", "yes"}

        # Phase 1 is intentionally impossible to switch to live trading through
        # configuration alone. Enabling live requires a reviewed code change.
        if mode != "DRY_RUN" or live:
            raise RuntimeError(
                "Phase 1 safety lock: mode must be DRY_RUN and live_trading=false"
            )

        return cls(
            mode=mode,
            live_trading=live,
            openai_model=os.getenv(
                "AI_INVESTOR_OPENAI_MODEL", str(raw["openai_model"])
            ),
            max_output_tokens=int(raw["max_output_tokens"]),
            max_llm_calls_per_run=int(raw["max_llm_calls_per_run"]),
            max_tool_calls_per_run=int(raw["max_tool_calls_per_run"]),
            max_estimated_cost_per_run_usd=float(
                raw["max_estimated_cost_per_run_usd"]
            ),
            max_monthly_llm_budget_usd=float(raw["max_monthly_llm_budget_usd"]),
        )
