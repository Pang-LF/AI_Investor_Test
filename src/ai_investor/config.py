from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9 and 3.10
    import tomli as tomllib


@dataclass(frozen=True)
class MonitorSettings:
    interval_minutes: int
    universe_size: int
    quote_batch_size: int
    max_mcp_calls_per_cycle: int
    quote_batch_delay_seconds: float
    market_timezone: str
    core_target: int
    event_target: int
    position_reserve: int
    core_min_price: float
    core_min_market_cap: int
    core_min_average_volume: int
    core_min_average_dollar_volume: int
    event_min_market_cap: int
    event_min_average_volume: int
    event_min_relative_volume: float
    event_min_absolute_change: float
    general_move_trigger: float
    position_move_trigger: float
    fixed_etfs: Tuple[str, ...]

    @classmethod
    def from_dict(cls, raw: dict) -> "MonitorSettings":
        settings = cls(
            interval_minutes=int(raw["interval_minutes"]),
            universe_size=int(raw["universe_size"]),
            quote_batch_size=int(raw["quote_batch_size"]),
            max_mcp_calls_per_cycle=int(raw["max_mcp_calls_per_cycle"]),
            quote_batch_delay_seconds=float(raw["quote_batch_delay_seconds"]),
            market_timezone=str(raw["market_timezone"]),
            core_target=int(raw["core_target"]),
            event_target=int(raw["event_target"]),
            position_reserve=int(raw["position_reserve"]),
            core_min_price=float(raw["core_min_price"]),
            core_min_market_cap=int(raw["core_min_market_cap"]),
            core_min_average_volume=int(raw["core_min_average_volume"]),
            core_min_average_dollar_volume=int(
                raw["core_min_average_dollar_volume"]
            ),
            event_min_market_cap=int(raw["event_min_market_cap"]),
            event_min_average_volume=int(raw["event_min_average_volume"]),
            event_min_relative_volume=float(raw["event_min_relative_volume"]),
            event_min_absolute_change=float(raw["event_min_absolute_change"]),
            general_move_trigger=float(raw["general_move_trigger"]),
            position_move_trigger=float(raw["position_move_trigger"]),
            fixed_etfs=tuple(str(symbol).upper() for symbol in raw["fixed_etfs"]),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.interval_minutes != 15:
            raise RuntimeError("Phase 1 monitor interval must remain 15 minutes")
        if self.universe_size != 60:
            raise RuntimeError("Phase 1 monitor universe must remain 60 symbols")
        if self.quote_batch_size < 1 or self.quote_batch_size > 20:
            raise RuntimeError("Robinhood quote batches must contain 1-20 symbols")
        required_batches = (
            self.universe_size + self.quote_batch_size - 1
        ) // self.quote_batch_size
        if required_batches != 3:
            raise RuntimeError("The 60-symbol monitor must use exactly three batches")
        if self.max_mcp_calls_per_cycle < 7:
            raise RuntimeError("MCP call budget is too small for a full refresh cycle")
        if len(set(self.fixed_etfs)) != len(self.fixed_etfs):
            raise RuntimeError("fixed_etfs contains duplicates")
        if len(self.fixed_etfs) != 12:
            raise RuntimeError("The confirmed design requires exactly 12 fixed ETFs")
        if self.core_target + self.event_target + self.position_reserve != 48:
            raise RuntimeError("Dynamic buckets plus position reserve must total 48")


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
    monitor: MonitorSettings

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
            monitor=MonitorSettings.from_dict(raw["monitor"]),
        )
