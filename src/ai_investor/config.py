from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9 and 3.10
    import tomli as tomllib


@dataclass(frozen=True)
class HealthSettings:
    persistent_failure_repeat_minutes: int
    degraded_cycles_before_alert: int

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "HealthSettings":
        settings = cls(
            persistent_failure_repeat_minutes=int(
                raw.get("persistent_failure_repeat_minutes", 360)
            ),
            degraded_cycles_before_alert=int(
                raw.get("degraded_cycles_before_alert", 2)
            ),
        )
        if not 15 <= settings.persistent_failure_repeat_minutes <= 1440:
            raise RuntimeError("Failure reminder interval must be 15-1440 minutes")
        if not 1 <= settings.degraded_cycles_before_alert <= 4:
            raise RuntimeError("Degraded-cycle alert threshold must be 1-4")
        return settings


@dataclass(frozen=True)
class MonitorSettings:
    interval_minutes: int
    universe_size: int
    quote_batch_size: int
    max_mcp_calls_per_cycle: int
    quote_batch_delay_seconds: float
    max_quote_age_minutes: int
    market_timezone: str
    large_target: int
    mid_target: int
    small_target: int
    event_target: int
    position_reserve: int
    large_min_price: float
    large_min_market_cap: int
    large_min_average_volume: int
    large_min_average_dollar_volume: int
    mid_min_price: float
    mid_min_market_cap: int
    mid_max_market_cap: int
    mid_min_average_volume: int
    mid_min_average_dollar_volume: int
    small_min_price: float
    small_min_market_cap: int
    small_max_market_cap: int
    small_min_average_volume: int
    small_min_average_dollar_volume: int
    small_min_float_ratio: float
    small_median_liquidity_candidate_limit: int
    stable_min_ipo_age_calendar_days: int
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
            max_quote_age_minutes=int(raw["max_quote_age_minutes"]),
            market_timezone=str(raw["market_timezone"]),
            large_target=int(raw["large_target"]),
            mid_target=int(raw["mid_target"]),
            small_target=int(raw["small_target"]),
            event_target=int(raw["event_target"]),
            position_reserve=int(raw["position_reserve"]),
            large_min_price=float(raw["large_min_price"]),
            large_min_market_cap=int(raw["large_min_market_cap"]),
            large_min_average_volume=int(raw["large_min_average_volume"]),
            large_min_average_dollar_volume=int(raw["large_min_average_dollar_volume"]),
            mid_min_price=float(raw["mid_min_price"]),
            mid_min_market_cap=int(raw["mid_min_market_cap"]),
            mid_max_market_cap=int(raw["mid_max_market_cap"]),
            mid_min_average_volume=int(raw["mid_min_average_volume"]),
            mid_min_average_dollar_volume=int(raw["mid_min_average_dollar_volume"]),
            small_min_price=float(raw["small_min_price"]),
            small_min_market_cap=int(raw["small_min_market_cap"]),
            small_max_market_cap=int(raw["small_max_market_cap"]),
            small_min_average_volume=int(raw["small_min_average_volume"]),
            small_min_average_dollar_volume=int(raw["small_min_average_dollar_volume"]),
            small_min_float_ratio=float(raw["small_min_float_ratio"]),
            small_median_liquidity_candidate_limit=int(
                raw["small_median_liquidity_candidate_limit"]
            ),
            stable_min_ipo_age_calendar_days=int(
                raw["stable_min_ipo_age_calendar_days"]
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
        if self.max_mcp_calls_per_cycle < 13:
            raise RuntimeError("MCP call budget is too small for a full refresh cycle")
        if self.max_quote_age_minutes < self.interval_minutes:
            raise RuntimeError("Quote freshness window must cover at least one interval")
        if len(set(self.fixed_etfs)) != len(self.fixed_etfs):
            raise RuntimeError("fixed_etfs contains duplicates")
        if len(self.fixed_etfs) != 12:
            raise RuntimeError("The confirmed design requires exactly 12 fixed ETFs")
        if (
            self.large_target + self.mid_target + self.small_target
            + self.event_target + self.position_reserve != 48
        ):
            raise RuntimeError("Dynamic buckets plus position reserve must total 48")
        if not 0 < self.small_min_float_ratio <= 1:
            raise RuntimeError("Small-cap float ratio must be in (0, 1]")
        if not 8 <= self.small_median_liquidity_candidate_limit <= 20:
            raise RuntimeError("Small-cap eligibility shortlist must be 8-20 names")
        if self.stable_min_ipo_age_calendar_days < 252:
            raise RuntimeError("Stable pools require about 180 trading days of history")
        if not (
            self.small_min_market_cap < self.small_max_market_cap
            <= self.mid_min_market_cap < self.mid_max_market_cap
            <= self.large_min_market_cap
        ):
            raise RuntimeError("Size-bucket market-cap boundaries must not overlap")


@dataclass(frozen=True)
class ForecastSettings:
    history_calendar_days: int
    min_history_bars: int
    ridge_penalty: float
    min_training_samples: int
    shrinkage: float
    research_candidate_count: int
    research_min_raw_probability_positive: float
    research_min_raw_expected_excess_return_20d: float
    execution_calibration_approved: bool
    execution_min_calibrated_probability_positive: float
    execution_min_bias_adjusted_excess_return_20d: float
    execution_min_calibration_date_blocks: int
    max_abs_forecast_20d: float

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ForecastSettings":
        settings = cls(
            history_calendar_days=int(raw["history_calendar_days"]),
            min_history_bars=int(raw["min_history_bars"]),
            ridge_penalty=float(raw["ridge_penalty"]),
            min_training_samples=int(raw["min_training_samples"]),
            shrinkage=float(raw["shrinkage"]),
            research_candidate_count=int(raw["research_candidate_count"]),
            research_min_raw_probability_positive=float(
                raw["research_min_raw_probability_positive"]
            ),
            research_min_raw_expected_excess_return_20d=float(
                raw["research_min_raw_expected_excess_return_20d"]
            ),
            execution_calibration_approved=bool(
                raw["execution_calibration_approved"]
            ),
            execution_min_calibrated_probability_positive=float(
                raw["execution_min_calibrated_probability_positive"]
            ),
            execution_min_bias_adjusted_excess_return_20d=float(
                raw["execution_min_bias_adjusted_excess_return_20d"]
            ),
            execution_min_calibration_date_blocks=int(
                raw["execution_min_calibration_date_blocks"]
            ),
            max_abs_forecast_20d=float(raw["max_abs_forecast_20d"]),
        )
        if settings.history_calendar_days < 120:
            raise RuntimeError("Forecast history must cover at least 120 calendar days")
        if settings.min_history_bars < 80:
            raise RuntimeError("Forecasting requires at least 80 completed daily bars")
        if not 0 < settings.shrinkage <= 1:
            raise RuntimeError("Forecast shrinkage must be in (0, 1]")
        if settings.research_candidate_count < 1 or settings.research_candidate_count > 5:
            raise RuntimeError("Research candidate count must remain between one and five")
        if not 0 < settings.research_min_raw_probability_positive < 1:
            raise RuntimeError("Research probability threshold must be in (0, 1)")
        if not 0 < settings.execution_min_calibrated_probability_positive < 1:
            raise RuntimeError("Execution probability threshold must be in (0, 1)")
        if settings.execution_min_calibration_date_blocks < 5:
            raise RuntimeError("Execution calibration needs at least five non-overlapping blocks")
        return settings


@dataclass(frozen=True)
class ResearchSettings:
    deep_candidate_count: int

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ResearchSettings":
        settings = cls(deep_candidate_count=int(raw["deep_candidate_count"]))
        if not 1 <= settings.deep_candidate_count <= 5:
            raise RuntimeError("Deep news/earnings research must cover 1-5 names")
        return settings


@dataclass(frozen=True)
class PortfolioSettings:
    max_invested_fraction: float
    soft_max_position_fraction: float
    soft_max_sector_fraction: float
    risk_aversion: float
    uncertainty_penalty: float
    optimizer_iterations: int
    optimizer_step_size: float
    min_trade_usd: float

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "PortfolioSettings":
        settings = cls(
            max_invested_fraction=float(raw["max_invested_fraction"]),
            soft_max_position_fraction=float(raw["soft_max_position_fraction"]),
            soft_max_sector_fraction=float(raw["soft_max_sector_fraction"]),
            risk_aversion=float(raw["risk_aversion"]),
            uncertainty_penalty=float(raw["uncertainty_penalty"]),
            optimizer_iterations=int(raw["optimizer_iterations"]),
            optimizer_step_size=float(raw["optimizer_step_size"]),
            min_trade_usd=float(raw["min_trade_usd"]),
        )
        if not 0 < settings.max_invested_fraction <= 1:
            raise RuntimeError("max_invested_fraction must be in (0, 1]")
        if not 0 < settings.soft_max_position_fraction <= 0.50:
            raise RuntimeError("Strategy soft position cap must remain in (0, 0.50]")
        if not settings.soft_max_position_fraction <= settings.soft_max_sector_fraction <= 1:
            raise RuntimeError("Strategy sector cap must be between position cap and 100%")
        if settings.optimizer_iterations < 10:
            raise RuntimeError("Optimizer iteration count is too small")
        if settings.min_trade_usd < 1:
            raise RuntimeError("Minimum trade must be at least $1")
        return settings


@dataclass(frozen=True)
class RiskSettings:
    policy_version: str
    allow_margin: bool
    allow_options: bool
    allow_short_selling: bool
    allow_crypto: bool
    allow_leveraged_etfs: bool
    max_position_fraction: float
    max_positions: int
    max_trade_usd: float
    max_trade_fraction: float
    max_daily_turnover_fraction: float
    max_portfolio_drawdown_fraction: float
    max_daily_loss_fraction: float
    max_live_orders_per_day: int
    max_spread_fraction: float
    max_execution_quote_age_seconds: int
    max_decision_age_seconds: int
    max_decision_price_drift_fraction: float

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "RiskSettings":
        settings = cls(
            policy_version=str(raw["policy_version"]),
            allow_margin=bool(raw["allow_margin"]),
            allow_options=bool(raw["allow_options"]),
            allow_short_selling=bool(raw["allow_short_selling"]),
            allow_crypto=bool(raw["allow_crypto"]),
            allow_leveraged_etfs=bool(raw["allow_leveraged_etfs"]),
            max_position_fraction=float(raw["max_position_fraction"]),
            max_positions=int(raw["max_positions"]),
            max_trade_usd=float(raw["max_trade_usd"]),
            max_trade_fraction=float(raw["max_trade_fraction"]),
            max_daily_turnover_fraction=float(raw["max_daily_turnover_fraction"]),
            max_portfolio_drawdown_fraction=float(
                raw["max_portfolio_drawdown_fraction"]
            ),
            max_daily_loss_fraction=float(raw["max_daily_loss_fraction"]),
            max_live_orders_per_day=int(raw["max_live_orders_per_day"]),
            max_spread_fraction=float(raw["max_spread_fraction"]),
            max_execution_quote_age_seconds=int(
                raw["max_execution_quote_age_seconds"]
            ),
            max_decision_age_seconds=int(raw["max_decision_age_seconds"]),
            max_decision_price_drift_fraction=float(
                raw["max_decision_price_drift_fraction"]
            ),
        )
        forbidden = (
            settings.allow_margin,
            settings.allow_options,
            settings.allow_short_selling,
            settings.allow_crypto,
            settings.allow_leveraged_etfs,
        )
        if any(forbidden):
            raise RuntimeError("Hard risk v1 must remain cash-equity long-only")
        if not 0 < settings.max_position_fraction <= 1.0:
            raise RuntimeError("Hard position cap must be in (0, 100%]")
        if settings.max_positions < 1 or settings.max_positions > 10:
            raise RuntimeError("Hard position count must remain between one and ten")
        if settings.max_trade_usd <= 0 or not 0 < settings.max_trade_fraction <= 1.0:
            raise RuntimeError("Trade caps must be positive and no more than 100% of NAV")
        if not 0 < settings.max_daily_turnover_fraction <= 1.0:
            raise RuntimeError("Hard daily turnover cap may not exceed 100%")
        if not 0 < settings.max_daily_loss_fraction <= 0.08:
            raise RuntimeError("Daily loss circuit breaker may not exceed 8%")
        if not 0 < settings.max_portfolio_drawdown_fraction <= 0.20:
            raise RuntimeError("Portfolio drawdown circuit breaker may not exceed 20%")
        if not 30 <= settings.max_decision_age_seconds <= 600:
            raise RuntimeError("Decision-to-order age must remain between 30 and 600 seconds")
        if not 0 < settings.max_decision_price_drift_fraction <= 0.02:
            raise RuntimeError("Decision price drift cap may not exceed 2%")
        return settings


@dataclass(frozen=True)
class ExecutionSettings:
    max_decision_runs_per_day: int
    decision_windows: Tuple[str, ...]
    order_type: str
    time_in_force: str
    market_hours: str

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ExecutionSettings":
        settings = cls(
            max_decision_runs_per_day=int(raw["max_decision_runs_per_day"]),
            decision_windows=tuple(str(value) for value in raw["decision_windows"]),
            order_type=str(raw["order_type"]),
            time_in_force=str(raw["time_in_force"]),
            market_hours=str(raw["market_hours"]),
        )
        if settings.max_decision_runs_per_day > 3:
            raise RuntimeError("Phase 2 permits at most three decision runs per day")
        if settings.order_type != "market" or settings.market_hours != "regular_hours":
            raise RuntimeError("Phase 2 supports regular-hours market orders only")
        if settings.time_in_force != "gfd":
            raise RuntimeError("Phase 2 orders must be good-for-day")
        return settings


@dataclass(frozen=True)
class CapitalPlan:
    target_value_usd: float
    target_horizon_years: int
    planned_contributions_usd: Tuple[float, ...]
    planned_contribution_months: Tuple[int, ...]

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "CapitalPlan":
        settings = cls(
            target_value_usd=float(raw["target_value_usd"]),
            target_horizon_years=int(raw["target_horizon_years"]),
            planned_contributions_usd=tuple(
                float(value) for value in raw["planned_contributions_usd"]
            ),
            planned_contribution_months=tuple(
                int(value) for value in raw["planned_contribution_months"]
            ),
        )
        if settings.target_horizon_years > 5:
            raise RuntimeError("The confirmed target horizon may not exceed five years")
        if len(settings.planned_contributions_usd) != len(
            settings.planned_contribution_months
        ):
            raise RuntimeError("Contribution amounts and months must align")
        return settings


@dataclass(frozen=True)
class Settings:
    mode: str
    live_trading: bool
    strategy_version: str
    prompt_version: str
    openai_model: str
    max_input_tokens: int
    max_output_tokens: int
    max_llm_calls_per_run: int
    max_tool_calls_per_run: int
    max_mcp_calls_per_decision_run: int
    max_estimated_cost_per_run_usd: float
    max_monthly_llm_budget_usd: float
    notification_required: bool
    health: HealthSettings
    monitor: MonitorSettings
    forecast: ForecastSettings
    research: ResearchSettings
    portfolio: PortfolioSettings
    risk: RiskSettings
    execution: ExecutionSettings
    capital_plan: CapitalPlan

    @classmethod
    def load(cls, path: Path) -> "Settings":
        with path.open("rb") as handle:
            raw = tomllib.load(handle)

        mode = os.getenv("AI_INVESTOR_MODE", str(raw["mode"]))
        live = os.getenv(
            "AI_INVESTOR_LIVE_TRADING", str(raw["live_trading"])
        ).lower() in {"1", "true", "yes"}

        if mode not in {"DRY_RUN", "SHADOW", "LIVE"}:
            raise RuntimeError("mode must be DRY_RUN, SHADOW, or LIVE")
        if (mode == "LIVE") != live:
            raise RuntimeError(
                "LIVE requires both mode=LIVE and live_trading=true; all other "
                "modes require live_trading=false"
            )
        if int(raw["max_llm_calls_per_run"]) != 1:
            raise RuntimeError("Decision runs must use exactly one bounded LLM call")
        if not 1 <= int(raw["max_tool_calls_per_run"]) <= 13:
            raise RuntimeError("Research tool budget may not exceed thirteen")
        if not 1 <= int(raw["max_mcp_calls_per_decision_run"]) <= 50:
            raise RuntimeError("Decision MCP call budget must be between one and fifty")

        return cls(
            mode=mode,
            live_trading=live,
            strategy_version=str(raw["strategy_version"]),
            prompt_version=str(raw["prompt_version"]),
            openai_model=os.getenv(
                "AI_INVESTOR_OPENAI_MODEL", str(raw["openai_model"])
            ),
            max_input_tokens=int(raw["max_input_tokens"]),
            max_output_tokens=int(raw["max_output_tokens"]),
            max_llm_calls_per_run=int(raw["max_llm_calls_per_run"]),
            max_tool_calls_per_run=int(raw["max_tool_calls_per_run"]),
            max_mcp_calls_per_decision_run=int(
                raw["max_mcp_calls_per_decision_run"]
            ),
            max_estimated_cost_per_run_usd=float(
                raw["max_estimated_cost_per_run_usd"]
            ),
            max_monthly_llm_budget_usd=float(raw["max_monthly_llm_budget_usd"]),
            notification_required=bool(raw["notification_required"]),
            health=HealthSettings.from_dict(raw.get("health", {})),
            monitor=MonitorSettings.from_dict(raw["monitor"]),
            forecast=ForecastSettings.from_dict(raw["forecast"]),
            research=ResearchSettings.from_dict(raw["research"]),
            portfolio=PortfolioSettings.from_dict(raw["portfolio"]),
            risk=RiskSettings.from_dict(raw["risk"]),
            execution=ExecutionSettings.from_dict(raw["execution"]),
            capital_plan=CapitalPlan.from_dict(raw["capital_plan"]),
        )
