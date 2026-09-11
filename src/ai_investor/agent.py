from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, time as clock_time, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from zoneinfo import ZoneInfo

from .config import Settings
from .execution import (
    assert_live_armed,
    execute_orders,
    execution_toolset,
    fetch_broker_state,
    plan_orders,
    reconcile_orders,
)
from .forecasting import candidate_forecasts, forecast_assets, infer_market_regime
from .ledger import Ledger
from .market_data import HistoricalCache, get_daily_histories
from .monitor import run_monitor_cycle
from .notification import (
    build_decision_email,
    flush_outbox,
    require_email_configuration,
    send_or_queue,
)
from .portfolio import optimize_portfolio
from .research import allowed_symbols, analyze_candidates, collect_research
from .robinhood_mcp import RobinhoodMCPClient


@dataclass(frozen=True)
class AgentResult:
    status: str
    run_id: str = ""
    decision_key: str = ""
    monitor_status: str = ""
    candidates: tuple[str, ...] = ()
    orders: tuple[Dict[str, Any], ...] = ()
    llm_cost_usd: float = 0.0
    detail: str = ""


def _latest_monitor_payload(path: str) -> Dict[str, Any]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return json.loads(lines[-1]) if lines else {}


def _decision_reason(now: datetime, settings: Settings, has_trigger: bool) -> Optional[str]:
    local = now.astimezone(ZoneInfo(settings.monitor.market_timezone))
    minute = local.hour * 60 + local.minute
    for value in settings.execution.decision_windows:
        hour, minute_value = (int(part) for part in value.split(":"))
        start = hour * 60 + minute_value
        if start <= minute < start + settings.monitor.interval_minutes:
            return "scheduled_" + value.replace(":", "")
    return "market_trigger" if has_trigger else None


def _quote_map(payload: Mapping[str, Any]) -> Dict[str, Dict[str, object]]:
    return {
        str(item.get("symbol", "")).upper(): dict(item)
        for item in payload.get("quotes", [])
        if item.get("symbol")
    }


def _tradability_map(payload: Mapping[str, Any], symbols: list[str]) -> Dict[str, bool]:
    rows = (payload.get("data") or {}).get("results") or []
    result: Dict[str, bool] = {}
    for row in rows:
        symbol = str(row.get("symbol") or row.get("ticker") or "").upper()
        if not symbol:
            continue
        result[symbol] = bool(
            row.get("tradable", row.get("tradeable", row.get("state") in {"active", "tradable"}))
        )
    return {symbol: result.get(symbol, False) for symbol in symbols}


def _fresh_execution_quotes(client: RobinhoodMCPClient, symbols: list[str]) -> Dict[str, Dict[str, object]]:
    result: Dict[str, Dict[str, object]] = {}
    for index in range(0, len(symbols), 20):
        payload = client.call_tool("get_equity_quotes", {"symbols": symbols[index:index + 20]})
        rows = (payload.get("data") or {}).get("results") or []
        for item in rows:
            quote = dict(item.get("quote") or item)
            symbol = str(quote.get("symbol", "")).upper()
            if symbol:
                result[symbol] = quote
    return result


def run_agent_cycle(
    settings: Settings,
    root: Path,
    *,
    force_monitor: bool = False,
    no_delay: bool = False,
    now: Optional[datetime] = None,
) -> AgentResult:
    cycle_started = time.monotonic()
    flush_outbox(root)
    current = now or datetime.now(timezone.utc)
    monitor = run_monitor_cycle(settings, root, force=force_monitor, no_delay=no_delay, now=current)
    timings: Dict[str, float] = {
        "monitor_seconds": round(time.monotonic() - cycle_started, 3)
    }
    if monitor.status != "completed_read_only":
        return AgentResult(status="monitor_only", monitor_status=monitor.status)
    monitor_payload = _latest_monitor_payload(monitor.log_path)
    reason = _decision_reason(current, settings, bool(monitor.triggers))
    if reason is None:
        return AgentResult(status="monitored_no_decision_trigger", monitor_status=monitor.status)
    local = current.astimezone(ZoneInfo(settings.monitor.market_timezone))
    trading_date = local.date().isoformat()
    decision_key = f"{trading_date}|{reason}|{settings.strategy_version}"
    run_id = str(uuid.uuid4())
    ledger_path = root / ".local" / "state" / "ledger.sqlite"
    with Ledger(ledger_path) as ledger:
        if ledger.decision_exists(decision_key):
            return AgentResult(status="duplicate_decision_suppressed", decision_key=decision_key)
        if ledger.decision_runs_today(trading_date) >= settings.execution.max_decision_runs_per_day:
            return AgentResult(status="daily_decision_budget_exhausted", decision_key=decision_key)
        with RobinhoodMCPClient(
            max_calls=settings.max_mcp_calls_per_decision_run,
            allowed_tools=execution_toolset(settings.mode == "LIVE"),
            allow_order_submission=settings.mode == "LIVE" and settings.live_trading,
        ) as client:
            state = fetch_broker_state(client)
            timings["account_state_seconds"] = round(
                time.monotonic() - cycle_started - timings["monitor_seconds"], 3
            )
            if settings.mode == "LIVE":
                # Fail before research spend, then check again immediately
                # before every placement inside execute_orders.
                assert_live_armed(
                    settings, root, state.account_number, trading_date
                )
            ledger.record_equity_snapshot(current.isoformat(), trading_date, state.portfolio_value, state.cash)
            reconcile_orders(client, ledger, state, trading_date)
            entries = monitor_payload.get("universe") or []
            symbols = list(dict.fromkeys([str(item.get("symbol", "")).upper() for item in entries if item.get("symbol")] + [p.symbol for p in state.positions]))
            forecast_started = time.monotonic()
            histories = get_daily_histories(
                client, HistoricalCache(root / ".local" / "state" / "historicals"),
                symbols, trading_date, settings.forecast.history_calendar_days,
                settings.forecast.min_history_bars,
            )
            missing_history = [symbol for symbol in symbols if len(histories.get(symbol, [])) < settings.forecast.min_history_bars]
            if "SPY" in missing_history:
                payload = {"status": "insufficient_history", "missing": missing_history}
                ledger.record_run(
                    run_id=run_id, decision_key=decision_key, trading_date=trading_date,
                    mode=settings.mode, status="insufficient_history",
                    strategy_version=settings.strategy_version,
                    risk_policy_version=settings.risk.policy_version,
                    portfolio_value=state.portfolio_value, cash=state.cash, payload=payload,
                )
                return AgentResult(status="insufficient_benchmark_history", run_id=run_id, decision_key=decision_key, detail=",".join(missing_history))
            forecasts, models = forecast_assets(histories, settings.forecast)
            candidates = candidate_forecasts(forecasts, settings.forecast)
            timings["history_and_forecast_seconds"] = round(
                time.monotonic() - forecast_started, 3
            )
            by_symbol = {item.symbol: item for item in forecasts}
            # Existing holdings take priority so every held name receives an
            # explicit retain/exit review before new opportunities consume the
            # ten-name research budget.
            research_set = [
                by_symbol[position.symbol]
                for position in state.positions
                if position.symbol in by_symbol
            ]
            for candidate in candidates:
                if candidate.symbol not in {item.symbol for item in research_set}:
                    research_set.append(candidate)
            research_set = research_set[: settings.risk.max_positions]
            if not research_set:
                payload = {"status": "no_quantitative_opportunity", "excluded_short_history": missing_history, "model_samples": {key: value.training_samples for key, value in models.items()}}
                ledger.record_run(
                    run_id=run_id, decision_key=decision_key, trading_date=trading_date,
                    mode=settings.mode, status="no_trade", strategy_version=settings.strategy_version,
                    risk_policy_version=settings.risk.policy_version,
                    portfolio_value=state.portfolio_value, cash=state.cash, payload=payload,
                )
                return AgentResult(status="no_quantitative_opportunity", run_id=run_id, decision_key=decision_key)
            regime = infer_market_regime(histories)
            try:
                email_config = (
                    require_email_configuration()
                    if settings.notification_required
                    else None
                )
            except RuntimeError as exc:
                ledger.record_run(
                    run_id=run_id,
                    decision_key=decision_key,
                    trading_date=trading_date,
                    mode=settings.mode,
                    status="blocked_notification_not_configured",
                    strategy_version=settings.strategy_version,
                    risk_policy_version=settings.risk.policy_version,
                    portfolio_value=state.portfolio_value,
                    cash=state.cash,
                    payload={"status": "blocked_notification_not_configured"},
                )
                return AgentResult(
                    status="blocked_notification_not_configured",
                    run_id=run_id,
                    decision_key=decision_key,
                    detail=str(exc),
                )
            research_started = time.monotonic()
            expected_research_calls = 2 + 2 * min(
                len(research_set), settings.research.deep_candidate_count
            )
            if expected_research_calls > settings.max_tool_calls_per_run:
                raise RuntimeError(
                    "Configured research plan exceeds the per-run tool-call budget"
                )
            research_payload, source_tools = collect_research(
                client,
                [item.symbol for item in research_set],
                settings.research.deep_candidate_count,
            )
            timings["research_tools_seconds"] = round(
                time.monotonic() - research_started, 3
            )
            try:
                research = analyze_candidates(
                    settings=settings, ledger=ledger, run_id=run_id,
                    forecasts=research_set, regime=regime,
                    market_context=monitor_payload.get("market_summary") or {},
                    research=research_payload, source_tools=source_tools,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                delivery = send_or_queue(
                    root,
                    run_id=run_id,
                    subject=f"[AI Investor] LLM call failed | {current.isoformat()[:16]}",
                    body=(
                        f"Run: {run_id}\nTime: {current.isoformat()}\n"
                        f"Candidates: {', '.join(item.symbol for item in research_set)}\n"
                        f"The LLM call failed or violated its budget. No order was created.\n"
                        f"Error: {error}"
                    ),
                    config=email_config,
                )
                ledger.record_run(
                    run_id=run_id, decision_key=decision_key,
                    trading_date=trading_date, mode=settings.mode,
                    status="llm_failed_no_trade",
                    strategy_version=settings.strategy_version,
                    risk_policy_version=settings.risk.policy_version,
                    portfolio_value=state.portfolio_value, cash=state.cash,
                    payload={
                        "status": "llm_failed_no_trade",
                        "error": error,
                        "notification": asdict(delivery),
                    },
                )
                return AgentResult(
                    status="llm_failed_no_trade", run_id=run_id,
                    decision_key=decision_key, detail=error,
                )
            timings["llm_seconds"] = round(research.latency_seconds, 3)
            allowed = allowed_symbols(research)
            eligible = [item for item in research_set if item.symbol in allowed]
            target = optimize_portfolio(eligible, histories, settings.portfolio, settings.risk)
            monitor_quotes = _quote_map(monitor_payload)
            prices = {symbol: float(item.get("last_trade_price") or 0) for symbol, item in monitor_quotes.items()}
            orders = plan_orders(
                decision_key=decision_key, target_weights=target.weights, state=state,
                prices=prices, minimum_trade_usd=settings.portfolio.min_trade_usd,
            )
            symbols_to_trade = [item.symbol for item in orders]
            execution_quotes = _fresh_execution_quotes(client, symbols_to_trade) if symbols_to_trade else {}
            tradability: Dict[str, bool] = {}
            for index in range(0, len(symbols_to_trade), 10):
                batch = symbols_to_trade[index:index + 10]
                payload = client.call_tool(
                    "get_equity_tradability",
                    {"symbols": batch, "account_number": state.account_number},
                )
                tradability.update(_tradability_map(payload, batch))
            outcomes = execute_orders(
                settings=settings, root=root, ledger=ledger, client=client, state=state,
                orders=orders, quotes=execution_quotes, tradability=tradability,
                trading_date=trading_date, now=datetime.now(timezone.utc),
                allowed_buy_symbols={
                    str(item.get("symbol", "")).upper()
                    for item in entries
                    if item.get("bucket") != "position"
                },
                decision_started_at=current,
                reference_prices=prices,
            )
            timings["decision_to_orders_seconds"] = round(
                time.monotonic() - cycle_started, 3
            )
            subject, body = build_decision_email(
                run_id=run_id,
                timestamp=current.isoformat(),
                mode=settings.mode,
                portfolio_value=state.portfolio_value,
                cash=state.cash,
                quote_count=monitor.quote_count,
                triggers=monitor.triggers,
                intraday_market_summary=monitor_payload.get("market_summary") or {},
                regime=regime,
                forecasts=research_set,
                research=research,
                target_weights=target.weights,
                orders=outcomes,
                timings=timings,
            )
            delivery = send_or_queue(
                root,
                run_id=run_id,
                subject=subject,
                body=body,
                config=email_config,
            )
            payload = {
                "run_id": run_id, "timestamp": current.isoformat(), "portfolio": "agentic",
                "strategy_version": settings.strategy_version, "risk_policy_version": settings.risk.policy_version,
                "llm_model": settings.openai_model, "llm_usage": asdict(research),
                "market_snapshot": monitor_payload.get("quotes", []),
                "intraday_market_summary": monitor_payload.get("market_summary") or {},
                "current_positions": [asdict(item) for item in state.positions],
                "candidate_stocks": [item.to_dict() for item in research_set],
                "excluded_short_history": missing_history,
                "reasoning_summary": research.assessment,
                "target_portfolio": target.to_dict(), "orders": outcomes,
                "portfolio_value": state.portfolio_value, "cash_balance": state.cash,
                "timings": timings,
                "notification": asdict(delivery),
            }
            ledger.record_run(
                run_id=run_id, decision_key=decision_key, trading_date=trading_date,
                mode=settings.mode, status="completed", strategy_version=settings.strategy_version,
                risk_policy_version=settings.risk.policy_version,
                portfolio_value=state.portfolio_value, cash=state.cash, payload=payload,
            )
            log_path = root / "logs" / "decisions" / f"{trading_date}.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, separators=(",", ":"), default=str) + "\n")
            return AgentResult(
                status="completed", run_id=run_id, decision_key=decision_key,
                monitor_status=monitor.status,
                candidates=tuple(item.symbol for item in research_set),
                orders=tuple(outcomes), llm_cost_usd=research.estimated_cost_usd,
            )
