from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, time as clock_time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from zoneinfo import ZoneInfo

from .config import Settings
from .decision_funnel import build_candidate_funnel, finalize_candidate_funnel
from .execution import (
    assert_live_armed,
    assert_live_armed_fingerprint,
    execute_orders,
    execution_toolset,
    fetch_broker_state,
    plan_orders,
    reconcile_orders,
)
from .event_engine import build_event_shadow_forecasts
from .forecasting import (
    calibration_diagnostics,
    candidate_forecasts,
    investment_candidate_forecasts,
    investment_eligibility_failures,
    forecast_assets,
    history_integrity_issues,
    holding_gate_failures,
    infer_market_regime,
    model_integrity_issues,
)
from .health import clear_operational_failures, report_operational_failure
from .ledger import Ledger
from .market_data import HistoricalCache, get_daily_histories
from .monitor import UniverseCache, run_monitor_cycle
from .model_snapshot import get_or_create_daily_models
from .notification import (
    build_decision_email,
    classify_intraday_tone,
    flush_outbox,
    require_email_configuration,
    send_or_queue,
)
from .portfolio import TargetPortfolio, optimize_portfolio
from .public_filings import collect_sec_filings
from .research import (
    ResearchResult,
    allowed_symbols,
    analyze_candidates,
    collect_research,
    persist_security_facts,
    plan_deep_research,
    plan_sec_research,
    research_context_signature,
)
from .robinhood_mcp import RobinhoodMCPClient
from .universe import assemble_research_universe, entries_to_json


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


def _decision_reason(
    now: datetime,
    settings: Settings,
    has_trigger: bool,
    completed_scheduled_reasons: set[str],
) -> Optional[str]:
    local = now.astimezone(ZoneInfo(settings.monitor.market_timezone))
    minute = local.hour * 60 + local.minute
    due_windows = []
    for value in sorted(settings.execution.decision_windows):
        hour, minute_value = (int(part) for part in value.split(":"))
        start = hour * 60 + minute_value
        if start <= minute:
            due_windows.append("scheduled_" + value.replace(":", ""))
    for reason in due_windows:
        if reason not in completed_scheduled_reasons:
            return reason
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
    if settings.mode == "LIVE" and monitor.account_fingerprint:
        # Validate persistent authorization on every in-hours observation cycle,
        # not only after an expensive LLM decision has already been made.
        assert_live_armed_fingerprint(
            settings, root, monitor.account_fingerprint
        )
        clear_operational_failures(root, ("live_authorization",))
    if monitor.status != "completed_read_only":
        if monitor.status in {
            "skipped_stale_market_data",
            "skipped_incomplete_market_data",
        }:
            report_operational_failure(
                root,
                component="market_data",
                error=f"Monitor cycle stopped: {monitor.status}",
                repeat_minutes=settings.health.persistent_failure_repeat_minutes,
                occurrence_threshold=settings.health.degraded_cycles_before_alert,
                now=current,
            )
        return AgentResult(status="monitor_only", monitor_status=monitor.status)
    clear_operational_failures(
        root,
        ("market_data", "robinhood_service", "robinhood_authentication"),
    )
    monitor_payload = _latest_monitor_payload(monitor.log_path)
    local = current.astimezone(ZoneInfo(settings.monitor.market_timezone))
    trading_date = local.date().isoformat()
    ledger_path = root / ".local" / "state" / "ledger.sqlite"
    with Ledger(ledger_path) as ledger:
        reason = _decision_reason(
            current,
            settings,
            bool(monitor.triggers),
            ledger.scheduled_decision_reasons(trading_date),
        )
        if reason is None:
            return AgentResult(
                status="monitored_no_decision_trigger", monitor_status=monitor.status
            )
        last_decision = ledger.last_decision_at(trading_date)
        if last_decision is not None:
            elapsed_minutes = (
                current.astimezone(timezone.utc) - last_decision.astimezone(timezone.utc)
            ).total_seconds() / 60.0
            if elapsed_minutes < settings.execution.minimum_minutes_between_decisions:
                remaining = settings.execution.minimum_minutes_between_decisions - elapsed_minutes
                return AgentResult(
                    status="decision_cooldown",
                    monitor_status=monitor.status,
                    detail=f"{remaining:.1f} minutes remaining",
                )
        if reason == "market_trigger":
            slot = local.strftime("%H%M")
            decision_reason_key = f"market_trigger_{slot}"
        else:
            decision_reason_key = reason
        decision_key = (
            f"{trading_date}|{decision_reason_key}|{settings.strategy_version}"
        )
        run_id = str(uuid.uuid4())
        if ledger.decision_exists(decision_key):
            return AgentResult(status="duplicate_decision_suppressed", decision_key=decision_key)
        if (
            reason == "market_trigger"
            and ledger.event_decision_runs_today(trading_date)
            >= settings.execution.max_event_decision_runs_per_day
        ):
            return AgentResult(
                status="event_decision_budget_exhausted", decision_key=decision_key
            )
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
            ledger.prune_holding_exit_states(
                {position.symbol for position in state.positions}
            )
            entries = monitor_payload.get("universe") or []
            cached_universe = UniverseCache(
                root / ".local" / "state" / "universe.json"
            ).load(trading_date)
            if cached_universe is None:
                raise RuntimeError(
                    "Broad research universe unavailable after monitor refresh"
                )
            broad_entries = assemble_research_universe(
                settings.research.quantitative_universe_size,
                [position.symbol for position in state.positions],
                cached_universe["large_candidates"],
                cached_universe["mid_candidates"],
                cached_universe["small_candidates"],
                cached_universe["event_candidates"],
            )
            broad_entries_json = entries_to_json(broad_entries)
            context_entries = [
                item for item in entries if not item.get("investable", True)
            ]
            analysis_entries = broad_entries_json + context_entries
            core_symbols = list(dict.fromkeys(
                [
                    str(item.get("symbol", "")).upper()
                    for item in entries
                    if item.get("symbol")
                ]
                + [position.symbol for position in state.positions]
            ))
            symbols = list(dict.fromkeys(
                [
                    str(item.get("symbol", "")).upper()
                    for item in analysis_entries
                    if item.get("symbol")
                ]
                + core_symbols
            ))
            forecast_started = time.monotonic()
            histories = get_daily_histories(
                client, HistoricalCache(root / ".local" / "state" / "historicals"),
                symbols, trading_date, settings.forecast.history_calendar_days,
                settings.forecast.min_history_bars,
            )
            history_integrity = history_integrity_issues(
                histories, settings.forecast.max_daily_close_ratio
            )
            if "SPY" in history_integrity:
                raise RuntimeError(
                    "MODEL_INTEGRITY_BLOCKED: benchmark history failed: "
                    + ";".join(history_integrity["SPY"])
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
            core_investable_symbols = {
                str(item.get("symbol", "")).upper()
                for item in entries
                if item.get("investable", True)
            }
            core_investable_symbols.update(
                position.symbol for position in state.positions
            )
            investable_symbols = {
                str(item.get("symbol", "")).upper()
                for item in analysis_entries
                if item.get("investable", True)
            }
            investable_symbols.update(position.symbol for position in state.positions)
            forecast_histories = {
                symbol: bars
                for symbol, bars in histories.items()
                if (symbol == "SPY" or symbol in core_investable_symbols)
                and symbol not in history_integrity
            }
            models, model_snapshot = get_or_create_daily_models(
                root, trading_date, forecast_histories, settings.forecast
            )
            forecasts, models = forecast_assets(
                forecast_histories, settings.forecast, models=models
            )
            broad_forecast_histories = {
                symbol: bars
                for symbol, bars in histories.items()
                if (symbol == "SPY" or symbol in investable_symbols)
                and symbol not in history_integrity
            }
            broad_forecasts, _ = forecast_assets(
                broad_forecast_histories, settings.forecast, models=models
            )
            broad_model_integrity_issues = model_integrity_issues(
                broad_forecasts, models, settings.forecast
            )
            broad_quarantine_symbols = {
                issue.split(":", 1)[0]
                for issue in broad_model_integrity_issues
                if ":" in issue
            }
            broad_forecasts = [
                item for item in broad_forecasts
                if item.symbol not in broad_quarantine_symbols
            ]
            monitor_quotes = _quote_map(monitor_payload)
            event_shadow_forecasts = build_event_shadow_forecasts(
                histories=forecast_histories,
                quotes=monitor_quotes,
                entries=entries,
                triggers=monitor.triggers,
                settings=settings.event_engine,
                observed_at=current.isoformat(),
            )
            event_shadow_resolutions = ledger.resolve_event_shadow_signals(
                forecast_histories
            )
            event_shadow_insertions = ledger.record_event_shadow_signals(
                run_id,
                trading_date,
                [item.to_dict() for item in event_shadow_forecasts],
            )
            integrity_issues = model_integrity_issues(
                forecasts, models, settings.forecast
            )
            for position in state.positions:
                if position.symbol in history_integrity:
                    integrity_issues.append(
                        f"held_symbol_history_integrity={position.symbol}"
                    )
            integrity_issues = sorted(set(integrity_issues))
            event_symbols = {
                str(item.get("symbol", "")).upper()
                for item in analysis_entries
                if item.get("bucket") == "event" and item.get("symbol")
            }
            event_symbols.update(item.symbol for item in event_shadow_forecasts)
            universe_security_state = ledger.active_security_events(
                investable_symbols, trading_date
            )
            continuity_symbols = set(universe_security_state)
            continuity_order = sorted(
                continuity_symbols,
                key=lambda symbol: (
                    max(
                        str(item.get("last_seen_at", ""))
                        for item in universe_security_state[symbol]
                    ),
                    symbol,
                ),
                reverse=True,
            )
            ranked_candidates = candidate_forecasts(
                broad_forecasts,
                settings.forecast,
                event_symbols=event_symbols | continuity_symbols,
                limit=settings.research.quantitative_shortlist_size,
            )
            forecast_by_symbol = {
                item.symbol: item for item in broad_forecasts
            }
            continuity_candidates = [
                forecast_by_symbol[symbol]
                for symbol in continuity_order
                if symbol in forecast_by_symbol
            ]
            event_shadow_candidates = [
                forecast_by_symbol[item.symbol]
                for item in event_shadow_forecasts
                if item.symbol in forecast_by_symbol
            ]
            candidate_shortlist = list(
                {
                    item.symbol: item
                    for item in (
                        continuity_candidates
                        + event_shadow_candidates
                        + ranked_candidates
                    )
                }.values()
            )[: settings.research.quantitative_shortlist_size]
            timings["history_and_forecast_seconds"] = round(
                time.monotonic() - forecast_started, 3
            )
            by_symbol = {item.symbol: item for item in broad_forecasts}
            # Existing holdings take priority so every held name receives an
            # explicit retain/exit review. The top five opportunities remain in
            # scope, while unused fresh-LLM capacity rotates through the broader
            # shortlist instead of repeatedly paying for identical names.
            holding_forecasts = [
                by_symbol[position.symbol]
                for position in state.positions
                if position.symbol in by_symbol
            ]
            top_candidates = candidate_shortlist[
                : settings.forecast.research_candidate_count
            ]
            event_shadow_by_symbol = {
                item.symbol: item.to_dict() for item in event_shadow_forecasts
            }
            signature_candidates = list({
                item.symbol: item
                for item in holding_forecasts + candidate_shortlist
            }.values())
            research_signatures = {
                item.symbol: research_context_signature(
                    symbol=item.symbol,
                    trading_date=trading_date,
                    model=settings.openai_model,
                    prompt_version=settings.prompt_version,
                    forecast=item,
                    persistent_state=universe_security_state.get(item.symbol, []),
                    triggers=monitor.triggers,
                    event_shadow_forecast=event_shadow_by_symbol.get(item.symbol),
                )
                for item in signature_candidates
            }
            cached_assessments = ledger.cached_research_assessments(
                research_signatures, current.isoformat()
            )
            research_set = []
            fresh_symbols: list[str] = []

            def add_research_candidate(item: Any) -> None:
                if item.symbol in {candidate.symbol for candidate in research_set}:
                    return
                is_fresh = item.symbol not in cached_assessments
                if (
                    is_fresh
                    and len(fresh_symbols)
                    >= settings.research.max_fresh_llm_symbols_per_run
                ):
                    return
                if len(research_set) >= settings.risk.max_positions:
                    return
                research_set.append(item)
                if is_fresh:
                    fresh_symbols.append(item.symbol)

            for item in holding_forecasts:
                add_research_candidate(item)
            for item in top_candidates:
                add_research_candidate(item)
            for item in candidate_shortlist:
                if len(fresh_symbols) >= settings.research.max_fresh_llm_symbols_per_run:
                    break
                if item.symbol in cached_assessments:
                    continue
                add_research_candidate(item)
            candidate_funnel = build_candidate_funnel(
                analysis_entries,
                broad_forecasts,
                settings.forecast,
                event_symbols=event_symbols,
                persistent_symbols=continuity_symbols,
                research_symbols={item.symbol for item in research_set},
                holding_symbols={position.symbol for position in state.positions},
            )
            if not research_set:
                payload = {
                    "status": "no_quantitative_opportunity",
                    "excluded_short_history": missing_history,
                    "model_samples": {
                        key: value.training_samples for key, value in models.items()
                    },
                    "model_snapshot": model_snapshot,
                    "candidate_funnel": candidate_funnel,
                }
                ledger.record_run(
                    run_id=run_id, decision_key=decision_key, trading_date=trading_date,
                    mode=settings.mode, status="no_trade", strategy_version=settings.strategy_version,
                    risk_policy_version=settings.risk.policy_version,
                    portfolio_value=state.portfolio_value, cash=state.cash, payload=payload,
                )
                return AgentResult(status="no_quantitative_opportunity", run_id=run_id, decision_key=decision_key)
            regime = infer_market_regime(forecast_histories)
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
            research_symbols_ordered = [item.symbol for item in research_set]
            fresh_research_set = [
                item for item in research_set if item.symbol in fresh_symbols
            ]
            fresh_research_symbols = [item.symbol for item in fresh_research_set]
            persistent_security_state = {
                symbol: universe_security_state[symbol]
                for symbol in research_symbols_ordered
                if symbol in universe_security_state
            }
            deep_symbols = plan_deep_research(
                fresh_research_symbols,
                entries=analysis_entries,
                triggers=monitor.triggers,
                holding_symbols={position.symbol for position in state.positions},
                persistent_state=persistent_security_state,
                limit=settings.research.deep_candidate_count,
            )
            sec_symbols = plan_sec_research(
                deep_symbols,
                entries=analysis_entries,
                triggers=monitor.triggers,
                persistent_state=persistent_security_state,
                limit=settings.research.max_sec_symbols_per_run,
            )
            expected_research_calls = (
                (3 if fresh_research_symbols else 0)
                + 2 * len(deep_symbols)
                + (1 + 2 * len(sec_symbols) if sec_symbols else 0)
            )
            if expected_research_calls > settings.max_tool_calls_per_run:
                raise RuntimeError(
                    "Configured research plan exceeds the per-run tool-call budget"
                )
            if fresh_research_symbols:
                research_payload, source_tools = collect_research(
                    client,
                    fresh_research_symbols,
                    settings.research.deep_candidate_count,
                    account_number=state.account_number,
                    deep_symbols=deep_symbols,
                )
                sec_payload, sec_tools = collect_sec_filings(
                    root,
                    sec_symbols,
                    user_agent=(
                        f"AIInvestorTest/0.8 ({email_config.sender})"
                        if email_config is not None
                        else "AIInvestorTest/0.8"
                    ),
                    max_symbols=settings.research.max_sec_symbols_per_run,
                )
            else:
                research_payload, source_tools = {}, ()
                sec_payload, sec_tools = {}, ()
            source_tools = tuple(source_tools) + tuple(sec_tools)
            if len(source_tools) > settings.max_tool_calls_per_run:
                raise RuntimeError("Actual research tool-call budget exceeded")
            research_payload = {
                "research_plan": {
                    "deep_symbols": deep_symbols,
                    "sec_symbols": sec_symbols,
                    "fresh_llm_symbols": fresh_research_symbols,
                    "cached_symbols": [
                        symbol for symbol in research_symbols_ordered
                        if symbol in cached_assessments
                    ],
                },
                "sec_filings": sec_payload,
                **research_payload,
            }
            research_symbols = {item.symbol for item in research_set}
            research_payload["universe_eligibility"] = [
                item
                for item in analysis_entries
                if str(item.get("symbol", "")).upper() in set(fresh_research_symbols)
            ]
            timings["research_tools_seconds"] = round(
                time.monotonic() - research_started, 3
            )
            try:
                if fresh_research_set:
                    fresh_research = analyze_candidates(
                        settings=settings, ledger=ledger, run_id=run_id,
                        forecasts=fresh_research_set, regime=regime,
                        market_context=monitor_payload.get("market_summary") or {},
                        research=research_payload, source_tools=source_tools,
                        persistent_security_state={
                            symbol: persistent_security_state[symbol]
                            for symbol in fresh_research_symbols
                            if symbol in persistent_security_state
                        },
                        event_shadow_forecasts=[
                            item.to_dict() for item in event_shadow_forecasts
                            if item.symbol in set(fresh_research_symbols)
                        ],
                    )
                else:
                    fresh_research = ResearchResult(
                        model=settings.openai_model,
                        assessment={"market_summary": "", "candidates": []},
                        source_tools=(),
                        input_tokens=0,
                        output_tokens=0,
                        estimated_cost_usd=0.0,
                        latency_seconds=0.0,
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
            fresh_by_symbol = {
                str(item.get("symbol") or "").upper(): item
                for item in fresh_research.assessment.get("candidates", [])
            }
            expires_at = (
                current + timedelta(minutes=settings.research.assessment_ttl_minutes)
            ).isoformat()
            combined_candidates = []
            for symbol in research_symbols_ordered:
                assessment = fresh_by_symbol.get(symbol)
                if assessment is None and symbol in cached_assessments:
                    assessment = cached_assessments[symbol]["assessment"]
                if assessment is None:
                    raise RuntimeError(
                        f"Missing fresh or cached research assessment for {symbol}"
                    )
                combined_candidates.append(assessment)
            research = ResearchResult(
                model=settings.openai_model,
                assessment={
                    "market_summary": (
                        fresh_research.assessment.get("market_summary")
                        or "No material research-context change; cached assessments reused."
                    ),
                    "candidates": combined_candidates,
                },
                source_tools=fresh_research.source_tools,
                input_tokens=fresh_research.input_tokens,
                output_tokens=fresh_research.output_tokens,
                estimated_cost_usd=fresh_research.estimated_cost_usd,
                latency_seconds=fresh_research.latency_seconds,
            )
            timings["llm_seconds"] = round(research.latency_seconds, 3)
            persisted_security_facts = persist_security_facts(
                ledger,
                fresh_research.assessment,
                current,
                research_evidence=research_payload,
                persistent_state=persistent_security_state,
            )
            updated_security_state = ledger.active_security_events(
                set(research_symbols_ordered), trading_date
            )
            fresh_store_signatures = {
                symbol: research_context_signature(
                    symbol=symbol,
                    trading_date=trading_date,
                    model=settings.openai_model,
                    prompt_version=settings.prompt_version,
                    forecast=by_symbol[symbol],
                    persistent_state=updated_security_state.get(symbol, []),
                    triggers=monitor.triggers,
                    event_shadow_forecast=event_shadow_by_symbol.get(symbol),
                )
                for symbol in fresh_research_symbols
                if symbol in by_symbol
            }
            ledger.store_research_assessments(
                list(fresh_by_symbol.values()),
                fresh_store_signatures,
                researched_at=current.isoformat(),
                expires_at=expires_at,
                model=settings.openai_model,
                prompt_version=settings.prompt_version,
            )
            allowed = allowed_symbols(research)
            llm_reviewed = [
                item for item in research_set if item.symbol in allowed
            ]
            quantitatively_entry_eligible = investment_candidate_forecasts(
                llm_reviewed,
                settings.forecast,
                allowed_symbols=investable_symbols,
            )
            entry_eligible = quantitatively_entry_eligible
            investment_failures_by_symbol = {
                item.symbol: list(
                    investment_eligibility_failures(item, settings.forecast)
                )
                for item in llm_reviewed
            }
            if integrity_issues:
                for item in llm_reviewed:
                    investment_failures_by_symbol.setdefault(item.symbol, []).append(
                        "model_integrity_blocked"
                    )
            assessments = {
                str(item.get("symbol", "")).upper(): item
                for item in research.assessment.get("candidates", [])
            }
            eligible_by_symbol = {item.symbol: item for item in entry_eligible}
            holding_decisions: Dict[str, Dict[str, Any]] = {}
            minimum_weights: Dict[str, float] = {}
            for position in state.positions:
                if integrity_issues:
                    current_weight = (
                        position.market_value / state.portfolio_value
                        if state.portfolio_value > 0 else 0.0
                    )
                    forecast = by_symbol.get(position.symbol)
                    if forecast is not None:
                        eligible_by_symbol.setdefault(position.symbol, forecast)
                    minimum_weights[position.symbol] = current_weight
                    holding_decisions[position.symbol] = {
                        "status": "model_integrity_blocked_retained",
                        "immediate_exit": False,
                        "consecutive_failures": 0,
                        "required_confirmations": (
                            settings.forecast.holding_exit_confirmation_runs
                        ),
                        "reasons": list(integrity_issues),
                    }
                    continue
                forecast = by_symbol.get(position.symbol)
                assessment = assessments.get(position.symbol, {})
                verdict = str(assessment.get("verdict", "missing_assessment"))
                severity = str(
                    assessment.get("data_quality_severity", "none")
                ).lower()
                quant_failures = (
                    list(holding_gate_failures(forecast, settings.forecast))
                    if forecast is not None
                    else ["missing_forecast"]
                )
                immediate_exit = verdict == "veto" or severity == "critical"
                failing = immediate_exit or verdict != "allow" or bool(quant_failures)
                reason_parts = quant_failures + (
                    [f"llm_verdict={verdict}"] if verdict != "allow" else []
                ) + (["critical_data_conflict"] if severity == "critical" else [])
                confirmations = ledger.update_holding_exit_signal(
                    symbol=position.symbol,
                    decision_key=decision_key,
                    failing=failing,
                    reason=",".join(reason_parts) or "holding_gate_passed",
                )
                exit_confirmed = immediate_exit or (
                    failing
                    and confirmations >= settings.forecast.holding_exit_confirmation_runs
                )
                if forecast is not None and not exit_confirmed:
                    eligible_by_symbol.setdefault(position.symbol, forecast)
                    if failing and state.portfolio_value > 0:
                        # The first non-critical failure cannot force a sale.
                        minimum_weights[position.symbol] = (
                            position.market_value / state.portfolio_value
                        )
                holding_decisions[position.symbol] = {
                    "status": (
                        "exit_confirmed" if exit_confirmed
                        else "pending_exit_retained" if failing
                        else "holding_gate_passed"
                    ),
                    "immediate_exit": immediate_exit,
                    "consecutive_failures": confirmations,
                    "required_confirmations": (
                        settings.forecast.holding_exit_confirmation_runs
                    ),
                    "reasons": reason_parts,
                }
            eligible = list(eligible_by_symbol.values())
            # Broad-universe candidates are discovered from completed daily bars,
            # but LIVE decisions must establish a fresh decision-time quote before
            # optimization. A second quote is fetched immediately before execution
            # and the hard risk engine rejects excessive drift between the two.
            decision_quotes = _fresh_execution_quotes(
                client, sorted({item.symbol for item in eligible})
            ) if eligible else {}
            sectors = {
                str(item.get("symbol", "")).upper(): str(item.get("sector", ""))
                for item in analysis_entries
                if item.get("symbol")
            }
            current_weights = {
                position.symbol: position.market_value / state.portfolio_value
                for position in state.positions
                if state.portfolio_value > 0 and position.market_value > 0
            }
            if integrity_issues:
                target = TargetPortfolio(
                    weights=current_weights,
                    cash_weight=max(0.0, state.cash / state.portfolio_value)
                    if state.portfolio_value > 0 else 1.0,
                    objective_value=0.0,
                    diagnostics={
                        "model_integrity_gate": "BLOCKED",
                        "issues": integrity_issues,
                    },
                )
            elif settings.forecast.execution_calibration_approved:
                target = optimize_portfolio(
                    eligible,
                    broad_forecast_histories,
                    settings.portfolio,
                    settings.risk,
                    sectors,
                    current_weights=current_weights,
                    minimum_weights=minimum_weights,
                )
            else:
                # Research-only calibration mode must never liquidate existing
                # positions as a side effect of withholding unapproved buys.
                target = TargetPortfolio(
                    weights=current_weights,
                    cash_weight=max(0.0, state.cash / state.portfolio_value)
                    if state.portfolio_value > 0
                    else 1.0,
                    objective_value=0.0,
                )
            reference_quotes = dict(monitor_quotes)
            reference_quotes.update(decision_quotes)
            prices = {
                symbol: float(item.get("last_trade_price") or 0)
                for symbol, item in reference_quotes.items()
            }
            orders = [] if integrity_issues else plan_orders(
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
                    for item in analysis_entries
                    if item.get("investable", True)
                },
                decision_started_at=current,
                reference_prices=prices,
            )
            candidate_funnel = finalize_candidate_funnel(
                candidate_funnel,
                assessments=assessments,
                investment_failures=investment_failures_by_symbol,
                holding_decisions=holding_decisions,
                target_weights=target.weights,
                orders=outcomes,
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
                investment_eligibility_failures=investment_failures_by_symbol,
                holding_decisions=holding_decisions,
                portfolio_diagnostics=target.diagnostics,
                event_shadow_forecasts=[
                    item.to_dict() for item in event_shadow_forecasts
                ],
                research_cache_summary={
                    "quantitative_universe_size": len(broad_entries),
                    "quantitative_forecasts": len(broad_forecasts),
                    "shortlist_size": len(candidate_shortlist),
                    "fresh_llm_symbols": fresh_research_symbols,
                    "cached_symbols": [
                        symbol for symbol in research_symbols_ordered
                        if symbol in cached_assessments
                    ],
                    "ttl_minutes": settings.research.assessment_ttl_minutes,
                    "forecast_quarantine": sorted(broad_quarantine_symbols),
                },
                execution_calibration_approved=(
                    settings.forecast.execution_calibration_approved
                ),
                twenty_day_new_entry_live_enabled=(
                    settings.forecast.twenty_day_new_entry_live_enabled
                ),
                pipeline_error=(
                    "MODEL_INTEGRITY_BLOCKED: " + "; ".join(integrity_issues)
                    if integrity_issues else None
                ),
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
                "structural_regime": regime.to_dict(),
                "intraday_tone": classify_intraday_tone(
                    monitor_payload.get("market_summary") or {}
                ),
                "current_positions": [asdict(item) for item in state.positions],
                "candidate_stocks": [item.to_dict() for item in research_set],
                "execution_calibration_approved": (
                    settings.forecast.execution_calibration_approved
                ),
                "alpha_permissions": {
                    "twenty_day_new_entry_live_enabled": (
                        settings.forecast.twenty_day_new_entry_live_enabled
                    ),
                    "twenty_day_existing_position_management_enabled": (
                        settings.forecast.twenty_day_existing_position_management_enabled
                    ),
                    "twenty_day_shadow_decisions_enabled": (
                        settings.forecast.twenty_day_shadow_decisions_enabled
                    ),
                },
                "event_shadow_forecasts": [
                    item.to_dict() for item in event_shadow_forecasts
                ],
                "event_shadow_tracking": {
                    "new_signals": event_shadow_insertions,
                    "updated_realizations": event_shadow_resolutions,
                },
                "broad_research_universe": {
                    "size": len(broad_entries),
                    "forecast_count": len(broad_forecasts),
                    "shortlist_count": len(candidate_shortlist),
                    "fresh_llm_symbols": fresh_research_symbols,
                    "cached_symbols": [
                        symbol for symbol in research_symbols_ordered
                        if symbol in cached_assessments
                    ],
                    "assessment_ttl_minutes": (
                        settings.research.assessment_ttl_minutes
                    ),
                    "forecast_quarantine": sorted(broad_quarantine_symbols),
                },
                "portfolio_eligible_stocks": [item.symbol for item in eligible],
                "entry_eligible_stocks": [item.symbol for item in entry_eligible],
                "investment_eligibility_failures": investment_failures_by_symbol,
                "holding_decisions": holding_decisions,
                "research_plan": research_payload.get("research_plan", {}),
                "research_sources": {
                    "tools": list(source_tools),
                    "sec_filings": {
                        symbol: {
                            "status": item.get("status"),
                            "cik": item.get("cik"),
                            "company_name": item.get("company_name"),
                            "recent_filings": item.get("recent_filings", []),
                            "latest_filing_excerpt_chars": len(
                                str(item.get("latest_filing_excerpt", ""))
                            ),
                        }
                        for symbol, item in sec_payload.items()
                    },
                },
                "persistent_security_state_before": persistent_security_state,
                "persistent_security_state_after": updated_security_state,
                "persisted_security_fact_count": persisted_security_facts,
                "candidate_funnel": candidate_funnel,
                "model_snapshot": model_snapshot,
                "history_integrity_quarantine": history_integrity,
                "broad_model_integrity_quarantine": {
                    "symbols": sorted(broad_quarantine_symbols),
                    "issues": broad_model_integrity_issues,
                },
                "model_integrity": {
                    "status": "BLOCKED" if integrity_issues else "PASSED",
                    "issues": integrity_issues,
                },
                "model_calibration": {
                    key: calibration_diagnostics(model)
                    for key, model in models.items()
                },
                "excluded_short_history": missing_history,
                "reasoning_summary": research.assessment,
                "target_portfolio": target.to_dict(), "orders": outcomes,
                "portfolio_value": state.portfolio_value, "cash_balance": state.cash,
                "timings": timings,
                "notification": asdict(delivery),
            }
            run_status = (
                "model_integrity_blocked" if integrity_issues else "completed"
            )
            payload["status"] = run_status
            ledger.record_run(
                run_id=run_id, decision_key=decision_key, trading_date=trading_date,
                mode=settings.mode, status=run_status, strategy_version=settings.strategy_version,
                risk_policy_version=settings.risk.policy_version,
                portfolio_value=state.portfolio_value, cash=state.cash, payload=payload,
            )
            log_path = root / "logs" / "decisions" / f"{trading_date}.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, separators=(",", ":"), default=str) + "\n")
            return AgentResult(
                status=run_status, run_id=run_id, decision_key=decision_key,
                monitor_status=monitor.status,
                candidates=tuple(item.symbol for item in research_set),
                orders=tuple(outcomes), llm_cost_usd=research.estimated_cost_usd,
            )
