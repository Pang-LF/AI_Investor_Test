from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, Mapping, Sequence

from .config import Settings
from .forecasting import (
    _aligned_series,
    _date_block_positive_probability,
    _features,
    _quantile,
    _raw_positive_probability,
    _rolling_beta,
    build_training_samples,
    fit_ridge_model,
    history_integrity_issues,
)
from .market_data import DailyBar, HistoricalCache, get_daily_histories
from .robinhood_mcp import RobinhoodMCPClient


@dataclass(frozen=True)
class AuditDefinition:
    signal_days: int = 60
    winner_quantile: float = 0.90
    minimum_excess_return_5d: float = 0.05
    minimum_excess_return_20d: float = 0.08
    history_calendar_days: int = 550


def _universe_symbols(payload: Mapping[str, Any]) -> tuple[list[str], set[str]]:
    broad: list[str] = []
    for key in (
        "large_candidates", "mid_candidates", "small_candidates",
        "event_candidates", "event_bucket", "entries",
    ):
        for item in payload.get(key) or []:
            symbol = str(
                item if isinstance(item, str) else item.get("symbol") or ""
            ).upper()
            if symbol and symbol not in broad:
                broad.append(symbol)
    active = {
        str(item.get("symbol") or "").upper()
        for item in payload.get("entries") or []
        if item.get("symbol")
    }
    if "SPY" not in broad:
        broad.append("SPY")
    active.add("SPY")
    return broad, active


def fetch_audit_histories(
    root: Path, settings: Settings, definition: AuditDefinition
) -> tuple[Dict[str, list[DailyBar]], Dict[str, Any]]:
    universe = json.loads((root / ".local/state/universe.json").read_text())
    broad, active = _universe_symbols(universe)
    required_calls = math.ceil(len(broad) / 10)
    if required_calls > 48:
        raise RuntimeError(f"Audit universe needs too many broker calls: {required_calls}")
    with RobinhoodMCPClient(
        max_calls=required_calls + 2,
        allowed_tools={"get_equity_historicals"},
        allow_order_submission=False,
    ) as client:
        histories = get_daily_histories(
            client,
            HistoricalCache(root / ".local/state/audits/historicals"),
            broad,
            str(universe["trading_date"]),
            definition.history_calendar_days,
            settings.forecast.min_history_bars,
        )
    return histories, {
        "universe_date": universe["trading_date"],
        "broad_static_symbols": len(broad) - 1,
        "active_static_symbols": len(active) - 1,
        "historical_calls": required_calls,
    }


def _forward_excess(
    closes: Sequence[float], benchmark: Sequence[float], index: int, horizon: int
) -> float:
    beta = _rolling_beta(closes, benchmark, index)
    return sum(
        closes[future] / closes[future - 1] - 1.0
        - beta * (benchmark[future] / benchmark[future - 1] - 1.0)
        for future in range(index + 1, index + horizon + 1)
    )


def _event_proxy(closes: Sequence[float], volumes: Sequence[float], index: int) -> bool:
    if index < 20 or closes[index - 1] <= 0:
        return False
    daily_return = closes[index] / closes[index - 1] - 1.0
    average_volume = fmean(volumes[index - 20:index])
    relative_volume = volumes[index] / average_volume if average_volume > 0 else 0.0
    return abs(daily_return) >= 0.04 or (
        abs(daily_return) >= 0.02 and relative_volume >= 2.0
    )


def _live_funnel_summary(root: Path) -> Dict[str, Any]:
    dispositions: Counter[str] = Counter()
    decision_runs = 0
    for path in sorted((root / "logs/decisions").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            decision_runs += 1
            for row in payload.get("candidate_funnel") or []:
                dispositions[str(row.get("final_disposition") or "unknown")] += 1
    return {
        "decision_runs": decision_runs,
        "dispositions": dict(sorted(dispositions.items())),
        "limitation": "Only actual logged runs contain historical LLM and optimizer outcomes.",
    }


def run_opportunity_capture_audit(
    root: Path,
    settings: Settings,
    histories: Mapping[str, Sequence[DailyBar]],
    definition: AuditDefinition,
    source_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    universe = json.loads((root / ".local/state/universe.json").read_text())
    broad_symbols, active_symbols = _universe_symbols(universe)
    integrity = history_integrity_issues(
        histories, settings.forecast.max_daily_close_ratio
    )
    if "SPY" in integrity:
        raise RuntimeError("Audit benchmark failed history integrity checks")
    clean = {symbol: bars for symbol, bars in histories.items() if symbol not in integrity}
    benchmark_bars = clean["SPY"]
    benchmark_dates = [bar.begins_at[:10] for bar in benchmark_bars]
    eligible_dates = benchmark_dates[
        settings.forecast.min_history_bars:-20
    ]
    signal_dates = eligible_dates[-definition.signal_days:]
    if len(signal_dates) < definition.signal_days:
        raise RuntimeError("Insufficient completed dates for a 60-day audit")
    audit_start_index = benchmark_dates.index(signal_dates[0])
    training_cutoff = benchmark_dates[audit_start_index - 20]
    active_histories = {
        symbol: bars for symbol, bars in clean.items() if symbol in active_symbols
    }
    training = [
        sample
        for sample in build_training_samples(
            active_histories, 20, settings.forecast.max_daily_close_ratio
        )
        if sample[0] < training_cutoff
    ]
    model = fit_ridge_model(
        training,
        20,
        settings.forecast.ridge_penalty,
        settings.forecast.min_training_samples,
        settings.forecast.shrinkage,
        settings.forecast.minimum_bias_correction_date_blocks,
        settings.forecast.bias_bootstrap_samples,
    )

    rows_by_date: Dict[str, list[Dict[str, Any]]] = {date: [] for date in signal_dates}
    for symbol in broad_symbols:
        bars = clean.get(symbol)
        if not bars or symbol == "SPY":
            continue
        dates, closes, volumes, benchmark = _aligned_series(bars, benchmark_bars)
        positions = {date: index for index, date in enumerate(dates)}
        for date in signal_dates:
            index = positions.get(date)
            if index is None or index < settings.forecast.min_history_bars or index + 20 >= len(dates):
                continue
            row: Dict[str, Any] = {
                "date": date,
                "symbol": symbol,
                "in_static_active_60": symbol in active_symbols,
                "event_proxy": _event_proxy(closes, volumes, index),
                "realized_excess_5d": _forward_excess(closes, benchmark, index, 5),
                "realized_excess_20d": _forward_excess(closes, benchmark, index, 20),
            }
            if symbol in active_symbols:
                features = _features(closes, volumes, benchmark, index)
                model_prediction = model.predict(features)
                raw_expected = settings.forecast.shrinkage * model_prediction
                uncapped_adjusted = raw_expected + model.applied_validation_bias
                adjusted = max(
                    -settings.forecast.max_abs_forecast_20d,
                    min(uncapped_adjusted, settings.forecast.max_abs_forecast_20d),
                )
                calibrated_probability = _date_block_positive_probability(
                    raw_expected,
                    model.validation_errors,
                    model.validation_dates,
                    settings.forecast.calibration_prior_date_blocks,
                )
                row.update({
                    "model_prediction_20d": model_prediction,
                    "raw_expected_20d": raw_expected,
                    "adjusted_expected_20d": adjusted,
                    "uncertainty_20d": model.residual_std,
                    "raw_probability": _raw_positive_probability(
                        raw_expected, model.residual_std
                    ),
                    "calibrated_probability": calibrated_probability,
                    "edge_ratio": adjusted / max(model.residual_std, 1e-9),
                })
            rows_by_date[date].append(row)

    all_rows: list[Dict[str, Any]] = []
    for date, rows in rows_by_date.items():
        for horizon, minimum in ((5, definition.minimum_excess_return_5d),
                                 (20, definition.minimum_excess_return_20d)):
            key = f"realized_excess_{horizon}d"
            threshold = max(
                minimum,
                _quantile([float(row[key]) for row in rows], definition.winner_quantile),
            )
            for row in rows:
                row[f"winner_{horizon}d"] = float(row[key]) >= threshold
                row[f"winner_threshold_{horizon}d"] = threshold
        qualified = [
            row for row in rows
            if row["in_static_active_60"] and (
                row["event_proxy"] or (
                    row.get("raw_expected_20d", -math.inf)
                    >= settings.forecast.research_min_raw_expected_excess_return_20d
                    and row.get("raw_probability", 0.0)
                    >= settings.forecast.research_min_raw_probability_positive
                )
            )
        ]
        selected = {
            row["symbol"] for row in sorted(
                qualified,
                key=lambda item: (
                    item.get("raw_expected_20d", -math.inf)
                    / max(item.get("uncertainty_20d", 1e-9), 1e-9),
                    item.get("raw_expected_20d", -math.inf),
                ),
                reverse=True,
            )[:settings.forecast.research_candidate_count]
        }
        for row in rows:
            if not row["in_static_active_60"]:
                stage = "outside_static_active_60"
            elif row["symbol"] not in selected:
                stage = (
                    "research_capacity"
                    if row in qualified else "quant_research_gate"
                )
            elif model.validation_date_blocks < settings.forecast.execution_min_calibration_date_blocks:
                stage = "insufficient_calibration"
            elif row["adjusted_expected_20d"] <= settings.forecast.execution_min_bias_adjusted_excess_return_20d:
                stage = "investment_alpha_gate"
            elif row["calibrated_probability"] < settings.forecast.execution_min_calibrated_probability_positive:
                stage = "investment_probability_gate"
            elif row["edge_ratio"] < settings.forecast.execution_min_edge_ratio_20d:
                stage = "investment_edge_gate"
            else:
                stage = "quant_pipeline_pass"
            row["quant_stage"] = stage
            row["selected_for_research_capacity"] = row["symbol"] in selected
            all_rows.append(row)

    horizon_summaries: Dict[str, Any] = {}
    for horizon in (5, 20):
        winners = [row for row in all_rows if row[f"winner_{horizon}d"]]
        active_winners = [row for row in winners if row["in_static_active_60"]]
        stage_counts = Counter(row["quant_stage"] for row in winners)
        event_count = sum(bool(row["event_proxy"]) for row in winners)
        passed = [row for row in winners if row["quant_stage"] == "quant_pipeline_pass"]
        all_passed = [
            row for row in all_rows if row["quant_stage"] == "quant_pipeline_pass"
        ]
        active_rows = [row for row in all_rows if row["in_static_active_60"]]
        active_winner_rate = len(active_winners) / len(active_rows) if active_rows else 0.0
        pass_precision = len(passed) / len(all_passed) if all_passed else 0.0
        largest_by_symbol: Dict[str, Dict[str, Any]] = {}
        for row in winners:
            if row["quant_stage"] == "quant_pipeline_pass":
                continue
            prior = largest_by_symbol.get(row["symbol"])
            if prior is None or row[f"realized_excess_{horizon}d"] > prior[f"realized_excess_{horizon}d"]:
                largest_by_symbol[row["symbol"]] = row
        horizon_summaries[f"{horizon}d"] = {
            "winner_opportunities": len(winners),
            "unique_winner_symbols": len({row["symbol"] for row in winners}),
            "static_active_pool_winner_opportunities": len(active_winners),
            "static_active_pool_coverage": (
                len(active_winners) / len(winners) if winners else 0.0
            ),
            "event_proxy_detected": event_count,
            "event_proxy_recall": event_count / len(winners) if winners else 0.0,
            "quant_pipeline_passed": len(passed),
            "quant_pipeline_recall": len(passed) / len(winners) if winners else 0.0,
            "conditional_quant_recall_inside_static_active_pool": (
                len(passed) / len(active_winners) if active_winners else 0.0
            ),
            "quant_pass_opportunities": len(all_passed),
            "quant_pass_winner_precision": pass_precision,
            "active_pool_winner_base_rate": active_winner_rate,
            "winner_precision_lift": (
                pass_precision / active_winner_rate if active_winner_rate else 0.0
            ),
            "quant_pass_mean_realized_excess": (
                fmean(row[f"realized_excess_{horizon}d"] for row in all_passed)
                if all_passed else 0.0
            ),
            "quant_pass_positive_rate": (
                fmean(row[f"realized_excess_{horizon}d"] > 0 for row in all_passed)
                if all_passed else 0.0
            ),
            "stage_counts": dict(sorted(stage_counts.items())),
            "largest_missed": sorted(
                (
                    {
                        "date": row["date"], "symbol": row["symbol"],
                        "realized_excess": row[f"realized_excess_{horizon}d"],
                        "stage": row["quant_stage"],
                        "event_proxy": row["event_proxy"],
                        "raw_expected_20d": row.get("raw_expected_20d"),
                        "calibrated_probability": row.get("calibrated_probability"),
                        "edge_ratio": row.get("edge_ratio"),
                    }
                    for row in largest_by_symbol.values()
                ),
                key=lambda item: item["realized_excess"],
                reverse=True,
            )[:20],
        }

    return {
        "audit_type": "static-universe_purged_oos_opportunity_capture",
        "definition": asdict(definition),
        "source": dict(source_metadata),
        "limitations": [
            "Broad and active universes are the current scanner snapshot, not historical constituents.",
            "This has survivorship and selection bias and is not a performance backtest.",
            "Daily event_proxy is not the production intraday event scanner.",
            "Overlapping signal dates count repeated opportunities and are not independent events.",
            "Historical news, LLM verdicts, portfolio state, and optimizer decisions are unavailable before logging began.",
        ],
        "signal_window": {
            "first": signal_dates[0], "last": signal_dates[-1],
            "training_cutoff_exclusive": training_cutoff,
            "purge_days": 20,
        },
        "training": {
            "samples": len(training),
            "validation_date_blocks": model.validation_date_blocks,
            "uncertainty_20d_decimal": model.residual_std,
            "applied_validation_bias_decimal": model.applied_validation_bias,
        },
        "history_integrity_quarantine": integrity,
        "horizons": horizon_summaries,
        "actual_logged_funnel": _live_funnel_summary(root),
        "rows": all_rows,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Opportunity Capture Audit",
        "",
        f"Signal window: {report['signal_window']['first']} through {report['signal_window']['last']}",
        "",
        "This is a diagnostic of opportunity coverage, not a return backtest. The broad and",
        "active universes are frozen from the current scanner snapshot, so results contain",
        "survivorship and selection bias. Repeated overlapping signal dates are opportunity",
        "observations, not independent events.",
        "",
        "## Funnel results",
        "",
        "| Horizon | Winner observations | Active-60 coverage | Conditional quant recall | Pass precision/lift | Pass mean excess |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for horizon in ("5d", "20d"):
        item = report["horizons"][horizon]
        lines.append(
            f"| {horizon} | {item['winner_opportunities']} | "
            f"{item['static_active_pool_coverage']:.1%} | "
            f"{item['conditional_quant_recall_inside_static_active_pool']:.1%} | "
            f"{item['quant_pass_winner_precision']:.1%} / {item['winner_precision_lift']:.2f}x | "
            f"{item['quant_pass_mean_realized_excess']:.2%} |"
        )
    lines.extend([
        "",
        "Active-60 coverage measures discovery, while conditional quant recall measures",
        "what the model/funnel retained after a name was already inside the active pool.",
        "Pass precision alone is not profitability: the realized return distribution also matters.",
        "",
        "| Horizon | Event-proxy recall | Quant passes | Positive rate | Active-pool base rate |",
        "|---|---:|---:|---:|---:|",
    ])
    for horizon in ("5d", "20d"):
        item = report["horizons"][horizon]
        lines.append(
            f"| {horizon} | {item['event_proxy_recall']:.1%} | "
            f"{item['quant_pass_opportunities']} | "
            f"{item['quant_pass_positive_rate']:.1%} | "
            f"{item['active_pool_winner_base_rate']:.1%} |"
        )
    five_day = report["horizons"]["5d"]
    twenty_day = report["horizons"]["20d"]
    lines.extend([
        "",
        "## Diagnostic interpretation",
        "",
        "- Within this biased static-universe diagnostic, the active 60 captured only "
        f"{five_day['static_active_pool_coverage']:.1%} of 5d and "
        f"{twenty_day['static_active_pool_coverage']:.1%} of 20d winner observations. "
        "Discovery is therefore a larger measured bottleneck than final execution gates.",
        "- Inside the active pool, the complete quantitative funnel retained only "
        f"{five_day['conditional_quant_recall_inside_static_active_pool']:.1%} of 5d and "
        f"{twenty_day['conditional_quant_recall_inside_static_active_pool']:.1%} of 20d winners; "
        "research capacity is the largest downstream rejection stage.",
        "- Quantitative passes enriched the winner rate by about "
        f"{five_day['winner_precision_lift']:.2f}x / "
        f"{twenty_day['winner_precision_lift']:.2f}x, but their mean realized excess was "
        f"{five_day['quant_pass_mean_realized_excess']:.2%} / "
        f"{twenty_day['quant_pass_mean_realized_excess']:.2%}. "
        "This does not establish tradable alpha and does not justify simply loosening gates.",
    ])
    for horizon in ("5d", "20d"):
        item = report["horizons"][horizon]
        lines.extend(["", f"## {horizon} attribution", "", "| Stage | Count |", "|---|---:|"])
        for stage, count in item["stage_counts"].items():
            lines.append(f"| {stage} | {count} |")
        lines.extend(["", "Largest missed winners:", "", "| Date | Symbol | Excess return | Stage | Event proxy |", "|---|---|---:|---|---|"])
        for row in item["largest_missed"][:15]:
            lines.append(
                f"| {row['date']} | {row['symbol']} | {row['realized_excess']:.2%} | "
                f"{row['stage']} | {row['event_proxy']} |"
            )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    actual = report["actual_logged_funnel"]
    lines.extend([
        "",
        "## Actual logged production funnel",
        "",
        f"Decision runs available: {actual['decision_runs']}",
        "",
        "| Disposition | Count |",
        "|---|---:|",
    ])
    for disposition, count in actual["dispositions"].items():
        lines.append(f"| {disposition} | {count} |")
    lines.extend(["", actual["limitation"]])
    return "\n".join(lines) + "\n"
