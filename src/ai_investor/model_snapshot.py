from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

from .config import ForecastSettings
from .forecasting import FORECAST_MODEL_VERSION, RidgeModel, forecast_assets
from .market_data import DailyBar


SNAPSHOT_SCHEMA_VERSION = 1
_TUPLE_FIELDS = {
    "coefficients",
    "feature_means",
    "feature_scales",
    "validation_errors",
    "validation_bias_interval",
    "validation_dates",
    "validation_predictions",
    "validation_targets",
}


def _settings_fingerprint(settings: ForecastSettings) -> str:
    encoded = json.dumps(
        {"model_version": FORECAST_MODEL_VERSION, "settings": asdict(settings)},
        sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _deserialize_model(payload: Mapping[str, object]) -> RidgeModel:
    values = dict(payload)
    for field in _TUPLE_FIELDS:
        values[field] = tuple(values[field])
    return RidgeModel(**values)  # type: ignore[arg-type]


def get_or_create_daily_models(
    root: Path,
    trading_date: str,
    histories: Mapping[str, Sequence[DailyBar]],
    settings: ForecastSettings,
) -> Tuple[Dict[str, RidgeModel], Dict[str, object]]:
    """Freeze fitted models and calibration for a trading day/config pair."""
    fingerprint = _settings_fingerprint(settings)
    directory = root / ".local" / "state" / "model_snapshots"
    path = directory / f"{trading_date}-{fingerprint[:12]}.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != SNAPSHOT_SCHEMA_VERSION
            or payload.get("trading_date") != trading_date
            or payload.get("forecast_settings_fingerprint") != fingerprint
        ):
            raise RuntimeError(f"Invalid daily model snapshot: {path}")
        models = {
            name: _deserialize_model(model)
            for name, model in dict(payload["models"]).items()
        }
        return models, {
            "status": "reused",
            "path": str(path),
            "created_at": payload.get("created_at"),
            "training_symbols": payload.get("training_symbols", []),
            "forecast_settings_fingerprint": fingerprint,
        }

    _, models = forecast_assets(histories, settings)
    data_dates = [
        bar.begins_at[:10]
        for bars in histories.values()
        for bar in bars[-1:]
    ]
    data_as_of = max(data_dates) if data_dates else trading_date
    created_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "trading_date": trading_date,
        "created_at": created_at,
        "data_as_of": data_as_of,
        "forecast_settings_fingerprint": fingerprint,
        "training_symbols": sorted(histories),
        "models": {name: asdict(model) for name, model in models.items()},
    }
    directory.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.replace(path)
    return models, {
        "status": "created",
        "path": str(path),
        "created_at": created_at,
        "training_symbols": payload["training_symbols"],
        "forecast_settings_fingerprint": fingerprint,
    }
