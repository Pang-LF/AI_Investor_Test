from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class Ledger:
    """Local durable state for budgets, idempotency, and reconciliation."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(str(path), timeout=30)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self._create_schema()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *_args: object) -> None:
        self.connection.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                decision_key TEXT UNIQUE,
                created_at TEXT NOT NULL,
                trading_date TEXT NOT NULL,
                mode TEXT NOT NULL,
                status TEXT NOT NULL,
                strategy_version TEXT NOT NULL,
                risk_policy_version TEXT NOT NULL,
                portfolio_value REAL,
                cash REAL,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS runs_date_idx
                ON runs(trading_date, created_at);

            CREATE TABLE IF NOT EXISTS equity_snapshots (
                observed_at TEXT PRIMARY KEY,
                trading_date TEXT NOT NULL,
                total_value REAL NOT NULL,
                cash REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS equity_date_idx
                ON equity_snapshots(trading_date, observed_at);

            CREATE TABLE IF NOT EXISTS llm_usage (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                estimated_cost_usd REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS orders (
                ref_id TEXT PRIMARY KEY,
                decision_key TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                trading_date TEXT NOT NULL,
                mode TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                order_type TEXT NOT NULL,
                quantity TEXT,
                dollar_amount TEXT,
                planned_notional REAL NOT NULL,
                status TEXT NOT NULL,
                broker_order_id TEXT,
                filled_quantity REAL,
                average_fill_price REAL,
                response_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS orders_date_idx
                ON orders(trading_date, created_at);

            CREATE TABLE IF NOT EXISTS holding_exit_state (
                symbol TEXT PRIMARY KEY,
                consecutive_failures INTEGER NOT NULL,
                last_decision_key TEXT NOT NULL,
                reason TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS security_event_state (
                symbol TEXT NOT NULL,
                event_type TEXT NOT NULL,
                status TEXT NOT NULL,
                summary TEXT NOT NULL,
                confidence TEXT NOT NULL,
                source_basis_json TEXT NOT NULL,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                valid_until TEXT NOT NULL,
                invalidation_condition TEXT NOT NULL,
                PRIMARY KEY(symbol, event_type)
            );
            CREATE INDEX IF NOT EXISTS security_event_validity_idx
                ON security_event_state(status, valid_until);
            """
        )
        columns = {
            row[1]
            for row in self.connection.execute("PRAGMA table_info(orders)").fetchall()
        }
        if "filled_quantity" not in columns:
            self.connection.execute("ALTER TABLE orders ADD COLUMN filled_quantity REAL")
        if "average_fill_price" not in columns:
            self.connection.execute("ALTER TABLE orders ADD COLUMN average_fill_price REAL")
        # Robinhood's placement response nests the order at data.order. Older
        # builds stored the full response but missed its id; repair those rows
        # so subsequent get_equity_orders responses can be matched safely.
        rows = self.connection.execute(
            "SELECT ref_id, response_json FROM orders WHERE broker_order_id IS NULL"
        ).fetchall()
        for row in rows:
            try:
                payload = json.loads(row["response_json"])
                data = payload.get("data") or {}
                order = data.get("order") or data
                broker_order_id = order.get("id") or order.get("order_id")
            except (AttributeError, json.JSONDecodeError, TypeError):
                broker_order_id = None
            if broker_order_id:
                self.connection.execute(
                    "UPDATE orders SET broker_order_id=? WHERE ref_id=?",
                    (str(broker_order_id), row["ref_id"]),
                )
        self.connection.commit()

    @staticmethod
    def _json(payload: Dict[str, Any]) -> str:
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)

    def decision_exists(self, decision_key: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM runs WHERE decision_key=?", (decision_key,)
        ).fetchone()
        return row is not None

    def event_decision_runs_today(self, trading_date: str) -> int:
        row = self.connection.execute(
            """
            SELECT COUNT(*) AS count FROM runs
            WHERE trading_date=? AND decision_key LIKE ?
            """,
            (trading_date, f"{trading_date}|market_trigger%"),
        ).fetchone()
        return int(row["count"])

    def scheduled_decision_reasons(self, trading_date: str) -> set[str]:
        rows = self.connection.execute(
            "SELECT decision_key FROM runs WHERE trading_date=?",
            (trading_date,),
        ).fetchall()
        return {
            parts[1]
            for row in rows
            for parts in [str(row["decision_key"]).split("|")]
            if len(parts) >= 2 and parts[1].startswith("scheduled_")
        }

    def last_decision_at(self, trading_date: str) -> Optional[datetime]:
        row = self.connection.execute(
            """
            SELECT created_at FROM runs
            WHERE trading_date=? ORDER BY created_at DESC LIMIT 1
            """,
            (trading_date,),
        ).fetchone()
        if not row:
            return None
        return datetime.fromisoformat(str(row["created_at"]).replace("Z", "+00:00"))

    def record_run(
        self,
        *,
        run_id: str,
        decision_key: str,
        trading_date: str,
        mode: str,
        status: str,
        strategy_version: str,
        risk_policy_version: str,
        portfolio_value: float,
        cash: float,
        payload: Dict[str, Any],
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO runs(
                run_id, decision_key, created_at, trading_date, mode, status,
                strategy_version, risk_policy_version, portfolio_value, cash,
                payload_json
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(decision_key) DO UPDATE SET
                status=excluded.status,
                portfolio_value=excluded.portfolio_value,
                cash=excluded.cash,
                payload_json=excluded.payload_json
            """,
            (
                run_id,
                decision_key,
                datetime.now(timezone.utc).isoformat(),
                trading_date,
                mode,
                status,
                strategy_version,
                risk_policy_version,
                portfolio_value,
                cash,
                self._json(payload),
            ),
        )
        self.connection.commit()

    def record_equity_snapshot(
        self, observed_at: str, trading_date: str, total_value: float, cash: float
    ) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO equity_snapshots(
                observed_at, trading_date, total_value, cash
            ) VALUES(?,?,?,?)
            """,
            (observed_at, trading_date, total_value, cash),
        )
        self.connection.commit()

    def daily_open_value(self, trading_date: str) -> Optional[float]:
        row = self.connection.execute(
            """
            SELECT total_value FROM equity_snapshots
            WHERE trading_date=? ORDER BY observed_at ASC LIMIT 1
            """,
            (trading_date,),
        ).fetchone()
        return float(row["total_value"]) if row else None

    def high_watermark(self) -> Optional[float]:
        row = self.connection.execute(
            "SELECT MAX(total_value) AS value FROM equity_snapshots"
        ).fetchone()
        value = row["value"] if row else None
        return float(value) if value is not None else None

    def monthly_llm_cost(self, month_prefix: str) -> float:
        row = self.connection.execute(
            """
            SELECT COALESCE(SUM(estimated_cost_usd), 0) AS cost
            FROM llm_usage WHERE substr(created_at, 1, 7)=?
            """,
            (month_prefix,),
        ).fetchone()
        return float(row["cost"])

    def record_llm_usage(
        self,
        run_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        estimated_cost_usd: float,
    ) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO llm_usage(
                run_id, created_at, model, input_tokens, output_tokens,
                estimated_cost_usd
            ) VALUES(?,?,?,?,?,?)
            """,
            (
                run_id,
                datetime.now(timezone.utc).isoformat(),
                model,
                input_tokens,
                output_tokens,
                estimated_cost_usd,
            ),
        )
        self.connection.commit()

    def daily_order_notional(self, trading_date: str) -> float:
        row = self.connection.execute(
            """
            SELECT COALESCE(SUM(planned_notional), 0) AS value
            FROM orders
            WHERE trading_date=? AND status NOT IN ('blocked','cancelled','rejected','failed')
            """,
            (trading_date,),
        ).fetchone()
        return float(row["value"])

    def daily_order_count(self, trading_date: str) -> int:
        row = self.connection.execute(
            """
            SELECT COUNT(*) AS count FROM orders
            WHERE trading_date=? AND status NOT IN ('blocked','cancelled','rejected','failed')
            """,
            (trading_date,),
        ).fetchone()
        return int(row["count"])

    def get_order(self, ref_id: str) -> Optional[Dict[str, Any]]:
        row = self.connection.execute(
            "SELECT * FROM orders WHERE ref_id=?", (ref_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_order_by_broker_id(self, broker_order_id: str) -> Optional[Dict[str, Any]]:
        row = self.connection.execute(
            "SELECT * FROM orders WHERE broker_order_id=?", (broker_order_id,)
        ).fetchone()
        return dict(row) if row else None

    def update_holding_exit_signal(
        self,
        *,
        symbol: str,
        decision_key: str,
        failing: bool,
        reason: str,
    ) -> int:
        existing = self.connection.execute(
            "SELECT * FROM holding_exit_state WHERE symbol=?", (symbol,)
        ).fetchone()
        if existing and existing["last_decision_key"] == decision_key:
            return int(existing["consecutive_failures"])
        failures = (
            int(existing["consecutive_failures"]) + 1
            if failing and existing
            else 1 if failing else 0
        )
        self.connection.execute(
            """
            INSERT INTO holding_exit_state(
                symbol, consecutive_failures, last_decision_key, reason, updated_at
            ) VALUES(?,?,?,?,?)
            ON CONFLICT(symbol) DO UPDATE SET
                consecutive_failures=excluded.consecutive_failures,
                last_decision_key=excluded.last_decision_key,
                reason=excluded.reason,
                updated_at=excluded.updated_at
            """,
            (
                symbol,
                failures,
                decision_key,
                reason,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.connection.commit()
        return failures

    def get_holding_exit_state(self, symbol: str) -> Optional[Dict[str, Any]]:
        row = self.connection.execute(
            "SELECT * FROM holding_exit_state WHERE symbol=?", (symbol,)
        ).fetchone()
        return dict(row) if row else None

    def prune_holding_exit_states(self, active_symbols: set[str]) -> None:
        if not active_symbols:
            self.connection.execute("DELETE FROM holding_exit_state")
        else:
            placeholders = ",".join("?" for _ in active_symbols)
            self.connection.execute(
                f"DELETE FROM holding_exit_state WHERE symbol NOT IN ({placeholders})",
                tuple(sorted(active_symbols)),
            )
        self.connection.commit()

    def active_security_events(
        self, symbols: set[str], as_of: str
    ) -> Dict[str, list[Dict[str, Any]]]:
        if not symbols:
            return {}
        placeholders = ",".join("?" for _ in symbols)
        rows = self.connection.execute(
            f"""
            SELECT * FROM security_event_state
            WHERE symbol IN ({placeholders})
              AND status='active' AND valid_until>=?
            ORDER BY symbol, event_type
            """,
            (*sorted(symbols), as_of),
        ).fetchall()
        result: Dict[str, list[Dict[str, Any]]] = {}
        for raw in rows:
            item = dict(raw)
            try:
                source_payload = json.loads(item.pop("source_basis_json"))
                item["source_basis"] = list(source_payload.get("items", []))
            except (AttributeError, json.JSONDecodeError, TypeError):
                item["source_basis"] = []
            result.setdefault(str(item["symbol"]), []).append(item)
        return result

    def upsert_security_event(
        self,
        *,
        symbol: str,
        event_type: str,
        status: str,
        summary: str,
        confidence: str,
        source_basis: list[str],
        observed_at: str,
        valid_until: str,
        invalidation_condition: str,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO security_event_state(
                symbol, event_type, status, summary, confidence,
                source_basis_json, first_seen_at, last_seen_at, valid_until,
                invalidation_condition
            ) VALUES(?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(symbol,event_type) DO UPDATE SET
                status=excluded.status,
                summary=excluded.summary,
                confidence=excluded.confidence,
                source_basis_json=excluded.source_basis_json,
                last_seen_at=excluded.last_seen_at,
                valid_until=excluded.valid_until,
                invalidation_condition=excluded.invalidation_condition
            """,
            (
                symbol.upper(),
                event_type,
                status,
                summary,
                confidence,
                self._json({"items": source_basis}),
                observed_at,
                observed_at,
                valid_until,
                invalidation_condition,
            ),
        )
        self.connection.commit()

    def upsert_order(
        self,
        *,
        ref_id: str,
        decision_key: str,
        trading_date: str,
        mode: str,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Optional[str],
        dollar_amount: Optional[str],
        planned_notional: float,
        status: str,
        broker_order_id: Optional[str] = None,
        filled_quantity: Optional[float] = None,
        average_fill_price: Optional[float] = None,
        response: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.connection.execute(
            """
            INSERT INTO orders(
                ref_id, decision_key, created_at, updated_at, trading_date,
                mode, symbol, side, order_type, quantity, dollar_amount,
                planned_notional, status, broker_order_id, filled_quantity,
                average_fill_price, response_json
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(ref_id) DO UPDATE SET
                updated_at=excluded.updated_at,
                status=excluded.status,
                broker_order_id=COALESCE(excluded.broker_order_id, orders.broker_order_id),
                filled_quantity=COALESCE(excluded.filled_quantity, orders.filled_quantity),
                average_fill_price=COALESCE(excluded.average_fill_price, orders.average_fill_price),
                response_json=excluded.response_json
            """,
            (
                ref_id,
                decision_key,
                now,
                now,
                trading_date,
                mode,
                symbol,
                side,
                order_type,
                quantity,
                dollar_amount,
                planned_notional,
                status,
                broker_order_id,
                filled_quantity,
                average_fill_price,
                self._json(response or {}),
            ),
        )
        self.connection.commit()
