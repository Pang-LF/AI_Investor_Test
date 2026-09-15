#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_investor.config import Settings
from ai_investor.opportunity_audit import (
    AuditDefinition,
    fetch_audit_histories,
    render_markdown,
    run_opportunity_capture_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    settings = Settings.load(root / "config/settings.toml")
    definition = AuditDefinition()
    histories, metadata = fetch_audit_histories(root, settings, definition)
    report = run_opportunity_capture_audit(
        root, settings, histories, definition, metadata
    )
    date = metadata["universe_date"]
    data_path = root / ".local/state/audits" / f"opportunity_capture_{date}.json"
    report_path = root / "docs/audits" / f"opportunity_capture_{date}.md"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps(report, separators=(",", ":")), encoding="utf-8")
    report_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({
        "data_path": str(data_path),
        "report_path": str(report_path),
        "horizons": report["horizons"],
    }, indent=2))


if __name__ == "__main__":
    main()
