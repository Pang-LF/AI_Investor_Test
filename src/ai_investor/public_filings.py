from __future__ import annotations

import hashlib
import json
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import httpx


SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
RELEVANT_FORMS = frozenset(
    {
        "8-K", "8-K/A", "10-K", "10-K/A", "10-Q", "10-Q/A",
        "S-3", "S-4", "424B3", "424B5", "SC 13D", "SC 13D/A",
        "SC 14D9", "SC 14D9/A", "PREM14A", "DEFA14A",
    }
)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.ignored_depth = 0

    def handle_starttag(
        self, tag: str, _attrs: list[tuple[str, Optional[str]]]
    ) -> None:
        if tag in {"script", "style", "ix:hidden"}:
            self.ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "ix:hidden"} and self.ignored_depth:
            self.ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.ignored_depth and data.strip():
            self.parts.append(data.strip())


def _document_excerpt(text: str, max_chars: int = 12_000) -> str:
    parser = _TextExtractor()
    parser.feed(text)
    compact = re.sub(r"\s+", " ", " ".join(parser.parts)).strip()
    return compact[:max_chars]


def _fresh(path: Path, ttl_seconds: int) -> bool:
    return path.exists() and time.time() - path.stat().st_mtime <= ttl_seconds


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.replace(path)


def _ticker_map(payload: Mapping[str, Mapping[str, Any]]) -> Dict[str, str]:
    return {
        str(item.get("ticker", "")).upper(): f"{int(item['cik_str']):010d}"
        for item in payload.values()
        if item.get("ticker") and item.get("cik_str") is not None
    }


def _recent_filings(payload: Mapping[str, Any], limit: int = 12) -> list[Dict[str, str]]:
    recent = ((payload.get("filings") or {}).get("recent") or {})
    forms = recent.get("form") or []
    accessions = recent.get("accessionNumber") or []
    filing_dates = recent.get("filingDate") or []
    reports = recent.get("reportDate") or []
    documents = recent.get("primaryDocument") or []
    cik = str(payload.get("cik") or "").lstrip("0")
    rows: list[Dict[str, str]] = []
    for form, accession, filing_date, report_date, document in zip(
        forms, accessions, filing_dates, reports, documents
    ):
        if form not in RELEVANT_FORMS:
            continue
        compact_accession = str(accession).replace("-", "")
        rows.append(
            {
                "form": str(form),
                "filing_date": str(filing_date),
                "report_date": str(report_date),
                "accession_number": str(accession),
                "primary_document": str(document),
                "filing_url": (
                    f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                    f"{compact_accession}/{document}"
                ),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def collect_sec_filings(
    root: Path,
    symbols: Sequence[str],
    *,
    user_agent: str,
    max_symbols: int = 2,
    client: Optional[httpx.Client] = None,
) -> Tuple[Dict[str, Any], tuple[str, ...]]:
    """Collect bounded SEC metadata; failures remain explicit research evidence."""
    chosen = list(dict.fromkeys(symbol.upper() for symbol in symbols))[:max_symbols]
    if not chosen:
        return {}, ()
    cache = root / ".local" / "state" / "sec"
    tickers_path = cache / "company_tickers.json"
    calls: list[str] = []
    owned_client = client is None
    session = client or httpx.Client(
        timeout=12.0,
        headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
    )
    result: Dict[str, Any] = {}
    try:
        if not _fresh(tickers_path, 7 * 24 * 3600):
            calls.append("sec_company_tickers")
            response = session.get(SEC_TICKERS_URL)
            response.raise_for_status()
            _write_json(tickers_path, response.json())
        mapping = _ticker_map(_read_json(tickers_path))
        for symbol in chosen:
            cik = mapping.get(symbol)
            if not cik:
                result[symbol] = {"status": "ticker_not_found"}
                continue
            path = cache / "submissions" / f"{cik}.json"
            try:
                if not _fresh(path, 3600):
                    calls.append("sec_company_submissions")
                    response = session.get(SEC_SUBMISSIONS_URL.format(cik=cik))
                    response.raise_for_status()
                    _write_json(path, response.json())
                payload = _read_json(path)
                filings = _recent_filings(payload)
                result[symbol] = {
                    "status": "ok",
                    "cik": cik,
                    "company_name": payload.get("name"),
                    "recent_filings": filings,
                }
                if filings:
                    latest = filings[0]
                    accession = latest["accession_number"].replace("-", "")
                    document_key = hashlib.sha256(
                        latest["filing_url"].encode("utf-8")
                    ).hexdigest()
                    document_path = cache / "documents" / f"{document_key}.json"
                    try:
                        if not document_path.exists():
                            calls.append("sec_filing_document")
                            response = session.get(latest["filing_url"])
                            response.raise_for_status()
                            _write_json(
                                document_path,
                                {"excerpt": _document_excerpt(response.text)},
                            )
                        result[symbol]["latest_filing_excerpt"] = _read_json(
                            document_path
                        ).get("excerpt", "")
                    except (OSError, ValueError, httpx.HTTPError) as exc:
                        result[symbol]["latest_filing_excerpt_error"] = (
                            f"{type(exc).__name__}: {exc}"
                        )
            except (OSError, ValueError, httpx.HTTPError) as exc:
                result[symbol] = {
                    "status": "unavailable",
                    "error": f"{type(exc).__name__}: {exc}",
                }
    except (OSError, ValueError, httpx.HTTPError) as exc:
        return {
            symbol: {
                "status": "unavailable",
                "error": f"{type(exc).__name__}: {exc}",
            }
            for symbol in chosen
        }, tuple(calls)
    finally:
        if owned_client:
            session.close()
    return result, tuple(calls)
