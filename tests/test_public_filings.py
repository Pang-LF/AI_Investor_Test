import unittest

from ai_investor.public_filings import (
    _document_excerpt,
    _recent_filings,
    _ticker_map,
)


class PublicFilingsTests(unittest.TestCase):
    def test_ticker_mapping_and_relevant_filing_extraction(self) -> None:
        self.assertEqual(
            _ticker_map({"0": {"ticker": "abc", "cik_str": 123}}),
            {"ABC": "0000000123"},
        )
        payload = {
            "cik": "0000000123",
            "filings": {
                "recent": {
                    "form": ["8-K", "4"],
                    "accessionNumber": ["0000000123-26-000001", "x"],
                    "filingDate": ["2026-09-14", "2026-09-14"],
                    "reportDate": ["2026-09-13", "2026-09-13"],
                    "primaryDocument": ["event.htm", "ownership.xml"],
                }
            },
        }
        rows = _recent_filings(payload)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["form"], "8-K")
        self.assertIn("/123/000000012326000001/event.htm", rows[0]["filing_url"])

    def test_html_excerpt_removes_markup_and_bounds_size(self) -> None:
        self.assertEqual(
            _document_excerpt("<html><body><b>Material</b> event</body></html>"),
            "Material event",
        )


if __name__ == "__main__":
    unittest.main()
