from __future__ import annotations

import re
from datetime import datetime
from typing import Any


_EXCESS_FRACTION = re.compile(r"(\.\d{6})\d+(?=(?:Z|[+-]\d{2}:\d{2})$)")


def parse_rfc3339(value: Any) -> datetime:
    """Parse RFC3339 timestamps, truncating nanoseconds to microseconds.

    Python versions used by the local runner reject Robinhood timestamps with
    nine fractional digits. Truncation preserves more precision than any
    freshness or execution-age policy in this project requires.
    """
    text = str(value).strip()
    text = _EXCESS_FRACTION.sub(r"\1", text)
    return datetime.fromisoformat(text.replace("Z", "+00:00"))
