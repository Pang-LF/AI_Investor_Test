import os
import tempfile
import unittest
from pathlib import Path

from ai_investor.config import Settings
from ai_investor.robinhood_readonly import READ_ONLY_TOOLS, WRITE_TOOLS


class SafetyTests(unittest.TestCase):
    def test_read_only_allowlist_has_no_write_tools(self) -> None:
        self.assertFalse(set(READ_ONLY_TOOLS) & WRITE_TOOLS)
        self.assertTrue(all(name.startswith("get_") for name in READ_ONLY_TOOLS))

    def test_live_mode_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.toml"
            path.write_text(
                'mode="DRY_RUN"\nlive_trading=false\nopenai_model="x"\n'
                "max_output_tokens=1\nmax_llm_calls_per_run=1\n"
                "max_tool_calls_per_run=1\n"
                "max_estimated_cost_per_run_usd=0.01\n"
                "max_monthly_llm_budget_usd=1.0\n",
                encoding="utf-8",
            )
            old = os.environ.get("AI_INVESTOR_LIVE_TRADING")
            os.environ["AI_INVESTOR_LIVE_TRADING"] = "true"
            try:
                with self.assertRaises(RuntimeError):
                    Settings.load(path)
            finally:
                if old is None:
                    os.environ.pop("AI_INVESTOR_LIVE_TRADING", None)
                else:
                    os.environ["AI_INVESTOR_LIVE_TRADING"] = old


if __name__ == "__main__":
    unittest.main()
