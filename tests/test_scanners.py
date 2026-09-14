import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_investor.config import Settings
from ai_investor.robinhood_mcp import MCPError, RobinhoodMCPClient
from ai_investor.scanners import (
    SCANNER_KEYS,
    ScannerRecord,
    ScannerRegistry,
    scanner_definitions,
)


class ScannerTests(unittest.TestCase):
    def setUp(self) -> None:
        settings = Settings.load(Path("config/settings.toml"))
        self.definitions = scanner_definitions(settings.monitor)

    def test_four_versioned_definitions_are_stable(self) -> None:
        self.assertEqual(set(self.definitions), set(SCANNER_KEYS))
        for definition in self.definitions.values():
            self.assertEqual(len(definition.fingerprint), 64)
            self.assertTrue(definition.title.endswith(definition.fingerprint[:8]))
            self.assertEqual(definition.create_arguments()["preset"], "INITIAL")

    def test_registry_rejects_configuration_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scanners.json"
            registry = ScannerRegistry(
                {
                    key: ScannerRecord(
                        scan_id=f"scan-{key}",
                        title=definition.title,
                        fingerprint=definition.fingerprint,
                        server_fingerprint=f"remote-{key}",
                    )
                    for key, definition in self.definitions.items()
                }
            )
            registry.save(path)
            loaded = ScannerRegistry.load(path, self.definitions)
            self.assertEqual(loaded.scan_id("large"), "scan-large")
            changed = dict(self.definitions)
            original = changed["large"]
            changed["large"] = type(original)(
                original.key,
                original.title_prefix,
                original.filters + ({"predicate": ">", "values": ["1"]},),
                original.columns,
            )
            with self.assertRaises(MCPError):
                ScannerRegistry.load(path, changed)

    def test_scanner_creation_requires_separate_explicit_capability(self) -> None:
        with patch(
            "ai_investor.robinhood_mcp.RobinhoodOAuth.access_token",
            return_value="test-token",
        ):
            client = RobinhoodMCPClient(
                max_calls=1,
                allowed_tools={"create_scan"},
                allow_scanner_configuration=False,
            )
            try:
                with self.assertRaises(MCPError):
                    client.call_tool("create_scan", {})
            finally:
                client._http.close()


if __name__ == "__main__":
    unittest.main()
