import json
import unittest

from growthagent.tools import (
    calculator,
    create_campaign,
    default_registry,
    fetch_url,
    get_campaign_status,
    get_metrics,
    reset_state,
    safe_eval,
    search,
)
from growthagent.tools.base import ToolRegistry


class CalculatorTest(unittest.TestCase):
    def test_basic(self):
        self.assertAlmostEqual(safe_eval("3.2 * 4.7"), 15.04)
        self.assertAlmostEqual(safe_eval("2 + 3 * 4"), 14)

    def test_rejects_import(self):
        with self.assertRaises(Exception):
            safe_eval("__import__('os').system('ls')")

    def test_rejects_attributes(self):
        with self.assertRaises(Exception):
            safe_eval("().__class__")

    def test_tool_returns_result(self):
        out = calculator(expression="3.2*4.7")
        self.assertAlmostEqual(out["result"], 15.04)


class MockToolsTest(unittest.TestCase):
    def test_search_returns_results(self):
        out = search(query="budget optimization")
        self.assertIn("results", out)
        self.assertTrue(out["results"])

    def test_fetch_known_and_unknown(self):
        ok = fetch_url(url="https://example.com/landing/focusapp")
        self.assertIn("title", ok)
        missing = fetch_url(url="https://nope.example/x")
        self.assertIn("error", missing)

    def test_ad_platform_flow(self):
        reset_state()
        c = create_campaign(objective="app_installs", budget=1000.0, creative="x")
        cid = c["campaign_id"]
        st = get_campaign_status(campaign_id=cid)
        self.assertEqual(st["status"], "active")
        self.assertEqual(st["objective"], "app_installs")

    def test_analytics_deterministic(self):
        a = get_metrics(campaign_id="camp_1")
        b = get_metrics(campaign_id="camp_1")
        self.assertEqual(a["ctr"], b["ctr"])
        self.assertIn("roas", a)


class RegistryTest(unittest.TestCase):
    def test_default_registry_has_tools(self):
        reg = default_registry()
        for name in ["search", "fetch_url", "calculator", "create_campaign", "get_campaign_status", "get_metrics"]:
            self.assertIn(name, reg.names())

    def test_execute_unknown_tool_returns_error(self):
        reg = default_registry()
        result = reg.execute("nope", {})
        data = json.loads(result)
        self.assertIn("error", data)

    def test_execute_calculator(self):
        reg = default_registry()
        result = reg.execute("calculator", {"expression": "2+2"})
        self.assertEqual(json.loads(result)["result"], 4)


if __name__ == "__main__":
    unittest.main()
