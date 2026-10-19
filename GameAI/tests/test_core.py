"""Device-free checks for repository packaging and core parsing."""

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class CoreTests(unittest.TestCase):
    def test_environment_key_only(self):
        with patch.dict(sys.modules, {"openai": types.SimpleNamespace(OpenAI=object)}):
            import gpt

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-placeholder"}):
            self.assertEqual(gpt.load_api_key(), "test-placeholder")
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                gpt.load_api_key()

    def test_example_config_parses(self):
        with patch.dict(sys.modules, {"openai": types.SimpleNamespace(OpenAI=object)}):
            import gpt

        settings = gpt.get_stage_settings("L_mind", profile_dir=ROOT / "dataset" / "AOV")
        self.assertTrue(settings["enabled"])
        self.assertIn("左手", settings["system_prompt"])

    def test_action_normalization(self):
        with patch.dict(sys.modules, {"pyautogui": types.SimpleNamespace()}):
            import action

        self.assertEqual(action.parse_plan_text("c(4,5)|e", 3), ["c(4,5)", "e", "e"])
        self.assertEqual(action.normalize_command("invalid"), "e")


if __name__ == "__main__":
    unittest.main()
