import copy
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from assistant import Chat, Config, main
from errors import AssistantError
from knowledge import Evidence
from personality import Persona


ROOT = Path(__file__).resolve().parents[1]


def profile():
    return Persona.load(ROOT / "persona.example.json")


def config():
    return Config("Legacy", "Be honest.", "test", "http://127.0.0.1:11434", "", ())


class PersonaTests(unittest.TestCase):
    def test_round_trip_and_immutable_lists(self):
        persona = profile()
        self.assertEqual(Persona.from_dict(persona.to_dict()), persona)
        self.assertIsInstance(persona.traits, tuple)
        data = persona.to_dict()
        data["traits"].append("new")
        self.assertNotIn("new", persona.traits)

    def test_legacy_config_remains_supported(self):
        persona = Chat(config(), Mock()).persona
        self.assertEqual(persona.name, "Legacy")
        self.assertEqual(persona.description, "Be honest.")

    def test_invalid_profiles_are_rejected(self):
        original = profile().to_dict()
        for field, value in [
            ("version", True), ("version", 2), ("name", ""),
            ("name", "x" * 81), ("description", None), ("greeting", " "),
            ("traits", []), ("traits", ["x"] * 9), ("traits", [{}]),
            ("boundaries", "not a list"), ("tone", 5),
            ("verbosity", "unlimited"), ("verbosity", []),
            ("humor", "cruel"), ("humor", {}),
            ("permissions", ["turn_on"]),
        ]:
            with self.subTest(field=field, value=value):
                data = copy.deepcopy(original)
                data[field] = value
                with self.assertRaises(AssistantError):
                    Persona.from_dict(data)
        for data in [[], {}, {key: value for key, value in original.items() if key != "name"}]:
            with self.assertRaises(AssistantError):
                Persona.from_dict(data)

    def test_prompt_has_style_and_honest_identity_boundary(self):
        persona = profile()
        prompt = persona.prompt()
        self.assertIn(persona.name, prompt)
        self.assertIn("拟人化", prompt)
        self.assertIn("不构成家庭事实", prompt)
        self.assertIn("不冒充真人", prompt)
        for verbosity, expected in [
            ("brief", "简短回答"), ("balanced", "适量细节"), ("detailed", "展开解释")
        ]:
            data = persona.to_dict()
            data.update(verbosity=verbosity, humor="none")
            text = Persona.from_dict(data).prompt()
            self.assertIn(expected, text)
            self.assertIn("不主动开玩笑", text)

    def test_persona_applies_to_every_turn_and_grounded_answer(self):
        client = Mock()
        client.request.return_value = {"message": {"role": "assistant", "content": "Hello"}}
        chat = Chat(config(), client, profile())
        chat.reply("First")
        chat.reply("Second")
        for call in client.request.call_args_list:
            system = call.kwargs["payload"]["messages"][0]["content"]
            self.assertIn("小家", system)
            self.assertIn("不虚构家庭事实", system)
            self.assertIn("没有自动长期偏好记忆", system)
        evidence = Evidence("doc", 1, 1, "source.md", "Example evidence")
        client.request.return_value = {
            "message": {
                "role": "assistant",
                "content": json.dumps({"answer": "Answer", "citations": [evidence.citation]}),
            }
        }
        chat.ask("question", [evidence])
        prompt = client.request.call_args.kwargs["payload"]["messages"][0]["content"]
        self.assertIn("小家", prompt)
        self.assertIn("不可信", prompt)

    def test_cli_loads_profile_or_reports_failure(self):
        with patch("assistant.run", return_value=0) as run:
            self.assertEqual(main([
                "--config", str(ROOT / "config.example.json"),
                "--persona", str(ROOT / "persona.example.json"),
            ]), 0)
        self.assertEqual(run.call_args.kwargs["persona"], profile())
        with (
            patch("sys.stderr"),
            patch("assistant.run") as run,
        ):
            self.assertEqual(main([
                "--config", str(ROOT / "config.example.json"),
                "--persona", str(ROOT / "missing-persona.json"),
            ]), 1)
            run.assert_not_called()


class ModelStatusTests(unittest.TestCase):
    def test_installed_and_implicit_latest_models(self):
        for model, installed in [("test", "test:latest"), ("test:v1", "test:v1")]:
            settings = Config("Test", "Helpful", model, "http://127.0.0.1:11434", "", ())
            client = Mock()
            client.request.return_value = {"models": [{"name": installed}]}
            text = Chat(settings, client).status()
            self.assertIn("已安装", text)
            self.assertIn("尚未验证实际生成效果", text)
            client.request.assert_called_once_with("http://127.0.0.1:11434/api/tags")

    def test_missing_or_malformed_model_list_is_not_reported_ready(self):
        client = Mock()
        for response in [
            {}, [], {"models": []}, {"models": "invalid"},
            {"models": [None]}, {"models": [{"name": ""}]},
            {"models": [{"name": "other-model"}]},
        ]:
            with self.subTest(response=response):
                client.request.return_value = response
                with self.assertRaises(AssistantError):
                    Chat(config(), client).status()


if __name__ == "__main__":
    unittest.main()
