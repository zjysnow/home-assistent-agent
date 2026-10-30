import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from assistant import Chat, Config, run
from errors import AssistantError
from reasoning import decode_json, load_json, reason, validate_rules


def rules():
    return {
        "version": 1,
        "rules": [
            {
                "id": "suggest_light",
                "when": {
                    "all": [
                        {"fact": "room.occupied"},
                        {"fact": "room.dark"},
                        {"not": {"fact": "room.sleeping"}},
                    ]
                },
                "suggestion": "Suggest light only.",
            }
        ],
    }


def facts():
    return {"facts": {"room.occupied": True, "room.dark": True, "room.sleeping": False}}


def config():
    return Config("Test", "Be honest.", "test", "http://127.0.0.1:11434", "", ())


class RuleEvaluationTests(unittest.TestCase):
    def test_true_has_explanation_and_suggestion(self):
        report = reason(rules(), facts())[0]
        self.assertEqual(report["status"], "true")
        self.assertEqual(report["suggestion"], "Suggest light only.")
        self.assertEqual(report["evidence"]["room.sleeping"], "false")
        self.assertEqual(report["expression"], rules()["rules"][0]["when"])

    def test_fact_swap_without_changing_rules(self):
        fixed_rules = rules()
        changed_facts = facts()
        changed_facts["facts"]["room.sleeping"] = True
        self.assertEqual(reason(fixed_rules, facts())[0]["status"], "true")
        self.assertEqual(reason(fixed_rules, changed_facts)[0]["status"], "false")
        self.assertIsNone(reason(fixed_rules, changed_facts)[0]["suggestion"])

    def test_rule_swap_without_changing_facts(self):
        changed_rules = rules()
        changed_rules["rules"][0]["when"]["all"][-1] = {"fact": "room.sleeping"}
        self.assertEqual(reason(rules(), facts())[0]["status"], "true")
        self.assertEqual(reason(changed_rules, facts())[0]["status"], "false")

    def test_unknown_sleep_is_not_assumed_false(self):
        for value in ["missing", None]:
            changed = facts()
            if value == "missing":
                del changed["facts"]["room.sleeping"]
            else:
                changed["facts"]["room.sleeping"] = value
            report = reason(rules(), changed)[0]
            self.assertEqual(report["status"], "unknown")
            self.assertIsNone(report["suggestion"])

    def test_conflict_blocks_suggestions_even_in_decisive_branches(self):
        changed_facts = facts()
        changed_facts["facts"]["room.sleeping"] = "conflict"
        self.assertEqual(reason(rules(), changed_facts)[0]["status"], "conflict")
        changed_rules = rules()
        changed_rules["rules"][0]["when"] = {
            "any": [{"fact": "room.occupied"}, {"fact": "room.sleeping"}]
        }
        self.assertEqual(reason(changed_rules, changed_facts)[0]["status"], "conflict")
        changed_rules["rules"][0]["when"] = {
            "all": [{"not": {"fact": "room.occupied"}}, {"fact": "room.sleeping"}]
        }
        self.assertEqual(reason(changed_rules, changed_facts)[0]["status"], "conflict")

    def test_all_any_and_negation_truth_tables(self):
        for operator, cases in {
            "all": [
                ([True, True], "true"),
                ([True, False], "false"),
                ([None, False], "false"),
                ([True, None], "unknown"),
            ],
            "any": [
                ([False, False], "false"),
                ([True, False], "true"),
                ([None, True], "true"),
                ([False, None], "unknown"),
            ],
        }.items():
            for values, expected in cases:
                with self.subTest(operator=operator, values=values):
                    custom = rules()
                    custom["rules"][0]["when"] = {
                        operator: [{"fact": "a"}, {"fact": "b"}]
                    }
                    self.assertEqual(
                        reason(custom, {"facts": dict(zip(["a", "b"], values))})[0]["status"],
                        expected,
                    )

    def test_composed_rules_in_a_fictional_world(self):
        custom = rules()
        custom["rules"][0]["when"] = {
            "all": [
                {"any": [{"fact": "zorb.blue"}, {"fact": "mip.ready"}]},
                {"not": {"fact": "mip.blocked"}},
            ]
        }
        result = reason(
            custom,
            {"facts": {"zorb.blue": False, "mip.ready": True, "mip.blocked": False}},
        )
        self.assertEqual(result[0]["status"], "true")
        self.assertEqual(len(result[0]["evidence"]), 3)

    def test_input_objects_are_not_mutated(self):
        rule_data, fact_data = rules(), facts()
        originals = copy.deepcopy((rule_data, fact_data))
        reason(rule_data, fact_data)
        self.assertEqual((rule_data, fact_data), originals)


class ValidationTests(unittest.TestCase):
    def test_invalid_expression_operators_and_shapes(self):
        for expression in [
            {},
            {"fact": "bad name"},
            {"fact": "__import__()"},
            {"all": []},
            {"any": "not a list"},
            {"all": [{"fact": "a"}] * 17},
            {"eval": "run code"},
            {"fact": "a", "not": {"fact": "b"}},
        ]:
            with self.subTest(expression=expression):
                custom = rules()
                custom["rules"][0]["when"] = expression
                with self.assertRaises(AssistantError):
                    validate_rules(custom)

    def test_depth_and_node_limits(self):
        deep = {"fact": "a"}
        for _ in range(10):
            deep = {"not": deep}
        wide = {"all": [{"all": [{"fact": "a"}] * 16}] * 8}
        for expression in [deep, wide]:
            custom = rules()
            custom["rules"][0]["when"] = expression
            with self.assertRaises(AssistantError):
                validate_rules(custom)

    def test_rule_schema_rejects_actions_and_duplicate_ids(self):
        for modify in [
            lambda data: data.update(version=True),
            lambda data: data.update(rules=[]),
            lambda data: data["rules"].append(copy.deepcopy(data["rules"][0])),
            lambda data: data["rules"][0].update(action="turn_on"),
            lambda data: data["rules"][0].update(suggestion=" "),
            lambda data: data["rules"][0].update(suggestion="x" * 501),
        ]:
            custom = rules()
            modify(custom)
            with self.assertRaises(AssistantError):
                validate_rules(custom)

    def test_invalid_facts_are_not_silently_coerced(self):
        for invalid in [
            [],
            {"facts": [], "extra": True},
            {"facts": {"a": 1}},
            {"facts": {"a": "false"}},
            {"facts": {"a": []}},
            {"facts": {"bad name": True}},
        ]:
            with self.subTest(invalid=invalid):
                with self.assertRaises(AssistantError):
                    reason(rules(), invalid)

    def test_duplicate_json_keys_are_rejected(self):
        with self.assertRaises(AssistantError):
            decode_json('{"facts":{"a":true,"a":false}}')

    def test_file_errors_and_size_limits(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as directory:
            path = Path(directory) / "rules.json"
            for contents in [b"bad-json", b"\xff", b"x" * 100_001]:
                path.write_bytes(contents)
                with self.assertRaises(AssistantError):
                    load_json(path)
            with self.assertRaises(AssistantError):
                load_json(path.with_name("missing.json"))


class CandidateAndTerminalTests(unittest.TestCase):
    def test_candidate_uses_structured_format_without_saving_or_execution(self):
        client = Mock()
        client.request.return_value = {
            "message": {"role": "assistant", "content": json.dumps(rules())}
        }
        chat = Chat(config(), client)
        self.assertEqual(chat.propose("If dark and occupied, suggest light."), rules())
        self.assertIn("$defs", client.request.call_args.kwargs["payload"]["format"])
        self.assertEqual(chat.history, [])
        self.assertEqual(client.request.call_count, 1)

    def test_invalid_model_candidate_cannot_be_enabled(self):
        client = Mock()
        candidate = rules()
        candidate["rules"][0]["action"] = "turn_on"
        client.request.return_value = {
            "message": {"role": "assistant", "content": json.dumps(candidate)}
        }
        with self.assertRaises(AssistantError):
            Chat(config(), client).propose("turn on a light")

    def test_terminal_reason_and_propose(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as directory:
            root = Path(directory)
            rule_path, fact_path = root / "rules.json", root / "facts.json"
            rule_path.write_text(json.dumps(rules()), encoding="utf-8")
            fact_path.write_text(json.dumps(facts()), encoding="utf-8")
            client = Mock()
            client.request.return_value = {
                "message": {"role": "assistant", "content": json.dumps(rules())}
            }
            output, errors = io.StringIO(), io.StringIO()
            with (
                patch("builtins.input", side_effect=[
                    f'/reason "{rule_path}" "{fact_path}"',
                    "/propose Suggest light if occupied",
                    "/quit",
                ]),
                patch("assistant.JsonClient", return_value=client),
                patch("sys.stdout", output),
                patch("sys.stderr", errors),
            ):
                self.assertEqual(run(config(), data_dir=root / "runtime"), 0)
            self.assertIn('"status": "true"', output.getvalue())
            self.assertIn("未保存、未启用", output.getvalue())
            self.assertEqual(errors.getvalue(), "")
            self.assertFalse((root / "runtime").exists())
            self.assertEqual(client.request.call_count, 1)


if __name__ == "__main__":
    unittest.main()
