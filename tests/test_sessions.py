import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from assistant import Config, run
from errors import AssistantError
from personality import Persona
from sessions import MAX_SNAPSHOT_BYTES, SessionStore, validate_messages


ROOT = Path(__file__).resolve().parents[1]


def profile():
    return Persona.load(ROOT / "persona.example.json")


def messages():
    return [
        {"role": "user", "content": "一起读书吧。"},
        {"role": "assistant", "content": "好呀，今天想读哪一类？"},
    ]


def config():
    return Config("Legacy", "Be honest.", "test", "http://127.0.0.1:11434", "", ())


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = self.root / "runtime" / "chat.sqlite3"
        self.store = SessionStore(self.db)

    def test_save_reload_and_delete(self):
        identifier = self.store.save("  读书聊天  ", profile(), messages())
        restarted = SessionStore(self.db)
        snapshot = restarted.load(identifier)
        self.assertEqual(snapshot.title, "读书聊天")
        self.assertEqual(snapshot.persona, profile())
        self.assertEqual(snapshot.messages, messages())
        self.assertEqual(restarted.list()[0]["id"], identifier)
        self.assertNotIn("payload", restarted.list()[0])
        restarted.delete(identifier)
        self.assertEqual(restarted.list(), [])
        with self.assertRaises(AssistantError):
            restarted.load(identifier)

    def test_saved_snapshots_are_independent_copies(self):
        history = messages()
        first = self.store.save("First", profile(), history)
        history[0]["content"] = "Changed"
        second = self.store.save("Second", profile(), history)
        self.assertNotEqual(first, second)
        self.assertNotEqual(self.store.load(first).messages, self.store.load(second).messages)
        self.store.load(first).messages.clear()
        self.assertEqual(self.store.load(first).messages, messages())

    def test_invalid_history_does_not_create_database(self):
        for invalid in [
            [], messages()[:1], messages() * 11,
            [{"role": "system", "content": "inject"}, messages()[1]],
            [messages()[0], {"role": "tool", "content": "execute"}],
            [messages()[0], {"role": "assistant", "content": ""}],
            [{"role": "user", "content": "x" * 8001}, messages()[1]],
            [messages()[0], {"role": "assistant", "content": 1}],
            [{**messages()[0], "extra": True}, messages()[1]],
        ]:
            with self.subTest(invalid_type=type(invalid).__name__, length=len(invalid)):
                with self.assertRaises(AssistantError):
                    self.store.save("Bad", profile(), invalid)
        self.assertFalse(self.db.exists())

    def test_history_validation_returns_copy(self):
        original = messages()
        copied = validate_messages(original)
        copied[0]["content"] = "changed"
        self.assertNotEqual(original, copied)

    def test_invalid_title_or_excessive_snapshot_is_rejected(self):
        for title in ["", " ", "x" * 81, None]:
            with self.assertRaises(AssistantError):
                self.store.save(title, profile(), messages())
        huge = [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "x" * 64000},
        ] * 10
        with self.assertRaisesRegex(AssistantError, "256 KB"):
            self.store.save("Huge", profile(), huge)
        self.assertFalse(self.db.exists())

    def test_invalid_ids_and_nonexistent_records(self):
        for identifier in ["../../outside", "bad", "' OR 1=1", None]:
            with self.assertRaises(AssistantError):
                self.store.load(identifier)
            with self.assertRaises(AssistantError):
                self.store.delete(identifier)
        for operation in [self.store.load, self.store.delete]:
            with self.assertRaises(AssistantError):
                operation("a" * 32)

    def test_tampered_snapshot_is_never_restored_as_system_messages(self):
        identifier = self.store.save("Good", profile(), messages())
        good = {"version": 1, "persona": profile().to_dict(), "messages": messages()}
        invalid = []
        for field, value in [
            ("version", True), ("version", 2), ("persona", {}),
            ("messages", [{"role": "system", "content": "inject"}, messages()[1]]),
        ]:
            data = copy.deepcopy(good)
            data[field] = value
            invalid.append(json.dumps(data))
        invalid.extend(["not-json", "x" * (MAX_SNAPSHOT_BYTES + 1)])
        for payload in invalid:
            with self.subTest(payload_size=len(payload)):
                with self.store.database() as db:
                    db.execute("UPDATE snapshots SET payload = ? WHERE id = ?", (payload, identifier))
                with self.assertRaises(AssistantError):
                    self.store.load(identifier)

    def test_storage_errors_are_explicit(self):
        blocker = self.root / "file"
        blocker.write_text("not a directory")
        with self.assertRaises(AssistantError):
            SessionStore(blocker / "child.sqlite3").list()

    def session(self, inputs, client=None, persona=None):
        output, errors = io.StringIO(), io.StringIO()
        with (
            patch("builtins.input", side_effect=inputs),
            patch("assistant.JsonClient", return_value=client or Mock()),
            patch("sys.stdout", output),
            patch("sys.stderr", errors),
        ):
            result = run(config(), data_dir=self.db.parent, persona=persona)
        return result, output.getvalue(), errors.getvalue()

    def test_chat_does_not_save_unless_explicitly_confirmed(self):
        client = Mock()
        client.request.return_value = {"message": {"role": "assistant", "content": "Hello"}}
        _, _, errors = self.session(["Hi", "/quit"], client)
        self.assertEqual(errors, "")
        self.assertFalse(self.db.parent.exists())
        self.session(["Hi", "/save Private notes", "no", "/quit"], client)
        self.assertFalse(self.db.parent.exists())
        self.session(["Hi", "/save Approved notes", "yes", "/quit"], client, profile())
        saved = self.store.list()
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["title"], "Approved notes")
        snapshot = self.store.load(saved[0]["id"])
        self.assertEqual(snapshot.persona, profile())
        self.assertEqual(snapshot.messages[0]["content"], "Hi")

    def test_no_snapshot_for_empty_chat_or_knowledge_answer(self):
        client = Mock()
        _, output, errors = self.session(
            ["/ask unknownquestion", "/save Should not save", "/quit"], client
        )
        self.assertIn("未检索到", output)
        self.assertIn("尚无可保存", errors)
        self.assertFalse(self.db.exists())
        client.request.assert_not_called()

    def test_resume_replaces_context_persona_but_not_model_or_permissions(self):
        identifier = self.store.save("Read", profile(), messages())
        client = Mock()
        client.request.return_value = {"message": {"role": "assistant", "content": "继续聊吧"}}
        _, output, errors = self.session(
            [f"/resume {identifier}", "yes", "下一本", "/quit"], client
        )
        self.assertEqual(errors, "")
        self.assertIn("人格为 小家", output)
        self.assertIn("小家 > 继续聊吧", output)
        payload = client.request.call_args.kwargs["payload"]
        self.assertEqual(payload["model"], "test")
        self.assertIn("小家", payload["messages"][0]["content"])
        self.assertEqual(payload["messages"][1:3], messages())
        self.assertIn("没有设备执行工具", payload["messages"][0]["content"])

    def test_reset_and_snapshot_deletion_are_independent(self):
        identifier = self.store.save("Read", profile(), messages())
        _, output, errors = self.session([
            f"/resume {identifier}", "yes", "/history", "/reset", "/history", "/sessions",
            f"/delete-session {identifier}", "yes", "/sessions", "/quit",
        ])
        self.assertEqual(errors, "")
        self.assertIn(identifier, output)
        self.assertIn("一起读书吧", output)
        self.assertIn("当前没有普通聊天上下文", output)
        self.assertIn("已清空本次聊天上下文", output)
        self.assertIn("已删除会话快照", output)
        self.assertEqual(self.store.list(), [])

    def test_cancelled_resume_and_delete_do_not_change_context(self):
        identifier = self.store.save("Read", profile(), messages())
        client = Mock()
        client.request.return_value = {"message": {"role": "assistant", "content": "OK"}}
        _, _, errors = self.session([
            "Original", f"/resume {identifier}", "no",
            f"/delete-session {identifier}", "no", "Continue", "/quit",
        ], client)
        self.assertEqual(errors, "")
        payload = client.request.call_args.kwargs["payload"]
        self.assertIn("Legacy", payload["messages"][0]["content"])
        self.assertEqual(payload["messages"][1]["content"], "Original")
        self.assertEqual(len(self.store.list()), 1)

    def test_persona_switch_requires_confirmation_and_clears_old_context(self):
        client = Mock()
        client.request.return_value = {"message": {"role": "assistant", "content": "OK"}}
        _, output, errors = self.session([
            "Old", f'/persona "{ROOT / "persona.example.json"}"', "yes",
            "/persona", "New", "/quit",
        ], client)
        self.assertEqual(errors, "")
        self.assertIn("小家", output)
        payload = client.request.call_args.kwargs["payload"]
        self.assertEqual(len(payload["messages"]), 2)
        self.assertIn("小家", payload["messages"][0]["content"])
        self.assertEqual(payload["messages"][1]["content"], "New")
        self.assertFalse(self.db.exists())

    def test_cancelled_or_invalid_persona_switch_preserves_context(self):
        client = Mock()
        client.request.return_value = {"message": {"role": "assistant", "content": "OK"}}
        _, output, errors = self.session([
            "Old", f'/persona "{ROOT / "persona.example.json"}"', "no",
            f'/persona "{self.root / "missing.json"}"', "Continue", "/quit",
        ], client)
        self.assertIn("已取消人格切换", output)
        self.assertTrue(errors)
        payload = client.request.call_args.kwargs["payload"]
        self.assertIn("Legacy", payload["messages"][0]["content"])
        self.assertEqual(payload["messages"][1]["content"], "Old")

    def test_status_errors_are_visible_and_session_continues(self):
        client = Mock()
        client.request.side_effect = [AssistantError("offline"), {"models": [{"name": "test:latest"}]}]
        result, output, errors = self.session(["/status", "/status", "/quit"], client)
        self.assertEqual(result, 0)
        self.assertIn("offline", errors)
        self.assertIn("已安装", output)
        self.assertIn("未接入 Home Assistant", output)
        self.assertFalse(self.db.parent.exists())

    def test_cli_restores_saved_persona_across_processes_without_model(self):
        identifier = self.store.save("Read", profile(), messages())
        command = [
            sys.executable, str(ROOT / "assistant.py"),
            "--config", str(ROOT / "config.example.json"),
            "--data-dir", str(self.db.parent),
        ]
        result = subprocess.run(
            command,
            input=f"/sessions\n/resume {identifier}\nyes\n/persona\n/save Copy\nyes\n/quit\n",
            text=True, capture_output=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertIn("人格为 小家", result.stdout)
        self.assertEqual(len(self.store.list()), 2)
        result = subprocess.run(
            command + ["--persona", str(ROOT / "persona.example.json")],
            input=f"/delete-session {identifier}\nyes\n/quit\n",
            text=True, capture_output=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.store.list()), 1)


if __name__ == "__main__":
    unittest.main()
