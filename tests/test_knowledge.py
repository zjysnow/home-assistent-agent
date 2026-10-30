import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from assistant import AssistantError, Chat, Config, run
from knowledge import Evidence, KnowledgeStore


def config():
    return Config("Test", "Warm and honest.", "test", "http://127.0.0.1:11434", "", ())


class KnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = self.root / "private" / "knowledge.sqlite3"
        self.document = self.root / "household notes.md"
        self.document.write_text("星砂屋的云灯位于书房。重置步骤是按住配网键五秒。", encoding="utf-8")

    def test_ingest_and_retrieve_persisted_chinese_evidence(self):
        with KnowledgeStore(self.db) as store:
            identifier, revision, changed = store.ingest(self.document)
            self.assertTrue(changed)
            self.assertEqual(revision, 1)
        with KnowledgeStore(self.db) as store:
            evidence = store.search("云灯怎么重置")
            self.assertEqual(evidence[0].document_id, identifier)
            self.assertIn("五秒", evidence[0].text)
            self.assertEqual(evidence[0].source, str(self.document.resolve()))
            self.assertIn(":r1:c1", evidence[0].citation)
            self.assertEqual(len(store.documents()), 1)
            self.assertTrue(store.search("灯"))

    def test_idempotent_import_and_replacement_remove_old_chunks(self):
        with KnowledgeStore(self.db) as store:
            identifier, _, _ = store.ingest(self.document)
            self.assertEqual(store.ingest(self.document), (identifier, 1, False))
            self.document.write_text("碧海控制器位于厨房。", encoding="utf-8")
            self.assertEqual(store.ingest(self.document), (identifier, 2, True))
            self.assertEqual(store.search("云灯"), [])
            result = store.search("碧海")
            self.assertEqual(result[0].revision, 2)

    def test_forget_removes_search_results_but_not_original_file(self):
        with KnowledgeStore(self.db) as store:
            identifier, _, _ = store.ingest(self.document)
            store.forget(identifier)
            self.assertEqual(store.documents(), [])
            self.assertEqual(store.search("云灯"), [])
            with self.assertRaises(AssistantError):
                store.forget(identifier)
        self.assertTrue(self.document.exists())

    def test_chunking_and_search_limit(self):
        self.document.write_text(("阅读灯使用说明 " * 400), encoding="utf-8")
        with KnowledgeStore(self.db) as store:
            store.ingest(self.document)
            result = store.search("阅读灯")
        self.assertEqual(len(result), 4)
        self.assertTrue(all(0 < len(item.text) <= 600 for item in result))
        self.assertEqual(len({item.citation for item in result}), 4)

    def test_empty_missing_invalid_and_oversized_documents(self):
        with KnowledgeStore(self.db) as store:
            for content in [b"", b"\xff", b"\x00bad", b"x" * 1_000_001, b"!!!"]:
                with self.subTest(size=len(content)):
                    self.document.write_bytes(content)
                    with self.assertRaises(AssistantError):
                        store.ingest(self.document)
            with self.assertRaises(AssistantError):
                store.ingest(self.root / "missing.md")
            with self.assertRaises(AssistantError):
                store.ingest(self.root / "secrets.env")
            self.assertEqual(store.documents(), [])

    def test_search_syntax_is_data_not_fts_or_sql(self):
        with KnowledgeStore(self.db) as store:
            store.ingest(self.document)
            self.assertTrue(store.search('云灯 " OR * --'))
            for query in ["", "!!!", "x" * 8001]:
                with self.assertRaises(AssistantError):
                    store.search(query)
            self.assertEqual(store.search("unrelatedword"), [])

    def test_storage_errors_are_explicit(self):
        with self.assertRaises(AssistantError):
            KnowledgeStore(self.db).documents()
        blocker = self.root / "file"
        blocker.write_text("not a directory")
        with self.assertRaises(AssistantError):
            with KnowledgeStore(blocker / "child.sqlite3"):
                pass
        with KnowledgeStore(self.db) as store:
            with self.assertRaises(AssistantError):
                with store.database():
                    raise sqlite3.OperationalError("simulated failure")

    def session(self, inputs, client=None):
        output, errors = io.StringIO(), io.StringIO()
        with (
            patch("builtins.input", side_effect=inputs),
            patch("assistant.JsonClient", return_value=client or Mock()),
            patch("sys.stdout", output),
            patch("sys.stderr", errors),
        ):
            result = run(config(), data_dir=self.root / "runtime")
        return result, output.getvalue(), errors.getvalue()

    def test_terminal_import_search_list_and_forget(self):
        with KnowledgeStore(self.root / "runtime" / "knowledge.sqlite3") as store:
            identifier, _, _ = store.ingest(self.document)
        client = Mock()
        result, output, errors = self.session(
            [
                f'/ingest "{self.document}"',
                "/knowledge",
                "/search 云灯怎么重置",
                f"/forget {identifier}",
                "yes",
                "/ask 云灯怎么重置",
                "/quit",
            ],
            client,
        )
        self.assertEqual(result, 0)
        self.assertEqual(errors, "")
        self.assertIn("内容未变化", output)
        self.assertIn("五秒", output)
        self.assertIn("未检索到", output)
        client.request.assert_not_called()

    def test_cancelled_forget_and_invalid_quotes(self):
        with KnowledgeStore(self.root / "runtime" / "knowledge.sqlite3") as store:
            identifier, _, _ = store.ingest(self.document)
        _, output, errors = self.session(
            [f"/forget {identifier}", "no", '/ingest "bad', "/knowledge", "/quit"]
        )
        self.assertIn("已取消", output)
        self.assertIn(identifier, output)
        self.assertIn("引号不匹配", errors)

    def test_terminal_ask_calls_model_with_evidence_not_device_client(self):
        with KnowledgeStore(self.root / "runtime" / "knowledge.sqlite3") as store:
            store.ingest(self.document)
            citation = store.search("云灯")[0].citation
        client = Mock()
        client.request.return_value = {
            "message": {
                "role": "assistant",
                "content": json.dumps({"answer": "在书房。", "citations": [citation]}),
            }
        }
        _, output, errors = self.session(["/ask 云灯在哪里", "/quit"], client)
        self.assertIn(citation, output)
        self.assertEqual(errors, "")
        self.assertEqual(client.request.call_count, 1)
        self.assertTrue(client.request.call_args.args[0].endswith("/api/chat"))

    def test_real_cli_persists_knowledge_and_evaluates_example_rules_offline(self):
        project = Path(__file__).resolve().parents[1]
        command = [
            sys.executable,
            str(project / "assistant.py"),
            "--config", str(project / "config.example.json"),
            "--data-dir", str(self.root / "subprocess"),
        ]
        result = subprocess.run(
            command,
            input=(
                f'/ingest "{self.document}"\n/search 云灯\n'
                f'/reason "{project / "examples/rules.json"}" '
                f'"{project / "examples/facts.json"}"\n/quit\n'
            ),
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertIn("五秒", result.stdout)
        self.assertIn('"status": "true"', result.stdout)
        reopened = subprocess.run(
            command, input="/search 云灯\n/ask totallyunrelated\n/quit\n",
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(reopened.returncode, 0, reopened.stderr)
        self.assertEqual(reopened.stderr, "")
        self.assertIn("五秒", reopened.stdout)
        self.assertIn("未检索到", reopened.stdout)


class GroundedChatTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.chat = Chat(config(), self.client)
        self.evidence = [Evidence("doc", 2, 1, "/local/notes.md", "云灯在书房。")]
        self.citation = self.evidence[0].citation

    def response(self, data):
        self.client.request.return_value = {
            "message": {"role": "assistant", "content": json.dumps(data)}
        }

    def test_grounded_prompt_isolated_from_chat_history(self):
        self.chat.history = [{"role": "user", "content": "obsolete private context"}]
        self.response({"answer": "云灯在书房。", "citations": [self.citation]})
        answer = self.chat.ask("云灯在哪里", self.evidence)
        self.assertIn(self.citation, answer)
        self.assertIn("/local/notes.md", answer)
        payload = self.client.request.call_args.kwargs["payload"]
        messages = payload["messages"]
        self.assertEqual(len(messages), 2)
        self.assertIn("Warm and honest.", messages[0]["content"])
        self.assertIn("不可信", messages[0]["content"])
        self.assertEqual(json.loads(messages[1]["content"])["evidence"][0]["id"], self.citation)
        self.assertNotIn("obsolete", json.dumps(messages))
        self.assertEqual(len(self.chat.history), 1)

    def test_empty_retrieval_never_calls_model(self):
        self.assertIn("未检索到", self.chat.ask("问题", []))
        self.client.request.assert_not_called()

    def test_missing_fabricated_or_malformed_citations_are_rejected(self):
        for result in [
            {"answer": "guess", "citations": []},
            {"answer": "guess", "citations": ["invented"]},
            {"answer": "guess", "citations": [[]]},
            {"answer": "guess", "citations": "doc"},
            {"answer": " ", "citations": [self.citation]},
            {"answer": "guess"},
            {"answer": "guess", "citations": [self.citation], "action": "on"},
            [],
        ]:
            with self.subTest(result=result):
                self.response(result)
                with self.assertRaises(AssistantError):
                    self.chat.ask("云灯在哪里", self.evidence)
        self.client.request.return_value = {
            "message": {"role": "assistant", "content": "not-json"}
        }
        with self.assertRaises(AssistantError):
            self.chat.ask("云灯在哪里", self.evidence)

    def test_removed_knowledge_is_not_replayed_from_rag_history(self):
        self.response({"answer": "云灯在书房。", "citations": [self.citation]})
        self.chat.ask("云灯在哪里", self.evidence)
        self.client.reset_mock()
        self.assertIn("未检索到", self.chat.ask("云灯在哪里", []))
        self.client.request.assert_not_called()
        self.assertEqual(self.chat.history, [])


if __name__ == "__main__":
    unittest.main()
