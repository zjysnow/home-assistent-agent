import io
from http.client import IncompleteRead
import json
import os
from pathlib import Path
import unittest
from unittest.mock import MagicMock, Mock, patch
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request

from assistant import (
    AssistantError,
    Chat,
    Config,
    HomeAssistant,
    JsonClient,
    NoRedirect,
    main,
    run,
    validate_url,
)


def example_data():
    return {
        "name": "Test",
        "persona": "Be helpful.",
        "model": "local-model",
        "ollama_url": "http://127.0.0.1:11434",
        "ha_url": "http://192.168.1.2:8123",
        "allowed_entities": ["light.desk", "fan.bedroom"],
    }


def config():
    data = example_data()
    data["allowed_entities"] = tuple(data["allowed_entities"])
    return Config(**data)


class ConfigTests(unittest.TestCase):
    def load(self, data):
        with patch.object(Path, "read_text", return_value=json.dumps(data)):
            return Config.load(Path("test.json"))

    def test_valid_config(self):
        loaded = self.load(example_data())
        self.assertEqual(loaded, config())

    def test_chat_only_configuration(self):
        data = example_data()
        data.update(ha_url="", allowed_entities=[])
        loaded = self.load(data)
        self.assertEqual(loaded.ha_url, "")
        self.assertEqual(loaded.allowed_entities, ())

    def test_invalid_configuration_fields(self):
        for field, value in [
            ("name", ""),
            ("model", 12),
            ("allowed_entities", "light.desk"),
            ("allowed_entities", ["lock.door"]),
            ("allowed_entities", ["switch.heater"]),
            ("allowed_entities", ["light.desk", "light.desk"]),
            ("allowed_entities", ["light.desk/../lock.door"]),
            ("allowed_entities", [None]),
            ("unexpected", True),
        ]:
            with self.subTest(field=field, value=value):
                data = example_data()
                data[field] = value
                with self.assertRaises(AssistantError):
                    self.load(data)
        with self.assertRaises(AssistantError):
            self.load([])
        data = example_data()
        del data["model"]
        with self.assertRaises(AssistantError):
            self.load(data)

    def test_missing_or_malformed_file(self):
        for error in [FileNotFoundError(), UnicodeError(), json.JSONDecodeError("bad", "", 0)]:
            with self.subTest(error=type(error).__name__):
                with patch.object(Path, "read_text", side_effect=error):
                    with self.assertRaises(AssistantError):
                        Config.load(Path("missing.json"))

    def test_url_boundaries(self):
        for url in [
            "https://example.com",
            "file:///etc/passwd",
            "http://localhost:0",
            "http://localhost:bad",
            "http://user:pass@localhost",
            "http://localhost/api",
            "http://localhost?secret=1",
            "http://localhost/#fragment",
            "http://[bad",
        ]:
            with self.subTest(url=url):
                with self.assertRaises(AssistantError):
                    validate_url(url, local_only=True)
        self.assertEqual(
            validate_url("http://[::1]:11434/", local_only=True),
            "http://[::1]:11434",
        )


class JsonClientTests(unittest.TestCase):
    def setUp(self):
        self.client = JsonClient()
        self.client.opener = MagicMock()
        self.response = self.client.opener.open.return_value.__enter__.return_value = Mock()
        self.response.read.return_value = b'{"ok": true}'

    def test_get_and_post_headers(self):
        result = self.client.request("http://localhost/api", token="test-token")
        self.assertEqual(result, {"ok": True})
        request = self.client.opener.open.call_args.args[0]
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-token")
        self.client.request("http://localhost/api", payload={"text": "hello"})
        request = self.client.opener.open.call_args.args[0]
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(json.loads(request.data), {"text": "hello"})
        self.assertEqual(request.get_header("Content-type"), "application/json")
        self.assertIsNone(request.get_header("Authorization"))

    def test_transport_errors_do_not_expose_secrets(self):
        for error in [
            HTTPError("http://localhost", 401, "private-response", {}, None),
            URLError("private-response"),
            TimeoutError("private-response"),
            IncompleteRead(b"private-response", 50),
        ]:
            with self.subTest(error=type(error).__name__):
                self.client.opener.open.side_effect = error
                with self.assertRaises(AssistantError) as caught:
                    self.client.request("http://localhost/api", token="private-token")
                self.assertNotIn("private-response", str(caught.exception))
                self.assertNotIn("private-token", str(caught.exception))

    def test_invalid_or_oversized_response(self):
        for raw in [b"not-json", b"\xff", b"x" * 2_000_001]:
            with self.subTest(size=len(raw)):
                self.response.read.return_value = raw
                with self.assertRaises(AssistantError):
                    self.client.request("http://localhost/api")
        self.response.read.assert_called_with(2_000_001)

    def test_redirect_is_not_followed(self):
        result = NoRedirect().redirect_request(
            Request("http://localhost"), None, 302, "", {}, "http://other-host"
        )
        self.assertIsNone(result)

    def test_environment_proxy_is_disabled(self):
        with patch("assistant.ProxyHandler", wraps=ProxyHandler) as handler:
            JsonClient()
        handler.assert_called_once_with({})


class ChatTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.client.request.return_value = {
            "message": {"role": "assistant", "content": "Hello"}
        }
        self.chat = Chat(config(), self.client)

    def test_persona_and_context(self):
        self.assertEqual(self.chat.reply("First"), "Hello")
        self.chat.reply("Second")
        request = self.client.request.call_args
        self.assertEqual(request.args[0], "http://127.0.0.1:11434/api/chat")
        payload = request.kwargs["payload"]
        self.assertEqual(payload["model"], "local-model")
        self.assertFalse(payload["stream"])
        self.assertIn("Be helpful.", payload["messages"][0]["content"])
        self.assertEqual(payload["messages"][1]["content"], "First")
        self.assertEqual(payload["messages"][-1]["content"], "Second")
        self.assertNotIn("token", request.kwargs)

    def test_history_is_bounded_and_session_only(self):
        for index in range(15):
            self.chat.reply(str(index))
        self.assertEqual(len(self.chat.history), 20)
        self.assertEqual(self.chat.history[0]["content"], "5")
        self.assertEqual(Chat(config(), self.client).history, [])

    def test_invalid_inputs_do_not_call_model(self):
        for text in ["", "  ", "x" * 8001]:
            with self.assertRaises(AssistantError):
                self.chat.reply(text)
        self.client.request.assert_not_called()

    def test_invalid_model_results_do_not_pollute_history(self):
        for result in [
            {},
            [],
            {"message": None},
            {"message": {"role": "user", "content": "wrong"}},
            {"message": {"role": "assistant", "content": ""}},
            {"message": {"role": "assistant", "content": 1}},
        ]:
            with self.subTest(result=result):
                self.client.request.return_value = result
                with self.assertRaises(AssistantError):
                    self.chat.reply("Hello")
                self.assertEqual(self.chat.history, [])
        self.client.request.side_effect = AssistantError("offline")
        with self.assertRaises(AssistantError):
            self.chat.reply("Hello")
        self.assertEqual(self.chat.history, [])


class HomeAssistantTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.home = HomeAssistant(config(), self.client)
        env = patch.dict(os.environ, {"HA_TOKEN": "test-token"})
        env.start()
        self.addCleanup(env.stop)

    def state(self, value="off"):
        return {"entity_id": "light.desk", "state": value}

    def test_state_request(self):
        self.client.request.return_value = self.state()
        self.assertEqual(self.home.state("light.desk"), "off")
        self.client.request.assert_called_once_with(
            "http://192.168.1.2:8123/api/states/light.desk",
            payload=None,
            token="test-token",
        )

    def test_unauthorized_entities_never_reach_network(self):
        for entity in ["light.other", "lock.door", "light.desk/../../lock.door"]:
            with self.assertRaises(AssistantError):
                self.home.power(entity, "on")
        self.client.request.assert_not_called()

    def test_missing_token_or_url(self):
        with patch.dict(os.environ, {"HA_TOKEN": ""}):
            with self.assertRaises(AssistantError):
                self.home.state("light.desk")
        data = example_data()
        data["ha_url"] = ""
        with self.assertRaises(AssistantError):
            HomeAssistant(Config(**data), self.client).state("light.desk")
        self.client.request.assert_not_called()

    def test_invalid_state(self):
        for result in [[], {}, {"entity_id": "light.other", "state": "on"}, self.state(2)]:
            self.client.request.return_value = result
            with self.assertRaises(AssistantError):
                self.home.state("light.desk")

    def test_power_checks_state_and_targets_only_one_entity(self):
        self.client.request.side_effect = [self.state(), [], self.state("on")]
        result = self.home.power("light.desk", "on")
        call = self.client.request.call_args_list[1]
        self.assertEqual(
            call.args[0], "http://192.168.1.2:8123/api/services/light/turn_on"
        )
        self.assertEqual(call.kwargs["payload"], {"entity_id": "light.desk"})
        self.assertIn("on", result)
        self.assertIn("不代表物理设备已完成动作", result)

    def test_fan_and_turn_off(self):
        self.client.request.side_effect = [
            {"entity_id": "fan.bedroom", "state": "on"},
            [],
            {"entity_id": "fan.bedroom", "state": "off"},
        ]
        self.home.power("fan.bedroom", "off")
        self.assertTrue(
            self.client.request.call_args_list[1].args[0].endswith("/fan/turn_off")
        )

    def test_unknown_or_unavailable_prevents_control(self):
        for state in ["unknown", "unavailable"]:
            self.client.reset_mock()
            self.client.request.return_value = self.state(state)
            with self.assertRaises(AssistantError):
                self.home.power("light.desk", "on")
            self.assertEqual(self.client.request.call_count, 1)

    def test_invalid_action_is_rejected(self):
        with self.assertRaises(AssistantError):
            self.home.power("light.desk", "toggle")
        self.client.request.assert_not_called()

    def test_failure_after_control_is_not_reported_as_success(self):
        for response, text in [
            (AssistantError("timeout"), "结果未确认"),
            ({}, "服务响应无效"),
            ([], "实际结果未确认"),
        ]:
            with self.subTest(response=response):
                self.client.request.side_effect = [
                    self.state(),
                    response,
                    AssistantError("offline"),
                ]
                with self.assertRaisesRegex(AssistantError, text):
                    self.home.power("light.desk", "on")


class TerminalTests(unittest.TestCase):
    def session(self, inputs, client=None):
        output, errors = io.StringIO(), io.StringIO()
        with (
            patch("builtins.input", side_effect=inputs),
            patch("assistant.JsonClient", return_value=client or Mock()),
            patch("sys.stdout", output),
            patch("sys.stderr", errors),
        ):
            result = run(config())
        return result, output.getvalue(), errors.getvalue()

    def test_cancel_does_not_send_request(self):
        client = Mock()
        result, output, errors = self.session(
            ["/on light.desk", "no", "/quit"], client
        )
        self.assertEqual(result, 0)
        self.assertIn("已取消", output)
        self.assertEqual(errors, "")
        client.request.assert_not_called()

    def test_confirmed_control(self):
        client = Mock()
        client.request.side_effect = [
            {"entity_id": "light.desk", "state": "off"},
            [],
            {"entity_id": "light.desk", "state": "on"},
        ]
        with patch.dict(os.environ, {"HA_TOKEN": "test-token"}):
            _, output, errors = self.session(
                ["/on light.desk", "yes", "/quit"], client
            )
        self.assertIn("操作请求已被接受", output)
        self.assertEqual(errors, "")
        self.assertEqual(client.request.call_count, 3)

    def test_device_list_and_state_do_not_call_ollama(self):
        client = Mock()
        client.request.side_effect = [
            {"entity_id": "light.desk", "state": "on"},
            {"entity_id": "fan.bedroom", "state": "off"},
            {"entity_id": "light.desk", "state": "on"},
        ]
        with patch.dict(os.environ, {"HA_TOKEN": "test-token"}):
            _, output, errors = self.session(
                ["/devices", "/state light.desk", "/quit"], client
            )
        self.assertIn("light.desk: on", output)
        self.assertIn("fan.bedroom: off", output)
        self.assertEqual(errors, "")
        self.assertTrue(all("/api/states/" in call.args[0] for call in client.request.call_args_list))

    def test_chat_and_reset(self):
        client = Mock()
        client.request.return_value = {
            "message": {"role": "assistant", "content": "Hello"}
        }
        _, output, errors = self.session(
            ["First", "/reset", "Second", "/quit"], client
        )
        self.assertIn("Hello", output)
        self.assertEqual(errors, "")
        messages = client.request.call_args.kwargs["payload"]["messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[-1]["content"], "Second")

    def test_error_does_not_end_session(self):
        client = Mock()
        client.request.side_effect = AssistantError("offline")
        result, output, errors = self.session(
            ["hello", "/wrong", "/on lock.door", "/help", "/quit"], client
        )
        self.assertEqual(result, 0)
        self.assertIn("offline", errors)
        self.assertIn("未知命令", errors)
        self.assertIn("未获授权", errors)
        self.assertIn("/devices", output)
        self.assertEqual(client.request.call_count, 1)

    def test_eof_and_interrupt_exit_without_control(self):
        for exception in [EOFError(), KeyboardInterrupt()]:
            result, output, _ = self.session(["/on light.desk", exception])
            self.assertEqual(result, 0)
            self.assertIn("已退出", output)

    def test_main_reports_configuration_failure(self):
        with (
            patch.object(Config, "load", side_effect=AssistantError("bad config")),
            patch("sys.stderr", io.StringIO()) as errors,
        ):
            self.assertEqual(main([]), 1)
            self.assertIn("bad config", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
