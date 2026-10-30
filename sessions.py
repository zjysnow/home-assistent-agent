"""Explicit, opt-in snapshots of bounded ordinary-chat context."""

from contextlib import closing, contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sqlite3
import uuid

from errors import AssistantError
from personality import Persona
from reasoning import decode_json


MAX_SNAPSHOT_BYTES = 256_000


def validate_messages(messages) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not 2 <= len(messages) <= 20 or len(messages) % 2:
        raise AssistantError("会话快照必须包含 1 到 10 轮完整的普通聊天。")
    for index, message in enumerate(messages):
        if (
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message["role"] != ("user" if index % 2 == 0 else "assistant")
            or not isinstance(message["content"], str)
            or not message["content"].strip()
            or len(message["content"]) > (8000 if index % 2 == 0 else 64000)
        ):
            raise AssistantError("会话内容或角色顺序无效；不允许恢复 system、工具或额外字段。")
    return [dict(message) for message in messages]


def validate_title(title):
    if not isinstance(title, str) or not 1 <= len(title.strip()) <= 80:
        raise AssistantError("会话标题必须是 1 到 80 个字符的非空文本。")


@dataclass(frozen=True)
class Snapshot:
    identifier: str
    title: str
    created_at: str
    persona: Persona
    messages: list[dict[str, str]]


class SessionStore:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def database(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(self.path)) as db:
                db.row_factory = sqlite3.Row
                with db:
                    db.execute(
                        """CREATE TABLE IF NOT EXISTS snapshots (
                            id TEXT PRIMARY KEY, title TEXT NOT NULL,
                            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                            payload TEXT NOT NULL
                        )"""
                    )
                    yield db
        except (OSError, sqlite3.Error) as exc:
            raise AssistantError("会话快照操作失败；请检查本地目录权限、磁盘和数据库状态。") from exc

    def save(self, title: str, persona: Persona, messages: list[dict[str, str]]) -> str:
        validate_title(title)
        validated = validate_messages(messages)
        profile = Persona.from_dict(persona.to_dict())
        payload = json.dumps(
            {"version": 1, "persona": profile.to_dict(), "messages": validated},
            ensure_ascii=False,
        )
        if len(payload.encode("utf-8")) > MAX_SNAPSHOT_BYTES:
            raise AssistantError("会话快照超过 256 KB，未保存；请缩短会话内容。")
        identifier = uuid.uuid4().hex
        with self.database() as db:
            db.execute(
                "INSERT INTO snapshots (id, title, payload) VALUES (?, ?, ?)",
                (identifier, title.strip(), payload),
            )
        return identifier

    def list(self) -> list[dict]:
        with self.database() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT id, title, created_at FROM snapshots ORDER BY created_at DESC, id"
                )
            ]

    def validate_id(self, identifier):
        if not isinstance(identifier, str) or not re.fullmatch(r"[a-f0-9]{32}", identifier):
            raise AssistantError("会话 ID 格式无效；输入 /sessions 查看。")

    def load(self, identifier: str) -> Snapshot:
        self.validate_id(identifier)
        with self.database() as db:
            row = db.execute(
                "SELECT id, title, created_at, payload FROM snapshots WHERE id = ?",
                (identifier,),
            ).fetchone()
        if row is None:
            raise AssistantError("会话快照不存在；输入 /sessions 查看。")
        payload = row["payload"]
        if not isinstance(payload, str) or len(payload.encode("utf-8")) > MAX_SNAPSHOT_BYTES:
            raise AssistantError("会话快照过大或损坏，已拒绝恢复。")
        data = decode_json(payload)
        if (
            not isinstance(data, dict)
            or set(data) != {"version", "persona", "messages"}
            or type(data["version"]) is not int
            or data["version"] != 1
        ):
            raise AssistantError("会话快照版本或字段无效，已拒绝恢复。")
        validate_title(row["title"])
        return Snapshot(
            row["id"], row["title"], row["created_at"],
            Persona.from_dict(data["persona"]), validate_messages(data["messages"]),
        )

    def delete(self, identifier: str):
        self.validate_id(identifier)
        with self.database() as db:
            result = db.execute("DELETE FROM snapshots WHERE id = ?", (identifier,))
            if result.rowcount == 0:
                raise AssistantError("会话快照不存在，未删除任何内容。")
