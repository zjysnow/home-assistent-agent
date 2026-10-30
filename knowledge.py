"""Local, versioned document retrieval using SQLite FTS5, not embeddings."""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import sqlite3

from errors import AssistantError


def tokens(text: str) -> list[str]:
    result = []
    for word in re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]+", text.lower()):
        if re.fullmatch(r"[a-z0-9_]+", word) or len(word) == 1:
            result.append(word)
        else:
            result.extend(word[index : index + 2] for index in range(len(word) - 1))
    return result


@dataclass(frozen=True)
class Evidence:
    document_id: str
    revision: int
    chunk: int
    source: str
    text: str

    @property
    def citation(self) -> str:
        return f"{self.document_id}:r{self.revision}:c{self.chunk}"


class KnowledgeStore:
    def __init__(self, path: Path):
        self.path = path
        self.connection: sqlite3.Connection | None = None

    def __enter__(self) -> "KnowledgeStore":
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.path)
            self.connection.row_factory = sqlite3.Row
            with self.connection:
                self.connection.execute(
                    """CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY, source TEXT NOT NULL UNIQUE,
                        digest TEXT NOT NULL, revision INTEGER NOT NULL,
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )"""
                )
                self.connection.execute(
                    """CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
                        terms, text UNINDEXED, document_id UNINDEXED, ordinal UNINDEXED
                    )"""
                )
        except (OSError, sqlite3.Error) as exc:
            self.__exit__(None, None, None)
            raise AssistantError("无法初始化本地知识库；请检查目录权限和 SQLite FTS5 支持。") from exc
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    @contextmanager
    def database(self):
        if self.connection is None:
            raise AssistantError("知识库尚未打开。")
        try:
            with self.connection:
                yield self.connection
        except sqlite3.Error as exc:
            raise AssistantError("本地知识库操作失败；请检查磁盘、权限和数据库状态。") from exc

    def ingest(self, path: Path) -> tuple[str, int, bool]:
        if path.suffix.lower() not in {".txt", ".md"}:
            raise AssistantError("只允许导入 UTF-8 的 .txt 或 .md 文件。")
        try:
            source = str(path.resolve(strict=True))
            with path.open("rb") as stream:
                raw = stream.read(1_000_001)
        except OSError as exc:
            raise AssistantError("无法读取知识文档；请检查文件路径和权限。") from exc
        if len(raw) > 1_000_000:
            raise AssistantError("单个知识文档最多为 1 MB。")
        try:
            text = raw.decode("utf-8").strip()
        except UnicodeError as exc:
            raise AssistantError("知识文档必须使用 UTF-8 编码。") from exc
        if not text or "\x00" in text or not tokens(text):
            raise AssistantError("知识文档为空或不包含可检索的中英文文本。")
        identifier = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
        digest = hashlib.sha256(raw).hexdigest()
        with self.database() as db:
            previous = db.execute(
                "SELECT revision, digest FROM documents WHERE id = ?", (identifier,)
            ).fetchone()
            if previous and previous["digest"] == digest:
                return identifier, previous["revision"], False
            revision = previous["revision"] + 1 if previous else 1
            db.execute(
                """INSERT INTO documents (id, source, digest, revision) VALUES (?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET digest = excluded.digest,
                   revision = excluded.revision, updated_at = CURRENT_TIMESTAMP""",
                (identifier, source, digest, revision),
            )
            db.execute("DELETE FROM chunks WHERE document_id = ?", (identifier,))
            for ordinal, start in enumerate(range(0, len(text), 500), 1):
                chunk = text[start : start + 600]
                db.execute(
                    "INSERT INTO chunks (terms, text, document_id, ordinal) VALUES (?, ?, ?, ?)",
                    (
                        " ".join(tokens(chunk) + re.findall(r"[\u3400-\u9fff]", chunk)),
                        chunk,
                        identifier,
                        ordinal,
                    ),
                )
        return identifier, revision, True

    def search(self, query: str) -> list[Evidence]:
        if not query.strip() or len(query) > 8000:
            raise AssistantError("检索问题不能为空，且最多为 8000 个字符。")
        terms = list(dict.fromkeys(tokens(query)))[:64]
        if not terms:
            raise AssistantError("检索问题必须包含可检索的中英文文本。")
        expression = " OR ".join(f'"{term}"' for term in terms)
        with self.database() as db:
            rows = db.execute(
                """SELECT d.id, d.revision, c.ordinal, d.source, c.text
                   FROM chunks c JOIN documents d ON d.id = c.document_id
                   WHERE chunks MATCH ? ORDER BY bm25(chunks), d.id, c.ordinal LIMIT 4""",
                (expression,),
            ).fetchall()
        return [
            Evidence(row["id"], row["revision"], row["ordinal"], row["source"], row["text"])
            for row in rows
        ]

    def documents(self) -> list[dict]:
        with self.database() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT id, source, revision, updated_at FROM documents ORDER BY id"
                )
            ]

    def forget(self, identifier: str):
        with self.database() as db:
            result = db.execute("DELETE FROM documents WHERE id = ?", (identifier,))
            if result.rowcount == 0:
                raise AssistantError("文档 ID 不存在；输入 /knowledge 查看。")
            db.execute("DELETE FROM chunks WHERE document_id = ?", (identifier,))
