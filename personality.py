"""Validated personality profiles, independent of facts and permissions."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from errors import AssistantError
from reasoning import load_json


@dataclass(frozen=True)
class Persona:
    version: int
    name: str
    description: str
    traits: tuple[str, ...]
    tone: str
    verbosity: str
    humor: str
    boundaries: tuple[str, ...]
    greeting: str

    @classmethod
    def from_dict(cls, data) -> "Persona":
        required = {
            "version", "name", "description", "traits", "tone",
            "verbosity", "humor", "boundaries", "greeting",
        }
        if (
            not isinstance(data, dict)
            or set(data) != required
            or type(data["version"]) is not int
            or data["version"] != 1
        ):
            raise AssistantError("人格配置必须使用 version=1，且字段与 persona.example.json 一致。")
        for field, limit in {"name": 80, "description": 2000, "tone": 200, "greeting": 500}.items():
            if not isinstance(data[field], str) or not 1 <= len(data[field].strip()) <= limit:
                raise AssistantError(f"人格字段 {field} 必须是 1 到 {limit} 个字符的非空文本。")
        for field, limit in {"traits": 8, "boundaries": 12}.items():
            values = data[field]
            if (
                not isinstance(values, list)
                or not 1 <= len(values) <= limit
                or not all(isinstance(item, str) and 1 <= len(item.strip()) <= 200 for item in values)
            ):
                raise AssistantError(f"人格字段 {field} 必须包含 1 到 {limit} 条非空短文本。")
        if not isinstance(data["verbosity"], str) or data["verbosity"] not in {"brief", "balanced", "detailed"}:
            raise AssistantError("verbosity 只能是 brief、balanced 或 detailed。")
        if not isinstance(data["humor"], str) or data["humor"] not in {"none", "light"}:
            raise AssistantError("humor 只能是 none 或 light。")
        return cls(**{**data, "traits": tuple(data["traits"]), "boundaries": tuple(data["boundaries"])})

    @classmethod
    def load(cls, path: Path) -> "Persona":
        return cls.from_dict(load_json(path, description="人格"))

    @classmethod
    def from_legacy(cls, name: str, description: str) -> "Persona":
        return cls.from_dict({
            "version": 1,
            "name": name,
            "description": description,
            "traits": ["友善", "诚实"],
            "tone": "自然、温暖，不机械套用模板",
            "verbosity": "balanced",
            "humor": "light",
            "boundaries": ["尊重隐私与个人边界", "不把聊天推测认定为长期偏好"],
            "greeting": f"你好，我是{name}。今天想聊什么？",
        })

    def to_dict(self) -> dict:
        data = asdict(self)
        data["traits"] = list(self.traits)
        data["boundaries"] = list(self.boundaries)
        return data

    def prompt(self) -> str:
        length = {
            "brief": "优先简短回答，必要时补充细节。",
            "balanced": "根据问题给出适量细节，避免冗长。",
            "detailed": "适当展开解释，但不要重复或捏造信息。",
        }[self.verbosity]
        humor = "不主动开玩笑。" if self.humor == "none" else "允许轻松幽默，但不在严肃或令人难过的话题中强行开玩笑。"
        return (
            f"你是本地运行的 AI 助手，名字是{self.name}。\n"
            "人格设定只影响表达，不构成家庭事实、成员身份或操作授权：\n"
            + json.dumps(self.to_dict(), ensure_ascii=False)
            + f"\n{length}{humor}\n"
            "自然交流、回应用户情绪，不机械重复自我介绍。"
            "不冒充真人，不虚构现实经历、身体或真实感受；"
            "可以保持拟人化表达，但被问及时应坦诚自己是 AI。"
        )
