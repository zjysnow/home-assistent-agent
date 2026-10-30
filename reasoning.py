"""Bounded rule representation and conservative four-valued evaluation."""

from enum import Enum
import json
from pathlib import Path
import re

from errors import AssistantError


FACT_PATTERN = r"[a-z][a-z0-9_.]{0,79}"
EXPRESSION_SCHEMA = {
    "oneOf": [
        {
            "type": "object",
            "properties": {"fact": {"type": "string", "pattern": f"^{FACT_PATTERN}$"}},
            "required": ["fact"],
            "additionalProperties": False,
        },
        *[
            {
                "type": "object",
                "properties": {
                    operator: {
                        "type": "array",
                        "items": {"$ref": "#/$defs/expression"},
                        "minItems": 1,
                        "maxItems": 16,
                    }
                },
                "required": [operator],
                "additionalProperties": False,
            }
            for operator in ("all", "any")
        ],
        {
            "type": "object",
            "properties": {"not": {"$ref": "#/$defs/expression"}},
            "required": ["not"],
            "additionalProperties": False,
        },
    ]
}
RULE_SCHEMA = {
    "type": "object",
    "properties": {
        "version": {"const": 1},
        "rules": {
            "type": "array",
            "minItems": 1,
            "maxItems": 32,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "pattern": f"^{FACT_PATTERN}$"},
                    "when": {"$ref": "#/$defs/expression"},
                    "suggestion": {"type": "string", "minLength": 1, "maxLength": 500},
                },
                "required": ["id", "when", "suggestion"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["version", "rules"],
    "additionalProperties": False,
    "$defs": {"expression": EXPRESSION_SCHEMA},
}


class Truth(Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"


def unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AssistantError("JSON 包含重复字段；请明确消除歧义或用 conflict 表示事实冲突。")
        result[key] = value
    return result


def decode_json(raw):
    try:
        return json.loads(raw, object_pairs_hook=unique_fields)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise AssistantError("内容不是有效或可处理的 JSON。") from exc


def load_json(path: Path):
    try:
        with path.open("rb") as stream:
            raw = stream.read(100_001)
        if len(raw) > 100_000:
            raise AssistantError("规则或事实文件最多为 100 KB。")
        return decode_json(raw)
    except OSError as exc:
        raise AssistantError("无法读取规则或事实文件；请检查路径及 JSON 格式。") from exc


def validate_expression(node, *, depth=0, budget=None):
    if budget is None:
        budget = [64]
    budget[0] -= 1
    if depth > 8 or budget[0] < 0:
        raise AssistantError("规则表达式最多 8 层、64 个节点。")
    if not isinstance(node, dict) or len(node) != 1:
        raise AssistantError("规则表达式只能有一个 fact、all、any 或 not 字段。")
    operator, value = next(iter(node.items()))
    if operator == "fact":
        if not isinstance(value, str) or not re.fullmatch(FACT_PATTERN, value):
            raise AssistantError("事实名称必须是小写英文、数字、下划线或点，且以字母开头。")
    elif operator in {"all", "any"}:
        if not isinstance(value, list) or not 1 <= len(value) <= 16:
            raise AssistantError("all／any 必须包含 1 到 16 个条件。")
        for child in value:
            validate_expression(child, depth=depth + 1, budget=budget)
    elif operator == "not":
        validate_expression(value, depth=depth + 1, budget=budget)
    else:
        raise AssistantError("规则包含不支持的运算符；不允许代码或设备动作。")


def validate_rules(data) -> dict:
    if (
        not isinstance(data, dict)
        or set(data) != {"version", "rules"}
        or type(data["version"]) is not int
        or data["version"] != 1
        or not isinstance(data["rules"], list)
        or not 1 <= len(data["rules"]) <= 32
    ):
        raise AssistantError("规则文件必须使用 version=1 并包含 1 到 32 条 rules。")
    identifiers = set()
    for rule in data["rules"]:
        if not isinstance(rule, dict) or set(rule) != {"id", "when", "suggestion"}:
            raise AssistantError("每条规则必须且只能包含 id、when、suggestion。")
        identifier = rule["id"]
        if (
            not isinstance(identifier, str)
            or not re.fullmatch(FACT_PATTERN, identifier)
            or identifier in identifiers
        ):
            raise AssistantError("规则 ID 无效或重复。")
        identifiers.add(identifier)
        if not isinstance(rule["suggestion"], str) or not 1 <= len(rule["suggestion"].strip()) <= 500:
            raise AssistantError("规则建议必须是 1 到 500 个字符的非空文本。")
        validate_expression(rule["when"])
    return data


def validate_facts(data) -> dict[str, Truth]:
    if (
        not isinstance(data, dict)
        or set(data) != {"facts"}
        or not isinstance(data["facts"], dict)
        or len(data["facts"]) > 256
    ):
        raise AssistantError("事实文件必须只包含 facts 对象，最多 256 项。")
    result = {}
    for name, value in data["facts"].items():
        if not isinstance(name, str) or not re.fullmatch(FACT_PATTERN, name):
            raise AssistantError("事实名称格式无效。")
        if value is True:
            result[name] = Truth.TRUE
        elif value is False:
            result[name] = Truth.FALSE
        elif value is None:
            result[name] = Truth.UNKNOWN
        elif isinstance(value, str) and value == "conflict":
            result[name] = Truth.CONFLICT
        else:
            raise AssistantError("事实值只能是 true、false、null 或字符串 conflict。")
    return result


def evaluate(node: dict, facts: dict[str, Truth]) -> Truth:
    operator, value = next(iter(node.items()))
    if operator == "fact":
        return facts.get(value, Truth.UNKNOWN)
    if operator == "not":
        child = evaluate(value, facts)
        return {Truth.TRUE: Truth.FALSE, Truth.FALSE: Truth.TRUE}.get(child, child)
    children = [evaluate(child, facts) for child in value]
    # Conflicting evidence blocks suggestions even if another branch is decisive.
    if Truth.CONFLICT in children:
        return Truth.CONFLICT
    decisive = Truth.FALSE if operator == "all" else Truth.TRUE
    if decisive in children:
        return decisive
    if Truth.UNKNOWN in children:
        return Truth.UNKNOWN
    return Truth.TRUE if operator == "all" else Truth.FALSE


def fact_names(node: dict) -> set[str]:
    operator, value = next(iter(node.items()))
    if operator == "fact":
        return {value}
    if operator == "not":
        return fact_names(value)
    return set().union(*(fact_names(child) for child in value))


def reason(rules, facts) -> list[dict]:
    validated = validate_rules(rules)
    known = validate_facts(facts)
    reports = []
    for rule in validated["rules"]:
        state = evaluate(rule["when"], known)
        reports.append(
            {
                "id": rule["id"],
                "status": state.value,
                "suggestion": rule["suggestion"] if state is Truth.TRUE else None,
                "expression": rule["when"],
                "evidence": {
                    name: known.get(name, Truth.UNKNOWN).value
                    for name in sorted(fact_names(rule["when"]))
                },
            }
        )
    return reports
