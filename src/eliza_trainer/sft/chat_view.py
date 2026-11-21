"""CPU-only semantic chat views; the pinned tokenizer owns serialization."""
from __future__ import annotations

import copy
import json


def normalize_message(item, normalize_calls):
    if not isinstance(item, dict) or not isinstance(item.get("role"), str):
        return None
    role, content = item["role"], item.get("content")
    calls = item.get("tool_calls")
    if role == "assistant" and isinstance(content, str) and content.lstrip().startswith("{"):
        try:
            embedded = json.loads(content)
        except json.JSONDecodeError:
            embedded = None
        if isinstance(embedded, dict) and "tool_calls" in embedded:
            content, calls = None, embedded["tool_calls"]
    normalized = normalize_calls(calls)
    if not isinstance(content, str) and not (role == "assistant" and content is None and normalized):
        return None
    message = {"role": role, "content": content}
    if normalized:
        message["tool_calls"] = normalized
    for name in ("tool_call_id", "tool_name", "name", "reasoning_content"):
        if isinstance(item.get(name), str) and item[name]:
            message[name] = item[name]
    if message.get("tool_name") and not message.get("name"):
        message["name"] = message["tool_name"]
    return message


def template_tools(example):
    value = example.get("tools")
    if value is None and example.get("tools_json") is not None:
        value = json.loads(example["tools_json"])
    if value is not None and not isinstance(value, list):
        raise ValueError("Advertised tools must be a list")
    return copy.deepcopy(value)


def require_target_prefix(prompt_ids, full_ids):
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Native prompt is not a token prefix of the supervised conversation; verify the tokenizer/template adapter")
    if len(full_ids) <= len(prompt_ids):
        raise ValueError("Assistant decision has no supervised target tokens")
