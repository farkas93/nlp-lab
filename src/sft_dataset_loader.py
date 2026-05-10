from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import fsspec
from datasets import Dataset, DownloadMode, load_dataset


@dataclass
class SFTManifestLoadResult:
    train_dataset: Dataset
    eval_dataset: Dataset
    manifest: dict[str, Any]
    manifest_uri: str
    manifest_sha256: str


LOGGER = logging.getLogger(__name__)


def _s3_storage_options() -> dict[str, Any]:
    options: dict[str, Any] = {}
    key = os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("STORAGE_ACCOUNT")
    secret = os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("STORAGE_KEY")
    endpoint_url = os.getenv("AWS_ENDPOINT_URL") or os.getenv("STORAGE_URL")
    region = os.getenv("AWS_REGION", "us-east-1")

    if key:
        options["key"] = key
    if secret:
        options["secret"] = secret
    if endpoint_url:
        options["client_kwargs"] = {"endpoint_url": endpoint_url, "region_name": region}
    return options


def _read_text_uri(uri: str) -> str:
    parsed = urlparse(uri)
    if parsed.scheme in {"s3", "s3a"}:
        with fsspec.open(uri, "rt", encoding="utf-8", **_s3_storage_options()) as handle:
            return handle.read()
    return Path(uri).read_text(encoding="utf-8")


def _resolve_split_file_uri(manifest_uri: str, split_key: str) -> str:
    parsed = urlparse(manifest_uri)
    if "://" in split_key:
        return split_key

    if parsed.scheme in {"s3", "s3a"}:
        bucket = parsed.netloc
        return f"s3://{bucket}/{split_key.lstrip('/')}"

    manifest_path = Path(manifest_uri)
    base_dir = manifest_path.parent
    return str((base_dir / split_key).resolve())


def _load_manifest(manifest_uri: str) -> tuple[dict[str, Any], str]:
    raw_text = _read_text_uri(manifest_uri)
    manifest = json.loads(raw_text)
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be a JSON object")
    digest = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    return manifest, digest


def _split_files_from_manifest(manifest: dict[str, Any], split_name: str) -> list[str]:
    splits = manifest.get("splits")
    if not isinstance(splits, list):
        return []
    return [
        item.get("key")
        for item in splits
        if isinstance(item, dict)
        and item.get("split") == split_name
        and isinstance(item.get("key"), str)
    ]


def load_sft_manifest_dataset(
    *,
    manifest_uri: str,
    train_split: str,
    eval_split: str,
    max_train_samples: int | None,
    max_eval_samples: int | None,
    cache_mode: str = "reuse",
) -> SFTManifestLoadResult:
    manifest, manifest_sha256 = _load_manifest(manifest_uri)

    train_files = _split_files_from_manifest(manifest, train_split)
    eval_files = _split_files_from_manifest(manifest, eval_split)
    if not train_files:
        raise ValueError(f"No train files found in manifest split '{train_split}'")
    if not eval_files:
        raise ValueError(f"No eval files found in manifest split '{eval_split}'")

    train_uris = [_resolve_split_file_uri(manifest_uri, key) for key in train_files]
    eval_uris = [_resolve_split_file_uri(manifest_uri, key) for key in eval_files]

    download_mode = DownloadMode.REUSE_DATASET_IF_EXISTS
    if cache_mode == "refresh":
        download_mode = DownloadMode.FORCE_REDOWNLOAD

    dataset_dict = load_dataset(
        "parquet",
        data_files={"train": train_uris, "eval": eval_uris},
        storage_options=_s3_storage_options(),
        download_mode=download_mode,
    )
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["eval"]

    if max_train_samples is not None and len(train_dataset) > max_train_samples:
        train_dataset = train_dataset.select(range(max_train_samples))
    if max_eval_samples is not None and len(eval_dataset) > max_eval_samples:
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    return SFTManifestLoadResult(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        manifest=manifest,
        manifest_uri=manifest_uri,
        manifest_sha256=manifest_sha256,
    )


def _truncate_text(value: str, limit: int = 240) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "..."


def _build_template_debug_payload(
    *,
    split_name: str,
    example: dict[str, Any],
    prompt_messages: list[dict[str, str]],
    target_text: str,
    reason: str,
    error: Exception | None = None,
) -> dict[str, Any]:
    role_sequence = [msg.get("role") for msg in prompt_messages]
    last_non_system_role = None
    for msg in reversed(prompt_messages):
        role = msg.get("role")
        if role != "system":
            last_non_system_role = role
            break

    tail_preview = [
        {
            "role": msg.get("role"),
            "content": _truncate_text(msg.get("content") or ""),
        }
        for msg in prompt_messages[-3:]
    ]

    payload: dict[str, Any] = {
        "split": split_name,
        "reason": reason,
        "example_id": example.get("example_id"),
        "session_id": example.get("session_id"),
        "target_turn_index": example.get("target_turn_index"),
        "message_count": len(prompt_messages),
        "role_sequence": role_sequence,
        "last_non_system_role": last_non_system_role,
        "target_preview": _truncate_text(target_text),
        "messages_tail": tail_preview,
    }
    if error is not None:
        payload["error_type"] = error.__class__.__name__
        payload["error"] = str(error)
    return payload


def tokenize_with_assistant_only_loss(
    dataset: Dataset,
    tokenizer,
    max_seq_len: int,
    *,
    split_name: str = "unknown",
    fail_on_template_error: bool = False,
) -> Dataset:
    def _normalize_chat_message(item: Any) -> dict[str, Any] | None:
        if not isinstance(item, dict):
            return None
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            return None
        message: dict[str, Any] = {"role": role, "content": content}
        tool_calls = item.get("tool_calls")
        if isinstance(tool_calls, list):
            message["tool_calls"] = tool_calls
        tool_name = item.get("tool_name")
        if isinstance(tool_name, str) and tool_name.strip():
            message["tool_name"] = tool_name.strip()
            message["name"] = tool_name.strip()
        name = item.get("name")
        if isinstance(name, str) and name.strip():
            message["name"] = name.strip()
        return message

    def _tokenize(example: dict[str, Any]) -> dict[str, Any]:
        messages = example.get("messages")
        target_text = example.get("target_text")
        target_message = example.get("target_message")
        target_message_normalized = _normalize_chat_message(target_message)
        if not isinstance(messages, list) or (
            not isinstance(target_text, str) and target_message_normalized is None
        ):
            payload = _build_template_debug_payload(
                split_name=split_name,
                example=example,
                prompt_messages=[],
                target_text=target_text if isinstance(target_text, str) else "",
                reason="invalid_row_shape",
            )
            LOGGER.warning("Dropping SFT row: %s", json.dumps(payload, ensure_ascii=True))
            return {
                "__drop__": 1,
                "__drop_reason": "invalid_row_shape",
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
            }

        prompt_messages = []
        for item in messages:
            normalized = _normalize_chat_message(item)
            if normalized is None:
                continue
            prompt_messages.append(normalized)

        if not prompt_messages:
            payload = _build_template_debug_payload(
                split_name=split_name,
                example=example,
                prompt_messages=prompt_messages,
                target_text=target_text,
                reason="empty_prompt_messages",
            )
            LOGGER.warning("Dropping SFT row: %s", json.dumps(payload, ensure_ascii=True))
            return {
                "__drop__": 1,
                "__drop_reason": "empty_prompt_messages",
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
            }

        has_user_query = any(
            msg.get("role") == "user" and isinstance(msg.get("content"), str) and msg.get("content").strip()
            for msg in prompt_messages
        )
        if not has_user_query:
            payload = _build_template_debug_payload(
                split_name=split_name,
                example=example,
                prompt_messages=prompt_messages,
                target_text=target_text,
                reason="missing_user_query",
            )
            LOGGER.warning("Dropping SFT row: %s", json.dumps(payload, ensure_ascii=True))
            return {
                "__drop__": 1,
                "__drop_reason": "missing_user_query",
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
            }

        if target_message_normalized is not None:
            assistant_target_message = target_message_normalized
            effective_target_text = str(assistant_target_message.get("content") or "")
        else:
            assistant_target_message = {"role": "assistant", "content": target_text}
            effective_target_text = target_text

        full_messages = prompt_messages + [assistant_target_message]

        try:
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as exc:
            payload = _build_template_debug_payload(
                split_name=split_name,
                example=example,
                prompt_messages=prompt_messages,
                target_text=effective_target_text,
                reason="chat_template_error",
                error=exc,
            )
            LOGGER.warning("Dropping SFT row: %s", json.dumps(payload, ensure_ascii=True))
            if fail_on_template_error:
                raise
            return {
                "__drop__": 1,
                "__drop_reason": "chat_template_error",
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
            }

        if callable(tokenizer):
            prompt_encoded = tokenizer(prompt_text, add_special_tokens=False)
            full_encoded = tokenizer(full_text, add_special_tokens=False)
            prompt_ids = prompt_encoded["input_ids"]
            full_ids = full_encoded["input_ids"]
        else:
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)

        if not isinstance(prompt_ids, list) or not isinstance(full_ids, list):
            raise ValueError("Tokenizer returned non-list input_ids")

        prompt_len = min(len(prompt_ids), len(full_ids))

        if len(full_ids) > max_seq_len:
            overflow = len(full_ids) - max_seq_len
            full_ids = full_ids[overflow:]
            prompt_len = max(0, prompt_len - overflow)

        attention_mask = [1] * len(full_ids)
        labels = full_ids.copy()
        for idx in range(prompt_len):
            labels[idx] = -100

        return {
            "__drop__": 0,
            "__drop_reason": "",
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    tokenized = dataset.map(_tokenize)

    reasons = tokenized["__drop_reason"] if "__drop_reason" in tokenized.column_names else []
    dropped_by_reason: dict[str, int] = {}
    for reason in reasons:
        if not reason:
            continue
        dropped_by_reason[reason] = dropped_by_reason.get(reason, 0) + 1

    filtered = tokenized.filter(lambda row: row["__drop__"] == 0)
    dropped_count = len(tokenized) - len(filtered)
    if dropped_count > 0:
        LOGGER.warning(
            "Dropped %s/%s rows during assistant-only tokenization for split=%s reasons=%s",
            dropped_count,
            len(tokenized),
            split_name,
            json.dumps(dropped_by_reason, ensure_ascii=True, sort_keys=True),
        )

    remove_columns = [
        column
        for column in filtered.column_names
        if column not in {"input_ids", "attention_mask", "labels"}
    ]
    return filtered.remove_columns(remove_columns)
