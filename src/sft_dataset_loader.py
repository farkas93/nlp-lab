from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import fsspec
from datasets import Dataset, load_dataset


@dataclass
class SFTManifestLoadResult:
    train_dataset: Dataset
    eval_dataset: Dataset
    manifest: dict[str, Any]
    manifest_uri: str
    manifest_sha256: str


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

    dataset_dict = load_dataset(
        "parquet",
        data_files={"train": train_uris, "eval": eval_uris},
        storage_options=_s3_storage_options(),
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


def tokenize_with_assistant_only_loss(dataset: Dataset, tokenizer, max_seq_len: int) -> Dataset:
    def _tokenize(example: dict[str, Any]) -> dict[str, Any]:
        messages = example.get("messages")
        target_text = example.get("target_text")
        if not isinstance(messages, list) or not isinstance(target_text, str):
            raise ValueError("Row must include 'messages' (list) and 'target_text' (str)")

        prompt_messages = []
        for item in messages:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                continue
            prompt_messages.append({"role": role, "content": content})

        if not prompt_messages:
            raise ValueError("No prompt messages available for tokenization")

        full_messages = prompt_messages + [{"role": "assistant", "content": target_text}]

        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        full_ids = tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
        )

        if len(full_ids) > max_seq_len:
            full_ids = full_ids[-max_seq_len:]
        attention_mask = [1] * len(full_ids)

        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = full_ids.copy()
        for idx in range(prompt_len):
            labels[idx] = -100

        return {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return dataset.map(_tokenize, remove_columns=dataset.column_names)
