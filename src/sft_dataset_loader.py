from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import io
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import fsspec
from datasets import Dataset, DownloadMode, load_dataset
import pyarrow.parquet as pq
from src.eliza_trainer.sft.chat_view import normalize_message, template_tools, require_target_prefix


# =============================================================================
# Tool Call Normalizer System
# =============================================================================
# Different tokenizers expect different tool call formats:
# - Qwen: flat structure, arguments as dict (iterates with .items())
# - OpenAI/GPT: nested function wrapper, arguments as JSON string
# - Gemma: similar to Qwen, flat structure with dict arguments
# - Generic: safe fallback, minimal changes, dict arguments
# =============================================================================


class ToolCallNormalizer(ABC):
    """Abstract base for tool call normalization strategies.
    
    Different tokenizer families have different requirements for how tool calls
    are structured in chat messages. This class hierarchy allows us to adapt
    the normalization based on the target tokenizer.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging"""
        pass
    
    @property
    @abstractmethod
    def should_stringify_arguments(self) -> bool:
        """Whether to convert arguments dict → JSON string.
        
        Returns:
            True for OpenAI-style tokenizers (expect JSON string)
            False for Qwen-style tokenizers (expect dict for .items() iteration)
        """
        pass
    
    @abstractmethod
    def transform_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """Transform a single tool call to the target format.
        
        Args:
            tool_call: Original tool call dict with 'name' and 'arguments'
            
        Returns:
            Transformed tool call dict in tokenizer-specific format
        """
        pass
    
    def normalize_tool_calls(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize a list of tool calls.
        
        Args:
            tool_calls: List of tool call dicts
            
        Returns:
            List of normalized tool call dicts
        """
        return [self.transform_tool_call(tc) for tc in tool_calls]


class QwenNormalizer(ToolCallNormalizer):
    """Normalizer for Qwen family tokenizers (Qwen, Qwen2, Qwen3, Qwen3.5).
    
    Qwen's chat template expects:
    - Flat structure: tool_calls[].name (not nested in 'function')
    - Arguments as dict (template iterates with .items())
    - No required 'id' or 'type' fields
    
    Template excerpt:
        {%- for args_name, args_value in tool_call.arguments|items %}
    """
    
    @property
    def name(self) -> str:
        return "QwenNormalizer"
    
    @property
    def should_stringify_arguments(self) -> bool:
        return False  # Keep as dict for .items() iteration
    
    def transform_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        # Qwen expects flat structure - keep as-is
        # Handle both flat and nested formats gracefully
        if "function" in tool_call:
            # Convert nested to flat
            func = tool_call["function"]
            arguments = func.get("arguments", {})
        else:
            arguments = tool_call.get("arguments", {})
        
        # Parse JSON string arguments (from Parquet storage) back to dict
        # Qwen's chat template expects dict to iterate with .items()
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = {}
        
        return {
            "name": tool_call.get("name") if "function" not in tool_call else tool_call["function"].get("name"),
            "arguments": arguments,
        }


class GemmaNormalizer(ToolCallNormalizer):
    """Normalizer for Google Gemma family tokenizers.
    
    Gemma's tool call handling is similar to Qwen:
    - Flat structure with 'name' and 'arguments'
    - Arguments as dict
    - No special wrappers required
    """
    
    @property
    def name(self) -> str:
        return "GemmaNormalizer"
    
    @property
    def should_stringify_arguments(self) -> bool:
        return False  # Keep as dict
    
    def transform_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        # Similar to Qwen - flat structure
        if "function" in tool_call:
            func = tool_call["function"]
            arguments = func.get("arguments", {})
        else:
            arguments = tool_call.get("arguments", {})
        
        # Parse JSON string arguments (from Parquet storage) back to dict
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = {}
        
        return {
            "name": tool_call.get("name") if "function" not in tool_call else tool_call["function"].get("name"),
            "arguments": arguments,
        }


class OpenAINormalizer(ToolCallNormalizer):
    """Normalizer for OpenAI/GPT family tokenizers.
    
    OpenAI's format expects:
    - Nested structure: tool_calls[].function.name
    - Arguments as JSON string (not dict)
    - Required 'id' and 'type' fields
    
    Example output:
        {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "turn_on_lights",
                "arguments": '{"room": "living"}'  # JSON string!
            }
        }
    """
    
    @property
    def name(self) -> str:
        return "OpenAINormalizer"
    
    @property
    def should_stringify_arguments(self) -> bool:
        return True  # Convert to JSON string
    
    def _generate_call_id(self, tool_call: dict[str, Any]) -> str:
        """Generate a deterministic call ID based on tool call content."""
        # Use existing ID if present
        if tool_call.get("id"):
            return str(tool_call["id"])
        # Generate based on content hash
        content = json.dumps(tool_call, sort_keys=True, ensure_ascii=True)
        return f"call_{hashlib.md5(content.encode()).hexdigest()[:12]}"
    
    def transform_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        # Handle already-nested format
        if "function" in tool_call:
            func = tool_call["function"]
            name = func.get("name")
            arguments = func.get("arguments", {})
        else:
            name = tool_call.get("name")
            arguments = tool_call.get("arguments", {})
        
        # Convert arguments to JSON string if not already
        if isinstance(arguments, (dict, list)):
            arguments_str = json.dumps(arguments, ensure_ascii=True, sort_keys=True)
        else:
            arguments_str = str(arguments) if arguments else "{}"
        
        return {
            "id": self._generate_call_id(tool_call),
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments_str,
            }
        }


class GenericNormalizer(ToolCallNormalizer):
    """Safe fallback normalizer for unknown tokenizers.
    
    Uses minimal transformation:
    - Keeps flat structure (most common)
    - Keeps arguments as dict (safer than string)
    - No required wrapper fields
    
    This is a safe default that works with most tokenizers that support
    tool calls, though it may not be optimal for all.
    """
    
    @property
    def name(self) -> str:
        return "GenericNormalizer"
    
    @property
    def should_stringify_arguments(self) -> bool:
        return False  # Safe default: keep as dict
    
    def transform_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        # Flatten if nested, otherwise keep as-is
        if "function" in tool_call:
            func = tool_call["function"]
            return {
                "name": func.get("name"),
                "arguments": func.get("arguments", {}),
            }
        return {
            "name": tool_call.get("name"),
            "arguments": tool_call.get("arguments", {}),
        }


# =============================================================================
# Tokenizer Detection
# =============================================================================


def detect_tokenizer_type(tokenizer: Any, explicit_type: str | None = None) -> str:
    """Detect tokenizer type from model name or use explicit override.
    
    Args:
        tokenizer: HuggingFace tokenizer instance
        explicit_type: Optional explicit type override from config
        
    Returns:
        Detected tokenizer type string: 'qwen', 'gemma', 'openai', or 'generic'
    """
    if explicit_type:
        normalized = explicit_type.strip().lower()
        LOGGER.info("Using explicit tokenizer_type from config: %s", normalized)
        return normalized
    
    # Get model name from tokenizer
    model_name = getattr(tokenizer, 'name_or_path', '') or ''
    model_name_lower = model_name.lower()
    
    # Pattern matching for known model families
    if 'qwen' in model_name_lower:
        detected = 'qwen'
    elif 'gemma' in model_name_lower:
        detected = 'gemma'
    elif any(x in model_name_lower for x in ['gpt', 'openai', 'davinci', 'turbo']):
        detected = 'openai'
    elif 'claude' in model_name_lower or 'anthropic' in model_name_lower:
        detected = 'openai'  # Claude uses similar format to OpenAI
    else:
        detected = 'generic'
    
    LOGGER.info(
        "Auto-detected tokenizer_type=%s from model_name=%s",
        detected, model_name
    )
    return detected


def get_normalizer(tokenizer_type: str) -> ToolCallNormalizer:
    """Factory function to get appropriate normalizer for tokenizer type.
    
    Args:
        tokenizer_type: Type string from detection or config
        
    Returns:
        Appropriate ToolCallNormalizer instance
    """
    normalizers: dict[str, ToolCallNormalizer] = {
        'qwen': QwenNormalizer(),
        'gemma': GemmaNormalizer(),
        'openai': OpenAINormalizer(),
        'gpt': OpenAINormalizer(),
        'generic': GenericNormalizer(),
    }
    
    normalizer = normalizers.get(tokenizer_type.lower(), GenericNormalizer())
    LOGGER.debug("Using %s for tokenizer_type=%s", normalizer.name, tokenizer_type)
    return normalizer


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class SFTManifestLoadResult:
    train_dataset: Dataset
    eval_dataset: Dataset
    manifest: dict[str, Any]
    manifest_uri: str
    manifest_sha256: str


@dataclass
class TokenizationStats:
    """Statistics from tokenization with configurable loss mode.
    
    Renamed from AssistantOnlyTokenizationStats to reflect support for
    multiple loss modes: assistant_only, full_conversation, weighted.
    """
    rows_total: int
    rows_valid_before_filter: int
    rows_fit_fully: int
    rows_truncated: int
    rows_dropped: int
    dropped_by_reason: dict[str, int]
    loss_mode: str = "full_conversation"
    prompt_loss_weight: float | None = None  # Only for weighted mode

    @property
    def fit_pct(self) -> float:
        if self.rows_valid_before_filter <= 0:
            return 0.0
        return (self.rows_fit_fully / self.rows_valid_before_filter) * 100.0

    @property
    def truncated_pct(self) -> float:
        if self.rows_valid_before_filter <= 0:
            return 0.0
        return (self.rows_truncated / self.rows_valid_before_filter) * 100.0


# Backward compatibility alias
AssistantOnlyTokenizationStats = TokenizationStats


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
    options = _s3_storage_options() if urlparse(manifest_uri).scheme in {"s3", "s3a"} else {}
    with fsspec.open(manifest_uri, "rb", **options) as handle:
        payload = handle.read()
    manifest = json.loads(payload)
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be a JSON object")
    digest = hashlib.sha256(payload).hexdigest()
    return manifest, digest


def _load_verified_hass_split(manifest_uri, manifest, split_name, storage_options):
    """Decode the same bounded bytes that passed the pinned split digest."""
    entries = [item for item in manifest["splits"] if item.get("split") == split_name]
    if len(entries) != 1 or split_name not in {"train", "eval"}:
        raise ValueError("Expected one HASS Parquet object per train/eval split")
    entry = entries[0]
    prefix = f"dataset_id={manifest['dataset_id']}/dataset_version={manifest['dataset_version']}"
    if entry.get("key") != f"{prefix}/split={split_name}/part-00000.parquet" or not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", "")):
        raise ValueError("Invalid HASS split identity or checksum")
    parsed = urlparse(manifest_uri)
    uri = _resolve_split_file_uri(manifest_uri, entry["key"])
    if not parsed.scheme:  # local mirror of the exported manifest and split directories
        uri = str(Path(manifest_uri).parent / f"split={split_name}/part-00000.parquet")
    options = storage_options if parsed.scheme in {"s3", "s3a"} else {}
    limit = 32 * 1024 * 1024
    with fsspec.open(uri, "rb", **options) as handle:
        payload = handle.read(limit + 1)
    if len(payload) > limit or hashlib.sha256(payload).hexdigest() != entry["sha256"]:
        raise ValueError(f"HASS {split_name} split byte checksum mismatch or size limit exceeded")
    table = pq.read_table(io.BytesIO(payload))
    if not 1 <= table.num_rows <= 49 or table.num_rows != entry.get("rows"):
        raise ValueError(f"HASS {split_name} split row count mismatch")
    if any(mode != "assistant_decision" for mode in table["target_mode"].to_pylist()):
        raise ValueError("HASS split contains a row outside assistant_decision mode")
    return Dataset(table)


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


def _load_parquet_split(
    *,
    split_uris: list[str],
    storage_options: dict[str, Any],
    download_mode: DownloadMode,
) -> Dataset:
    return load_dataset(
        "parquet",
        data_files=split_uris,
        split="train",
        storage_options=storage_options,
        download_mode=download_mode,
    )


def load_sft_manifest_dataset(
    *,
    manifest_uri: str,
    train_split: str,
    eval_split: str,
    max_train_samples: int | None,
    max_eval_samples: int | None,
    cache_mode: str = "reuse",
    expected_manifest_sha256: str | None = None,
) -> SFTManifestLoadResult:
    manifest, manifest_sha256 = _load_manifest(manifest_uri)
    if expected_manifest_sha256 is not None and manifest_sha256 != expected_manifest_sha256:
        raise ValueError("Dataset manifest checksum mismatch")
    hass = manifest.get("kind") == "hass-experimental-sft-v1"
    if hass:
        if expected_manifest_sha256 is None or (train_split, eval_split) != ("train", "eval"):
            raise ValueError("HASS smoke requires an exact manifest pin and train/eval splits")
        if max_train_samples is not None or max_eval_samples is not None:
            raise ValueError("HASS smoke must preflight every frozen decision")

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

    storage_options = _s3_storage_options()
    train_dataset = _load_verified_hass_split(manifest_uri, manifest, train_split, storage_options) if hass else _load_parquet_split(
        split_uris=train_uris,
        storage_options=storage_options,
        download_mode=download_mode,
    )
    eval_dataset = _load_verified_hass_split(manifest_uri, manifest, eval_split, storage_options) if hass else _load_parquet_split(
        split_uris=eval_uris,
        storage_options=storage_options,
        download_mode=download_mode,
    )

    # Location 1: Log raw dataset info after Parquet load
    LOGGER.debug(
        "Loaded raw datasets from manifest: train_rows=%s eval_rows=%s train_files=%s eval_files=%s",
        len(train_dataset),
        len(eval_dataset),
        len(train_uris),
        len(eval_uris),
    )

    # Log first row structure to verify schema
    if len(train_dataset) > 0:
        first_row = train_dataset[0]
        LOGGER.debug(
            "First train row keys: %s, has_messages=%s, has_target_text=%s, has_target_message=%s",
            list(first_row.keys()),
            'messages' in first_row,
            'target_text' in first_row,
            'target_message' in first_row,
        )
        if 'messages' in first_row:
            messages = first_row['messages']
            if isinstance(messages, list) and len(messages) > 0:
                first_msg = messages[0]
                if isinstance(first_msg, dict):
                    LOGGER.debug(
                        "First message structure: keys=%s types=%s",
                        list(first_msg.keys()),
                        {k: type(v).__name__ for k, v in first_msg.items()},
                    )
                else:
                    LOGGER.debug("First message is not a dict: type=%s", type(first_msg).__name__)
    else:
        LOGGER.warning("Train dataset is EMPTY after Parquet load!")

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


def tokenize_with_loss_mode(
    dataset: Dataset,
    tokenizer,
    max_seq_len: int,
    loss_mode: str = "full_conversation",
    prompt_loss_weight: float = 0.0,
    *,
    split_name: str = "unknown",
    fail_on_template_error: bool = False,
    return_stats: bool = False,
    tokenizer_type: str | None = None,
    enable_thinking: bool = True,
    measure_only: bool = False,
) -> Dataset | tuple[Dataset, TokenizationStats]:
    """Tokenize dataset with configurable loss masking.
    
    This function supports three loss modes for SFT training:
    - assistant_only: Traditional SFT, loss only on response tokens (prompt masked)
    - full_conversation: Loss on all tokens (no masking except padding)
    - weighted: Fractional loss weight on prompt tokens, full weight on response
    
    Args:
        dataset: Input dataset with 'messages' and 'target_text'/'target_message' columns
        tokenizer: HuggingFace tokenizer instance
        max_seq_len: Maximum sequence length for tokenization
        loss_mode: Loss computation strategy. Options:
            - "assistant_only": Mask prompt tokens (traditional SFT)
            - "full_conversation": No masking, loss on all tokens
            - "weighted": Apply prompt_loss_weight to prompt, 1.0 to response
        prompt_loss_weight: Weight for prompt tokens when loss_mode="weighted".
            Range 0.0-1.0. Research suggests 0.1-0.5 is often optimal. Ignored for
            other loss modes.
        split_name: Name of the split (for logging)
        fail_on_template_error: Whether to raise on chat template errors
        return_stats: Whether to return tokenization statistics
        tokenizer_type: Explicit tokenizer type override (auto-detected if None).
                       Supported: 'qwen', 'gemma', 'openai', 'generic'
        enable_thinking: Whether to enable thinking/reasoning in the model output.
                        Set to False for Qwen3/3.5 models to disable <think> tags
                        in Home Assistant and other tool-calling scenarios.
                        When False, the model is trained to skip reasoning.
        measure_only: For assistant-decision rows, return token counts and IDs
                      without truncation, admission checks, or retaining token IDs.
    
    Returns:
        Tokenized dataset, optionally with tokenization statistics
        
    References:
        - Shi et al. (2024) "Instruction Tuning With Loss Over Instructions"
        - Huerta-Enochian & Ko (2024) "Instruction Fine-Tuning: Does Prompt Loss Matter?"
    """
    # Validate loss_mode
    valid_loss_modes = {"assistant_only", "full_conversation", "weighted"}
    if loss_mode not in valid_loss_modes:
        raise ValueError(
            f"Invalid loss_mode: {loss_mode}. "
            f"Must be one of: {', '.join(sorted(valid_loss_modes))}"
        )
    
    # Validate prompt_loss_weight for weighted mode
    if loss_mode == "weighted":
        if not 0.0 <= prompt_loss_weight <= 1.0:
            raise ValueError(
                f"prompt_loss_weight must be between 0.0 and 1.0, got {prompt_loss_weight}"
            )
    
    # Detect tokenizer type and get appropriate normalizer
    detected_type = detect_tokenizer_type(tokenizer, tokenizer_type)
    normalizer = get_normalizer(detected_type)
    
    LOGGER.debug(
        "Tokenization using %s (detected_type=%s, explicit=%s, loss_mode=%s, prompt_weight=%s, enable_thinking=%s)",
        normalizer.name,
        detected_type,
        tokenizer_type,
        loss_mode,
        prompt_loss_weight if loss_mode == "weighted" else "N/A",
        enable_thinking,
    )
    
    # Determine if we should pass enable_thinking to apply_chat_template
    # Only Qwen3/3.5 tokenizers support this parameter
    use_enable_thinking = detected_type == 'qwen'
    if use_enable_thinking:
        LOGGER.info(
            "Qwen chat-template enable_thinking=%s", enable_thinking
        )

    def _normalize_tool_calls(value: Any) -> list[Any] | None:
        """Normalize tool calls using the detected normalizer strategy."""
        if not isinstance(value, list):
            return None
        
        # First, do basic structural normalization
        basic_normalized: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue  # Skip non-dict items
            basic_normalized.append(item)
        
        if not basic_normalized:
            return None
        
        # Reject corrupt stored arguments rather than silently replacing them.
        for call in basic_normalized:
            arguments = call.get("function", call).get("arguments")
            if isinstance(arguments, str) and not isinstance(json.loads(arguments), dict):
                raise ValueError("Tool arguments must decode to an object")
        normalized = normalizer.normalize_tool_calls(basic_normalized)
        for source, target in zip(basic_normalized, normalized):
            for field in ("id", "type"):
                if source.get(field) is not None:
                    target[field] = source[field]
        return normalized

    def _normalize_chat_message(item: Any) -> dict[str, Any] | None:
        return normalize_message(item, _normalize_tool_calls)

    def _tokenize(example: dict[str, Any]) -> dict[str, Any]:
        decision_row = example.get("target_mode") == "assistant_decision"
        if measure_only and not decision_row:
            raise ValueError("Context measurement requires assistant-decision rows")
        if decision_row and not (loss_mode == "assistant_only" or (loss_mode == "weighted" and prompt_loss_weight == 0)):
            raise ValueError("Assistant-decision rows require current-target-only loss")
        messages = example.get("messages")
        target_text = example.get("target_text")
        target_message = example.get("target_message")
        target_message_normalized = _normalize_chat_message(target_message)
        if decision_row and (target_message_normalized is None or target_message_normalized["role"] != "assistant"):
            raise ValueError("Assistant-decision row has no valid native assistant target")
        if not isinstance(messages, list) or (
            not isinstance(target_text, str) and target_message_normalized is None
        ):
            if decision_row:
                raise ValueError("Assistant-decision row has invalid messages")
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
                "__is_truncated__": 0,
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
            }

        prompt_messages = []
        for item in messages:
            normalized = _normalize_chat_message(item)
            if normalized is None:
                if decision_row:
                    raise ValueError("Assistant-decision row contains an invalid visible message")
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
                "__is_truncated__": 0,
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
            }

        has_user_query = any(
            msg.get("role") == "user" and isinstance(msg.get("content"), str) and msg.get("content").strip()
            for msg in prompt_messages
        )
        if not has_user_query:
            if decision_row:
                raise ValueError("Assistant-decision row has no visible user query")
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
                "__is_truncated__": 0,
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
            # Build template kwargs - Qwen3/3.5 supports enable_thinking parameter
            template_kwargs: dict[str, Any] = {
                "tokenize": False,
            }
            if use_enable_thinking:
                template_kwargs["enable_thinking"] = enable_thinking
            tools = template_tools(example)
            if tools is not None:
                template_kwargs["tools"] = tools
            
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                add_generation_prompt=True,
                **template_kwargs,
            )
            full_text = tokenizer.apply_chat_template(
                full_messages,
                add_generation_prompt=False,
                **template_kwargs,
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
            if fail_on_template_error or decision_row:
                raise
            return {
                "__drop__": 1,
                "__drop_reason": "chat_template_error",
                "__is_truncated__": 0,
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
        if decision_row:
            require_target_prefix(prompt_ids, full_ids)
            if measure_only:
                return {"example_id": example.get("example_id"), "episode_id": example.get("episode_id"),
                        "prompt_tokens": len(prompt_ids), "target_tokens": len(full_ids) - len(prompt_ids),
                        "total_tokens": len(full_ids)}
            if len(full_ids) > max_seq_len:
                raise ValueError(f"Assistant decision {example.get('example_id')} in {split_name} needs "
                                 f"{len(full_ids)} tokens ({len(prompt_ids)} prompt), exceeds max_seq_len={max_seq_len}; "
                                 "measure the frozen export and preserve its full visible context")

        prompt_len = min(len(prompt_ids), len(full_ids))
        is_truncated = 0

        if len(full_ids) > max_seq_len:
            overflow = len(full_ids) - max_seq_len
            full_ids = full_ids[overflow:]
            prompt_len = max(0, prompt_len - overflow)
            is_truncated = 1

        attention_mask = [1] * len(full_ids)
        labels = full_ids.copy()
        
        # Apply loss masking based on mode
        loss_weights: list[float] | None = None
        
        if loss_mode == "assistant_only":
            # Traditional SFT: mask prompt tokens completely
            for idx in range(prompt_len):
                labels[idx] = -100
        elif loss_mode == "full_conversation":
            # No masking: model learns from all tokens
            # Research shows this can reduce overfitting and improve performance
            pass
        elif loss_mode == "weighted":
            # Weighted loss: prompt tokens get fractional weight, response tokens get 1.0
            loss_weights = []
            for idx in range(len(full_ids)):
                if idx < prompt_len:
                    loss_weights.append(prompt_loss_weight)
                else:
                    loss_weights.append(1.0)
        else:
            # Should never reach here due to validation, but defensive
            raise ValueError(f"Unhandled loss_mode: {loss_mode}")

        result: dict[str, Any] = {
            "__drop__": 0,
            "__drop_reason": "",
            "__is_truncated__": is_truncated,
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        
        # Only include loss_weights if using weighted mode
        if loss_weights is not None:
            result["loss_weights"] = loss_weights
        
        return result

    if measure_only:
        return dataset.map(_tokenize, remove_columns=dataset.column_names, load_from_cache_file=False)
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
            "Dropped %s/%s rows during tokenization (loss_mode=%s) for split=%s reasons=%s",
            dropped_count,
            len(tokenized),
            loss_mode,
            split_name,
            json.dumps(dropped_by_reason, ensure_ascii=True, sort_keys=True),
        )

    # Location 2: Enhanced tokenization stats logging
    rows_truncated = 0
    if "__is_truncated__" in filtered.column_names:
        rows_truncated = int(sum(int(value) for value in filtered["__is_truncated__"]))
    
    LOGGER.debug(
        "Tokenization stats for split=%s loss_mode=%s: raw_rows=%s valid_rows=%s dropped=%s fit_fully=%s truncated=%s",
        split_name,
        loss_mode,
        len(tokenized),
        len(filtered),
        dropped_count,
        len(filtered) - rows_truncated if len(filtered) > 0 else 0,
        rows_truncated,
    )

    # If everything was dropped, log detailed diagnostics
    if len(filtered) == 0 and len(tokenized) > 0:
        LOGGER.error(
            "CRITICAL: All %s rows dropped during tokenization for split=%s! Reasons: %s",
            len(tokenized),
            split_name,
            json.dumps(dropped_by_reason, ensure_ascii=True, sort_keys=True, indent=2),
        )
        # Log a sample of dropped rows for debugging
        if "__drop_reason__" in tokenized.column_names:
            sample_size = min(3, len(tokenized))
            LOGGER.error("Sample of dropped rows (first %s):", sample_size)
            for i in range(sample_size):
                row_data = tokenized[i]
                LOGGER.error(
                    "  Row %s: reason=%s, example_id=%s, session_id=%s",
                    i,
                    row_data.get("__drop_reason__", "unknown"),
                    row_data.get("example_id", "missing"),
                    row_data.get("session_id", "missing"),
                )

    # Columns to keep in the final dataset
    keep_columns = {"input_ids", "attention_mask", "labels"}
    if loss_mode == "weighted":
        keep_columns.add("loss_weights")
    
    remove_columns = [
        column
        for column in filtered.column_names
        if column not in keep_columns
    ]
    filtered_clean = filtered.remove_columns(remove_columns)

    if not return_stats:
        return filtered_clean

    rows_total = len(tokenized)
    rows_valid_before_filter = len(filtered)
    # rows_truncated already computed above in Location 2 logging
    rows_fit_fully = max(0, rows_valid_before_filter - rows_truncated)

    stats = TokenizationStats(
        rows_total=rows_total,
        rows_valid_before_filter=rows_valid_before_filter,
        rows_fit_fully=rows_fit_fully,
        rows_truncated=rows_truncated,
        rows_dropped=dropped_count,
        dropped_by_reason=dropped_by_reason,
        loss_mode=loss_mode,
        prompt_loss_weight=prompt_loss_weight if loss_mode == "weighted" else None,
    )
    return filtered_clean, stats


# Backward compatibility alias
def tokenize_with_assistant_only_loss(
    dataset: Dataset,
    tokenizer,
    max_seq_len: int,
    *,
    split_name: str = "unknown",
    fail_on_template_error: bool = False,
    return_stats: bool = False,
    tokenizer_type: str | None = None,
    enable_thinking: bool = True,
) -> Dataset | tuple[Dataset, TokenizationStats]:
    """Backward-compatible wrapper for tokenize_with_loss_mode.
    
    Deprecated: Use tokenize_with_loss_mode with loss_mode="assistant_only" instead.
    
    This function maintains backward compatibility for existing code that uses
    tokenize_with_assistant_only_loss. It delegates to tokenize_with_loss_mode
    with loss_mode="assistant_only".
    """
    return tokenize_with_loss_mode(
        dataset=dataset,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        loss_mode="assistant_only",
        prompt_loss_weight=0.0,
        split_name=split_name,
        fail_on_template_error=fail_on_template_error,
        return_stats=return_stats,
        tokenizer_type=tokenizer_type,
        enable_thinking=enable_thinking,
    )
