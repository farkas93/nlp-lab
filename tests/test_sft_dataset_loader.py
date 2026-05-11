from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset

from src.eliza_trainer.sft.dataset_loader import (
    load_sft_manifest_dataset,
    tokenize_with_assistant_only_loss,
)
from src.eliza_trainer.losses import AssistantOnlyDataCollator


class _DummyTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        joined = []
        for item in messages:
            joined.append(f"{item['role']}:{item['content']}")
        if add_generation_prompt:
            joined.append("assistant:")
        text = "\n".join(joined)
        tokens = [ord(ch) % 97 + 1 for ch in text]
        return tokens if tokenize else text

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) % 97 + 1 for ch in text]


class SFTDatasetLoaderTests(unittest.TestCase):
    def test_load_manifest_dataset_local_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_path = tmp_path / "train.parquet"
            eval_path = tmp_path / "eval.parquet"
            manifest_path = tmp_path / "manifest.json"

            row = {
                "messages": [{"role": "user", "content": "hi"}],
                "target_text": "hello",
            }
            table = pa.Table.from_pylist([row])
            pq.write_table(table, train_path)
            pq.write_table(table, eval_path)

            manifest = {
                "dataset_id": "general",
                "dataset_version": "v1",
                "splits": [
                    {"split": "train", "key": str(train_path)},
                    {"split": "eval", "key": str(eval_path)},
                ],
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            result = load_sft_manifest_dataset(
                manifest_uri=str(manifest_path),
                train_split="train",
                eval_split="eval",
                max_train_samples=None,
                max_eval_samples=None,
            )

            self.assertEqual(len(result.train_dataset), 1)
            self.assertEqual(len(result.eval_dataset), 1)
            self.assertEqual(result.manifest["dataset_id"], "general")

    def test_tokenize_and_collate_assistant_only(self) -> None:
        tokenizer = _DummyTokenizer()
        ds = Dataset.from_list(
            [
                {
                    "messages": [
                        {"role": "system", "content": "be concise"},
                        {"role": "user", "content": "say hi"},
                    ],
                    "target_text": "hi",
                }
            ]
        )

        tokenized = tokenize_with_assistant_only_loss(ds, tokenizer=tokenizer, max_seq_len=512)
        row = tokenized[0]
        self.assertEqual(len(row["input_ids"]), len(row["labels"]))
        self.assertIn(-100, row["labels"])

        collator = AssistantOnlyDataCollator(tokenizer=tokenizer)
        batch = collator([row])
        self.assertEqual(batch["input_ids"].shape[0], 1)
        self.assertEqual(batch["labels"].shape[1], batch["input_ids"].shape[1])

    def test_tokenize_supports_structured_target_message_with_tool_calls(self) -> None:
        tokenizer = _DummyTokenizer()
        ds = Dataset.from_list(
            [
                {
                    "messages": [
                        {"role": "user", "content": "Turn off all lights"},
                    ],
                    "target_text": '{"tool_calls":[{"name":"HassTurnOff","arguments":{"domain":"light"}}]}',
                    "target_message": {
                        "role": "assistant",
                        "content": '{"tool_calls":[{"name":"HassTurnOff","arguments":{"domain":"light"}}]}',
                        "tool_calls": [
                            {"name": "HassTurnOff", "arguments": {"domain": "light"}}
                        ],
                    },
                }
            ]
        )

        tokenized = tokenize_with_assistant_only_loss(ds, tokenizer=tokenizer, max_seq_len=512)
        self.assertEqual(len(tokenized), 1)
        row = tokenized[0]
        self.assertEqual(len(row["input_ids"]), len(row["labels"]))

    def test_load_manifest_dataset_supports_different_nested_tool_argument_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_path = tmp_path / "train.parquet"
            eval_path = tmp_path / "eval.parquet"
            manifest_path = tmp_path / "manifest.json"

            train_row = {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "tool call",
                        "tool_calls": [
                            {
                                "name": "hass_control_light",
                                "arguments": {
                                    "action": "off",
                                    "domain": "light",
                                    "light_entity_id": "light.kitchen",
                                },
                            }
                        ],
                    }
                ],
                "target_text": "done",
            }
            eval_row = {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "tool call",
                        "tool_calls": [
                            {
                                "name": "hass_trigger_automation",
                                "arguments": {
                                    "action": "trigger",
                                    "domain": "automation",
                                    "automation_entity_id": "automation.night",
                                },
                            }
                        ],
                    }
                ],
                "target_text": "done",
            }

            pq.write_table(pa.Table.from_pylist([train_row]), train_path)
            pq.write_table(pa.Table.from_pylist([eval_row]), eval_path)

            manifest = {
                "dataset_id": "hass_tools_v2",
                "dataset_version": "v1.0.0",
                "splits": [
                    {"split": "train", "key": str(train_path)},
                    {"split": "eval", "key": str(eval_path)},
                ],
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            result = load_sft_manifest_dataset(
                manifest_uri=str(manifest_path),
                train_split="train",
                eval_split="eval",
                max_train_samples=None,
                max_eval_samples=None,
            )

            self.assertEqual(len(result.train_dataset), 1)
            self.assertEqual(len(result.eval_dataset), 1)


if __name__ == "__main__":
    unittest.main()
