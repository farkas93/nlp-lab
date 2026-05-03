from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset

from src.sft_dataset_loader import (
    load_sft_manifest_dataset,
    tokenize_with_assistant_only_loss,
)
from src.training.loss_masking import AssistantOnlyDataCollator


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


if __name__ == "__main__":
    unittest.main()
