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
    tokenize_with_loss_mode,
    tokenize_with_assistant_only_loss,
)
from src.eliza_trainer.losses import AssistantOnlyDataCollator, DataCollatorWithLossMode


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

    def test_tokenize_returns_context_fit_stats(self) -> None:
        tokenizer = _DummyTokenizer()
        ds = Dataset.from_list(
            [
                {
                    "messages": [{"role": "user", "content": "short"}],
                    "target_text": "ok",
                },
                {
                    "messages": [{"role": "user", "content": "x" * 400}],
                    "target_text": "y" * 400,
                },
            ]
        )

        tokenized, stats = tokenize_with_assistant_only_loss(
            ds,
            tokenizer=tokenizer,
            max_seq_len=128,
            return_stats=True,
        )
        self.assertEqual(len(tokenized), 2)
        self.assertEqual(stats.rows_total, 2)
        self.assertEqual(stats.rows_valid_before_filter, 2)
        self.assertEqual(stats.rows_truncated, 1)
        self.assertEqual(stats.rows_fit_fully, 1)
        self.assertGreater(stats.fit_pct, 0.0)
        self.assertLess(stats.fit_pct, 100.0)

    def test_tokenize_return_stats_handles_dropped_rows_schema(self) -> None:
        tokenizer = _DummyTokenizer()
        ds = Dataset.from_list(
            [
                {
                    "messages": [{"role": "user", "content": "valid prompt"}],
                    "target_text": "ok",
                },
                {
                    "messages": [{"role": "assistant", "content": "no user message"}],
                    "target_text": "bad row",
                },
            ]
        )

        tokenized, stats = tokenize_with_assistant_only_loss(
            ds,
            tokenizer=tokenizer,
            max_seq_len=128,
            return_stats=True,
        )

        self.assertEqual(len(tokenized), 1)
        self.assertEqual(stats.rows_total, 2)
        self.assertEqual(stats.rows_valid_before_filter, 1)
        self.assertEqual(stats.rows_dropped, 1)

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


class LossModeTests(unittest.TestCase):
    """Tests for the three loss modes: assistant_only, full_conversation, weighted."""

    def test_tokenize_assistant_only_mode_masks_prompt(self) -> None:
        """Test that assistant_only mode masks prompt tokens with -100."""
        tokenizer = _DummyTokenizer()
        ds = Dataset.from_list([
            {
                "messages": [
                    {"role": "system", "content": "be concise"},
                    {"role": "user", "content": "say hi"},
                ],
                "target_text": "hello",
            }
        ])

        tokenized = tokenize_with_loss_mode(
            ds, 
            tokenizer=tokenizer, 
            max_seq_len=512,
            loss_mode="assistant_only"
        )
        row = tokenized[0]

        # In assistant_only mode, prompt tokens should be masked with -100
        self.assertEqual(len(row["input_ids"]), len(row["labels"]))
        self.assertIn(-100, row["labels"])
        # Response tokens should NOT be -100
        self.assertTrue(any(label != -100 for label in row["labels"]))

    def test_tokenize_full_conversation_mode_no_masking(self) -> None:
        """Test that full_conversation mode does not mask any tokens."""
        tokenizer = _DummyTokenizer()
        ds = Dataset.from_list([
            {
                "messages": [
                    {"role": "system", "content": "be concise"},
                    {"role": "user", "content": "say hi"},
                ],
                "target_text": "hello",
            }
        ])

        tokenized = tokenize_with_loss_mode(
            ds, 
            tokenizer=tokenizer, 
            max_seq_len=512,
            loss_mode="full_conversation"
        )
        row = tokenized[0]

        # In full_conversation mode, NO tokens should be masked
        self.assertEqual(len(row["input_ids"]), len(row["labels"]))
        self.assertNotIn(-100, row["labels"])
        # Labels should equal input_ids (no masking)
        self.assertEqual(row["input_ids"], row["labels"])

    def test_tokenize_weighted_mode_creates_loss_weights(self) -> None:
        """Test that weighted mode creates loss_weights tensor."""
        tokenizer = _DummyTokenizer()
        ds = Dataset.from_list([
            {
                "messages": [{"role": "user", "content": "hi"}],
                "target_text": "hello",
            }
        ])

        tokenized = tokenize_with_loss_mode(
            ds, 
            tokenizer=tokenizer, 
            max_seq_len=512,
            loss_mode="weighted",
            prompt_loss_weight=0.3
        )
        row = tokenized[0]

        # Should have loss_weights column
        self.assertIn("loss_weights", row)
        self.assertEqual(len(row["loss_weights"]), len(row["input_ids"]))
        
        # Labels should not be masked (weighted mode uses weights instead)
        self.assertNotIn(-100, row["labels"])
        
        # Should have both weight values: 0.3 for prompt, 1.0 for response
        weights = row["loss_weights"]
        self.assertTrue(any(abs(w - 0.3) < 0.01 for w in weights))
        self.assertTrue(any(abs(w - 1.0) < 0.01 for w in weights))

    def test_tokenize_weighted_mode_respects_weight_value(self) -> None:
        """Test that different prompt_loss_weight values are applied correctly."""
        tokenizer = _DummyTokenizer()
        ds = Dataset.from_list([
            {
                "messages": [{"role": "user", "content": "hi"}],
                "target_text": "hello",
            }
        ])

        for weight in [0.0, 0.25, 0.5, 1.0]:
            tokenized = tokenize_with_loss_mode(
                ds, 
                tokenizer=tokenizer, 
                max_seq_len=512,
                loss_mode="weighted",
                prompt_loss_weight=weight
            )
            row = tokenized[0]
            
            # Verify weight is applied
            self.assertTrue(any(abs(w - weight) < 0.01 for w in row["loss_weights"]))

    def test_tokenize_invalid_loss_mode_raises_error(self) -> None:
        """Test that invalid loss_mode raises ValueError."""
        tokenizer = _DummyTokenizer()
        ds = Dataset.from_list([
            {
                "messages": [{"role": "user", "content": "hi"}],
                "target_text": "hello",
            }
        ])

        with self.assertRaises(ValueError) as ctx:
            tokenize_with_loss_mode(
                ds, 
                tokenizer=tokenizer, 
                max_seq_len=512,
                loss_mode="invalid_mode"
            )
        self.assertIn("Invalid loss_mode", str(ctx.exception))

    def test_tokenize_weighted_mode_invalid_weight_raises_error(self) -> None:
        """Test that invalid prompt_loss_weight raises ValueError."""
        tokenizer = _DummyTokenizer()
        ds = Dataset.from_list([
            {
                "messages": [{"role": "user", "content": "hi"}],
                "target_text": "hello",
            }
        ])

        # Test weight > 1.0
        with self.assertRaises(ValueError) as ctx:
            tokenize_with_loss_mode(
                ds, 
                tokenizer=tokenizer, 
                max_seq_len=512,
                loss_mode="weighted",
                prompt_loss_weight=1.5
            )
        self.assertIn("prompt_loss_weight", str(ctx.exception))

        # Test weight < 0.0
        with self.assertRaises(ValueError) as ctx:
            tokenize_with_loss_mode(
                ds, 
                tokenizer=tokenizer, 
                max_seq_len=512,
                loss_mode="weighted",
                prompt_loss_weight=-0.1
            )
        self.assertIn("prompt_loss_weight", str(ctx.exception))

    def test_collator_handles_loss_weights(self) -> None:
        """Test that collator properly pads loss_weights."""
        tokenizer = _DummyTokenizer()
        
        # Create two features with different lengths to test padding
        feature1 = {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
            "labels": [1, 2, 3],
            "loss_weights": [0.25, 0.25, 1.0],
        }
        feature2 = {
            "input_ids": [4, 5, 6, 7, 8],
            "attention_mask": [1, 1, 1, 1, 1],
            "labels": [4, 5, 6, 7, 8],
            "loss_weights": [0.25, 0.25, 1.0, 1.0, 1.0],
        }
        
        collator = DataCollatorWithLossMode(tokenizer=tokenizer)
        batch = collator([feature1, feature2])
        
        # Should have loss_weights in output
        self.assertIn("loss_weights", batch)
        self.assertEqual(batch["loss_weights"].shape, batch["input_ids"].shape)
        
        # Padding should have weight 0.0
        # feature1 gets padded from length 3 to 5
        self.assertEqual(batch["loss_weights"][0, 3].item(), 0.0)
        self.assertEqual(batch["loss_weights"][0, 4].item(), 0.0)

    def test_collator_works_without_loss_weights(self) -> None:
        """Test that collator works for non-weighted modes (no loss_weights)."""
        tokenizer = _DummyTokenizer()
        
        # Features without loss_weights (assistant_only or full_conversation mode)
        feature1 = {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
            "labels": [-100, -100, 3],  # assistant_only style
        }
        feature2 = {
            "input_ids": [4, 5, 6, 7],
            "attention_mask": [1, 1, 1, 1],
            "labels": [-100, -100, 6, 7],
        }
        
        collator = DataCollatorWithLossMode(tokenizer=tokenizer)
        batch = collator([feature1, feature2])
        
        # Should NOT have loss_weights
        self.assertNotIn("loss_weights", batch)
        # Should have standard fields
        self.assertIn("input_ids", batch)
        self.assertIn("attention_mask", batch)
        self.assertIn("labels", batch)

    def test_backward_compatibility_alias(self) -> None:
        """Test that tokenize_with_assistant_only_loss still works as alias."""
        tokenizer = _DummyTokenizer()
        ds = Dataset.from_list([
            {
                "messages": [{"role": "user", "content": "hi"}],
                "target_text": "hello",
            }
        ])

        # Use the old function name
        tokenized = tokenize_with_assistant_only_loss(
            ds, 
            tokenizer=tokenizer, 
            max_seq_len=512
        )
        row = tokenized[0]

        # Should behave like assistant_only mode
        self.assertIn(-100, row["labels"])

    def test_tokenize_stats_include_loss_mode(self) -> None:
        """Test that tokenization stats include loss_mode information."""
        tokenizer = _DummyTokenizer()
        ds = Dataset.from_list([
            {
                "messages": [{"role": "user", "content": "hi"}],
                "target_text": "hello",
            }
        ])

        # Test assistant_only
        _, stats = tokenize_with_loss_mode(
            ds, tokenizer=tokenizer, max_seq_len=512,
            loss_mode="assistant_only", return_stats=True
        )
        self.assertEqual(stats.loss_mode, "assistant_only")
        self.assertIsNone(stats.prompt_loss_weight)

        # Test full_conversation
        _, stats = tokenize_with_loss_mode(
            ds, tokenizer=tokenizer, max_seq_len=512,
            loss_mode="full_conversation", return_stats=True
        )
        self.assertEqual(stats.loss_mode, "full_conversation")
        self.assertIsNone(stats.prompt_loss_weight)

        # Test weighted
        _, stats = tokenize_with_loss_mode(
            ds, tokenizer=tokenizer, max_seq_len=512,
            loss_mode="weighted", prompt_loss_weight=0.25, return_stats=True
        )
        self.assertEqual(stats.loss_mode, "weighted")
        self.assertEqual(stats.prompt_loss_weight, 0.25)


if __name__ == "__main__":
    unittest.main()
