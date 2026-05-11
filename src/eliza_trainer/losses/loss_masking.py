from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class AssistantOnlyDataCollator:
    tokenizer: Any
    label_pad_token_id: int = -100

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("No features provided to data collator")

        max_length = max(len(item["input_ids"]) for item in features)
        padded_input_ids: list[list[int]] = []
        padded_attention_mask: list[list[int]] = []
        padded_labels: list[list[int]] = []

        pad_id = self.tokenizer.pad_token_id
        for item in features:
            input_ids = list(item["input_ids"])
            attention_mask = list(item["attention_mask"])
            labels = list(item["labels"])
            pad_len = max_length - len(input_ids)

            padded_input_ids.append(input_ids + [pad_id] * pad_len)
            padded_attention_mask.append(attention_mask + [0] * pad_len)
            padded_labels.append(labels + [self.label_pad_token_id] * pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }
