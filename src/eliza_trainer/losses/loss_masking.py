from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class DataCollatorWithLossMode:
    """Data collator that handles padding for all loss modes including weighted.
    
    This collator is mode-agnostic. Loss masking is applied during tokenization
    via tokenize_with_loss_mode(). The collator handles:
    - Padding input_ids, attention_mask, and labels
    - Padding loss_weights (if present, for weighted mode)
    
    Padding tokens are always masked (label_pad_token_id = -100) regardless
    of loss_mode, following standard practice.
    
    Args:
        tokenizer: Tokenizer instance with pad_token_id attribute
        label_pad_token_id: Value for masked labels (default: -100, PyTorch ignore index)
    """
    tokenizer: Any
    label_pad_token_id: int = -100

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("No features provided to data collator")

        max_length = max(len(item["input_ids"]) for item in features)
        padded_input_ids: list[list[int]] = []
        padded_attention_mask: list[list[int]] = []
        padded_labels: list[list[int]] = []
        
        # Check if we have loss_weights (weighted mode)
        has_loss_weights = "loss_weights" in features[0]
        padded_loss_weights: list[list[float]] = [] if has_loss_weights else []

        pad_id = self.tokenizer.pad_token_id
        for item in features:
            input_ids = list(item["input_ids"])
            attention_mask = list(item["attention_mask"])
            labels = list(item["labels"])
            pad_len = max_length - len(input_ids)

            padded_input_ids.append(input_ids + [pad_id] * pad_len)
            padded_attention_mask.append(attention_mask + [0] * pad_len)
            # Padding positions always masked regardless of loss_mode
            padded_labels.append(labels + [self.label_pad_token_id] * pad_len)
            
            # Handle loss_weights if present (weighted mode)
            if has_loss_weights:
                loss_weights = list(item["loss_weights"])
                # Padding tokens get weight 0.0 (will also be masked by -100 labels)
                padded_loss_weights.append(loss_weights + [0.0] * pad_len)

        result: dict[str, torch.Tensor] = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }
        
        if has_loss_weights:
            result["loss_weights"] = torch.tensor(padded_loss_weights, dtype=torch.float32)
        
        return result


# Backward compatibility alias
AssistantOnlyDataCollator = DataCollatorWithLossMode
