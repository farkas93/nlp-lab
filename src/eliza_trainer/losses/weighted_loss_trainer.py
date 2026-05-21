"""Custom trainer for weighted loss computation in SFT.

This module provides a custom Trainer that supports per-token loss weighting,
which is used when loss_mode="weighted" to apply fractional weights to prompt
tokens during supervised fine-tuning.

References:
    - Huerta-Enochian & Ko (2024) "Instruction Fine-Tuning: Does Prompt Loss Matter?"
    - WIT (2025) "On the Effect of Instruction Tuning Loss on Generalization"
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from transformers import Trainer


class WeightedLossTrainer(Trainer):
    """Trainer that supports weighted loss computation for prompt tokens.
    
    When loss_weights is present in the batch, applies per-token weighting
    to the cross-entropy loss. This enables fractional weighting where:
    - Response tokens get weight 1.0 (full contribution to loss)
    - Prompt tokens get a configurable weight (e.g., 0.25)
    
    Research has shown that a low-to-moderate weight (0.1-0.5) on prompt tokens
    can improve generalization compared to both complete masking (assistant_only)
    and full inclusion (full_conversation).
    
    If loss_weights is not present in the batch, falls back to standard loss
    computation (compatible with assistant_only and full_conversation modes).
    """
    
    def compute_loss(
        self, 
        model, 
        inputs: dict[str, Any], 
        return_outputs: bool = False,
        **kwargs,
    ):
        """Compute loss with optional per-token weighting.
        
        Args:
            model: The model being trained
            inputs: Batch dict containing input_ids, attention_mask, labels,
                   and optionally loss_weights
            return_outputs: Whether to return model outputs along with loss
            **kwargs: Additional arguments (passed to parent)
        
        Returns:
            Loss tensor, or (loss, outputs) tuple if return_outputs=True
        """
        # Extract loss_weights if present, removing from inputs
        loss_weights = inputs.pop("loss_weights", None)
        
        # Get model outputs (standard forward pass)
        outputs = model(**inputs)
        
        if loss_weights is None:
            # Standard loss computation (assistant_only or full_conversation modes)
            # The model's built-in loss already handles -100 masking
            loss = outputs.loss
        else:
            # Weighted loss computation for weighted mode
            loss = self._compute_weighted_loss(outputs, inputs, loss_weights)
        
        return (loss, outputs) if return_outputs else loss
    
    def _compute_weighted_loss(
        self,
        outputs,
        inputs: dict[str, Any],
        loss_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted cross-entropy loss.
        
        Applies per-token weighting to the cross-entropy loss, allowing
        differential contribution from prompt vs response tokens.
        
        Args:
            outputs: Model outputs with logits
            inputs: Input batch with labels
            loss_weights: Per-token weights tensor [batch_size, seq_len]
        
        Returns:
            Weighted average loss
        """
        logits = outputs.logits
        labels = inputs["labels"]
        
        # Shift for causal LM: predict next token
        # logits[i] predicts labels[i+1]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = loss_weights[..., 1:].contiguous()
        
        # Flatten tensors for loss computation
        vocab_size = shift_logits.size(-1)
        flat_logits = shift_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)
        flat_weights = shift_weights.view(-1)
        
        # Compute per-token loss with reduction='none'
        # Note: ignore_index=-100 means those positions get loss=0
        per_token_loss = F.cross_entropy(
            flat_logits, 
            flat_labels, 
            reduction='none',
            ignore_index=-100
        )
        
        # Apply weights to per-token loss
        weighted_loss = per_token_loss * flat_weights
        
        # Compute normalization factor: sum of weights for valid (non-ignored) tokens
        # This ensures the loss scale is independent of how many tokens are weighted
        valid_mask = (flat_labels != -100).float()
        weight_sum = (flat_weights * valid_mask).sum()
        
        # Avoid division by zero (shouldn't happen with valid data)
        if weight_sum > 0:
            loss = weighted_loss.sum() / weight_sum
        else:
            # Fallback: sum without normalization
            loss = weighted_loss.sum()
        
        return loss


__all__ = ["WeightedLossTrainer"]
