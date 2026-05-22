"""
Tokenizer diagnostics for SFT pipeline analysis.

Analyzes tokenizer configuration to detect issues like pad_token == eos_token
that can cause infinite loops during inference with full_conversation loss mode.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from huggingface_hub import login

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eliza_trainer.sft.run_config import load_sft_run_config


@dataclass
class TokenizerDiagnostics:
    """Results from tokenizer configuration analysis."""
    eos_token: str | None
    eos_token_id: int | None
    bos_token: str | None
    bos_token_id: int | None
    pad_token: str | None
    pad_token_id: int | None
    unk_token: str | None
    unk_token_id: int | None
    chat_template: str | None
    pad_equals_eos: bool
    all_special_tokens: list[str]
    all_special_ids: list[int]


@dataclass
class TokenizerComparisonResult:
    """Comparison of tokenizer before and after train.py modifications."""
    model_name: str
    loss_mode: str
    before: TokenizerDiagnostics
    after: TokenizerDiagnostics
    issues: list[str]
    recommendations: list[str]


def _extract_tokenizer_diagnostics(tokenizer) -> TokenizerDiagnostics:
    """Extract diagnostic info from a tokenizer instance."""
    return TokenizerDiagnostics(
        eos_token=getattr(tokenizer, 'eos_token', None),
        eos_token_id=getattr(tokenizer, 'eos_token_id', None),
        bos_token=getattr(tokenizer, 'bos_token', None),
        bos_token_id=getattr(tokenizer, 'bos_token_id', None),
        pad_token=getattr(tokenizer, 'pad_token', None),
        pad_token_id=getattr(tokenizer, 'pad_token_id', None),
        unk_token=getattr(tokenizer, 'unk_token', None),
        unk_token_id=getattr(tokenizer, 'unk_token_id', None),
        chat_template=getattr(tokenizer, 'chat_template', None),
        pad_equals_eos=(
            getattr(tokenizer, 'pad_token_id', None) is not None and
            getattr(tokenizer, 'eos_token_id', None) is not None and
            getattr(tokenizer, 'pad_token_id', None) == getattr(tokenizer, 'eos_token_id', None)
        ),
        all_special_tokens=list(getattr(tokenizer, 'all_special_tokens', [])),
        all_special_ids=list(getattr(tokenizer, 'all_special_ids', [])),
    )


def analyze_tokenizer_config(config_path: str) -> TokenizerComparisonResult:
    """
    Analyze tokenizer configuration before and after train.py modifications.
    
    Args:
        config_path: Path to SFT YAML config file
        
    Returns:
        TokenizerComparisonResult with before/after comparison and detected issues
    """
    run_config = load_sft_run_config(config_path)
    
    # Login to HF if token available
    hf_token = os.environ.get("HF_HUB_TOKEN")
    if hf_token:
        login(token=hf_token)
    
    # Load tokenizer (BEFORE train.py modifications)
    tokenizer = AutoTokenizer.from_pretrained(
        run_config.model.model_name,
        add_eos_token=True,
        use_fast=True,
        cache_dir=run_config.hf_model_cache_dir,
    )
    before = _extract_tokenizer_diagnostics(tokenizer)
    
    # Apply train.py modifications
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    after = _extract_tokenizer_diagnostics(tokenizer)
    
    # Detect issues
    issues: list[str] = []
    recommendations: list[str] = []
    
    # Check if train.py overwrote a distinct pad_token
    if not before.pad_equals_eos and after.pad_equals_eos:
        issues.append(
            f"train.py overwrites distinct pad_token (was {before.pad_token!r} ID:{before.pad_token_id}) "
            f"with eos_token ({after.eos_token!r} ID:{after.eos_token_id})"
        )
    
    # Check for critical issue with full_conversation mode
    if after.pad_equals_eos and run_config.model.loss_mode == "full_conversation":
        issues.append(
            "CRITICAL: pad_token == eos_token with full_conversation loss mode. "
            "Padding tokens will contribute to loss, teaching model that EOS can appear mid-sequence."
        )
        recommendations.append(
            "Fix train.py: only set pad_token=eos_token if tokenizer.pad_token is None"
        )
    
    # Check for missing chat template
    if not after.chat_template:
        issues.append("Tokenizer has no chat_template defined")
        recommendations.append("Verify model supports chat template or provide custom template")
    
    return TokenizerComparisonResult(
        model_name=run_config.model.model_name,
        loss_mode=run_config.model.loss_mode,
        before=before,
        after=after,
        issues=issues,
        recommendations=recommendations,
    )


def print_tokenizer_report(result: TokenizerComparisonResult) -> None:
    """Print formatted tokenizer diagnostics report."""
    print("=" * 80)
    print("TOKENIZER DIAGNOSTICS")
    print("=" * 80)
    print()
    print(f"Model: {result.model_name}")
    print(f"Loss Mode: {result.loss_mode}")
    print()
    
    print("-" * 80)
    print("BEFORE train.py modifications:")
    print("-" * 80)
    print(f"  EOS Token: {result.before.eos_token!r} (ID: {result.before.eos_token_id})")
    print(f"  PAD Token: {result.before.pad_token!r} (ID: {result.before.pad_token_id})")
    print(f"  BOS Token: {result.before.bos_token!r} (ID: {result.before.bos_token_id})")
    print(f"  pad_token == eos_token: {result.before.pad_equals_eos}")
    print()
    
    print("-" * 80)
    print("AFTER train.py modifications:")
    print("-" * 80)
    print(f"  EOS Token: {result.after.eos_token!r} (ID: {result.after.eos_token_id})")
    print(f"  PAD Token: {result.after.pad_token!r} (ID: {result.after.pad_token_id})")
    print(f"  BOS Token: {result.after.bos_token!r} (ID: {result.after.bos_token_id})")
    print(f"  pad_token == eos_token: {result.after.pad_equals_eos}")
    print()
    
    print(f"All Special Tokens ({len(result.after.all_special_tokens)}):")
    for token, token_id in zip(result.after.all_special_tokens, result.after.all_special_ids):
        print(f"  {token!r} -> {token_id}")
    print()
    
    if result.after.chat_template:
        template_preview = result.after.chat_template[:500]
        if len(result.after.chat_template) > 500:
            template_preview += "... [truncated]"
        print(f"Chat Template:\n{template_preview}")
    else:
        print("Chat Template: NOT SET")
    print()
    
    print("-" * 80)
    print("ANALYSIS")
    print("-" * 80)
    
    if result.issues:
        print("\nIssues Found:")
        for issue in result.issues:
            print(f"  [X] {issue}")
    else:
        print("\n  [OK] No issues found")
    
    if result.recommendations:
        print("\nRecommendations:")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"  {i}. {rec}")
    
    print()
    print("=" * 80)


def get_tokenizer_diagnostics_json(result: TokenizerComparisonResult) -> str:
    """Return tokenizer diagnostics as JSON string."""
    return json.dumps({
        "model_name": result.model_name,
        "loss_mode": result.loss_mode,
        "before": asdict(result.before),
        "after": asdict(result.after),
        "issues": result.issues,
        "recommendations": result.recommendations,
    }, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # Allow running directly for testing
    if len(sys.argv) < 2:
        print("Usage: python test_tokenizer_diagnostics.py <config_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    result = analyze_tokenizer_config(config_path)
    print_tokenizer_report(result)
