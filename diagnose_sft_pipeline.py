#!/usr/bin/env python3
"""
SFT Pipeline Diagnostic Tool

Analyzes the training data pipeline to identify issues with tokenization,
special tokens, and chat template formatting that could cause inference problems
like infinite loops after EOS tokens.

Usage:
    ./diagnose_sft_pipeline.py --config configs/sft_hass_qwen3_5_0_8b.yaml
    ./diagnose_sft_pipeline.py --config configs/sft_hass_qwen3_5_0_8b.yaml --num-samples 10
    ./diagnose_sft_pipeline.py --config configs/sft_hass_qwen3_5_0_8b.yaml --gguf-path /path/to/model.gguf
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add src to path for imports
SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from huggingface_hub import login
from transformers import AutoTokenizer

from src.eliza_trainer.sft.run_config import load_sft_run_config
from src.eliza_trainer.sft.dataset_loader import load_sft_manifest_dataset


# =============================================================================
# Data Structures
# =============================================================================

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
    special_tokens_map: dict[str, Any]
    all_special_tokens: list[str]
    all_special_ids: list[int]


@dataclass
class SampleDiagnostics:
    """Results from analyzing a single training sample."""
    sample_index: int
    session_id: str | None
    raw_messages: list[dict[str, Any]]
    target_text: str | None
    full_text_decoded: str
    input_ids: list[int]
    labels: list[int]
    special_token_positions: dict[str, list[int]]
    eos_count: int
    mid_sequence_eos_count: int
    has_proper_eos_termination: bool
    issues: list[str]


@dataclass
class GGUFDiagnostics:
    """Results from GGUF metadata analysis."""
    file_path: str
    has_chat_template: bool
    chat_template: str | None
    eos_token_id: int | None
    bos_token_id: int | None
    pad_token_id: int | None
    metadata_keys: list[str]
    issues: list[str]


# =============================================================================
# Tokenizer Analysis
# =============================================================================

def analyze_tokenizer(tokenizer, add_eos_token: bool = True) -> TokenizerDiagnostics:
    """Analyze tokenizer configuration for potential issues."""
    
    eos_token = getattr(tokenizer, 'eos_token', None)
    eos_token_id = getattr(tokenizer, 'eos_token_id', None)
    bos_token = getattr(tokenizer, 'bos_token', None)
    bos_token_id = getattr(tokenizer, 'bos_token_id', None)
    pad_token = getattr(tokenizer, 'pad_token', None)
    pad_token_id = getattr(tokenizer, 'pad_token_id', None)
    unk_token = getattr(tokenizer, 'unk_token', None)
    unk_token_id = getattr(tokenizer, 'unk_token_id', None)
    
    chat_template = getattr(tokenizer, 'chat_template', None)
    
    special_tokens_map = {}
    if hasattr(tokenizer, 'special_tokens_map'):
        special_tokens_map = dict(tokenizer.special_tokens_map)
    
    all_special_tokens = list(getattr(tokenizer, 'all_special_tokens', []))
    all_special_ids = list(getattr(tokenizer, 'all_special_ids', []))
    
    pad_equals_eos = (
        pad_token_id is not None and 
        eos_token_id is not None and 
        pad_token_id == eos_token_id
    )
    
    return TokenizerDiagnostics(
        eos_token=eos_token,
        eos_token_id=eos_token_id,
        bos_token=bos_token,
        bos_token_id=bos_token_id,
        pad_token=pad_token,
        pad_token_id=pad_token_id,
        unk_token=unk_token,
        unk_token_id=unk_token_id,
        chat_template=chat_template,
        pad_equals_eos=pad_equals_eos,
        special_tokens_map=special_tokens_map,
        all_special_tokens=all_special_tokens,
        all_special_ids=all_special_ids,
    )


def print_tokenizer_diagnostics(diag: TokenizerDiagnostics) -> None:
    """Print tokenizer diagnostics in a formatted way."""
    print("-" * 80)
    print("PHASE 1: TOKENIZER CONFIGURATION")
    print("-" * 80)
    
    print(f"EOS Token: {diag.eos_token!r} (ID: {diag.eos_token_id})")
    print(f"BOS Token: {diag.bos_token!r} (ID: {diag.bos_token_id})")
    print(f"PAD Token: {diag.pad_token!r} (ID: {diag.pad_token_id})")
    print(f"UNK Token: {diag.unk_token!r} (ID: {diag.unk_token_id})")
    print()
    
    if diag.pad_equals_eos:
        print("⚠️  WARNING: pad_token_id == eos_token_id")
        print("   This causes issues with full_conversation loss mode because padding")
        print("   tokens train the model to continue generating after EOS.")
        print()
    
    print(f"All Special Tokens ({len(diag.all_special_tokens)}):")
    for token, token_id in zip(diag.all_special_tokens, diag.all_special_ids):
        print(f"  {token!r} -> {token_id}")
    print()
    
    if diag.chat_template:
        # Truncate for display
        template_preview = diag.chat_template[:500]
        if len(diag.chat_template) > 500:
            template_preview += "... [truncated]"
        print(f"Chat Template:\n{template_preview}")
    else:
        print("Chat Template: NOT SET")
    print()


# =============================================================================
# Training Data Analysis
# =============================================================================

def analyze_sample(
    sample: dict[str, Any],
    sample_index: int,
    tokenizer,
    max_seq_len: int,
    tokenizer_diag: TokenizerDiagnostics,
) -> SampleDiagnostics:
    """Analyze a single training sample for tokenization issues."""
    
    messages = sample.get("messages", [])
    target_text = sample.get("target_text")
    session_id = sample.get("session_id")
    
    issues: list[str] = []
    
    # Build prompt and full messages
    prompt_messages = []
    full_messages = []
    
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        normalized = {
            "role": msg.get("role"),
            "content": msg.get("content"),
        }
        if msg.get("tool_calls"):
            normalized["tool_calls"] = msg["tool_calls"]
        if msg.get("name"):
            normalized["name"] = msg["name"]
        full_messages.append(normalized)
    
    # Find where assistant response starts
    assistant_idx = None
    for i, msg in enumerate(full_messages):
        if msg.get("role") == "assistant":
            assistant_idx = i
            break
    
    if assistant_idx is not None:
        prompt_messages = full_messages[:assistant_idx]
    else:
        prompt_messages = full_messages
        issues.append("No assistant message found in sample")
    
    # Apply chat template
    try:
        full_text = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception as e:
        full_text = f"[CHAT TEMPLATE ERROR: {e}]"
        issues.append(f"Chat template error: {e}")
    
    # Tokenize
    try:
        encoded = tokenizer(full_text, add_special_tokens=False)
        input_ids = encoded["input_ids"]
    except Exception as e:
        input_ids = []
        issues.append(f"Tokenization error: {e}")
    
    # Find special token positions
    special_token_positions: dict[str, list[int]] = {}
    
    if tokenizer_diag.eos_token_id is not None:
        eos_positions = [i for i, tid in enumerate(input_ids) if tid == tokenizer_diag.eos_token_id]
        special_token_positions["eos"] = eos_positions
    
    if tokenizer_diag.bos_token_id is not None:
        bos_positions = [i for i, tid in enumerate(input_ids) if tid == tokenizer_diag.bos_token_id]
        special_token_positions["bos"] = bos_positions
    
    if tokenizer_diag.pad_token_id is not None and not tokenizer_diag.pad_equals_eos:
        pad_positions = [i for i, tid in enumerate(input_ids) if tid == tokenizer_diag.pad_token_id]
        special_token_positions["pad"] = pad_positions
    
    # Count EOS tokens
    eos_count = len(special_token_positions.get("eos", []))
    
    # Check for mid-sequence EOS (EOS tokens that aren't at the end)
    eos_positions = special_token_positions.get("eos", [])
    mid_sequence_eos_count = 0
    if eos_positions:
        # EOS at the very end is fine, any before that are mid-sequence
        # But in chat format, EOS after each turn is expected - need to check if final one exists
        last_eos_pos = max(eos_positions)
        seq_len = len(input_ids)
        # Check if there's an EOS near the end (within last 5 tokens)
        has_proper_eos_termination = (seq_len - last_eos_pos) <= 5
    else:
        has_proper_eos_termination = False
        issues.append("No EOS token found in sequence")
    
    # For chat format, multiple EOS is normal (one per turn), so we check differently
    # The issue is if we have EOS tokens that aren't at turn boundaries
    
    # Create labels (mimicking the actual training logic)
    labels = input_ids.copy()  # For full_conversation mode
    
    return SampleDiagnostics(
        sample_index=sample_index,
        session_id=session_id,
        raw_messages=messages,
        target_text=target_text,
        full_text_decoded=full_text if isinstance(full_text, str) else "",
        input_ids=input_ids,
        labels=labels,
        special_token_positions=special_token_positions,
        eos_count=eos_count,
        mid_sequence_eos_count=mid_sequence_eos_count,
        has_proper_eos_termination=has_proper_eos_termination,
        issues=issues,
    )


def print_sample_diagnostics(
    sample_diag: SampleDiagnostics,
    tokenizer,
    verbose: bool = False,
) -> None:
    """Print diagnostics for a single sample."""
    
    print(f"\nSample {sample_diag.sample_index}:")
    if sample_diag.session_id:
        print(f"  Session ID: {sample_diag.session_id}")
    print(f"  Sequence length: {len(sample_diag.input_ids)} tokens")
    print(f"  EOS tokens found: {sample_diag.eos_count}")
    
    # Show special token positions
    for token_name, positions in sample_diag.special_token_positions.items():
        if positions:
            if len(positions) <= 10:
                print(f"  {token_name.upper()} positions: {positions}")
            else:
                print(f"  {token_name.upper()} positions: {positions[:5]} ... {positions[-5:]} ({len(positions)} total)")
    
    # Show termination status
    if sample_diag.has_proper_eos_termination:
        print("  ✓ Proper EOS termination")
    else:
        print("  ✗ NO proper EOS termination")
    
    # Show issues
    for issue in sample_diag.issues:
        print(f"  ⚠️  {issue}")
    
    if verbose:
        # Show decoded text (truncated)
        decoded_preview = sample_diag.full_text_decoded[:1000]
        if len(sample_diag.full_text_decoded) > 1000:
            decoded_preview += "\n... [truncated]"
        print(f"\n  Decoded text:\n  {decoded_preview}")
        
        # Show first/last tokens with decoding
        if sample_diag.input_ids:
            first_tokens = sample_diag.input_ids[:20]
            last_tokens = sample_diag.input_ids[-20:]
            
            print(f"\n  First 20 tokens: {first_tokens}")
            print(f"  Decoded: {tokenizer.decode(first_tokens)!r}")
            
            print(f"\n  Last 20 tokens: {last_tokens}")
            print(f"  Decoded: {tokenizer.decode(last_tokens)!r}")


def print_sample_summary(samples: list[SampleDiagnostics]) -> None:
    """Print summary statistics across all samples."""
    
    print("\n" + "-" * 80)
    print("SAMPLE ANALYSIS SUMMARY")
    print("-" * 80)
    
    total = len(samples)
    proper_termination = sum(1 for s in samples if s.has_proper_eos_termination)
    samples_with_issues = sum(1 for s in samples if s.issues)
    
    print(f"Total samples analyzed: {total}")
    print(f"Samples with proper EOS termination: {proper_termination}/{total}")
    print(f"Samples with issues: {samples_with_issues}/{total}")
    
    # Collect all unique issues
    all_issues: Counter[str] = Counter()
    for s in samples:
        for issue in s.issues:
            all_issues[issue] += 1
    
    if all_issues:
        print("\nIssue summary:")
        for issue, count in all_issues.most_common():
            print(f"  - {issue}: {count} samples")
    
    # Token length statistics
    lengths = [len(s.input_ids) for s in samples if s.input_ids]
    if lengths:
        print(f"\nSequence length stats:")
        print(f"  Min: {min(lengths)}, Max: {max(lengths)}, Avg: {sum(lengths)/len(lengths):.1f}")
    
    # EOS count statistics
    eos_counts = [s.eos_count for s in samples]
    if eos_counts:
        print(f"\nEOS token count stats:")
        print(f"  Min: {min(eos_counts)}, Max: {max(eos_counts)}, Avg: {sum(eos_counts)/len(eos_counts):.1f}")


# =============================================================================
# GGUF Analysis
# =============================================================================

def analyze_gguf(gguf_path: str) -> GGUFDiagnostics:
    """Analyze GGUF file metadata for special tokens and chat template."""
    
    issues: list[str] = []
    metadata_keys: list[str] = []
    chat_template: str | None = None
    eos_token_id: int | None = None
    bos_token_id: int | None = None
    pad_token_id: int | None = None
    
    try:
        from gguf import GGUFReader
        
        reader = GGUFReader(gguf_path)
        
        # Extract metadata
        for field in reader.fields.values():
            metadata_keys.append(field.name)
            
            # Look for specific fields
            if field.name == "tokenizer.chat_template":
                # GGUF stores strings as bytes
                parts = field.parts
                if parts:
                    raw = parts[-1]
                    if isinstance(raw, bytes):
                        chat_template = raw.decode("utf-8", errors="replace")
                    elif isinstance(raw, (list, tuple)) and raw:
                        chat_template = bytes(raw).decode("utf-8", errors="replace")
            
            elif field.name == "tokenizer.ggml.eos_token_id":
                parts = field.parts
                if parts:
                    eos_token_id = int(parts[-1])
            
            elif field.name == "tokenizer.ggml.bos_token_id":
                parts = field.parts
                if parts:
                    bos_token_id = int(parts[-1])
            
            elif field.name == "tokenizer.ggml.padding_token_id":
                parts = field.parts
                if parts:
                    pad_token_id = int(parts[-1])
        
    except ImportError:
        issues.append("gguf package not installed - run: pip install gguf")
    except Exception as e:
        issues.append(f"Error reading GGUF: {e}")
    
    has_chat_template = chat_template is not None and len(chat_template.strip()) > 0
    
    return GGUFDiagnostics(
        file_path=gguf_path,
        has_chat_template=has_chat_template,
        chat_template=chat_template,
        eos_token_id=eos_token_id,
        bos_token_id=bos_token_id,
        pad_token_id=pad_token_id,
        metadata_keys=metadata_keys,
        issues=issues,
    )


def print_gguf_diagnostics(
    gguf_diag: GGUFDiagnostics,
    tokenizer_diag: TokenizerDiagnostics | None = None,
) -> None:
    """Print GGUF diagnostics."""
    
    print("-" * 80)
    print("PHASE 3: GGUF METADATA ANALYSIS")
    print("-" * 80)
    
    print(f"File: {gguf_diag.file_path}")
    print()
    
    if gguf_diag.issues:
        for issue in gguf_diag.issues:
            print(f"⚠️  {issue}")
        print()
        return
    
    print(f"EOS Token ID: {gguf_diag.eos_token_id}")
    print(f"BOS Token ID: {gguf_diag.bos_token_id}")
    print(f"PAD Token ID: {gguf_diag.pad_token_id}")
    print()
    
    if gguf_diag.has_chat_template:
        template_preview = gguf_diag.chat_template[:500] if gguf_diag.chat_template else ""
        if gguf_diag.chat_template and len(gguf_diag.chat_template) > 500:
            template_preview += "... [truncated]"
        print(f"Chat Template:\n{template_preview}")
    else:
        print("⚠️  NO CHAT TEMPLATE EMBEDDED IN GGUF")
        print("   This means Ollama will use its default template, which may not match training.")
    print()
    
    # Compare with tokenizer if available
    if tokenizer_diag:
        print("Comparison with training tokenizer:")
        
        eos_match = gguf_diag.eos_token_id == tokenizer_diag.eos_token_id
        bos_match = gguf_diag.bos_token_id == tokenizer_diag.bos_token_id
        
        print(f"  EOS ID match: {'✓' if eos_match else '✗'} (GGUF: {gguf_diag.eos_token_id}, Tokenizer: {tokenizer_diag.eos_token_id})")
        print(f"  BOS ID match: {'✓' if bos_match else '✗'} (GGUF: {gguf_diag.bos_token_id}, Tokenizer: {tokenizer_diag.bos_token_id})")
        
        if not eos_match:
            print("  ⚠️  EOS token mismatch may cause generation issues!")
    
    print()
    print(f"Total metadata keys found: {len(gguf_diag.metadata_keys)}")
    
    # Show tokenizer-related keys
    tokenizer_keys = [k for k in gguf_diag.metadata_keys if "token" in k.lower()]
    if tokenizer_keys:
        print("Tokenizer-related metadata keys:")
        for key in sorted(tokenizer_keys):
            print(f"  - {key}")


# =============================================================================
# Main Diagnostic Flow
# =============================================================================

def run_diagnostics(
    config_path: str,
    num_samples: int = 5,
    gguf_path: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run full diagnostic analysis."""
    
    results: dict[str, Any] = {
        "config_path": config_path,
        "issues_found": [],
        "recommendations": [],
    }
    
    print("=" * 80)
    print("SFT PIPELINE DIAGNOSTIC REPORT")
    print("=" * 80)
    print()
    
    # Load config
    print(f"Loading config: {config_path}")
    run_config = load_sft_run_config(config_path)
    print(f"Model: {run_config.model.model_name}")
    print(f"Dataset: {run_config.data.manifest_uri}")
    print(f"Loss Mode: {run_config.model.loss_mode}")
    print(f"Max Seq Len: {run_config.model.max_seq_len}")
    print()
    
    results["model_name"] = run_config.model.model_name
    results["loss_mode"] = run_config.model.loss_mode
    
    # Login to HF if token available
    hf_token = os.environ.get("HF_HUB_TOKEN")
    if hf_token:
        login(token=hf_token)
    
    # Load tokenizer (mimicking train.py behavior)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        run_config.model.model_name,
        add_eos_token=True,  # Same as train.py
        use_fast=True,
        cache_dir=run_config.hf_model_cache_dir,
    )
    
    # Analyze BEFORE applying train.py modifications
    print("Analyzing tokenizer (BEFORE train.py modifications)...")
    tokenizer_diag_before = analyze_tokenizer(tokenizer)
    
    # Apply train.py modifications
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    # Analyze AFTER modifications
    print("Analyzing tokenizer (AFTER train.py modifications)...")
    tokenizer_diag_after = analyze_tokenizer(tokenizer)
    
    print()
    print("-" * 80)
    print("TOKENIZER: BEFORE train.py MODIFICATIONS")
    print("-" * 80)
    print(f"PAD Token: {tokenizer_diag_before.pad_token!r} (ID: {tokenizer_diag_before.pad_token_id})")
    print(f"EOS Token: {tokenizer_diag_before.eos_token!r} (ID: {tokenizer_diag_before.eos_token_id})")
    print(f"pad_token == eos_token: {tokenizer_diag_before.pad_equals_eos}")
    
    print()
    print("-" * 80)
    print("TOKENIZER: AFTER train.py MODIFICATIONS")
    print("-" * 80)
    print(f"PAD Token: {tokenizer_diag_after.pad_token!r} (ID: {tokenizer_diag_after.pad_token_id})")
    print(f"EOS Token: {tokenizer_diag_after.eos_token!r} (ID: {tokenizer_diag_after.eos_token_id})")
    print(f"pad_token == eos_token: {tokenizer_diag_after.pad_equals_eos}")
    
    if tokenizer_diag_after.pad_equals_eos and run_config.model.loss_mode == "full_conversation":
        print()
        print("⚠️  CRITICAL ISSUE DETECTED:")
        print("   pad_token_id == eos_token_id WITH full_conversation loss mode!")
        print()
        print("   Impact: During batched training, shorter sequences are padded with EOS tokens.")
        print("   With full_conversation mode, these padding positions contribute to loss,")
        print("   teaching the model that EOS tokens can appear mid-sequence.")
        print()
        print("   Result: Model may not stop generating at EOS during inference.")
        results["issues_found"].append("pad_token == eos_token with full_conversation mode")
        results["recommendations"].append(
            "Fix pad_token handling in train.py: only set pad_token=eos_token if tokenizer.pad_token is None"
        )
    
    print()
    print_tokenizer_diagnostics(tokenizer_diag_after)
    
    # Load dataset
    print("-" * 80)
    print("PHASE 2: TRAINING DATA ANALYSIS")
    print("-" * 80)
    print()
    print(f"Loading dataset from: {run_config.data.manifest_uri}")
    
    try:
        dataset_result = load_sft_manifest_dataset(
            manifest_uri=run_config.data.manifest_uri,
            train_split=run_config.data.train_split,
            eval_split=run_config.data.eval_split,
            max_train_samples=num_samples,
            max_eval_samples=0,
            cache_mode="reuse",
        )
        
        train_dataset = dataset_result.train_dataset
        print(f"Loaded {len(train_dataset)} samples from train split")
        print()
        
        # Analyze samples
        sample_diagnostics: list[SampleDiagnostics] = []
        
        for i in range(min(num_samples, len(train_dataset))):
            sample = train_dataset[i]
            diag = analyze_sample(
                sample=sample,
                sample_index=i,
                tokenizer=tokenizer,
                max_seq_len=run_config.model.max_seq_len,
                tokenizer_diag=tokenizer_diag_after,
            )
            sample_diagnostics.append(diag)
            print_sample_diagnostics(diag, tokenizer, verbose=verbose)
        
        print_sample_summary(sample_diagnostics)
        
        # Check for issues in samples
        samples_without_eos = [s for s in sample_diagnostics if not s.has_proper_eos_termination]
        if samples_without_eos:
            results["issues_found"].append(f"{len(samples_without_eos)} samples lack proper EOS termination")
        
    except Exception as e:
        print(f"⚠️  Error loading dataset: {e}")
        results["issues_found"].append(f"Dataset load error: {e}")
    
    # GGUF analysis
    if gguf_path:
        print()
        gguf_diag = analyze_gguf(gguf_path)
        print_gguf_diagnostics(gguf_diag, tokenizer_diag_after)
        
        if not gguf_diag.has_chat_template:
            results["issues_found"].append("GGUF has no embedded chat template")
            results["recommendations"].append(
                "Add explicit TEMPLATE directive to Ollama Modelfile"
            )
        
        if gguf_diag.issues:
            results["issues_found"].extend(gguf_diag.issues)
    
    # Final summary
    print()
    print("=" * 80)
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)
    
    if results["issues_found"]:
        print("\nIssues Found:")
        for issue in results["issues_found"]:
            print(f"  ✗ {issue}")
    else:
        print("\n✓ No critical issues found")
    
    if results["recommendations"]:
        print("\nRecommendations:")
        for i, rec in enumerate(results["recommendations"], 1):
            print(f"  {i}. {rec}")
    
    print()
    print("=" * 80)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose SFT training pipeline for tokenization and special token issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic analysis
    %(prog)s --config configs/sft_hass_qwen3_5_0_8b.yaml
    
    # Analyze more samples
    %(prog)s --config configs/sft_hass_qwen3_5_0_8b.yaml --num-samples 20
    
    # Include GGUF analysis
    %(prog)s --config configs/sft_hass_qwen3_5_0_8b.yaml --gguf-path ~/models/model.gguf
    
    # Verbose output with full decoded sequences
    %(prog)s --config configs/sft_hass_qwen3_5_0_8b.yaml --verbose
        """
    )
    
    parser.add_argument(
        "--config",
        required=True,
        help="Path to SFT YAML config file (same format as start_sft.sh)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of training samples to analyze (default: 5)",
    )
    parser.add_argument(
        "--gguf-path",
        help="Optional path to GGUF file for metadata analysis",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show full decoded sequences and detailed token info",
    )
    
    args = parser.parse_args()
    
    # Load .env if present
    if os.path.exists(".env"):
        from dotenv import load_dotenv
        load_dotenv()
    
    # Configure logging
    logging.basicConfig(
        level=logging.WARNING,  # Suppress info logs from libraries
        format="%(levelname)s: %(message)s",
    )
    
    results = run_diagnostics(
        config_path=args.config,
        num_samples=args.num_samples,
        gguf_path=args.gguf_path,
        verbose=args.verbose,
    )
    
    # Exit with error code if issues found
    if results["issues_found"]:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
