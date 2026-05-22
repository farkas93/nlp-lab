"""
Training data diagnostics for SFT pipeline analysis.

Analyzes training samples to verify proper EOS token placement and
detect issues that could cause inference problems.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
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
from src.eliza_trainer.sft.dataset_loader import load_sft_manifest_dataset


@dataclass
class SampleAnalysis:
    """Analysis of a single training sample."""
    sample_index: int
    session_id: str | None
    sequence_length: int
    eos_token_positions: list[int]
    eos_count: int
    has_final_eos: bool
    decoded_first_50_tokens: str
    decoded_last_50_tokens: str
    issues: list[str]


@dataclass
class TrainingDataDiagnostics:
    """Results from training data analysis."""
    config_path: str
    model_name: str
    manifest_uri: str
    num_samples_analyzed: int
    samples: list[SampleAnalysis]
    summary: dict[str, Any]
    issues: list[str]
    recommendations: list[str]


def analyze_training_samples(config_path: str, num_samples: int = 5) -> TrainingDataDiagnostics:
    """
    Analyze training samples for EOS token placement and formatting issues.
    
    Args:
        config_path: Path to SFT YAML config file
        num_samples: Number of samples to analyze
        
    Returns:
        TrainingDataDiagnostics with sample analysis and detected issues
    """
    run_config = load_sft_run_config(config_path)
    
    # Login to HF if token available
    hf_token = os.environ.get("HF_HUB_TOKEN")
    if hf_token:
        login(token=hf_token)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        run_config.model.model_name,
        add_eos_token=True,
        use_fast=True,
        cache_dir=run_config.hf_model_cache_dir,
    )
    
    eos_token_id = tokenizer.eos_token_id
    
    # Load dataset
    dataset_result = load_sft_manifest_dataset(
        manifest_uri=run_config.data.manifest_uri,
        train_split=run_config.data.train_split,
        eval_split=run_config.data.eval_split,
        max_train_samples=num_samples,
        max_eval_samples=0,
        cache_mode="reuse",
    )
    
    train_dataset = dataset_result.train_dataset
    samples_to_analyze = min(num_samples, len(train_dataset))
    
    sample_analyses: list[SampleAnalysis] = []
    all_issues: list[str] = []
    
    for i in range(samples_to_analyze):
        sample = train_dataset[i]
        messages = sample.get("messages", [])
        session_id = sample.get("session_id")
        
        # Build full messages for chat template
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
        
        # Apply chat template
        issues: list[str] = []
        try:
            full_text = tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as e:
            full_text = ""
            issues.append(f"Chat template error: {e}")
        
        # Tokenize
        try:
            encoded = tokenizer(full_text, add_special_tokens=False)
            input_ids = encoded["input_ids"]
        except Exception as e:
            input_ids = []
            issues.append(f"Tokenization error: {e}")
        
        # Find EOS positions
        eos_positions = []
        if eos_token_id is not None:
            eos_positions = [idx for idx, tid in enumerate(input_ids) if tid == eos_token_id]
        
        # Check for final EOS
        has_final_eos = False
        if eos_positions and input_ids:
            last_eos = max(eos_positions)
            has_final_eos = (len(input_ids) - last_eos) <= 3
        
        if not has_final_eos and input_ids:
            issues.append("No EOS token at end of sequence")
        
        # Decode first/last tokens for inspection
        first_50 = ""
        last_50 = ""
        if input_ids:
            first_50 = tokenizer.decode(input_ids[:50], skip_special_tokens=False)
            last_50 = tokenizer.decode(input_ids[-50:], skip_special_tokens=False)
        
        sample_analyses.append(SampleAnalysis(
            sample_index=i,
            session_id=session_id,
            sequence_length=len(input_ids),
            eos_token_positions=eos_positions,
            eos_count=len(eos_positions),
            has_final_eos=has_final_eos,
            decoded_first_50_tokens=first_50,
            decoded_last_50_tokens=last_50,
            issues=issues,
        ))
        
        all_issues.extend(issues)
    
    # Compute summary
    samples_with_final_eos = sum(1 for s in sample_analyses if s.has_final_eos)
    samples_with_issues = sum(1 for s in sample_analyses if s.issues)
    lengths = [s.sequence_length for s in sample_analyses if s.sequence_length > 0]
    eos_counts = [s.eos_count for s in sample_analyses]
    
    summary = {
        "samples_analyzed": samples_to_analyze,
        "samples_with_final_eos": samples_with_final_eos,
        "samples_with_issues": samples_with_issues,
        "sequence_length_min": min(lengths) if lengths else 0,
        "sequence_length_max": max(lengths) if lengths else 0,
        "sequence_length_avg": sum(lengths) / len(lengths) if lengths else 0,
        "eos_count_min": min(eos_counts) if eos_counts else 0,
        "eos_count_max": max(eos_counts) if eos_counts else 0,
        "eos_count_avg": sum(eos_counts) / len(eos_counts) if eos_counts else 0,
    }
    
    # Aggregate issues
    issue_counts: Counter[str] = Counter()
    for s in sample_analyses:
        for issue in s.issues:
            issue_counts[issue] += 1
    
    aggregated_issues = [f"{issue} ({count}x)" for issue, count in issue_counts.most_common()]
    
    recommendations: list[str] = []
    if samples_with_final_eos < samples_to_analyze:
        recommendations.append(
            f"Only {samples_with_final_eos}/{samples_to_analyze} samples have proper EOS termination. "
            "Check chat template application."
        )
    
    return TrainingDataDiagnostics(
        config_path=config_path,
        model_name=run_config.model.model_name,
        manifest_uri=run_config.data.manifest_uri,
        num_samples_analyzed=samples_to_analyze,
        samples=sample_analyses,
        summary=summary,
        issues=aggregated_issues,
        recommendations=recommendations,
    )


def print_training_data_report(result: TrainingDataDiagnostics) -> None:
    """Print formatted training data diagnostics report."""
    print("=" * 80)
    print("TRAINING DATA DIAGNOSTICS")
    print("=" * 80)
    print()
    print(f"Model: {result.model_name}")
    print(f"Dataset: {result.manifest_uri}")
    print(f"Samples Analyzed: {result.num_samples_analyzed}")
    print()
    
    print("-" * 80)
    print("SAMPLE ANALYSIS")
    print("-" * 80)
    
    for sample in result.samples:
        print(f"\nSample {sample.sample_index}:")
        if sample.session_id:
            print(f"  Session ID: {sample.session_id}")
        print(f"  Sequence Length: {sample.sequence_length} tokens")
        print(f"  EOS Tokens: {sample.eos_count} at positions {sample.eos_token_positions[:10]}{'...' if len(sample.eos_token_positions) > 10 else ''}")
        print(f"  Final EOS: {'YES' if sample.has_final_eos else 'NO'}")
        
        if sample.issues:
            for issue in sample.issues:
                print(f"  [X] {issue}")
        
        print(f"\n  First 50 tokens: {sample.decoded_first_50_tokens!r}")
        print(f"  Last 50 tokens: {sample.decoded_last_50_tokens!r}")
    
    print()
    print("-" * 80)
    print("SUMMARY")
    print("-" * 80)
    print(f"  Samples with final EOS: {result.summary['samples_with_final_eos']}/{result.summary['samples_analyzed']}")
    print(f"  Samples with issues: {result.summary['samples_with_issues']}/{result.summary['samples_analyzed']}")
    print(f"  Sequence length: min={result.summary['sequence_length_min']}, max={result.summary['sequence_length_max']}, avg={result.summary['sequence_length_avg']:.1f}")
    print(f"  EOS count per sample: min={result.summary['eos_count_min']}, max={result.summary['eos_count_max']}, avg={result.summary['eos_count_avg']:.1f}")
    
    if result.issues:
        print("\nAggregated Issues:")
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


def get_training_data_diagnostics_json(result: TrainingDataDiagnostics) -> str:
    """Return training data diagnostics as JSON string."""
    return json.dumps({
        "config_path": result.config_path,
        "model_name": result.model_name,
        "manifest_uri": result.manifest_uri,
        "num_samples_analyzed": result.num_samples_analyzed,
        "samples": [asdict(s) for s in result.samples],
        "summary": result.summary,
        "issues": result.issues,
        "recommendations": result.recommendations,
    }, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_training_data_diagnostics.py <config_path> [num_samples]")
        sys.exit(1)
    
    config_path = sys.argv[1]
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    result = analyze_training_samples(config_path, num_samples)
    print_training_data_report(result)
