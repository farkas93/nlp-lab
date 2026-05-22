#!/usr/bin/env python3
"""
Model Inference Test Tool

Tests a trained LoRA adapter or GGUF model for inference issues like infinite
loops after EOS tokens. Uses the same configuration format as training.

Usage:
    # Test LoRA adapter from HuggingFace (inferred from config)
    ./test_model_inference.py --config configs/sft_hass_qwen3_5_0_8b.yaml

    # Test specific adapter repo
    ./test_model_inference.py --config configs/sft_hass_qwen3_5_0_8b.yaml --adapter-repo zskalo/qwen3.5-0.8b-lora-hass-tools

    # Test with custom prompts
    ./test_model_inference.py --config configs/sft_hass_qwen3_5_0_8b.yaml --prompt "Turn on the kitchen lights"

    # Test GGUF via Ollama
    ./test_model_inference.py --config configs/sft_hass_qwen3_5_0_8b.yaml --ollama-model qwen3_hass
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add src to path for imports
SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from src.eliza_trainer.sft.run_config import load_sft_run_config


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class InferenceResult:
    """Results from a single inference test."""
    prompt: str
    generated_text: str
    full_output: str
    generation_time_seconds: float
    output_token_count: int
    stopped_at_eos: bool
    detected_loop: bool
    loop_pattern: str | None
    raw_tokens: list[int] | None
    eos_positions: list[int]
    issues: list[str]


@dataclass
class InferenceTestSummary:
    """Summary of all inference tests."""
    total_tests: int
    successful_stops: int
    detected_loops: int
    issues: list[str]
    results: list[InferenceResult]


# =============================================================================
# Test Prompts
# =============================================================================

DEFAULT_TEST_PROMPTS = [
    # Simple greeting
    "Hello, how are you?",
    
    # Home Assistant tool use (if applicable)
    "Turn on the living room lights",
    "What's the temperature in the bedroom?",
    "Set the thermostat to 72 degrees",
    
    # General assistant tasks
    "What is the capital of France?",
    "Write a haiku about coding",
]


# =============================================================================
# Transformers/PEFT Inference
# =============================================================================

def test_with_transformers(
    config_path: str,
    adapter_repo: str | None = None,
    prompts: list[str] | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
) -> InferenceTestSummary:
    """Test inference using transformers + peft."""
    
    import torch
    from huggingface_hub import login
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    
    # Load config
    run_config = load_sft_run_config(config_path)
    
    # Determine adapter repo
    if adapter_repo is None:
        adapter_repo = run_config.hub.adapter_repo_name
        if not adapter_repo:
            raise ValueError(
                "Could not determine adapter repo from config. "
                "Either set hub.adapter_repo_name in config or pass --adapter-repo"
            )
    
    print(f"Testing adapter: {adapter_repo}")
    print(f"Base model: {run_config.model.model_name}")
    print()
    
    # Login to HF
    hf_token = os.environ.get("HF_HUB_TOKEN")
    if hf_token:
        login(token=hf_token)
    
    # Load tokenizer from base model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        run_config.model.model_name,
        cache_dir=run_config.hf_model_cache_dir,
    )
    
    # Get special token info
    eos_token_id = tokenizer.eos_token_id
    print(f"EOS token: {tokenizer.eos_token!r} (ID: {eos_token_id})")
    
    # Ensure pad token is set (for generation)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model with adapter
    print("Loading model with adapter...")
    peft_config = PeftConfig.from_pretrained(adapter_repo, token=hf_token)
    
    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=run_config.hf_model_cache_dir,
    )
    model = PeftModel.from_pretrained(model, adapter_repo, token=hf_token)
    model.eval()
    
    print("Model loaded successfully")
    print()
    
    # Prepare test prompts
    if prompts is None:
        prompts = DEFAULT_TEST_PROMPTS
    
    # Run tests
    results: list[InferenceResult] = []
    
    for i, prompt in enumerate(prompts):
        print(f"Test {i+1}/{len(prompts)}: {prompt[:50]}...")
        
        result = _run_single_inference(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            eos_token_id=eos_token_id,
        )
        results.append(result)
        
        # Print result
        status = "✓ STOPPED" if result.stopped_at_eos else "✗ DID NOT STOP"
        if result.detected_loop:
            status = "✗ LOOP DETECTED"
        
        print(f"  {status}")
        print(f"  Tokens generated: {result.output_token_count}")
        print(f"  Time: {result.generation_time_seconds:.2f}s")
        
        if result.detected_loop and result.loop_pattern:
            print(f"  Loop pattern: {result.loop_pattern[:100]}...")
        
        if result.issues:
            for issue in result.issues:
                print(f"  ⚠️  {issue}")
        
        # Show generated text (truncated)
        gen_preview = result.generated_text[:200]
        if len(result.generated_text) > 200:
            gen_preview += "..."
        print(f"  Generated: {gen_preview!r}")
        print()
    
    # Compute summary
    successful_stops = sum(1 for r in results if r.stopped_at_eos)
    detected_loops = sum(1 for r in results if r.detected_loop)
    
    all_issues: list[str] = []
    for r in results:
        all_issues.extend(r.issues)
    
    return InferenceTestSummary(
        total_tests=len(results),
        successful_stops=successful_stops,
        detected_loops=detected_loops,
        issues=all_issues,
        results=results,
    )


def _run_single_inference(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    eos_token_id: int | None,
) -> InferenceResult:
    """Run inference for a single prompt and analyze the result."""
    import torch
    
    issues: list[str] = []
    
    # Build chat messages
    messages = [
        {"role": "user", "content": prompt},
    ]
    
    # Apply chat template
    try:
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as e:
        issues.append(f"Chat template error: {e}")
        formatted_prompt = prompt
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]
    
    # Generate
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_token_id,
            return_dict_in_generate=True,
            output_scores=False,
        )
    
    generation_time = time.time() - start_time
    
    # Extract generated tokens
    generated_ids = outputs.sequences[0][input_length:].tolist()
    full_output_ids = outputs.sequences[0].tolist()
    
    # Decode
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    full_output = tokenizer.decode(full_output_ids, skip_special_tokens=False)
    
    # Find EOS positions in generated tokens
    eos_positions = []
    if eos_token_id is not None:
        eos_positions = [i for i, tid in enumerate(generated_ids) if tid == eos_token_id]
    
    # Check if stopped at EOS
    stopped_at_eos = False
    if eos_positions:
        # Check if final token is EOS or generation stopped shortly after EOS
        last_eos = max(eos_positions)
        stopped_at_eos = (len(generated_ids) - last_eos) <= 2
    
    # Detect loops
    detected_loop = False
    loop_pattern = None
    
    # Simple loop detection: check for repeated substrings
    if len(generated_text) > 50:
        # Check for repeated patterns
        for pattern_len in [10, 20, 30, 50]:
            if len(generated_text) >= pattern_len * 3:
                pattern = generated_text[-pattern_len:]
                count = generated_text.count(pattern)
                if count >= 3:
                    detected_loop = True
                    loop_pattern = pattern
                    break
    
    # Check for specific known patterns
    known_patterns = ["Answer:", "Question:", "<|im_start|>", "<|endoftext|>"]
    for pattern in known_patterns:
        count = full_output.count(pattern)
        if count > 5:  # More than 5 occurrences suggests looping
            detected_loop = True
            loop_pattern = loop_pattern or pattern
            issues.append(f"Pattern '{pattern}' appears {count} times")
    
    if not stopped_at_eos and not detected_loop:
        issues.append("Generation reached max_new_tokens without EOS")
    
    return InferenceResult(
        prompt=prompt,
        generated_text=generated_text,
        full_output=full_output,
        generation_time_seconds=generation_time,
        output_token_count=len(generated_ids),
        stopped_at_eos=stopped_at_eos,
        detected_loop=detected_loop,
        loop_pattern=loop_pattern,
        raw_tokens=generated_ids,
        eos_positions=eos_positions,
        issues=issues,
    )


# =============================================================================
# Ollama Inference
# =============================================================================

def test_with_ollama(
    model_name: str,
    prompts: list[str] | None = None,
    max_tokens: int = 256,
    temperature: float = 0.2,
    ollama_host: str = "http://localhost:11434",
) -> InferenceTestSummary:
    """Test inference using Ollama."""
    
    import requests
    
    print(f"Testing Ollama model: {model_name}")
    print(f"Ollama host: {ollama_host}")
    print()
    
    # Check if model exists
    try:
        response = requests.get(f"{ollama_host}/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        model_names = [m.get("name", "").split(":")[0] for m in models]
        
        if model_name not in model_names:
            print(f"⚠️  Model '{model_name}' not found in Ollama")
            print(f"Available models: {model_names}")
            return InferenceTestSummary(
                total_tests=0,
                successful_stops=0,
                detected_loops=0,
                issues=[f"Model '{model_name}' not found in Ollama"],
                results=[],
            )
    except Exception as e:
        print(f"⚠️  Could not connect to Ollama: {e}")
        return InferenceTestSummary(
            total_tests=0,
            successful_stops=0,
            detected_loops=0,
            issues=[f"Ollama connection error: {e}"],
            results=[],
        )
    
    # Prepare test prompts
    if prompts is None:
        prompts = DEFAULT_TEST_PROMPTS
    
    # Run tests
    results: list[InferenceResult] = []
    
    for i, prompt in enumerate(prompts):
        print(f"Test {i+1}/{len(prompts)}: {prompt[:50]}...")
        
        result = _run_ollama_inference(
            model_name=model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            ollama_host=ollama_host,
        )
        results.append(result)
        
        # Print result
        status = "✓ STOPPED" if result.stopped_at_eos else "✗ DID NOT STOP"
        if result.detected_loop:
            status = "✗ LOOP DETECTED"
        
        print(f"  {status}")
        print(f"  Time: {result.generation_time_seconds:.2f}s")
        
        if result.detected_loop and result.loop_pattern:
            print(f"  Loop pattern: {result.loop_pattern[:100]}...")
        
        if result.issues:
            for issue in result.issues:
                print(f"  ⚠️  {issue}")
        
        # Show generated text (truncated)
        gen_preview = result.generated_text[:200]
        if len(result.generated_text) > 200:
            gen_preview += "..."
        print(f"  Generated: {gen_preview!r}")
        print()
    
    # Compute summary
    successful_stops = sum(1 for r in results if r.stopped_at_eos)
    detected_loops = sum(1 for r in results if r.detected_loop)
    
    all_issues: list[str] = []
    for r in results:
        all_issues.extend(r.issues)
    
    return InferenceTestSummary(
        total_tests=len(results),
        successful_stops=successful_stops,
        detected_loops=detected_loops,
        issues=all_issues,
        results=results,
    )


def _run_ollama_inference(
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    ollama_host: str,
) -> InferenceResult:
    """Run inference for a single prompt via Ollama."""
    
    import requests
    
    issues: list[str] = []
    
    # Build request
    request_data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    
    # Call Ollama API
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{ollama_host}/api/generate",
            json=request_data,
            timeout=120,
        )
        response.raise_for_status()
        result_data = response.json()
    except Exception as e:
        return InferenceResult(
            prompt=prompt,
            generated_text="",
            full_output="",
            generation_time_seconds=time.time() - start_time,
            output_token_count=0,
            stopped_at_eos=False,
            detected_loop=False,
            loop_pattern=None,
            raw_tokens=None,
            eos_positions=[],
            issues=[f"Ollama API error: {e}"],
        )
    
    generation_time = time.time() - start_time
    
    generated_text = result_data.get("response", "")
    
    # Ollama doesn't return token IDs, so we can't do exact token analysis
    # We'll rely on text-based loop detection
    
    # Check done_reason
    done_reason = result_data.get("done_reason", "")
    stopped_at_eos = done_reason in ["stop", "eos"]
    
    if not stopped_at_eos:
        if done_reason == "length":
            issues.append("Generation stopped due to max_tokens limit")
        elif done_reason:
            issues.append(f"Unexpected done_reason: {done_reason}")
    
    # Detect loops via text patterns
    detected_loop = False
    loop_pattern = None
    
    if len(generated_text) > 50:
        for pattern_len in [10, 20, 30, 50]:
            if len(generated_text) >= pattern_len * 3:
                pattern = generated_text[-pattern_len:]
                count = generated_text.count(pattern)
                if count >= 3:
                    detected_loop = True
                    loop_pattern = pattern
                    break
    
    # Check for specific patterns in text
    known_patterns = ["Answer:", "Question:", "<|im_start|>", "<|endoftext|>", "<|im_end|>"]
    for pattern in known_patterns:
        count = generated_text.count(pattern)
        if count > 3:
            detected_loop = True
            loop_pattern = loop_pattern or pattern
            issues.append(f"Pattern '{pattern}' appears {count} times in output")
    
    return InferenceResult(
        prompt=prompt,
        generated_text=generated_text,
        full_output=generated_text,  # Ollama doesn't give us the full formatted output
        generation_time_seconds=generation_time,
        output_token_count=result_data.get("eval_count", 0),
        stopped_at_eos=stopped_at_eos,
        detected_loop=detected_loop,
        loop_pattern=loop_pattern,
        raw_tokens=None,  # Not available from Ollama
        eos_positions=[],  # Not available from Ollama
        issues=issues,
    )


# =============================================================================
# Summary Output
# =============================================================================

def print_summary(summary: InferenceTestSummary) -> None:
    """Print test summary."""
    
    print("=" * 80)
    print("INFERENCE TEST SUMMARY")
    print("=" * 80)
    print()
    print(f"Total tests: {summary.total_tests}")
    print(f"Successful stops at EOS: {summary.successful_stops}/{summary.total_tests}")
    print(f"Detected loops: {summary.detected_loops}/{summary.total_tests}")
    
    if summary.issues:
        print()
        print("All issues encountered:")
        for issue in set(summary.issues):
            count = summary.issues.count(issue)
            print(f"  - {issue} ({count}x)")
    
    print()
    
    if summary.detected_loops > 0:
        print("⚠️  INFINITE LOOP BUG DETECTED")
        print("   The model is not properly stopping at EOS tokens.")
        print()
        print("   Likely causes:")
        print("   1. pad_token == eos_token during training with full_conversation mode")
        print("   2. Missing or incorrect chat template in GGUF/Ollama config")
        print("   3. Mismatched EOS token between training and inference")
    elif summary.successful_stops == summary.total_tests:
        print("✓ All tests passed - model stops correctly at EOS")
    else:
        print("⚠️  Some tests did not stop at EOS - may need investigation")
    
    print()
    print("=" * 80)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test model inference for EOS token handling issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test LoRA adapter (inferred from config)
    %(prog)s --config configs/sft_hass_qwen3_5_0_8b.yaml
    
    # Test specific adapter
    %(prog)s --config configs/sft_hass_qwen3_5_0_8b.yaml --adapter-repo zskalo/qwen3.5-0.8b-lora-hass-tools
    
    # Test via Ollama
    %(prog)s --ollama-model qwen3_hass
    
    # Custom prompt
    %(prog)s --config configs/sft_hass_qwen3_5_0_8b.yaml --prompt "Turn on the lights"
        """
    )
    
    # Source options (mutually exclusive in practice)
    parser.add_argument(
        "--config",
        help="Path to SFT YAML config file (for transformers/peft testing)",
    )
    parser.add_argument(
        "--adapter-repo",
        help="HuggingFace adapter repo to test (overrides config inference)",
    )
    parser.add_argument(
        "--ollama-model",
        help="Ollama model name to test (alternative to transformers)",
    )
    parser.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        help="Ollama API host (default: http://localhost:11434)",
    )
    
    # Test options
    parser.add_argument(
        "--prompt",
        action="append",
        help="Custom prompt to test (can be specified multiple times)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Generation temperature (default: 0.2)",
    )
    
    args = parser.parse_args()
    
    # Validate args
    if not args.config and not args.ollama_model:
        parser.error("Either --config or --ollama-model is required")
    
    # Load .env if present
    if os.path.exists(".env"):
        from dotenv import load_dotenv
        load_dotenv()
    
    # Get prompts
    prompts = args.prompt if args.prompt else None
    
    # Run appropriate test
    if args.ollama_model:
        summary = test_with_ollama(
            model_name=args.ollama_model,
            prompts=prompts,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            ollama_host=args.ollama_host,
        )
    else:
        summary = test_with_transformers(
            config_path=args.config,
            adapter_repo=args.adapter_repo,
            prompts=prompts,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    
    print_summary(summary)
    
    # Exit with error if loops detected
    if summary.detected_loops > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
