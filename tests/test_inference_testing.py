"""
Inference testing for model diagnostics.

Tests trained models (LoRA adapters via transformers/peft or GGUF via Ollama)
for EOS token handling and detects infinite loop patterns.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import requests

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eliza_trainer.sft.run_config import load_sft_run_config


# Default test prompts
DEFAULT_PROMPTS = [
    "Hello, how are you?",
    "Turn on the living room lights",
    "What's the temperature in the bedroom?",
    "What is the capital of France?",
]


@dataclass
class InferenceResult:
    """Result from a single inference test."""
    prompt: str
    generated_text: str
    generation_time_seconds: float
    output_token_count: int
    stopped_at_eos: bool
    detected_loop: bool
    loop_pattern: str | None
    issues: list[str]


@dataclass
class InferenceTestResults:
    """Results from all inference tests."""
    test_type: str  # "transformers" or "ollama"
    model_identifier: str
    prompts_tested: int
    successful_stops: int
    detected_loops: int
    results: list[InferenceResult]
    issues: list[str]
    recommendations: list[str]


def _detect_loop_pattern(text: str) -> tuple[bool, str | None]:
    """Detect if text contains repetitive loop patterns."""
    if len(text) < 50:
        return False, None
    
    # Check for repeated substrings
    for pattern_len in [10, 20, 30, 50]:
        if len(text) >= pattern_len * 3:
            pattern = text[-pattern_len:]
            count = text.count(pattern)
            if count >= 3:
                return True, pattern
    
    # Check for known problematic patterns
    known_patterns = ["Answer:", "Question:", "<|im_start|>", "<|endoftext|>", "<|im_end|>"]
    for pattern in known_patterns:
        count = text.count(pattern)
        if count > 5:
            return True, pattern
    
    return False, None


def test_transformers_inference(
    config_path: str,
    adapter_repo: str | None = None,
    prompts: list[str] | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
) -> InferenceTestResults:
    """
    Test inference using transformers + peft.
    
    Args:
        config_path: Path to SFT YAML config file
        adapter_repo: HuggingFace adapter repo (inferred from config if None)
        prompts: List of test prompts (uses defaults if None)
        max_new_tokens: Maximum tokens to generate
        temperature: Generation temperature
        
    Returns:
        InferenceTestResults with test outcomes
    """
    import torch
    from huggingface_hub import login
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    run_config = load_sft_run_config(config_path)
    
    # Determine adapter repo
    if adapter_repo is None:
        adapter_repo = run_config.hub.adapter_repo_name
        if not adapter_repo:
            raise ValueError(
                "Could not determine adapter repo from config. "
                "Set hub.adapter_repo_name or pass adapter_repo argument."
            )
    
    # Login to HF
    hf_token = os.environ.get("HF_HUB_TOKEN")
    if hf_token:
        login(token=hf_token)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        run_config.model.model_name,
        cache_dir=run_config.hf_model_cache_dir,
    )
    
    eos_token_id = tokenizer.eos_token_id
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model with adapter
    peft_config = PeftConfig.from_pretrained(adapter_repo, token=hf_token)
    
    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=run_config.hf_model_cache_dir,
    )
    model = PeftModel.from_pretrained(model, adapter_repo, token=hf_token)
    model.eval()
    
    # Run tests
    if prompts is None:
        prompts = DEFAULT_PROMPTS
    
    results: list[InferenceResult] = []
    
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            results.append(InferenceResult(
                prompt=prompt,
                generated_text="",
                generation_time_seconds=0,
                output_token_count=0,
                stopped_at_eos=False,
                detected_loop=False,
                loop_pattern=None,
                issues=[f"Chat template error: {e}"],
            ))
            continue
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        input_length = inputs["input_ids"].shape[1]
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
            )
        
        generation_time = time.time() - start_time
        
        generated_ids = outputs[0][input_length:].tolist()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        
        # Check for EOS
        eos_positions = [i for i, tid in enumerate(generated_ids) if tid == eos_token_id]
        stopped_at_eos = bool(eos_positions) and (len(generated_ids) - max(eos_positions)) <= 2
        
        # Detect loops
        detected_loop, loop_pattern = _detect_loop_pattern(generated_text)
        
        issues: list[str] = []
        if not stopped_at_eos and not detected_loop:
            issues.append("Generation reached max_new_tokens without EOS")
        if detected_loop:
            issues.append(f"Loop pattern detected: {loop_pattern!r}")
        
        results.append(InferenceResult(
            prompt=prompt,
            generated_text=generated_text,
            generation_time_seconds=generation_time,
            output_token_count=len(generated_ids),
            stopped_at_eos=stopped_at_eos,
            detected_loop=detected_loop,
            loop_pattern=loop_pattern,
            issues=issues,
        ))
    
    # Aggregate results
    successful_stops = sum(1 for r in results if r.stopped_at_eos)
    detected_loops = sum(1 for r in results if r.detected_loop)
    all_issues = [issue for r in results for issue in r.issues]
    
    recommendations: list[str] = []
    if detected_loops > 0:
        recommendations.append(
            "Infinite loops detected. Check: 1) pad_token != eos_token in training, "
            "2) Chat template matches training format, 3) GGUF has correct EOS token ID."
        )
    
    return InferenceTestResults(
        test_type="transformers",
        model_identifier=adapter_repo,
        prompts_tested=len(prompts),
        successful_stops=successful_stops,
        detected_loops=detected_loops,
        results=results,
        issues=all_issues,
        recommendations=recommendations,
    )


def test_ollama_inference(
    model_name: str,
    prompts: list[str] | None = None,
    max_tokens: int = 256,
    temperature: float = 0.2,
    ollama_host: str = "http://localhost:11434",
) -> InferenceTestResults:
    """
    Test inference using Ollama.
    
    Args:
        model_name: Ollama model name
        prompts: List of test prompts (uses defaults if None)
        max_tokens: Maximum tokens to generate
        temperature: Generation temperature
        ollama_host: Ollama API host
        
    Returns:
        InferenceTestResults with test outcomes
    """
    # Check Ollama connectivity
    try:
        response = requests.get(f"{ollama_host}/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        model_names = [m.get("name", "").split(":")[0] for m in models]
        
        if model_name not in model_names:
            raise ValueError(f"Model '{model_name}' not found. Available: {model_names}")
    except requests.RequestException as e:
        raise ConnectionError(f"Could not connect to Ollama at {ollama_host}: {e}")
    
    if prompts is None:
        prompts = DEFAULT_PROMPTS
    
    results: list[InferenceResult] = []
    
    for prompt in prompts:
        request_data = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        
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
            results.append(InferenceResult(
                prompt=prompt,
                generated_text="",
                generation_time_seconds=time.time() - start_time,
                output_token_count=0,
                stopped_at_eos=False,
                detected_loop=False,
                loop_pattern=None,
                issues=[f"Ollama API error: {e}"],
            ))
            continue
        
        generation_time = time.time() - start_time
        generated_text = result_data.get("response", "")
        
        # Check done reason
        done_reason = result_data.get("done_reason", "")
        stopped_at_eos = done_reason in ["stop", "eos"]
        
        # Detect loops
        detected_loop, loop_pattern = _detect_loop_pattern(generated_text)
        
        issues: list[str] = []
        if not stopped_at_eos and done_reason == "length":
            issues.append("Generation stopped due to max_tokens limit")
        if detected_loop:
            issues.append(f"Loop pattern detected: {loop_pattern!r}")
        
        results.append(InferenceResult(
            prompt=prompt,
            generated_text=generated_text,
            generation_time_seconds=generation_time,
            output_token_count=result_data.get("eval_count", 0),
            stopped_at_eos=stopped_at_eos,
            detected_loop=detected_loop,
            loop_pattern=loop_pattern,
            issues=issues,
        ))
    
    # Aggregate results
    successful_stops = sum(1 for r in results if r.stopped_at_eos)
    detected_loops = sum(1 for r in results if r.detected_loop)
    all_issues = [issue for r in results for issue in r.issues]
    
    recommendations: list[str] = []
    if detected_loops > 0:
        recommendations.append(
            "Infinite loops detected. Check: 1) GGUF has correct chat template, "
            "2) Modelfile has correct stop tokens, 3) Training used distinct pad/eos tokens."
        )
    
    return InferenceTestResults(
        test_type="ollama",
        model_identifier=model_name,
        prompts_tested=len(prompts),
        successful_stops=successful_stops,
        detected_loops=detected_loops,
        results=results,
        issues=all_issues,
        recommendations=recommendations,
    )


def print_inference_report(result: InferenceTestResults) -> None:
    """Print formatted inference test report."""
    print("=" * 80)
    print("INFERENCE TEST DIAGNOSTICS")
    print("=" * 80)
    print()
    print(f"Test Type: {result.test_type}")
    print(f"Model: {result.model_identifier}")
    print(f"Prompts Tested: {result.prompts_tested}")
    print()
    
    print("-" * 80)
    print("TEST RESULTS")
    print("-" * 80)
    
    for i, r in enumerate(result.results):
        status = "OK" if r.stopped_at_eos else "FAILED"
        if r.detected_loop:
            status = "LOOP"
        
        print(f"\nTest {i+1}: [{status}]")
        print(f"  Prompt: {r.prompt[:50]}{'...' if len(r.prompt) > 50 else ''}")
        print(f"  Tokens: {r.output_token_count}, Time: {r.generation_time_seconds:.2f}s")
        print(f"  Stopped at EOS: {'YES' if r.stopped_at_eos else 'NO'}")
        
        if r.detected_loop:
            print(f"  Loop Pattern: {r.loop_pattern!r}")
        
        if r.issues:
            for issue in r.issues:
                print(f"  [X] {issue}")
        
        # Show generated text (truncated)
        gen_preview = r.generated_text[:200]
        if len(r.generated_text) > 200:
            gen_preview += "..."
        print(f"  Generated: {gen_preview!r}")
    
    print()
    print("-" * 80)
    print("SUMMARY")
    print("-" * 80)
    print(f"  Successful EOS stops: {result.successful_stops}/{result.prompts_tested}")
    print(f"  Detected loops: {result.detected_loops}/{result.prompts_tested}")
    
    if result.detected_loops > 0:
        print("\n  [X] INFINITE LOOP BUG DETECTED")
    elif result.successful_stops == result.prompts_tested:
        print("\n  [OK] All tests passed - model stops correctly at EOS")
    else:
        print("\n  [?] Some tests did not stop at EOS - may need investigation")
    
    if result.recommendations:
        print("\nRecommendations:")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"  {i}. {rec}")
    
    print()
    print("=" * 80)


def get_inference_diagnostics_json(result: InferenceTestResults) -> str:
    """Return inference diagnostics as JSON string."""
    data = {
        "test_type": result.test_type,
        "model_identifier": result.model_identifier,
        "prompts_tested": result.prompts_tested,
        "successful_stops": result.successful_stops,
        "detected_loops": result.detected_loops,
        "results": [asdict(r) for r in result.results],
        "issues": result.issues,
        "recommendations": result.recommendations,
    }
    # Truncate long generated texts
    for r in data["results"]:
        if len(r.get("generated_text", "")) > 500:
            r["generated_text"] = r["generated_text"][:500] + "... [truncated]"
    return json.dumps(data, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    print("Usage via run_model_diagnostics.sh:")
    print("  --inference-transformers  Test with transformers/peft (requires config)")
    print("  --inference-ollama <model>  Test with Ollama")
    sys.exit(0)
