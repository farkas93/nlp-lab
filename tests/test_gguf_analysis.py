"""
GGUF metadata analysis for model diagnostics.

Extracts and analyzes metadata from GGUF files to verify chat templates
and special token configurations match training setup.

Requires: gguf package (installed via --with gguf in uv command)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class GGUFDiagnostics:
    """Results from GGUF metadata analysis."""
    file_path: str
    file_size_mb: float
    has_chat_template: bool
    chat_template: str | None
    eos_token_id: int | None
    bos_token_id: int | None
    pad_token_id: int | None
    model_type: str | None
    architecture: str | None
    context_length: int | None
    vocab_size: int | None
    tokenizer_model: str | None
    all_metadata_keys: list[str]
    issues: list[str]
    recommendations: list[str]


def analyze_gguf_metadata(gguf_path: str) -> GGUFDiagnostics:
    """
    Extract and analyze GGUF file metadata.
    
    Args:
        gguf_path: Path to GGUF file
        
    Returns:
        GGUFDiagnostics with extracted metadata and detected issues
    """
    from gguf import GGUFReader
    
    path = Path(gguf_path)
    if not path.exists():
        raise FileNotFoundError(f"GGUF file not found: {gguf_path}")
    
    file_size_mb = path.stat().st_size / (1024 * 1024)
    
    reader = GGUFReader(gguf_path)
    
    # Extract metadata
    metadata: dict[str, Any] = {}
    all_keys: list[str] = []
    
    for field in reader.fields.values():
        all_keys.append(field.name)
        
        # Extract value from field parts
        parts = field.parts
        if not parts:
            continue
            
        raw = parts[-1]
        
        # Handle different value types
        if isinstance(raw, bytes):
            try:
                metadata[field.name] = raw.decode("utf-8", errors="replace")
            except Exception:
                metadata[field.name] = str(raw)
        elif isinstance(raw, (list, tuple)):
            if raw and isinstance(raw[0], int) and len(raw) < 100:
                # Likely a small int array
                metadata[field.name] = list(raw)
            elif raw:
                try:
                    metadata[field.name] = bytes(raw).decode("utf-8", errors="replace")
                except Exception:
                    metadata[field.name] = str(raw)
        else:
            metadata[field.name] = raw
    
    # Extract specific fields
    chat_template = metadata.get("tokenizer.chat_template")
    eos_token_id = metadata.get("tokenizer.ggml.eos_token_id")
    bos_token_id = metadata.get("tokenizer.ggml.bos_token_id")
    pad_token_id = metadata.get("tokenizer.ggml.padding_token_id")
    model_type = metadata.get("general.architecture")
    architecture = metadata.get("general.architecture")
    context_length = metadata.get(f"{architecture}.context_length") if architecture else None
    vocab_size = metadata.get(f"{architecture}.vocab_size") if architecture else None
    tokenizer_model = metadata.get("tokenizer.ggml.model")
    
    # Detect issues
    issues: list[str] = []
    recommendations: list[str] = []
    
    has_chat_template = bool(chat_template and str(chat_template).strip())
    
    if not has_chat_template:
        issues.append("No chat template embedded in GGUF")
        recommendations.append(
            "Add explicit TEMPLATE directive to Ollama Modelfile to ensure correct formatting"
        )
    
    if eos_token_id is None:
        issues.append("No EOS token ID found in GGUF metadata")
    
    if pad_token_id is not None and eos_token_id is not None and pad_token_id == eos_token_id:
        issues.append(f"PAD token ID ({pad_token_id}) equals EOS token ID ({eos_token_id})")
    
    return GGUFDiagnostics(
        file_path=gguf_path,
        file_size_mb=file_size_mb,
        has_chat_template=has_chat_template,
        chat_template=chat_template if has_chat_template else None,
        eos_token_id=int(eos_token_id) if eos_token_id is not None else None,
        bos_token_id=int(bos_token_id) if bos_token_id is not None else None,
        pad_token_id=int(pad_token_id) if pad_token_id is not None else None,
        model_type=model_type,
        architecture=architecture,
        context_length=int(context_length) if context_length is not None else None,
        vocab_size=int(vocab_size) if vocab_size is not None else None,
        tokenizer_model=tokenizer_model,
        all_metadata_keys=sorted(all_keys),
        issues=issues,
        recommendations=recommendations,
    )


def print_gguf_report(result: GGUFDiagnostics) -> None:
    """Print formatted GGUF diagnostics report."""
    print("=" * 80)
    print("GGUF METADATA DIAGNOSTICS")
    print("=" * 80)
    print()
    print(f"File: {result.file_path}")
    print(f"Size: {result.file_size_mb:.2f} MB")
    print()
    
    print("-" * 80)
    print("MODEL INFO")
    print("-" * 80)
    print(f"  Architecture: {result.architecture}")
    print(f"  Model Type: {result.model_type}")
    print(f"  Context Length: {result.context_length}")
    print(f"  Vocab Size: {result.vocab_size}")
    print(f"  Tokenizer Model: {result.tokenizer_model}")
    print()
    
    print("-" * 80)
    print("TOKENIZER CONFIG")
    print("-" * 80)
    print(f"  EOS Token ID: {result.eos_token_id}")
    print(f"  BOS Token ID: {result.bos_token_id}")
    print(f"  PAD Token ID: {result.pad_token_id}")
    print(f"  Has Chat Template: {'YES' if result.has_chat_template else 'NO'}")
    print()
    
    if result.has_chat_template and result.chat_template:
        template_preview = result.chat_template[:500]
        if len(result.chat_template) > 500:
            template_preview += "... [truncated]"
        print(f"Chat Template:\n{template_preview}")
    else:
        print("Chat Template: NOT EMBEDDED")
    print()
    
    print("-" * 80)
    print("METADATA KEYS")
    print("-" * 80)
    tokenizer_keys = [k for k in result.all_metadata_keys if "token" in k.lower()]
    print(f"Tokenizer-related keys ({len(tokenizer_keys)}):")
    for key in tokenizer_keys:
        print(f"  - {key}")
    print(f"\nTotal metadata keys: {len(result.all_metadata_keys)}")
    
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


def get_gguf_diagnostics_json(result: GGUFDiagnostics) -> str:
    """Return GGUF diagnostics as JSON string."""
    data = asdict(result)
    # Truncate chat template for JSON output
    if data.get("chat_template") and len(data["chat_template"]) > 1000:
        data["chat_template"] = data["chat_template"][:1000] + "... [truncated]"
    return json.dumps(data, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_gguf_analysis.py <gguf_path>")
        sys.exit(1)
    
    gguf_path = sys.argv[1]
    result = analyze_gguf_metadata(gguf_path)
    print_gguf_report(result)
