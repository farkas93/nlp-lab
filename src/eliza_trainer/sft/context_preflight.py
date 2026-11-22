"""Measure a pinned HASS export without loading model weights or truncating."""
import argparse
import json
import math
import re
from pathlib import Path
from statistics import median


def summarize(rows, budget):
    if not rows:
        raise ValueError("Context measurement requires a nonempty split")
    longest = max(rows, key=lambda row: row["total_tokens"])
    stats = {}
    for field in ("prompt_tokens", "target_tokens", "total_tokens"):
        lengths = sorted(row[field] for row in rows)
        stats[field] = {"median": median(lengths), "p95": lengths[math.ceil(.95 * len(lengths)) - 1],
                        "max": lengths[-1]}
    return {"rows": len(rows), "over_budget": sum(row["total_tokens"] > budget for row in rows),
            "longest": longest, **stats}


def measure(manifest_uri, digest, *, model, revision, tokenizer_revision, budget, tokenizer_type, enable_thinking):
    from transformers import AutoConfig, AutoTokenizer
    from src.sft_dataset_loader import load_sft_manifest_dataset, tokenize_with_loss_mode

    if budget < 1 or not all(re.fullmatch(r"[0-9a-f]{40}", value) for value in (revision, tokenizer_revision)):
        raise ValueError("Context measurement requires a positive baseline and pinned model/tokenizer revisions")
    config = AutoConfig.from_pretrained(model, revision=revision)
    native = config.get_text_config().max_position_embeddings
    tokenizer = AutoTokenizer.from_pretrained(model, revision=tokenizer_revision, add_eos_token=True, use_fast=True)
    data = load_sft_manifest_dataset(manifest_uri=manifest_uri, train_split="train", eval_split="eval",
                                     max_train_samples=None, max_eval_samples=None, expected_manifest_sha256=digest)
    if data.manifest.get("kind") != "hass-experimental-sft-v1":
        raise ValueError("Expected the private HASS smoke export")
    reports = {}
    for split, dataset in (("train", data.train_dataset), ("eval", data.eval_dataset)):
        measured = tokenize_with_loss_mode(dataset, tokenizer, budget, "assistant_only", split_name=split,
                                           tokenizer_type=tokenizer_type, enable_thinking=enable_thinking, measure_only=True)
        reports[split] = summarize(list(measured), budget)
    longest = max(item["total_tokens"]["max"] for item in reports.values())
    return {"manifest_sha256": digest, "model": model, "revision": revision, "tokenizer_revision": tokenizer_revision,
            "tokenizer_type": tokenizer_type, "enable_thinking": enable_thinking, "native_context": native,
            "baseline_budget": budget, "required_tokens": longest, "fits_native_context": longest <= native,
            "suggested_budget": min(native, max(budget, ((longest + 1023) // 1024) * 1024)),
            "splits": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--tokenizer-type", required=True)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = measure(args.manifest, args.sha256, model=args.model, revision=args.revision,
                     tokenizer_revision=args.tokenizer_revision, budget=args.budget,
                     tokenizer_type=args.tokenizer_type, enable_thinking=args.enable_thinking)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
