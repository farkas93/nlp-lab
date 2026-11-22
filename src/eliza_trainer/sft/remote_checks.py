"""Private context measurement and archived-adapter evaluation inside a job."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from uuid import UUID


def fetch_adapter(storage, bucket, prefix, source_run_id, pin, destination):
    from nlp_training.worker import download

    root = f"{prefix.rsplit('/', 1)[0]}/{UUID(source_run_id)}"
    result = json.loads(download(storage, bucket, root + "/result.json"))
    manifest = json.loads(download(storage, bucket, root + "/run-manifest.json"))
    if (result.get("outcome") != "succeeded" or result.get("run_id") != source_run_id
            or manifest.get("kind") != "sft" or manifest.get("run_id") != source_run_id
            or manifest.get("dataset_manifest") != pin):
        raise ValueError("Source must be a successful SFT run on this exact dataset")
    index = json.loads(download(storage, bucket, root + "/artifact-index.json"))
    allowed = {"adapter_config.json", "adapter_model.safetensors", "tokenizer.json",
               "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"}
    files = {}
    parents = set()
    for entry in index:
        relative = PurePosixPath(entry["path"])
        if relative.parent.name != "final" or relative.name not in allowed:
            continue
        if relative.is_absolute() or ".." in relative.parts or relative.name in files:
            raise ValueError("Invalid or ambiguous final adapter index")
        parents.add(relative.parent)
        files[relative.name] = entry
    if len(parents) != 1 or not {"adapter_config.json", "adapter_model.safetensors", "tokenizer.json", "tokenizer_config.json"} <= files.keys():
        raise ValueError("Source archive is missing a complete final adapter and tokenizer")
    destination.mkdir()
    for name, entry in files.items():
        if type(entry.get("size")) is not int or not 0 < entry["size"] <= 512 * 1024 * 1024:
            raise ValueError("Invalid or excessive adapter artifact size")
        response = storage.get_object(Bucket=bucket, Key=f"{root}/artifacts/{entry['path']}")
        body = response["Body"]
        size, checksum = 0, hashlib.sha256()
        try:
            with (destination / name).open("xb") as output:
                while chunk := body.read(1024 * 1024):
                    size += len(chunk)
                    if size > entry["size"]:
                        raise ValueError(f"Source adapter checksum mismatch: {name}")
                    checksum.update(chunk)
                    output.write(chunk)
        finally:
            body.close()
        if size != entry["size"] or checksum.hexdigest() != entry["sha256"]:
            raise ValueError(f"Source adapter checksum mismatch: {name}")
    return json.loads((destination / "adapter_config.json").read_bytes())


def evaluate(config, archive, directory):
    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoTokenizer
    from transformers import Qwen3_5ForConditionalGeneration
    from src.sft_dataset_loader import load_sft_manifest_dataset, tokenize_with_loss_mode

    model_config = config.model
    if model_config.architecture not in {"qwen3_5_conditional_generation", "gemma4_conditional_generation"} or not model_config.revision:
        raise ValueError("Archived-adapter evaluation requires a pinned qualified model architecture")
    if archive.get("base_model_name_or_path") != model_config.model_name or archive.get("revision") != model_config.revision:
        raise ValueError("Archived adapter base model or revision differs from the evaluation config")
    data = config.data
    held = load_sft_manifest_dataset(manifest_uri=data.manifest_uri, expected_manifest_sha256=data.expected_manifest_sha256,
                                     train_split="train", eval_split="eval", max_train_samples=None, max_eval_samples=None).eval_dataset
    ordered = sorted(range(len(held)), key=lambda i: held[i]["example_id"])
    call = next((i for i in ordered if held[i]["target_message"].get("tool_calls")), None)
    reply = next((i for i in ordered if held[i]["target_message"].get("content") and held[i]["messages"][-1]["role"] == "tool"), None)
    if call is None or reply is None:
        raise ValueError("Held-out split lacks a tool call or a post-tool reply")
    fixed = held.select(ordered)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    native = AutoConfig.from_pretrained(model_config.model_name, revision=model_config.revision).get_text_config().max_position_embeddings
    encoded = tokenize_with_loss_mode(fixed, tokenizer, native - 96, "assistant_only",
                                       tokenizer_type=config.runtime.tokenizer_type, enable_thinking=model_config.enable_thinking)
    loader = Qwen3_5ForConditionalGeneration
    if model_config.architecture == "gemma4_conditional_generation":
        from transformers import Gemma4ForConditionalGeneration
        loader = Gemma4ForConditionalGeneration
    if config.identity.backend == "unsloth":
        from .unsloth_model import load_model, verify_tokenizer
        from unsloth import FastModel
        base, loaded_tokenizer = load_model(config)
        verify_tokenizer(tokenizer, loaded_tokenizer)
    else:
        base = loader.from_pretrained(model_config.model_name, revision=model_config.revision,
                                      dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(base, directory).eval()
    if config.identity.backend == "unsloth":
        FastModel.for_inference(model)
    results = []
    for number, (example, row) in enumerate(zip(fixed, encoded, strict=True), start=1):
        boundary = next(i for i, label in enumerate(row["labels"]) if label != -100)
        prompt = torch.tensor([row["input_ids"][:boundary]], device=model.get_input_embeddings().weight.device)
        args = dict(input_ids=prompt, attention_mask=torch.ones_like(prompt), do_sample=False, max_new_tokens=96)
        with torch.inference_mode():
            with model.disable_adapter():
                original = model.generate(**args)[0, boundary:]
            tuned = model.generate(**args)[0, boundary:]
        target = tokenizer.decode(row["input_ids"][boundary:])
        results.append(dict(example_id=example["example_id"], family=example.get("family"), target=target,
                            base=tokenizer.decode(original), adapter=tokenizer.decode(tuned),
                            base_exact=original.tolist() == row["input_ids"][boundary:],
                            adapter_exact=tuned.tolist() == row["input_ids"][boundary:],
                            base_hit_output_budget=len(original) == 96, adapter_hit_output_budget=len(tuned) == 96))
        if number % 25 == 0 or number == len(fixed):
            print("ELIZA_PROGRESS " + json.dumps(dict(stage="evaluating_held_out", step=number, total=len(fixed))), flush=True)
    return results


def main():
    import yaml
    from .context_preflight import measure
    from .run_config import load_sft_run_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("hass-context", "hass-eval"), required=True)
    for name in ("config", "bucket", "prefix"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    config = load_sft_run_config(args.config)
    data, model = config.data, config.model
    pin = {"uri": data.manifest_uri, "sha256": data.expected_manifest_sha256}
    output = Path("artifacts")
    output.mkdir(exist_ok=True)
    if args.kind == "hass-context":
        from nlp_training.tracking import verify_tracking
        if not config.tracking.mlflow_tracking_uri:
            raise ValueError("Remote MLflow tracking must be configured before preflight")
        tracking = verify_tracking(config.tracking.mlflow_tracking_uri, config.training.experiment_name)
        report = measure(data.manifest_uri, data.expected_manifest_sha256, model=model.model_name,
                         revision=model.revision, tokenizer_revision=model.tokenizer_revision,
                         budget=model.max_seq_len, tokenizer_type=config.runtime.tokenizer_type,
                          enable_thinking=model.enable_thinking)
        report["mlflow"] = tracking
        name = "context-preflight.json"
    else:
        from nlp_training.worker import client

        raw = yaml.safe_load(Path(args.config).read_bytes())
        source_run_id = raw["evaluation"]["source_run_id"]
        directory = Path("source-adapter")
        adapter = fetch_adapter(client(), args.bucket, args.prefix, source_run_id, pin, directory)
        report = {"source_run_id": source_run_id, "dataset_manifest": pin, "model": model.model_name,
                  "revision": model.revision, "job_id": os.getenv("SLURM_JOB_ID"),
                  "examples": evaluate(config, adapter, directory)}
        results = report["examples"]
        report["metrics"] = {"rows": len(results), "base_exact_rate": sum(row["base_exact"] for row in results) / len(results),
                             "adapter_exact_rate": sum(row["adapter_exact"] for row in results) / len(results),
                             "note": "Exact token match is strict; not a semantic tool/entity correctness metric. No tools executed."}
        name = "held-out-comparison.json"
    (output / name).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
