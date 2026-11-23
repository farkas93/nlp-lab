"""Gemma 4 loading contract; called only after unsloth_entry patched the process."""
UNSLOTH_MODEL_OWNER = "unsloth"
UNSLOTH_MODEL_NAME = "gemma-4-E4B-it"
UNSLOTH_MODEL_REVISION = "4e22d7e59e078e63a14f351efdc5232ed366b621"


def expected_model_name():
    return f"{UNSLOTH_MODEL_OWNER}/{UNSLOTH_MODEL_NAME}"


def load_model(config):
    import sys
    if "unsloth" not in sys.modules:
        raise RuntimeError("Use nlp_training.unsloth_entry so Unsloth is imported before Transformers")
    if (config.model.model_name, config.model.revision, config.model.tokenizer_revision) != (
            expected_model_name(), UNSLOTH_MODEL_REVISION, UNSLOTH_MODEL_REVISION):
        raise ValueError("Gemma 4 Unsloth training requires the pinned Unsloth checkpoint and tokenizer")
    import torch
    from unsloth import FastModel

    model, processor = FastModel.from_pretrained(
        model_name=config.model.model_name, revision=config.model.revision,
        max_seq_length=config.model.max_seq_len, load_in_4bit=False, load_in_16bit=True,
        dtype=torch.bfloat16, full_finetuning=False, fast_inference=False,
        fix_tokenizer=False, use_exact_model_name=True,
        use_gradient_checkpointing="unsloth" if config.training.gradient_checkpointing else False,
    )
    return model, getattr(processor, "tokenizer", processor)


def attach_language_adapter(model, config):
    from unsloth import FastModel
    model = FastModel.get_peft_model(
        model, finetune_vision_layers=False, finetune_audio_layers=False,
        finetune_language_layers=True, finetune_attention_modules=True, finetune_mlp_modules=True,
        r=config.model.lora_r, lora_alpha=config.model.lora_alpha, lora_dropout=0,
        bias="none", random_state=config.training.seed,
        use_gradient_checkpointing="unsloth" if config.training.gradient_checkpointing else False,
    )
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name or "language_model" not in name or
                            "vision" in name or "audio" in name for name in trainable):
        raise RuntimeError("Unsloth did not produce exclusively language-model LoRA parameters")
    model.peft_config["default"].revision = config.model.revision
    return model


def verify_tokenizer(reference, loaded):
    if reference.get_vocab() != loaded.get_vocab() or reference.chat_template != loaded.chat_template:
        raise ValueError("Unsloth tokenizer/template differs from pinned preflight; refusing to change training serialization")
