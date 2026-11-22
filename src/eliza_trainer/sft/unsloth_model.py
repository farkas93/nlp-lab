"""Gemma 4 loading contract; called only after unsloth_entry patched the process."""
def load_model(config):
    import sys
    if "unsloth" not in sys.modules:
        raise RuntimeError("Use nlp_training.unsloth_entry so Unsloth is imported before Transformers")
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
