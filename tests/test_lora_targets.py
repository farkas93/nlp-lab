from eliza_trainer.sft.lora_targets import matched_names, targets_for


TEXT_LAYER = ("model.language_model.layers.{}.{}.{}.linear")
FAMILIES = (("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
            ("mlp", ("gate_proj", "up_proj", "down_proj")))


def text_names():
    return [TEXT_LAYER.format(layer, family, projection)
            for layer in (0, 11) for family, projections in FAMILIES for projection in projections]


def test_gemma4_targets_only_text_decoder_inner_linears():
    names = text_names() + [
        "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear",
        "model.audio_tower.layers.0.self_attn.q_proj.linear",
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.0.mlp.down_proj.linear.weight",
        "model.language_model.layers.0.self_attn.relative_k_proj",
    ]
    targets = targets_for("gemma4_conditional_generation", ["ignored"])
    assert isinstance(targets, str)
    assert matched_names(names, targets) == text_names()


def test_non_gemma_keeps_configured_suffix_targets():
    targets = targets_for("qwen3_5_conditional_generation", ("q_proj", "down_proj"))
    names = ["model.layers.0.self_attn.q_proj", "model.layers.0.mlp.down_proj", "model.vision.q_proj"]
    assert targets == ["q_proj", "down_proj"]
    assert matched_names(names, targets) == names
