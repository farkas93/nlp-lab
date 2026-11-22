"""LoRA target selection that does not import torch or PEFT."""
import re

# Gemma 4 wraps text projections in Gemma4ClippableLinear, which PEFT cannot
# target directly. Its text config disables clipping, so the inner nn.Linear is
# the whole computation and is safe to adapt. Scope to the text decoder so the
# vision/audio towers (also named q_proj/k_proj/...) stay frozen.
GEMMA4_LORA_TARGETS = (
    r"model\.language_model\.layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))\.linear"
)


def targets_for(architecture, configured):
    """Return the PEFT target spec: a regex for Gemma 4, a name list otherwise."""
    if architecture == "gemma4_conditional_generation":
        return GEMMA4_LORA_TARGETS
    return list(configured)


def matched_names(names, targets):
    """Select module names PEFT would adapt, without touching the model."""
    if isinstance(targets, str):
        return [name for name in names if re.fullmatch(targets, name)]
    suffixes = set(targets)
    return [name for name in names if name.rsplit(".", 1)[-1] in suffixes]
