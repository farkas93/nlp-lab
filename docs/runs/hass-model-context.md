# HASS model experiments and context budgets

The checked-in `eliza-small` profile serves Gemma 4 E4B-it Q4_K_M with
llama.cpp, `CTX_SIZE=131072`, and reasoning off. Verify the effective request
budget of the deployed server, including output reserve and any concurrency
settings. Job 7 used Qwen 0.8B and hit our 4,096-token admission guard before
loading weights. This is no evidence of a model limit or GPU memory failure.

| Checkpoint | Native window | Experiment and extension options |
| --- | ---: | --- |
| [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | 262,144 | Pinned queue smoke, non-thinking. Native capacity covers the serving target; do not assume the 9B scaling recipe works for this checkpoint. |
| [Gemma 4 E4B-it](https://huggingface.co/google/gemma-4-E4B-it) | 131,072 | Main Eliza Small target. Verify its native tool template, SFT backend and adapter deployment to llama.cpp. Local/global attention with proportional RoPE; no adopted extension beyond native. |
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | 262,144 | Larger baseline on the same frozen benchmark. Qwen documents YaRN to about 1.01M, with possible short-context degradation from static scaling. Treat as a separate experiment if native capacity is insufficient. |
| [Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base), [Qwen3.5-9B-Base](https://huggingface.co/Qwen/Qwen3.5-9B-Base), [Gemma 4 E4B](https://huggingface.co/google/gemma-4-E4B) | Qwen 262,144; Gemma 131,072 | Later base-versus-post-trained comparison. Qwen base cards advertise extension; it is not a verified DGX training capability. |

Gemma 4 E2B and Qwen3.5 2B/4B also have 128K serving profiles and are optional
latency baselines. Keep checkpoint revision, tokenizer, template, thinking,
loss policy, and serving quantization explicit in each comparison.

## Model capacity, training fit, and serving acceptance

- **Native window** is a checkpoint property. Shorter LoRA training does not
  change positional capacity, but long-context behavior can still regress.
- **Training fit** depends on the rendered examples, backend and allocation.
  Use exact token measurements and actual forward/backward evidence. Dynamic
  padding, batch size one and gradient checkpointing are the starting point.
  Benchmark efficient attention on the actual device. At very long lengths,
  validate chunked/fused target-only loss against the existing loss and gradients
  before adopting it. A 128K row with Qwen's vocabulary would require roughly
  61 GiB for full BF16 logits alone if materialized; masking labels by itself
  does not avoid this cost.
- **Serving acceptance** is measured on the intended llama.cpp profile. Compare
  original and tuned E4B at short and long lengths through 128K, placing HASS
  evidence early, middle and late. Record exact tool/entity/argument accuracy,
  grounded replies and latency. The current 49-row pilot verifies the machinery,
  not reliable 128K retention or generalization.

Order: finish Qwen 0.8B smoke, qualify Gemma E4B-it training and deployment,
then compare against Qwen 9B on a broader frozen benchmark. No RoPE scaling is
required for the 128K deployment target.
