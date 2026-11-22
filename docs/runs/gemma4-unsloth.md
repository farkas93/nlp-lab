# Gemma 4: Unsloth qualification before the full run

Training uses **`unsloth/gemma-4-E4B-it` Safetensors**, pinned to
`4e22d7e59e078e63a14f351efdc5232ed366b621`. The GGUF repository is for serving
or a later export/conversion step, not training input. Changing repositories also
changes experiment lineage: do not relabel previous Google-checkpoint runs.

## Notebook settings

Pull `feature/hass-full-runthrough`, update the lightweight client overlay, then
restart the Jupyter kernel:

```python
%pip install --target /home/jovyan/eliza/training/.client --upgrade /home/jovyan/eliza/nlp-lab/packages/training-client
```

Use the same frozen household release. In `hass-sft.ipynb` select:

```python
RUN_LABEL = 'hass-gemma4-unsloth-qualification-01'
MODEL_PRESET = 'sft_hass_gemma4_unsloth_qualification.yaml'
RESOURCE_OVERRIDES = {'memory_mb': 65536, 'time_minutes': 120}
```

Run setup, remote preflight and training. The operator automatically submits the
`unsloth-gemma4` runtime for Unsloth training/evaluation; preflight still uses the
CPU-only context runtime and the pinned Unsloth checkpoint tokenizer. No new
Jupyter/Dagster image or host-wide trainer installation is needed.

Qualification performs:

1. Unsloth import **before** Transformers and PEFT through a dedicated entrypoint.
2. Pinned `FastModel.from_pretrained` load without vLLM, quantization or silent
   repository remapping. Tokenizer vocabulary and template must match preflight.
3. `FastModel.get_peft_model` with language/attention/MLP enabled, vision/audio
   disabled. Reject any trainable parameters outside language-model LoRA.
4. Longest-row forward/backward with finite loss and gradients, using Unsloth's
   checkpointing instead of overwriting it with generic Trainer checkpointing.
5. Two optimizer steps on our already-tokenized examples. Preserve the current
   assistant decision masks; do not run text preparation or response-only masking
   a second time. There is no packing or prompt loss.
6. Save adapter and pinned tokenizer, release the trained model, reload the exact
   base and saved adapter, and compare greedy continuation tokens before/after.

The verified `training-summary.json` includes `qualification` with
`save_reload_tokens_equal: true`. A successful training process alone is not
enough if qualification raised. Check the longest-row probe too. The qualification
skips full held-out loss evaluation to keep this gate short; the full preset runs
it and the notebook's later evaluation job compares held-out generations remotely.

Only after qualification succeeds, select a **new RUN_LABEL** and
`sft_hass_household_gemma4.yaml` for the 100-step run. This uses the same pinned
checkpoint/runtime and BF16 language-only LoRA. Read-only evaluation reloads through
Unsloth too; plain Transformers reload would omit the model patches.

## Runtime pins and verification boundaries

The profile is isolated from the existing Qwen/TRL environment. It retains the
PyTorch 2.9.1 CUDA 13 ARM64 wheel and verified cuSPARSELt metadata repair, and adds
Unsloth/Unsloth Zoo 2026.11.9, PEFT 0.22.0, Triton 3.5.1, the matching official
ARM64 torchvision 0.24.1 wheel, and torchao 0.15.0. Transitive versions are resolved
in `packages/training-client/nlp_training/profiles/unsloth-gemma4.resolved.txt`.
The resolver was checked for CPython 3.12 / Linux aarch64 with binary wheels only,
jointly including the lightweight client's dependencies. Installation still runs
`uv pip check` and the pinned wheel verification. Runtime records archive the
actual installed package versions. No `--no-deps` installation or xformers source
build is used.

This proves dependency resolution and code contracts, **not** GB10 kernel execution.
The two-step remote job is the hardware acceptance gate. GEMMA4's upstream
`use_cache=False`/shared-KV training fixes are one reason to use its maintained
Unsloth path rather than only change PEFT target names.

Upstream references:
- https://unsloth.ai/docs/models/gemma-4/train
- https://unsloth.ai/docs/blog/fine-tuning-llms-with-nvidia-dgx-spark-and-unsloth

GGUF merge/conversion, quantization and llama.cpp behavior validation remain a
separate serving export. No Hub publication is enabled by these presets.
