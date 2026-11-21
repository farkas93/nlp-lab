# SFT Data Contract

This contract defines what `nlp-lab` expects from `materialize_sft_release` outputs.

## Storage layout

- `dataset_id=<id>/dataset_version=<version>/split=train/part-*.parquet`
- `dataset_id=<id>/dataset_version=<version>/split=eval/part-*.parquet`
- `dataset_id=<id>/dataset_version=<version>/manifest.json`

## Manifest requirements

Manifest must be a JSON object with:

- `dataset_id`
- `dataset_version`
- `splits` list where each item contains:
  - `split` (for example `train`, `eval`)
  - `key` (object key or URI to Parquet file)

Recommended manifest fields for lineage:

- `construction.target_mode`
- `construction.token_budget`
- `construction.context_overhead_tokens`
- `construction.split_seed`
- `construction.eval_percent`
- `stats.train_rows`, `stats.eval_rows`, `stats.total_rows`

## Row-level requirements

Each SFT row must include:

- `messages`: ordered context messages
- `target_text`: assistant target string (backward-compatible)

Expected `messages` item shape:

- `role`: `system|user|assistant|tool`
- `content`: text, or `null` for an assistant decision containing native `tool_calls`

Native tool-calling metadata is supported and preserved when present:

- `tool_calls` (assistant messages)
- `tool_name` / `name` (tool messages)
- `tool_call_id` (tool results, matching the preceding assistant call ID)
- `id` / `type` on calls, preserved through tokenizer-specific argument adaptation
- `tools_json`: optional row-level JSON string containing advertised OpenAI-style
  function schemas; passed as `tools` to both prompt and full chat-template renders

For Parquet schema stability, store arguments as JSON strings (including
`tool_calls[].function.arguments` for nested calls). Native Qwen rendering receives
decoded argument objects; malformed stored JSON is rejected.

Optional structured target shape:

- `target_message`: assistant target message object. When present, loader/tokenizer uses this as the supervised target and preserves native tool-call fields.

Optional but recommended metadata:

- `session_id`
- `source_export_ts`
- `cleaned_run_ts`
- `session_language`
- `target_mode`

## Training semantics

- Context messages are prompt context.
- `target_text` is the supervised assistant output when `target_message` is absent.
- `target_message` is the supervised assistant output when present (including structured tool-call fields).
- Loss is computed only on assistant target tokens.
- Context tokens are masked with `-100` labels.

## Target modes

- `single_turn`: each assistant turn becomes one target example.
- `merge_consecutive_assistant`: contiguous assistant turns are merged with separator `\n\n`.
- `assistant_decision`: full visible context followed by exactly one native assistant
  decision, which may contain multiple calls or speech. The private HASS experimental
  exporter uses this mode. It requires `assistant_only` loss (or weighted loss with
  zero prompt weight), a valid native target and preservation of every context
  message. Template failures, non-prefix prompt/full tokenizations and context
  overflow fail explicitly instead of dropping/truncating decisions. Increase the
  context budget after measuring the pinned tokenizer; do not infer fit from raw text.

Pin the exact model/tokenizer revision and inspect both call and post-tool speech
views before the first real run. CPU contract tests do not certify a model's native
template. Qwen's `enable_thinking` flag is passed explicitly for both values; a
non-thinking prefix can still contain the template's empty thinking delimiters.
