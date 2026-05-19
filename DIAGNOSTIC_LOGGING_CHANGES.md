# Diagnostic Logging Enhancements

## Summary

Added comprehensive diagnostic logging to quickly identify dataset loading and tokenization issues in nlp-lab training runs.

## Changes Made

### 1. **Runtime Configuration - Log Level Support**

Added `log_level` parameter to runtime configuration:

**Files Modified:**
- `src/eliza_trainer/sft/run_config.py`
- `src/eliza_trainer/dpo/run_config.py`
- `src/eliza_trainer/common/runtime.py`

**Usage in Config:**
```yaml
runtime:
  hf_model_cache_dir: ./hf_models
  log_level: DEBUG  # Options: DEBUG, INFO, WARNING, ERROR
```

**Default:** `INFO` if not specified

### 2. **Location 1: Dataset Load Diagnostics**

**File:** `src/sft_dataset_loader.py` (after line 160)

**What it logs (DEBUG level):**
- Raw dataset row counts after Parquet load
- Number of files loaded
- First row schema validation
- Message structure inspection
- Warning if dataset is empty after load

**Example Output:**
```
DEBUG: Loaded raw datasets from manifest: train_rows=266 eval_rows=44 train_files=1 eval_files=1
DEBUG: First train row keys: ['messages', 'target_text', 'session_id', 'example_id'], has_messages=True, has_target_text=True
DEBUG: First message structure: keys=['role', 'content', 'tool_calls'] types={'role': 'str', 'content': 'str', 'tool_calls': 'list'}
```

### 3. **Location 2: Tokenization Drop Diagnostics**

**File:** `src/sft_dataset_loader.py` (after line 449)

**What it logs:**
- DEBUG: Tokenization statistics (raw/valid/dropped/truncated counts)
- ERROR: Critical alert when ALL rows are dropped
- ERROR: Sample of dropped rows with reasons
- ERROR: Detailed drop reason breakdown

**Example Output:**
```
DEBUG: Tokenization stats for split=train: raw_rows=266 valid_rows=0 dropped=266 fit_fully=0 truncated=0
ERROR: CRITICAL: All 266 rows dropped during tokenization for split=train! Reasons: {
  "chat_template_error": 266
}
ERROR: Sample of dropped rows (first 3):
  Row 0: reason=chat_template_error, example_id=abc-123, session_id=xyz-789
  Row 1: reason=chat_template_error, example_id=def-456, session_id=uvw-012
  Row 2: reason=chat_template_error, example_id=ghi-789, session_id=rst-345
```

### 4. **Location 3: Train.py Enhanced Logging**

**File:** `src/eliza_trainer/sft/train.py` (after line 312)

**What it logs:**
- INFO: Tokenization flow summary (raw→final with drop counts)
- WARNING: Drop reasons if any rows were dropped
- ERROR: Early failure with clear message if training dataset is empty

**Example Output:**
```
INFO: Tokenization complete: train_raw=266->0 (dropped=266) eval_raw=44->0 (dropped=44)
WARNING: Train tokenization dropped 266/266 rows (100.0%). Reasons: {"chat_template_error": 266}
ERROR: Training dataset is empty after tokenization! Raw rows: 266, Dropped: 266, Reasons: {'chat_template_error': 266}. Check dataset schema compatibility with tokenizer.
ValueError: Training dataset is empty after tokenization! ...
```

**Benefits:**
- Clear error BEFORE hitting confusing PyTorch sampler error
- Actionable diagnostics with drop reasons
- No more cryptic "num_samples=0" errors

### 5. **Location 4: Tokenizer Compatibility Test**

**File:** `src/eliza_trainer/sft/train.py` (after line 248)

**What it logs (DEBUG level):**
- Tests tokenizer with a simple message structure
- Early warning if tokenizer doesn't support chat templates
- Helps identify incompatible models early

**Example Output:**
```
DEBUG: Testing tokenizer chat template compatibility...
DEBUG: Tokenizer chat template test passed: <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user...
```

Or if it fails:
```
DEBUG: Testing tokenizer chat template compatibility...
ERROR: Tokenizer chat template test FAILED: ... [error details]
ERROR: This tokenizer may not support your dataset's message format
```

## How to Use

### Enable Debug Logging

Add to your training config:

```yaml
runtime:
  hf_model_cache_dir: ./hf_models
  log_level: DEBUG
```

### Run Training

```bash
./start_sft.sh configs/your_config.yaml
```

### Interpret Output

**If you see:**
```
ERROR: CRITICAL: All X rows dropped during tokenization for split=train!
```

**Check the reasons:**
- `chat_template_error`: Tokenizer doesn't support your message format (tool calls, etc.)
- `invalid_row_shape`: Missing `messages` or `target_text` fields
- `missing_user_query`: No user messages in conversation
- `empty_prompt_messages`: All messages were filtered out

**Common Issues:**

| Error Reason | Likely Cause | Solution |
|--------------|--------------|----------|
| `chat_template_error` | Tokenizer incompatible with tool calls | Use a model that supports tool calls (Llama 3.3, Qwen 2.5, etc.) |
| `invalid_row_shape` | Dataset schema mismatch | Check manifest - dataset may be corrupted |
| `missing_user_query` | Dataset has only system/assistant messages | Review dataset generation in eliza-data-pipelines |
| `empty_prompt_messages` | Message normalization failed | Check for malformed message dicts |

## Files Modified

1. `src/eliza_trainer/common/runtime.py` - Added log_level parameter to configure_logging()
2. `src/eliza_trainer/sft/run_config.py` - Added log_level to SFTRuntimeConfig
3. `src/eliza_trainer/dpo/run_config.py` - Added log_level to DPORuntimeConfig
4. `src/eliza_trainer/sft/train.py` - Added Location 3, Location 4, log level reconfiguration
5. `src/eliza_trainer/dpo/train.py` - Added log level reconfiguration
6. `src/sft_dataset_loader.py` - Added Location 1, Location 2
7. `configs/sft_hass_qwen3_5_0_8b.yaml` - Example with DEBUG logging enabled

## Testing

To verify the changes work, run a training job that you know has issues:

```bash
cd /home/wseliza/private/nlp-lab
./start_sft.sh configs/sft_hass_qwen3_5_0_8b.yaml
```

You should now see:
1. ✅ Debug logs showing dataset load details
2. ✅ Clear error messages if rows are dropped
3. ✅ Actionable diagnostics before PyTorch errors
4. ✅ Sample dropped rows with reasons

## Rollback

If you need to disable DEBUG logging, simply change:

```yaml
runtime:
  log_level: INFO  # or remove the line entirely (defaults to INFO)
```

Or set environment variable:
```bash
export LOG_LEVEL=INFO
```
