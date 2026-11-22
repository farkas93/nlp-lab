import copy
import hashlib
import json
import os

import pytest

from eliza_trainer.sft.chat_view import normalize_message, template_tools, require_target_prefix


CALL = {"id": "decision-call-1", "type": "function", "function": {
    "name": "HassTurnOff", "arguments": '{"name":"Kitchen light"}'}}
TOOLS = [{"type": "function", "function": {"name": "HassTurnOff", "parameters": {
    "type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}}]


def decision_row():
    return {"messages": [{"role": "system", "content": "Use advertised entities."},
                         {"role": "user", "content": "Turn off the kitchen light."},
                         {"role": "assistant", "content": None, "tool_calls": [copy.deepcopy(CALL)]},
                         {"role": "tool", "content": '{"success":true}', "tool_call_id": CALL["id"], "name": "HassTurnOff"}],
            "target_message": {"role": "assistant", "content": "The kitchen light is off."},
            "target_text": "The kitchen light is off.", "target_mode": "assistant_decision",
            "tools_json": json.dumps(TOOLS)}


def test_null_call_and_result_identity_survive_semantic_view():
    row = decision_row()
    assert normalize_message(row["messages"][2], lambda calls: calls) == row["messages"][2]
    assert normalize_message(row["messages"][3], lambda calls: calls) == row["messages"][3]
    tools = template_tools(row)
    tools[0]["function"]["name"] = "changed"
    assert template_tools(row) == TOOLS


@pytest.mark.parametrize("prompt,full", [([1, 2], [1, 3, 4]), ([1, 2], [1, 2])])
def test_target_boundary_rejects_prefix_mismatch_and_empty_supervision(prompt, full):
    with pytest.raises(ValueError):
        require_target_prefix(prompt, full)


class NativeTokenizer:
    name_or_path = "Qwen/test-template"

    def __init__(self):
        self.views = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, tools, enable_thinking):
        assert tokenize is False and enable_thinking is False
        assert tools == TOOLS
        self.views.append(copy.deepcopy(messages))
        text = "".join(item["role"] + ":" + (item.get("content") or "") +
                       json.dumps(item.get("tool_calls") or []) + "\n" for item in messages)
        return text + "assistant:" if add_generation_prompt else text

    def encode(self, text, add_special_tokens):
        return list(text.encode())


@pytest.fixture
def tokenization():
    datasets = pytest.importorskip("datasets")
    from src.sft_dataset_loader import tokenize_with_loss_mode
    return datasets.Dataset, tokenize_with_loss_mode


def test_native_history_target_tools_and_mask_are_preserved(tokenization):
    Dataset, tokenize = tokenization
    tokenizer = NativeTokenizer()
    row = decision_row()
    result = tokenize(Dataset.from_list([row]), tokenizer, 4096, "assistant_only", enable_thinking=False)[0]
    history = tokenizer.views[0]
    assert len(history) == len(row["messages"])
    call = history[2]["tool_calls"][0]
    assert history[2]["content"] is None
    assert call["id"] == history[3]["tool_call_id"] == CALL["id"]
    assert call["arguments"] == {"name": "Kitchen light"}
    assert tokenizer.views[1][-1]["content"] == row["target_text"]
    first_target = next(index for index, value in enumerate(result["labels"]) if value != -100)
    assert all(value == -100 for value in result["labels"][:first_target])
    assert bytes(result["input_ids"][first_target:]).startswith(b"The kitchen light is off.")
    assert result["labels"][first_target:] == result["input_ids"][first_target:]


def test_null_content_call_is_the_target_instead_of_legacy_text(tokenization):
    Dataset, tokenize = tokenization
    row = decision_row()
    row["target_message"] = copy.deepcopy(row["messages"][2])
    row["target_message"]["tool_calls"][0]["id"] = "decision-call-2"
    row["target_text"] = "This legacy fallback must not be trained."
    tokenizer = NativeTokenizer()
    result = tokenize(Dataset.from_list([row]), tokenizer, 4096, "assistant_only", enable_thinking=False)[0]
    assert tokenizer.views[1][-1]["content"] is None
    assert tokenizer.views[1][-1]["tool_calls"][0]["id"] == "decision-call-2"
    trained = bytes(value for value in result["labels"] if value != -100)
    assert b"HassTurnOff" in trained and b"legacy fallback" not in trained


def test_malformed_arguments_cannot_be_silently_replaced(tokenization):
    Dataset, tokenize = tokenization
    row = decision_row()
    row["messages"][2]["tool_calls"][0]["function"]["arguments"] = "not JSON"
    with pytest.raises(ValueError):
        tokenize(Dataset.from_list([row]), NativeTokenizer(), 4096, "assistant_only", enable_thinking=False)


@pytest.mark.parametrize("loss_mode,max_length", [("full_conversation", 4096), ("assistant_only", 8)])
def test_decision_view_refuses_prompt_loss_and_context_truncation(tokenization, loss_mode, max_length):
    Dataset, tokenize = tokenization
    with pytest.raises(ValueError):
        tokenize(Dataset.from_list([decision_row()]), NativeTokenizer(), max_length, loss_mode, enable_thinking=False)


@pytest.mark.skipif(os.getenv("HASS_PINNED_TOKENIZER_TEST") != "1", reason="opt-in checkpoint tokenizer download")
def test_pinned_qwen_native_call_and_post_tool_speech(tokenization):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", revision="2fc06364715b967f1860aea9cf38778875588b17")
    Dataset, tokenize = tokenization
    speech = decision_row()
    action = copy.deepcopy(speech)
    action["messages"] = action["messages"][:2]
    action["target_message"] = speech["messages"][2]
    rows = tokenize(Dataset.from_list([action, speech]), tokenizer, 4096, "assistant_only", enable_thinking=False)
    targets = [tokenizer.decode([value for value in row["labels"] if value != -100]) for row in rows]
    assert "<function=HassTurnOff>" in targets[0] and "Kitchen light" in targets[0]
    assert targets[1].startswith(speech["target_text"])
    assert all("<think>" not in target for target in targets)

    measured = tokenize(Dataset.from_list([action, speech]), tokenizer, 32, "assistant_only",
                        enable_thinking=False, measure_only=True)
    assert measured.column_names == ["example_id", "episode_id", "prompt_tokens", "target_tokens", "total_tokens"]
    for stats, row in zip(measured, rows, strict=True):
        assert stats["total_tokens"] == len(row["input_ids"]) > 32
        assert stats["prompt_tokens"] == row["labels"].count(-100)
        assert stats["target_tokens"] == len(row["labels"]) - row["labels"].count(-100)


def test_hass_loader_consumes_only_pinned_parquet_bytes(tmp_path, tokenization):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from src.sft_dataset_loader import load_sft_manifest_dataset

    dataset_id, version = "hass_pilot_experimental", "experimental-" + "a" * 64
    prefix = f"dataset_id={dataset_id}/dataset_version={version}"
    entries = []
    for split in ("train", "eval"):
        path = tmp_path / f"split={split}/part-00000.parquet"
        path.parent.mkdir()
        pq.write_table(pa.Table.from_pylist([decision_row()]), path)
        entries.append({"split": split, "key": f"{prefix}/split={split}/part-00000.parquet",
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": 1})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"kind": "hass-experimental-sft-v1", "dataset_id": dataset_id,
                                    "dataset_version": version, "splits": entries}) + "\r\n")
    args = dict(manifest_uri=str(manifest), train_split="train", eval_split="eval", max_train_samples=None,
                max_eval_samples=None, expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    result = load_sft_manifest_dataset(**args)
    assert result.train_dataset[0]["messages"][2]["content"] is None
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="split byte checksum"):
        load_sft_manifest_dataset(**args)


def test_household_loader_accepts_multiple_verified_shards(tmp_path, tokenization):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from src.sft_dataset_loader import load_sft_manifest_dataset
    prefix = 'dataset_id=household/dataset_version=household-test'
    entries = []
    for split in ('train', 'eval'):
        for part in range(2):
            path = tmp_path / f'split={split}/part-{part:05d}.parquet'
            path.parent.mkdir(exist_ok=True)
            pq.write_table(pa.Table.from_pylist([decision_row()] * 40), path)
            entries.append(dict(split=split, key=f'{prefix}/split={split}/{path.name}', rows=40,
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(kind='hass-household-sft-v1', dataset_id='household',
                                       dataset_version='household-test', splits=entries)))
    args = dict(manifest_uri=str(manifest), train_split='train', eval_split='eval', max_train_samples=None,
                max_eval_samples=None, expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    loaded = load_sft_manifest_dataset(**args)
    assert len(loaded.train_dataset) == len(loaded.eval_dataset) == 80
    path.write_bytes(path.read_bytes() + b'corrupt')
    with pytest.raises(ValueError, match='checksum mismatch'):
        load_sft_manifest_dataset(**args)


@pytest.mark.skipif(os.getenv("HASS_PINNED_TOKENIZER_TEST") != "1", reason="opt-in tokenizer downloads")
@pytest.mark.parametrize("model,revision,kind", [
    ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a", "qwen"),
    ("google/gemma-4-E4B-it", "4ecc2ba63bcac23ba4a767fc87025b1450f84c9c", "gemma4"),
    ("unsloth/gemma-4-E4B-it", "4e22d7e59e078e63a14f351efdc5232ed366b621", "gemma4"),
])
def test_comparison_checkpoint_templates_preserve_call_and_speech(tokenization, model, revision, kind):
    from transformers import AutoTokenizer
    Dataset, tokenize = tokenization
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    speech = decision_row()
    action = copy.deepcopy(speech)
    action["messages"] = speech["messages"][:2]
    action["target_message"] = speech["messages"][2]
    rows = tokenize(Dataset.from_list([action, speech]), tokenizer, 131072, "assistant_only",
                    tokenizer_type=kind, enable_thinking=False)
    texts = [tokenizer.decode([value for value in row["labels"] if value != -100]) for row in rows]
    assert "HassTurnOff" in texts[0] and "Kitchen light" in texts[0]
    assert speech["target_text"] in texts[1]
