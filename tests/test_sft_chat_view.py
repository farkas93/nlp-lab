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
