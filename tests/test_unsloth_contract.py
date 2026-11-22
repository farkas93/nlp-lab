import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.eliza_trainer.sft.run_config import load_sft_run_config
from src.eliza_trainer.sft.unsloth_model import load_model, attach_language_adapter, verify_tokenizer

ROOT = Path(__file__).resolve().parents[1]


def test_gemma_unsloth_loader_preserves_pins_and_language_only(monkeypatch):
    config = load_sft_run_config(str(ROOT / 'configs/sft_hass_gemma4_unsloth_qualification.yaml'))
    assert config.training.max_steps == 2 and config.training.qualify_save_reload
    model = Mock()
    model.named_parameters.return_value = [('base_model.model.language_model.layers.0.q_proj.lora_A.default.weight',
                                            SimpleNamespace(requires_grad=True)),
                                           ('base_model.model.vision_tower.weight', SimpleNamespace(requires_grad=False))]
    model.peft_config = {'default': SimpleNamespace(revision=None)}
    tokenizer = Mock()
    fast = Mock()
    fast.from_pretrained.return_value = model, SimpleNamespace(tokenizer=tokenizer)
    fast.get_peft_model.return_value = model
    monkeypatch.setitem(sys.modules, 'unsloth', SimpleNamespace(FastModel=fast))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(bfloat16='bf16'))
    assert load_model(config) == (model, tokenizer)
    kwargs = fast.from_pretrained.call_args.kwargs
    assert kwargs['model_name'] == 'unsloth/gemma-4-E4B-it'
    assert kwargs['revision'] == config.model.revision and kwargs['use_exact_model_name']
    assert not kwargs['fix_tokenizer'] and not kwargs['load_in_4bit'] and not kwargs['fast_inference']
    attach_language_adapter(model, config)
    kwargs = fast.get_peft_model.call_args.kwargs
    assert not kwargs['finetune_vision_layers'] and not kwargs['finetune_audio_layers']
    assert kwargs['finetune_language_layers'] and kwargs['use_gradient_checkpointing'] == 'unsloth'
    model.named_parameters.return_value = [('vision.lora_A.weight', SimpleNamespace(requires_grad=True))]
    with pytest.raises(RuntimeError, match='exclusively language'):
        attach_language_adapter(model, config)


def test_serialization_mismatch_fails_instead_of_silently_changing_template():
    reference = SimpleNamespace(get_vocab=lambda: {'tool': 1}, chat_template='pinned')
    verify_tokenizer(reference, reference)
    with pytest.raises(ValueError, match='differs from pinned'):
        verify_tokenizer(reference, SimpleNamespace(get_vocab=lambda: {'tool': 1}, chat_template='patched'))


def test_unsloth_entry_patches_before_trainer_import(monkeypatch):
    from nlp_training import unsloth_entry
    monkeypatch.setattr(sys, 'argv', ['entry', 'src.eliza_trainer.sft.train', '--config', 'private.yaml'])
    monkeypatch.setitem(sys.modules, 'unsloth', SimpleNamespace())
    called = []
    def invoke(module, run_name):
        assert 'unsloth' in sys.modules
        assert sys.argv == [module, '--config', 'private.yaml']
        called.append(module)
    monkeypatch.setattr(unsloth_entry.runpy, 'run_module', invoke)
    unsloth_entry.main()
    assert called == ['src.eliza_trainer.sft.train']
