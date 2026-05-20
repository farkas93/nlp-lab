"""Unit tests for tool call normalizers.

Tests the tokenizer-aware normalization system that adapts tool call format
based on target tokenizer requirements.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import Mock

from src.sft_dataset_loader import (
    QwenNormalizer,
    GemmaNormalizer,
    OpenAINormalizer,
    GenericNormalizer,
    detect_tokenizer_type,
    get_normalizer,
)


class QwenNormalizerTests(unittest.TestCase):
    """Tests for QwenNormalizer - Qwen family tokenizers."""
    
    def setUp(self):
        self.normalizer = QwenNormalizer()
    
    def test_keeps_arguments_as_dict(self):
        """Qwen should keep arguments as dict for .items() iteration."""
        tool_calls = [
            {
                "name": "turn_on_lights",
                "arguments": {"room": "living_room", "brightness": 80}
            }
        ]
        
        result = self.normalizer.normalize_tool_calls(tool_calls)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "turn_on_lights")
        self.assertIsInstance(result[0]["arguments"], dict)
        self.assertEqual(result[0]["arguments"]["room"], "living_room")
        self.assertEqual(result[0]["arguments"]["brightness"], 80)
    
    def test_does_not_stringify_arguments(self):
        """Qwen normalizer should report it doesn't stringify arguments."""
        self.assertFalse(self.normalizer.should_stringify_arguments)
    
    def test_flattens_nested_function_format(self):
        """Qwen should convert nested (OpenAI) format to flat format."""
        tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "search_web",
                    "arguments": {"query": "weather"}
                }
            }
        ]
        
        result = self.normalizer.normalize_tool_calls(tool_calls)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "search_web")
        self.assertIsInstance(result[0]["arguments"], dict)
        self.assertNotIn("function", result[0])
        self.assertNotIn("id", result[0])
        self.assertNotIn("type", result[0])
    
    def test_handles_empty_arguments(self):
        """Qwen should handle tool calls with no arguments."""
        tool_calls = [
            {"name": "get_time"}
        ]
        
        result = self.normalizer.normalize_tool_calls(tool_calls)
        
        self.assertEqual(result[0]["name"], "get_time")
        self.assertEqual(result[0]["arguments"], {})
    
    def test_preserves_complex_argument_types(self):
        """Qwen should preserve nested dicts and lists in arguments."""
        tool_calls = [
            {
                "name": "create_event",
                "arguments": {
                    "title": "Meeting",
                    "attendees": ["alice", "bob"],
                    "details": {"location": "Room A", "priority": 1}
                }
            }
        ]
        
        result = self.normalizer.normalize_tool_calls(tool_calls)
        
        args = result[0]["arguments"]
        self.assertEqual(args["title"], "Meeting")
        self.assertEqual(args["attendees"], ["alice", "bob"])
        self.assertEqual(args["details"]["location"], "Room A")
    
    def test_name_property(self):
        """Normalizer should report correct name."""
        self.assertEqual(self.normalizer.name, "QwenNormalizer")


class GemmaNormalizerTests(unittest.TestCase):
    """Tests for GemmaNormalizer - Google Gemma family."""
    
    def setUp(self):
        self.normalizer = GemmaNormalizer()
    
    def test_keeps_arguments_as_dict(self):
        """Gemma should keep arguments as dict like Qwen."""
        tool_calls = [
            {
                "name": "calculate",
                "arguments": {"expression": "2 + 2"}
            }
        ]
        
        result = self.normalizer.normalize_tool_calls(tool_calls)
        
        self.assertIsInstance(result[0]["arguments"], dict)
        self.assertEqual(result[0]["arguments"]["expression"], "2 + 2")
    
    def test_does_not_stringify_arguments(self):
        """Gemma normalizer should not stringify arguments."""
        self.assertFalse(self.normalizer.should_stringify_arguments)
    
    def test_flattens_nested_format(self):
        """Gemma should flatten nested function format."""
        tool_calls = [
            {
                "function": {
                    "name": "test_tool",
                    "arguments": {"param": "value"}
                }
            }
        ]
        
        result = self.normalizer.normalize_tool_calls(tool_calls)
        
        self.assertEqual(result[0]["name"], "test_tool")
        self.assertNotIn("function", result[0])
    
    def test_name_property(self):
        """Normalizer should report correct name."""
        self.assertEqual(self.normalizer.name, "GemmaNormalizer")


class OpenAINormalizerTests(unittest.TestCase):
    """Tests for OpenAINormalizer - OpenAI/GPT family."""
    
    def setUp(self):
        self.normalizer = OpenAINormalizer()
    
    def test_converts_arguments_to_json_string(self):
        """OpenAI should convert arguments dict to JSON string."""
        tool_calls = [
            {
                "name": "turn_on_lights",
                "arguments": {"room": "living_room"}
            }
        ]
        
        result = self.normalizer.normalize_tool_calls(tool_calls)
        
        # Arguments should be JSON string
        self.assertIsInstance(result[0]["function"]["arguments"], str)
        parsed = json.loads(result[0]["function"]["arguments"])
        self.assertEqual(parsed["room"], "living_room")
    
    def test_adds_required_wrapper_fields(self):
        """OpenAI should add id, type, and function wrapper."""
        tool_calls = [
            {
                "name": "search",
                "arguments": {"query": "test"}
            }
        ]
        
        result = self.normalizer.normalize_tool_calls(tool_calls)
        
        self.assertIn("id", result[0])
        self.assertEqual(result[0]["type"], "function")
        self.assertIn("function", result[0])
        self.assertEqual(result[0]["function"]["name"], "search")
    
    def test_generates_deterministic_ids(self):
        """OpenAI should generate consistent IDs for same tool call."""
        tool_calls = [
            {"name": "test", "arguments": {"a": 1}}
        ]
        
        result1 = self.normalizer.normalize_tool_calls(tool_calls)
        result2 = self.normalizer.normalize_tool_calls(tool_calls)
        
        # Same input should produce same ID
        self.assertEqual(result1[0]["id"], result2[0]["id"])
    
    def test_preserves_existing_id(self):
        """OpenAI should preserve existing tool call ID."""
        tool_calls = [
            {
                "id": "custom_id_123",
                "name": "test",
                "arguments": {}
            }
        ]
        
        result = self.normalizer.normalize_tool_calls(tool_calls)
        
        self.assertEqual(result[0]["id"], "custom_id_123")
    
    def test_handles_already_nested_format(self):
        """OpenAI should handle already-nested input gracefully."""
        tool_calls = [
            {
                "id": "call_abc",
                "type": "function",
                "function": {
                    "name": "existing_tool",
                    "arguments": {"key": "value"}
                }
            }
        ]
        
        result = self.normalizer.normalize_tool_calls(tool_calls)
        
        self.assertEqual(result[0]["function"]["name"], "existing_tool")
        # Arguments should be stringified
        self.assertIsInstance(result[0]["function"]["arguments"], str)
    
    def test_should_stringify_arguments_true(self):
        """OpenAI normalizer should report it stringifies arguments."""
        self.assertTrue(self.normalizer.should_stringify_arguments)
    
    def test_name_property(self):
        """Normalizer should report correct name."""
        self.assertEqual(self.normalizer.name, "OpenAINormalizer")
    
    def test_handles_empty_arguments(self):
        """OpenAI should handle empty arguments gracefully."""
        tool_calls = [
            {"name": "no_args_tool"}
        ]
        
        result = self.normalizer.normalize_tool_calls(tool_calls)
        
        self.assertEqual(result[0]["function"]["arguments"], "{}")


class GenericNormalizerTests(unittest.TestCase):
    """Tests for GenericNormalizer - safe fallback."""
    
    def setUp(self):
        self.normalizer = GenericNormalizer()
    
    def test_keeps_arguments_as_dict(self):
        """Generic should keep arguments as dict (safe default)."""
        tool_calls = [
            {
                "name": "tool",
                "arguments": {"param": "value"}
            }
        ]
        
        result = self.normalizer.normalize_tool_calls(tool_calls)
        
        self.assertIsInstance(result[0]["arguments"], dict)
    
    def test_flattens_nested_format(self):
        """Generic should flatten nested format."""
        tool_calls = [
            {
                "function": {
                    "name": "nested_tool",
                    "arguments": {"x": 1}
                }
            }
        ]
        
        result = self.normalizer.normalize_tool_calls(tool_calls)
        
        self.assertEqual(result[0]["name"], "nested_tool")
        self.assertNotIn("function", result[0])
    
    def test_does_not_stringify(self):
        """Generic should not stringify (safe default)."""
        self.assertFalse(self.normalizer.should_stringify_arguments)
    
    def test_name_property(self):
        """Normalizer should report correct name."""
        self.assertEqual(self.normalizer.name, "GenericNormalizer")


class TokenizerDetectionTests(unittest.TestCase):
    """Tests for tokenizer type detection logic."""
    
    def test_detects_qwen_from_name(self):
        """Should detect Qwen family from model name."""
        tokenizer = Mock()
        
        for name in ["Qwen/Qwen3.5-0.8B", "qwen/qwen2-7b", "QWEN/Qwen-72B"]:
            tokenizer.name_or_path = name
            detected = detect_tokenizer_type(tokenizer)
            self.assertEqual(detected, "qwen", f"Failed for {name}")
    
    def test_detects_gemma_from_name(self):
        """Should detect Gemma family from model name."""
        tokenizer = Mock()
        
        for name in ["google/gemma-7b", "GEMMA-2-9B", "gemma-instruct"]:
            tokenizer.name_or_path = name
            detected = detect_tokenizer_type(tokenizer)
            self.assertEqual(detected, "gemma", f"Failed for {name}")
    
    def test_detects_openai_from_name(self):
        """Should detect OpenAI/GPT family from model name."""
        tokenizer = Mock()
        
        for name in ["openai/gpt-4", "GPT-3.5-turbo", "text-davinci-003"]:
            tokenizer.name_or_path = name
            detected = detect_tokenizer_type(tokenizer)
            self.assertEqual(detected, "openai", f"Failed for {name}")
    
    def test_detects_claude_as_openai_style(self):
        """Claude should use OpenAI-style format."""
        tokenizer = Mock()
        tokenizer.name_or_path = "anthropic/claude-3-sonnet"
        
        detected = detect_tokenizer_type(tokenizer)
        
        self.assertEqual(detected, "openai")
    
    def test_explicit_override_takes_precedence(self):
        """Explicit type should override auto-detection."""
        tokenizer = Mock()
        tokenizer.name_or_path = "openai/gpt-4"  # Would detect as openai
        
        detected = detect_tokenizer_type(tokenizer, explicit_type="qwen")
        
        self.assertEqual(detected, "qwen")
    
    def test_explicit_override_normalizes_case(self):
        """Explicit type should be normalized to lowercase."""
        tokenizer = Mock()
        tokenizer.name_or_path = "some/model"
        
        detected = detect_tokenizer_type(tokenizer, explicit_type="QWEN")
        
        self.assertEqual(detected, "qwen")
    
    def test_unknown_model_returns_generic(self):
        """Unknown models should return generic type."""
        tokenizer = Mock()
        tokenizer.name_or_path = "unknown/custom-model-v1"
        
        detected = detect_tokenizer_type(tokenizer)
        
        self.assertEqual(detected, "generic")
    
    def test_handles_missing_name_or_path(self):
        """Should handle tokenizers without name_or_path."""
        tokenizer = Mock(spec=[])  # No name_or_path attribute
        
        detected = detect_tokenizer_type(tokenizer)
        
        self.assertEqual(detected, "generic")


class NormalizerFactoryTests(unittest.TestCase):
    """Tests for get_normalizer factory function."""
    
    def test_returns_qwen_normalizer(self):
        """Should return QwenNormalizer for qwen type."""
        normalizer = get_normalizer("qwen")
        self.assertIsInstance(normalizer, QwenNormalizer)
    
    def test_returns_gemma_normalizer(self):
        """Should return GemmaNormalizer for gemma type."""
        normalizer = get_normalizer("gemma")
        self.assertIsInstance(normalizer, GemmaNormalizer)
    
    def test_returns_openai_normalizer(self):
        """Should return OpenAINormalizer for openai type."""
        normalizer = get_normalizer("openai")
        self.assertIsInstance(normalizer, OpenAINormalizer)
    
    def test_returns_openai_for_gpt_alias(self):
        """Should return OpenAINormalizer for gpt alias."""
        normalizer = get_normalizer("gpt")
        self.assertIsInstance(normalizer, OpenAINormalizer)
    
    def test_returns_generic_for_unknown(self):
        """Should return GenericNormalizer for unknown types."""
        normalizer = get_normalizer("unknown_type")
        self.assertIsInstance(normalizer, GenericNormalizer)
    
    def test_case_insensitive(self):
        """Factory should be case-insensitive."""
        normalizer = get_normalizer("QWEN")
        self.assertIsInstance(normalizer, QwenNormalizer)


class IntegrationTests(unittest.TestCase):
    """Integration tests for the full normalization flow."""
    
    def test_qwen_preserves_your_dataset_format(self):
        """Your custom format should pass through Qwen normalizer unchanged."""
        normalizer = QwenNormalizer()
        
        # Your actual format from eliza-data-pipelines
        your_tool_calls = [
            {
                "name": "HassTurnOff",
                "arguments": {"domain": "light", "device_class": "light"}
            }
        ]
        
        result = normalizer.normalize_tool_calls(your_tool_calls)
        
        # Should be identical structure
        self.assertEqual(result[0]["name"], "HassTurnOff")
        self.assertEqual(result[0]["arguments"]["domain"], "light")
        self.assertIsInstance(result[0]["arguments"], dict)
    
    def test_openai_converts_your_format(self):
        """Your format should be converted to OpenAI format."""
        normalizer = OpenAINormalizer()
        
        your_tool_calls = [
            {
                "name": "HassTurnOff",
                "arguments": {"domain": "light"}
            }
        ]
        
        result = normalizer.normalize_tool_calls(your_tool_calls)
        
        # Should have OpenAI structure
        self.assertIn("function", result[0])
        self.assertIn("id", result[0])
        self.assertEqual(result[0]["type"], "function")
        self.assertEqual(result[0]["function"]["name"], "HassTurnOff")
        # Arguments should be JSON string
        self.assertIsInstance(result[0]["function"]["arguments"], str)
        parsed = json.loads(result[0]["function"]["arguments"])
        self.assertEqual(parsed["domain"], "light")


if __name__ == "__main__":
    unittest.main()
