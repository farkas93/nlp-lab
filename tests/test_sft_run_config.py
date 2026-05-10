from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.nlp_lab.sft.run_config import load_sft_run_config


class SFTRunConfigTests(unittest.TestCase):
    def test_defaults_include_backend_and_cache_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.yaml"
            cfg_path.write_text(
                "\n".join(
                    [
                        "data:",
                        "  dataset_manifest_uri: s3://bucket/manifest.json",
                        "model:",
                        "  model_name: Qwen/Qwen3.5-0.8B",
                        "training:",
                        "  experiment_name: exp",
                        "  run_name: run",
                        "  output_dir: ./outputs",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_sft_run_config(str(cfg_path))
            self.assertEqual(config.training.backend, "trl")
            self.assertEqual(config.data.cache_mode, "reuse")
            self.assertFalse(config.hub.full_model)
            self.assertIsNone(config.hub.full_model_repo_name)
            self.assertIsNone(config.hub.adapter_tag)
            self.assertIsNone(config.hub.full_model_tag)

    def test_unsloth_backend_and_refresh_cache_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.yaml"
            cfg_path.write_text(
                "\n".join(
                    [
                        "data:",
                        "  dataset_manifest_uri: s3://bucket/manifest.json",
                        "  cache_mode: refresh",
                        "model:",
                        "  model_name: Qwen/Qwen3.5-0.8B",
                        "training:",
                        "  backend: unsloth",
                        "  experiment_name: exp",
                        "  run_name: run",
                        "  output_dir: ./outputs",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_sft_run_config(str(cfg_path))
            self.assertEqual(config.training.backend, "unsloth")
            self.assertEqual(config.data.cache_mode, "refresh")

    def test_full_model_hub_fields_parse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.yaml"
            cfg_path.write_text(
                "\n".join(
                    [
                        "data:",
                        "  dataset_manifest_uri: s3://bucket/manifest.json",
                        "model:",
                        "  model_name: Qwen/Qwen3.5-0.8B",
                        "training:",
                        "  experiment_name: exp",
                        "  run_name: run",
                        "  output_dir: ./outputs",
                        "hub:",
                        "  push_to_hub: true",
                        "  repo_name: user/adapter-repo",
                        "  full_model: true",
                        "  full_model_repo_name: user/full-model-repo",
                        "  adapter_tag: v1.0.0",
                        "  full_model_tag: v1.0.0",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_sft_run_config(str(cfg_path))
            self.assertTrue(config.hub.push_to_hub)
            self.assertEqual(config.hub.repo_name, "user/adapter-repo")
            self.assertTrue(config.hub.full_model)
            self.assertEqual(config.hub.full_model_repo_name, "user/full-model-repo")
            self.assertEqual(config.hub.adapter_tag, "v1.0.0")
            self.assertEqual(config.hub.full_model_tag, "v1.0.0")


if __name__ == "__main__":
    unittest.main()
