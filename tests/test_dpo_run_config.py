from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from src.eliza_trainer.dpo.run_config import load_dpo_run_config


def _write_config(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


class DPORunConfigTests(unittest.TestCase):
    def test_schema_v2_derives_run_identity_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.yaml"
            _write_config(
                cfg_path,
                """
                config_schema_version: 2
                identity:
                  experiment_tag: hass_dpo
                  backend: trl
                  run_label: nightly
                data:
                  bucket: dpo-datasets
                  dataset_id: dpo_hass_core_tools
                  dataset_version: v1_internal
                model:
                  owner: Qwen
                  name: Qwen3.5-0.8B
                training:
                  output_root: ./outputs
                hub:
                  push_to_hub: true
                  owner: zskalo
                  publish_adapter: true
                  publish_full_model: false
                """,
            )

            config = load_dpo_run_config(str(cfg_path))

            self.assertEqual(config.model.model_name, "Qwen/Qwen3.5-0.8B")
            self.assertEqual(
                config.data.manifest_uri,
                "s3://dpo-datasets/dataset_id=dpo_hass_core_tools/dataset_version=v1_internal/manifest.json",
            )
            self.assertEqual(config.training.backend, "trl")
            self.assertTrue(config.training.experiment_name.startswith("qwen3_5_0_8b_hass_dpo"))
            self.assertRegex(config.training.run_name, r"^hass_dpo-trl-\d{8}T\d{6}Z-nightly$")
            self.assertRegex(
                config.training.output_dir,
                r"^\./outputs/qwen3_5_0_8b/hass_dpo/trl/\d{8}T\d{6}Z$",
            )
            self.assertTrue(config.hub.adapter_repo_name.endswith("-lora"))
            self.assertIsNone(config.hub.full_model_repo_name)
            self.assertEqual(config.hub.adapter_tag, config.training.run_name)

    def test_rejects_legacy_model_name_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.yaml"
            _write_config(
                cfg_path,
                """
                config_schema_version: 2
                identity:
                  experiment_tag: hass_dpo
                  backend: trl
                data:
                  bucket: dpo-datasets
                  dataset_id: dpo_hass_core_tools
                  dataset_version: v1_internal
                model:
                  model_name: Qwen/Qwen3.5-0.8B
                """,
            )

            with self.assertRaisesRegex(ValueError, "Legacy config keys"):
                load_dpo_run_config(str(cfg_path))

    def test_push_to_hub_requires_owner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.yaml"
            _write_config(
                cfg_path,
                """
                config_schema_version: 2
                identity:
                  experiment_tag: hass_dpo
                  backend: trl
                data:
                  bucket: dpo-datasets
                  dataset_id: dpo_hass_core_tools
                  dataset_version: v1_internal
                model:
                  owner: Qwen
                  name: Qwen3.5-0.8B
                hub:
                  push_to_hub: true
                """,
            )

            with self.assertRaisesRegex(ValueError, "requires hub.owner"):
                load_dpo_run_config(str(cfg_path))

    def test_custom_adapter_tag_strategy_requires_adapter_tag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.yaml"
            _write_config(
                cfg_path,
                """
                config_schema_version: 2
                identity:
                  experiment_tag: hass_dpo
                  backend: trl
                data:
                  bucket: dpo-datasets
                  dataset_id: dpo_hass_core_tools
                  dataset_version: v1_internal
                model:
                  owner: Qwen
                  name: Qwen3.5-0.8B
                hub:
                  push_to_hub: true
                  owner: zskalo
                  publish_adapter: true
                  publish_full_model: false
                  adapter_tag_strategy: custom
                """,
            )

            with self.assertRaisesRegex(ValueError, "adapter_tag must be set"):
                load_dpo_run_config(str(cfg_path))


if __name__ == "__main__":
    unittest.main()
