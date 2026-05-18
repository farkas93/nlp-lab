from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from src.eliza_trainer.sft.run_config import load_sft_run_config


def _write_config(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


class SFTRunConfigTests(unittest.TestCase):
    def test_schema_v2_derives_run_identity_and_repos(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.yaml"
            _write_config(
                cfg_path,
                """
                config_schema_version: 2
                identity:
                  experiment_tag: hass_sft
                  backend: trl
                  run_label: run1
                data:
                  bucket: sft-datasets
                  dataset_id: general
                  dataset_version: v1.0.1
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

            config = load_sft_run_config(str(cfg_path))

            self.assertEqual(config.model.model_name, "Qwen/Qwen3.5-0.8B")
            self.assertEqual(
                config.data.manifest_uri,
                "s3://sft-datasets/dataset_id=general/dataset_version=v1.0.1/manifest.json",
            )
            self.assertEqual(config.training.backend, "trl")
            self.assertTrue(config.training.experiment_name.startswith("qwen3_5_0_8b_hass_sft"))
            self.assertRegex(
                config.training.run_name,
                r"^hass_sft-trl-\d{8}T\d{6}Z-run1$",
            )
            self.assertTrue(config.hub.adapter_repo_name.endswith("-lora"))
            self.assertEqual(config.hub.full_model_repo_name, None)
            self.assertEqual(config.hub.adapter_tag, config.training.run_name)
            self.assertIsNone(config.hub.full_model_tag)

    def test_same_owner_preserves_model_lineage_for_full_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.yaml"
            _write_config(
                cfg_path,
                """
                config_schema_version: 2
                identity:
                  experiment_tag: hass_sft
                  backend: trl
                data:
                  bucket: sft-datasets
                  dataset_id: general
                  dataset_version: v1.0.1
                model:
                  owner: zskalo
                  name: qwen3.5-0.8b-lora
                hub:
                  push_to_hub: true
                  owner: zskalo
                  publish_adapter: false
                  publish_full_model: true
                  full_model_tag_strategy: run_name
                """,
            )

            config = load_sft_run_config(str(cfg_path))
            self.assertEqual(config.hub.adapter_repo_name, None)
            self.assertEqual(config.hub.full_model_repo_name, "zskalo/qwen3.5-0.8b")
            self.assertEqual(config.hub.full_model_tag, config.training.run_name)

    def test_rejects_legacy_model_name_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.yaml"
            _write_config(
                cfg_path,
                """
                config_schema_version: 2
                identity:
                  experiment_tag: hass_sft
                  backend: trl
                data:
                  bucket: sft-datasets
                  dataset_id: general
                  dataset_version: v1.0.1
                model:
                  model_name: Qwen/Qwen3.5-0.8B
                """,
            )

            with self.assertRaisesRegex(ValueError, "Legacy config keys"):
                load_sft_run_config(str(cfg_path))

    def test_requires_schema_v2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.yaml"
            _write_config(
                cfg_path,
                """
                config_schema_version: 1
                identity:
                  experiment_tag: hass_sft
                  backend: trl
                data:
                  bucket: sft-datasets
                  dataset_id: general
                  dataset_version: v1.0.1
                model:
                  owner: Qwen
                  name: Qwen3.5-0.8B
                """,
            )

            with self.assertRaisesRegex(ValueError, "schema version"):
                load_sft_run_config(str(cfg_path))


if __name__ == "__main__":
    unittest.main()
