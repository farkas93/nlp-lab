from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, DownloadMode

from src.dpo_dataset_loader import load_dpo_manifest_dataset


class DPODatasetLoaderTests(unittest.TestCase):
    def test_load_manifest_dataset_local_parquet_with_limits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_path = tmp_path / "train.parquet"
            eval_path = tmp_path / "eval.parquet"
            manifest_path = tmp_path / "manifest.json"

            train_rows = [
                {"prompt": f"p{i}", "chosen": f"c{i}", "rejected": f"r{i}"}
                for i in range(3)
            ]
            eval_rows = [
                {"prompt": f"ep{i}", "chosen": f"ec{i}", "rejected": f"er{i}"}
                for i in range(2)
            ]
            pq.write_table(pa.Table.from_pylist(train_rows), train_path)
            pq.write_table(pa.Table.from_pylist(eval_rows), eval_path)

            manifest = {
                "dataset_id": "dpo_hass_core_tools",
                "dataset_version": "v1_internal",
                "splits": [
                    {"split": "train", "key": str(train_path)},
                    {"split": "eval", "key": str(eval_path)},
                ],
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            result = load_dpo_manifest_dataset(
                manifest_uri=str(manifest_path),
                train_split="train",
                eval_split="eval",
                max_train_samples=2,
                max_eval_samples=1,
            )

            self.assertEqual(len(result.train_dataset), 2)
            self.assertEqual(len(result.eval_dataset), 1)
            self.assertEqual(result.manifest["dataset_id"], "dpo_hass_core_tools")
            self.assertEqual(result.manifest_uri, str(manifest_path))
            self.assertEqual(len(result.manifest_sha256), 64)

    def test_missing_required_column_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_path = tmp_path / "train.parquet"
            eval_path = tmp_path / "eval.parquet"
            manifest_path = tmp_path / "manifest.json"

            pq.write_table(
                pa.Table.from_pylist([{"prompt": "p", "chosen": "c", "rejected": "r"}]),
                train_path,
            )
            pq.write_table(
                pa.Table.from_pylist([{"prompt": "p", "chosen": "c"}]),
                eval_path,
            )

            manifest = {
                "dataset_id": "dpo_hass_core_tools",
                "dataset_version": "v1_internal",
                "splits": [
                    {"split": "train", "key": str(train_path)},
                    {"split": "eval", "key": str(eval_path)},
                ],
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Missing required DPO columns"):
                load_dpo_manifest_dataset(
                    manifest_uri=str(manifest_path),
                    train_split="train",
                    eval_split="eval",
                    max_train_samples=None,
                    max_eval_samples=None,
                )

    def test_refresh_cache_mode_uses_force_redownload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "manifest.json"
            manifest = {
                "splits": [
                    {"split": "train", "key": "/tmp/train.parquet"},
                    {"split": "eval", "key": "/tmp/eval.parquet"},
                ]
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            sample = Dataset.from_list([{"prompt": "p", "chosen": "c", "rejected": "r"}])

            with mock.patch("src.dpo_dataset_loader._load_parquet_split", return_value=sample) as patched:
                load_dpo_manifest_dataset(
                    manifest_uri=str(manifest_path),
                    train_split="train",
                    eval_split="eval",
                    max_train_samples=None,
                    max_eval_samples=None,
                    cache_mode="refresh",
                )

            self.assertEqual(patched.call_count, 2)
            for call in patched.call_args_list:
                self.assertEqual(call.kwargs["download_mode"], DownloadMode.FORCE_REDOWNLOAD)


if __name__ == "__main__":
    unittest.main()
