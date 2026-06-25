from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from huggingface_hub import HfApi
from huggingface_hub import login
from transformers import AutoTokenizer

from ..common.data_policy import (
    confirm_or_abort,
    is_push_blocked_for_p4,
    requires_confirmation,
    summarize_manifest_policy,
)
from ..common.runtime import configure_logging, ensure_cuda_alloc_conf, load_project_env
from ..common.system_telemetry import (
    SystemTelemetrySampler,
    build_system_telemetry_artifact,
    collect_system_fingerprint,
    load_telemetry_config_from_env,
)
from .backends.trl_backend import run_trl_dpo_training
from .dataset_loader import load_dpo_manifest_dataset
from .run_config import apply_tracking_env, load_dpo_run_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DPO from manifest-based parquet dataset")
    parser.add_argument(
        "--config",
        default="configs/dpo_hass_qwen3_5_0_8b.yaml",
        help="Path to DPO YAML config",
    )
    parser.add_argument(
        "--dry-run-config",
        action="store_true",
        help="Resolve and validate config only, then exit",
    )
    parser.add_argument(
        "--assume-yes",
        action="store_true",
        help="Auto-confirm policy gates that would otherwise require interactive approval",
    )
    return parser.parse_args()


def _mlflow_log_dataset_lineage(manifest_uri: str, manifest_sha256: str, manifest: dict) -> None:
    mlflow.log_param("dataset_manifest_uri", manifest_uri)
    mlflow.log_param("dataset_manifest_sha256", manifest_sha256)
    mlflow.log_param("dataset_id", manifest.get("dataset_id"))
    mlflow.log_param("dataset_version", manifest.get("dataset_version"))

    construction = (
        manifest.get("construction") if isinstance(manifest.get("construction"), dict) else {}
    )
    for key in ("split_seed", "eval_percent"):
        if key in construction:
            mlflow.log_param(f"dataset_{key}", construction[key])

    stats = manifest.get("stats") if isinstance(manifest.get("stats"), dict) else {}
    for key in ("train_rows", "eval_rows", "total_rows"):
        if key in stats:
            mlflow.log_metric(key, float(stats[key]))


def _validate_manifest_expectation(expected_sha256: str | None, actual_sha256: str) -> None:
    if not expected_sha256:
        return
    expected = expected_sha256.strip().lower()
    actual = actual_sha256.strip().lower()
    if expected != actual:
        raise RuntimeError(
            "Dataset manifest SHA256 mismatch: "
            f"expected={expected} actual={actual}. "
            "Update data.expected_manifest_sha256 or refresh dataset manifest reference."
        )


def _copy_config_snapshot(config_path: str, artifacts_dir: Path) -> Path | None:
    try:
        source = Path(config_path)
        if not source.exists():
            return None
        target = artifacts_dir / "effective_config.yaml"
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        return target
    except Exception:
        logging.warning("Failed to capture config snapshot for diagnostics", exc_info=True)
        return None


def _write_crash_report(
    *,
    artifacts_dir: Path,
    config_path: str,
    backend: str | None,
    run_name: str | None,
    run_id: str | None,
    exc: BaseException,
) -> Path | None:
    try:
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "config_path": config_path,
            "backend": backend,
            "run_name": run_name,
            "mlflow_run_id": run_id,
            "host_log_file": os.getenv("DPO_RUN_LOG_FILE") or None,
        }
        path = artifacts_dir / "crash_report.json"
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        return path
    except Exception:
        logging.warning("Failed to write crash report diagnostics", exc_info=True)
        return None


def _log_diagnostics_artifacts_to_mlflow(run_id: str | None, artifact_paths: list[Path]) -> None:
    if not run_id:
        return

    existing = [path for path in artifact_paths if isinstance(path, Path) and path.exists()]
    if not existing:
        return

    try:
        client = mlflow.tracking.MlflowClient()
        for path in existing:
            client.log_artifact(run_id, str(path), artifact_path="run_diagnostics")
    except Exception:
        logging.warning(
            "Failed to upload one or more run diagnostics artifacts to MLflow",
            exc_info=True,
        )


def _mlflow_log_system_fingerprint(fingerprint: dict[str, object]) -> None:
    for key, value in sorted(fingerprint.items()):
        if value is None:
            continue
        mlflow.log_param(f"system_{key}", value)


def _log_telemetry_metrics_to_mlflow(run_id: str | None, metrics: dict[str, float]) -> None:
    if not run_id or not metrics:
        return
    try:
        client = mlflow.tracking.MlflowClient()
        for key, value in sorted(metrics.items()):
            client.log_metric(run_id, key, float(value))
    except Exception:
        logging.warning("Failed to log system telemetry metrics to MLflow", exc_info=True)


def _detect_source_repo_lineage(model_repo: str) -> str:
    try:
        files = set(HfApi().list_repo_files(repo_id=model_repo, repo_type="model"))
    except Exception:
        return "unknown"

    has_adapter = "adapter_config.json" in files or any(
        name.startswith("adapter_model") for name in files
    )
    has_full = "config.json" in files and any(
        name.startswith("model.safetensors") or name.startswith("pytorch_model")
        for name in files
    )
    if has_adapter and not has_full:
        return "adapter"
    if has_full and not has_adapter:
        return "full"
    if has_adapter and has_full:
        return "mixed"
    return "unknown"


def main() -> None:
    # Initial logging setup with INFO level
    configure_logging()
    ensure_cuda_alloc_conf()
    args = parse_args()

    diagnostics_dir = Path(tempfile.mkdtemp(prefix="dpo_run_diagnostics_"))
    diagnostics_paths: list[Path] = []
    config_snapshot = _copy_config_snapshot(args.config, diagnostics_dir)
    if config_snapshot:
        diagnostics_paths.append(config_snapshot)
    log_file_env = str(os.getenv("DPO_RUN_LOG_FILE") or "").strip()
    if log_file_env:
        diagnostics_paths.append(Path(log_file_env))
    run_id: str | None = None
    backend: str | None = None
    run_name: str | None = None
    telemetry_sampler: SystemTelemetrySampler | None = None
    telemetry_summary_metrics: dict[str, float] = {}
    telemetry_fingerprint: dict[str, object] = {}

    try:
        project_root = Path(__file__).resolve().parents[3]
        load_project_env(project_root)

        run_config = load_dpo_run_config(args.config)
        
        # Reconfigure logging with config-specified level
        configure_logging(run_config.runtime.log_level)
        
        apply_tracking_env(run_config)
        backend = run_config.training.backend
        run_name = run_config.training.run_name
        source_repo_lineage = _detect_source_repo_lineage(run_config.model.model_name)

        telemetry_config = load_telemetry_config_from_env()
        if telemetry_config.enabled:
            telemetry_fingerprint = collect_system_fingerprint(
                log_raw_hostname=telemetry_config.log_raw_hostname,
            )
            telemetry_sampler = SystemTelemetrySampler(
                interval_sec=telemetry_config.interval_sec,
            )
            telemetry_sampler.start()
            telemetry_fingerprint["host_name_mode"] = (
                "raw" if telemetry_config.log_raw_hostname else "anonymized"
            )
            telemetry_fingerprint["sampling_interval_sec"] = telemetry_config.interval_sec

        logging.info(
            "Resolved DPO run plan model=%s backend=%s experiment=%s run_name=%s output_dir=%s manifest_uri=%s adapter_repo=%s full_repo=%s source_repo_lineage=%s",
            run_config.model.model_name,
            run_config.training.backend,
            run_config.training.experiment_name,
            run_config.training.run_name,
            run_config.training.output_dir,
            run_config.data.manifest_uri,
            run_config.hub.adapter_repo_name,
            run_config.hub.full_model_repo_name,
            source_repo_lineage,
        )

        if args.dry_run_config:
            logging.info("Dry run successful; exiting before tokenizer/dataset/training")
            return

        hf_token = os.environ.get("HF_HUB_TOKEN")
        if hf_token:
            login(token=hf_token)

        tokenizer = AutoTokenizer.from_pretrained(
            run_config.model.model_name,
            add_eos_token=True,
            use_fast=True,
            cache_dir=run_config.hf_model_cache_dir,
        )
        # Only set pad_token if not already defined
        # Some models (e.g., Qwen) have distinct pad_token and eos_token by design
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logging.info(f"Set pad_token to eos_token: {tokenizer.eos_token!r} (ID: {tokenizer.eos_token_id})")
        else:
            logging.info(
                f"Using existing pad_token: {tokenizer.pad_token!r} (ID: {tokenizer.pad_token_id}), "
                f"eos_token: {tokenizer.eos_token!r} (ID: {tokenizer.eos_token_id})"
            )
        tokenizer.padding_side = "left"

        dataset_result = load_dpo_manifest_dataset(
            manifest_uri=run_config.data.manifest_uri,
            train_split=run_config.data.train_split,
            eval_split=run_config.data.eval_split,
            max_train_samples=run_config.data.max_train_samples,
            max_eval_samples=run_config.data.max_eval_samples,
            cache_mode=run_config.data.cache_mode,
        )
        _validate_manifest_expectation(
            run_config.data.expected_manifest_sha256,
            dataset_result.manifest_sha256,
        )
        policy_summary = summarize_manifest_policy(dataset_result.manifest)
        logging.info(
            "Dataset policy summary has_metadata=%s classes=%s max=%s counts=%s",
            policy_summary.has_metadata,
            policy_summary.classes_present,
            policy_summary.max_policy_class,
            policy_summary.class_counts,
        )
        if run_config.hub.push_to_hub and is_push_blocked_for_p4(policy_summary):
            raise RuntimeError(
                "Refusing to continue: dataset contains policy class P4 and hub.push_to_hub=true. "
                "Disable hub.push_to_hub or use a release manifest without P4."
            )
        if requires_confirmation(policy_summary):
            logging.warning(
                "Dataset includes high-restriction policy classes (P3/P4). "
                "Explicit confirmation is required unless --assume-yes is set."
            )
            confirm_or_abort(policy_summary, assume_yes=args.assume_yes, context="DPO training")

        logging.info(
            "Prepared DPO dataset backend=%s manifest_sha256=%s train_rows=%s eval_rows=%s cache_mode=%s",
            run_config.training.backend,
            dataset_result.manifest_sha256,
            len(dataset_result.train_dataset),
            len(dataset_result.eval_dataset),
            run_config.data.cache_mode,
        )

        mlflow.set_experiment(run_config.training.experiment_name)
        with mlflow.start_run(run_name=run_config.training.run_name) as active_run:
            run_id = active_run.info.run_id
            _mlflow_log_dataset_lineage(
                dataset_result.manifest_uri,
                dataset_result.manifest_sha256,
                dataset_result.manifest,
            )
            mlflow.log_param("model_name", run_config.model.model_name)
            mlflow.log_param("source_repo_lineage", source_repo_lineage)
            mlflow.log_param(
                "dataset_policy_classes",
                ",".join(policy_summary.classes_present) if policy_summary.classes_present else "unknown",
            )
            mlflow.log_param("dataset_policy_max_class", policy_summary.max_policy_class)
            mlflow.log_param(
                "dataset_policy_counts",
                json.dumps(policy_summary.class_counts, ensure_ascii=True, sort_keys=True),
            )
            mlflow.log_param("backend", run_config.training.backend)
            mlflow.log_param("cache_mode", run_config.data.cache_mode)
            mlflow.log_param("system_telemetry_enabled", bool(telemetry_config.enabled))
            if telemetry_config.enabled:
                _mlflow_log_system_fingerprint(telemetry_fingerprint)
            mlflow.log_param("dpo_beta", run_config.dpo.beta)
            mlflow.log_param("dpo_max_prompt_length", run_config.dpo.max_prompt_length)
            mlflow.log_param("dpo_max_length", run_config.dpo.max_length)
            mlflow.log_metric("train_rows_effective", float(len(dataset_result.train_dataset)))
            mlflow.log_metric("eval_rows_effective", float(len(dataset_result.eval_dataset)))

            run_trl_dpo_training(
                run_config=run_config,
                tokenizer=tokenizer,
                train_dataset=dataset_result.train_dataset,
                eval_dataset=dataset_result.eval_dataset,
            )
    except Exception as exc:
        crash_report = _write_crash_report(
            artifacts_dir=diagnostics_dir,
            config_path=args.config,
            backend=backend,
            run_name=run_name,
            run_id=run_id,
            exc=exc,
        )
        if crash_report:
            diagnostics_paths.append(crash_report)
        logging.exception("DPO run crashed")
        raise
    finally:
        if telemetry_sampler is not None:
            telemetry_summary_metrics = telemetry_sampler.stop()
            telemetry_path = diagnostics_dir / "system_telemetry.json"
            telemetry_path.write_text(
                build_system_telemetry_artifact(
                    fingerprint=telemetry_fingerprint,
                    summary_metrics=telemetry_summary_metrics,
                ),
                encoding="utf-8",
            )
            diagnostics_paths.append(telemetry_path)
        _log_telemetry_metrics_to_mlflow(run_id, telemetry_summary_metrics)
        _log_diagnostics_artifacts_to_mlflow(run_id, diagnostics_paths)


if __name__ == "__main__":
    main()
