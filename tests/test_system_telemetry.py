from __future__ import annotations

import os
import platform
import unittest
from unittest import mock

from src.eliza_trainer.common.system_telemetry import (
    collect_system_fingerprint,
    load_telemetry_config_from_env,
)


class SystemTelemetryConfigTests(unittest.TestCase):
    def test_defaults_are_enabled_and_raw_hostname(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = load_telemetry_config_from_env()
        self.assertTrue(cfg.enabled)
        self.assertTrue(cfg.log_raw_hostname)
        self.assertGreater(cfg.interval_sec, 0)

    def test_opt_out_env_values(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "NLP_LAB_LOG_SYSTEM_TELEMETRY": "0",
                "NLP_LAB_LOG_RAW_HOSTNAME": "0",
                "NLP_LAB_SYSTEM_TELEMETRY_INTERVAL_SEC": "7",
            },
            clear=True,
        ):
            cfg = load_telemetry_config_from_env()
        self.assertFalse(cfg.enabled)
        self.assertFalse(cfg.log_raw_hostname)
        self.assertEqual(cfg.interval_sec, 7.0)


class SystemTelemetryFingerprintTests(unittest.TestCase):
    def test_anonymized_hostname_mode(self) -> None:
        fingerprint = collect_system_fingerprint(log_raw_hostname=False)
        host_name = str(fingerprint.get("host_name") or "")
        self.assertTrue(host_name.startswith("sha256:"))

    def test_raw_hostname_mode(self) -> None:
        fingerprint = collect_system_fingerprint(log_raw_hostname=True)
        host_name = str(fingerprint.get("host_name") or "")
        self.assertEqual(host_name, platform.node() or "unknown-host")


if __name__ == "__main__":
    unittest.main()
