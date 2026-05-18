from __future__ import annotations

import unittest
from unittest.mock import patch

from src.eliza_trainer.common import data_policy


class SFTDataPolicyTests(unittest.TestCase):
    def test_summarize_manifest_policy_handles_missing_governance(self) -> None:
        summary = data_policy.summarize_manifest_policy({"dataset_id": "general"})
        self.assertFalse(summary.has_metadata)
        self.assertEqual(summary.classes_present, [])
        self.assertEqual(summary.class_counts, {})
        self.assertEqual(summary.max_policy_class, "unknown")

    def test_summarize_manifest_policy_normalizes_and_sorts(self) -> None:
        manifest = {
            "governance": {
                "policy_classes_present": ["p3", "P1", "P3", "p2"],
                "policy_class_counts": {"p2": 9, "P3": 2, "P1": 3},
                "max_policy_class": "p3",
            }
        }
        summary = data_policy.summarize_manifest_policy(manifest)
        self.assertTrue(summary.has_metadata)
        self.assertEqual(summary.classes_present, ["P1", "P2", "P3"])
        self.assertEqual(summary.class_counts, {"P2": 9, "P3": 2, "P1": 3})
        self.assertEqual(summary.max_policy_class, "P3")

    def test_requires_confirmation_and_push_block(self) -> None:
        summary = data_policy.PolicySummary(
            has_metadata=True,
            classes_present=["P2", "P3", "P4"],
            class_counts={"P2": 10, "P3": 4, "P4": 1},
            max_policy_class="P4",
        )
        self.assertTrue(data_policy.requires_confirmation(summary))
        self.assertTrue(data_policy.is_push_blocked_for_p4(summary))

    def test_confirm_or_abort_fail_closed_non_interactive(self) -> None:
        summary = data_policy.PolicySummary(
            has_metadata=True,
            classes_present=["P3"],
            class_counts={"P3": 1},
            max_policy_class="P3",
        )
        with patch.object(data_policy, "is_interactive_shell", return_value=False):
            with self.assertRaises(RuntimeError):
                data_policy.confirm_or_abort(summary, assume_yes=False)

    def test_confirm_or_abort_accepts_interactive_yes(self) -> None:
        summary = data_policy.PolicySummary(
            has_metadata=True,
            classes_present=["P3"],
            class_counts={"P3": 1},
            max_policy_class="P3",
        )
        with (
            patch.object(data_policy, "is_interactive_shell", return_value=True),
            patch("builtins.input", return_value="yes"),
        ):
            data_policy.confirm_or_abort(summary, assume_yes=False)


if __name__ == "__main__":
    unittest.main()
