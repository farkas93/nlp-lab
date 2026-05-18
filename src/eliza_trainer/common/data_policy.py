from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Any


POLICY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "P4": 4}
CONFIRMATION_POLICY_CLASSES = {"P3", "P4"}


@dataclass
class PolicySummary:
    has_metadata: bool
    classes_present: list[str]
    class_counts: dict[str, int]
    max_policy_class: str


def _policy_sort_key(value: str) -> tuple[int, str]:
    token = str(value or "").strip().upper()
    return (POLICY_ORDER.get(token, 999), token)


def summarize_manifest_policy(manifest: dict[str, Any]) -> PolicySummary:
    governance = manifest.get("governance") if isinstance(manifest.get("governance"), dict) else None
    if not isinstance(governance, dict):
        return PolicySummary(
            has_metadata=False,
            classes_present=[],
            class_counts={},
            max_policy_class="unknown",
        )

    counts_raw = governance.get("policy_class_counts")
    class_counts: dict[str, int] = {}
    if isinstance(counts_raw, dict):
        for key, value in counts_raw.items():
            token = str(key or "").strip().upper()
            if not token:
                continue
            try:
                class_counts[token] = int(value)
            except Exception:
                continue

    classes_raw = governance.get("policy_classes_present")
    classes_present: list[str] = []
    if isinstance(classes_raw, list):
        classes_present = [
            str(item or "").strip().upper()
            for item in classes_raw
            if str(item or "").strip()
        ]
    if not classes_present and class_counts:
        classes_present = sorted(list(class_counts.keys()), key=_policy_sort_key)
    else:
        classes_present = sorted(list(dict.fromkeys(classes_present)), key=_policy_sort_key)

    max_policy_class = str(governance.get("max_policy_class") or "").strip().upper()
    if not max_policy_class and classes_present:
        max_policy_class = classes_present[-1]
    if not max_policy_class:
        max_policy_class = "unknown"

    return PolicySummary(
        has_metadata=True,
        classes_present=classes_present,
        class_counts=class_counts,
        max_policy_class=max_policy_class,
    )


def requires_confirmation(summary: PolicySummary) -> bool:
    present = set(summary.classes_present)
    return bool(present & CONFIRMATION_POLICY_CLASSES)


def is_push_blocked_for_p4(summary: PolicySummary) -> bool:
    return "P4" in set(summary.classes_present)


def is_interactive_shell() -> bool:
    try:
        return bool(sys.stdin.isatty())
    except Exception:
        return False


def confirm_or_abort(summary: PolicySummary, *, assume_yes: bool, context: str = "training") -> None:
    if not requires_confirmation(summary):
        return
    if assume_yes:
        return
    if not is_interactive_shell():
        raise RuntimeError(
            "Dataset includes policy class P3/P4 and run is non-interactive. "
            "Failing closed. Re-run with --assume-yes to continue intentionally."
        )

    answer = input(
        f"Dataset includes policy class P3/P4. Proceed with {context}? [y/N]: "
    )
    token = str(answer or "").strip().lower()
    if token not in {"y", "yes"}:
        raise RuntimeError(f"Run aborted by user at policy confirmation gate ({context})")


__all__ = [
    "PolicySummary",
    "summarize_manifest_policy",
    "requires_confirmation",
    "is_push_blocked_for_p4",
    "is_interactive_shell",
    "confirm_or_abort",
]
