"""Shared checks for private bundle creation and worker startup."""
import re


def validate_dataset_pin(config, pin):
    data = config.get("data", {})
    for field in ("bucket", "dataset_id", "dataset_version"):
        if not isinstance(data.get(field), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", data[field]):
            raise ValueError(f"Private runs require a valid data.{field}")
    uri = (f"s3://{data['bucket']}/dataset_id={data['dataset_id']}/"
           f"dataset_version={data['dataset_version']}/manifest.json")
    digest = data.get("expected_manifest_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Private runs require data.expected_manifest_sha256")
    if data.get("manifest_uri", uri) != uri or not isinstance(pin, dict) or pin.get("uri") != uri or pin.get("sha256") != digest:
        raise ValueError("Dataset manifest pin differs from the training config")
    return uri, digest


def validate_acknowledgement(acknowledgement, digest):
    if acknowledgement is not None and acknowledgement != digest:
        raise ValueError("Restricted-data acknowledgement must match the pinned dataset manifest SHA-256")
