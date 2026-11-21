"""Repair one verified NVIDIA ARM64 wheel's invalid SBSA platform tag.

The binary/version are unchanged. Provenance and every vendor RECORD entry are
checked before writing; normal dependency/platform checks still run afterward.
"""
import base64
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import sys
import tempfile
from pathlib import Path, PurePosixPath
from urllib.request import urlopen

NAME = "nvidia-cusparselt-cu13"
VERSION = "0.8.0"
DIST_INFO = "nvidia_cusparselt_cu13-0.8.0.dist-info"
URL = "https://pypi.nvidia.com/nvidia-cusparselt-cu13/nvidia_cusparselt_cu13-0.8.0-py3-none-manylinux2014_aarch64.whl"
ARCHIVE_SHA256 = "400c6ed1cf6780fc6efedd64ec9f1345871767e6a1a0a552a1ea0578117ea77c"
OLD_WHEEL = b"Wheel-Version: 1.0\nGenerator: setuptools (80.9.0)\nRoot-Is-Purelib: true\nTag: py3-none-manylinux2014_sbsa\n\n"
NEW_WHEEL = OLD_WHEEL.replace(b"manylinux2014_sbsa", b"manylinux2014_aarch64")
BINARY = "nvidia/cusparselt/lib/libcusparseLt.so.0"
# From the checksum-pinned vendor wheel's RECORD, not from the installed copy.
OFFICIAL_FILES = {
    "nvidia/cusparselt/LICENSE.txt": ("6NFYiFpoG5Xsem_AbdjUpSmJ83TLE4DIpMj7J_09XV4", 17948),
    "nvidia/cusparselt/include/cusparseLt.h": ("WJLVrmtY9jrJWDll2LrvRIAex3ryFFFtyxVC6bOV7O8", 18157),
    BINARY: ("IN5rNsu_V1OyaCbux1OiAFYI4Uy8O_3UaBE4CYvoaUE", 273881681),
    DIST_INFO + "/METADATA": ("cum1u5GjgEfZkUWhqAKtlH_1VUMeO_M3UcsxpfqR9uI", 12641),
    DIST_INFO + "/WHEEL": ("9EEww1DEpSYdq-rReTwY88G51UoHWvABWuRF6Sy_8Bs", 106),
    DIST_INFO + "/top_level.txt": ("ofIC-dbK0s22jN55sgfVpbhHWT63ZdJLWeSb6K_1-BI", 18),
}


def digest_bytes(data):
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode()


def atomic(path, data):
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix="." + path.name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), path.stat().st_mode & 0o777 if path.exists() else 0o600)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def verify_reference_archive():
    """UV can omit PEP 610 hashes. Verify the pinned reference independently."""
    digest, size = hashlib.sha256(), 0
    try:
        with urlopen(URL, timeout=120) as response:
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > 256 * 1024 * 1024:
                    raise RuntimeError("NVIDIA reference archive exceeds size limit")
                digest.update(chunk)
    except OSError as exc:
        raise RuntimeError("Could not verify the pinned NVIDIA reference archive") from exc
    if digest.hexdigest() != ARCHIVE_SHA256:
        raise RuntimeError("Pinned NVIDIA reference archive SHA-256 mismatch")


def repair():
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise RuntimeError("cuSPARSELt metadata repair is restricted to Linux aarch64")
    dist = importlib.metadata.distribution(NAME)
    if dist.version != VERSION:
        raise RuntimeError("cuSPARSELt metadata repair requires the pinned 0.8.0 version")
    root = Path(dist.locate_file("")).resolve()

    def installed(name):
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError("Invalid cuSPARSELt RECORD path")
        location = Path(dist.locate_file(name))
        if location.is_symlink() or not location.resolve().is_relative_to(root):
            raise RuntimeError("cuSPARSELt installed file is outside its environment")
        return location

    origin = json.loads(installed(DIST_INFO + "/direct_url.json").read_text())
    if origin.get("url") != URL:
        raise RuntimeError("cuSPARSELt artifact URL does not match the pinned ARM64 wheel")
    archive = origin.get("archive_info", {})
    if not isinstance(archive, dict) or not isinstance(archive.get("hashes") or {}, dict):
        raise RuntimeError("Invalid cuSPARSELt archive provenance")
    hashes = archive.get("hashes") or {}
    recorded_hash = hashes.get("sha256")
    legacy_hash = archive.get("hash")
    if recorded_hash is not None and recorded_hash != ARCHIVE_SHA256:
        raise RuntimeError("Recorded cuSPARSELt SHA-256 does not match the pinned ARM64 wheel")
    if legacy_hash is not None and legacy_hash != "sha256=" + ARCHIVE_SHA256:
        raise RuntimeError("Recorded cuSPARSELt archive hash does not match the pinned ARM64 wheel")
    missing_hash = recorded_hash is None and legacy_hash is None
    wheel_path = installed(DIST_INFO + "/WHEEL")
    wheel = wheel_path.read_bytes()
    if wheel not in (OLD_WHEEL, NEW_WHEEL):
        raise RuntimeError("Unexpected cuSPARSELt WHEEL metadata; refusing repair")
    record_path = installed(DIST_INFO + "/RECORD")
    rows = list(csv.reader(io.StringIO(record_path.read_text())))
    indexed = {}
    extras = {DIST_INFO + "/" + filename for filename in
              ("INSTALLER", "REQUESTED", "direct_url.json", "uv_cache.json", "uv_build.json")}
    for row in rows:
        if len(row) != 3 or row[0] in indexed:
            raise RuntimeError("Invalid or duplicate cuSPARSELt RECORD entry")
        name, checksum, size = row
        indexed[name] = row
        path = installed(name)
        if name == DIST_INFO + "/RECORD":
            if checksum or size:
                raise RuntimeError("Unexpected RECORD self-checksum")
            continue
        if name not in OFFICIAL_FILES and name not in extras:
            raise RuntimeError("Unexpected installed cuSPARSELt file: " + name)
        actual = (file_digest(path), path.stat().st_size)
        if name in OFFICIAL_FILES:
            expected = OFFICIAL_FILES[name]
            # Accept the interrupted state where WHEEL was repaired but RECORD
            # still holds its original row; all binary/header bytes must match.
            if name == DIST_INFO + "/WHEEL" and wheel == NEW_WHEEL:
                expected_actual = (digest_bytes(NEW_WHEEL), len(NEW_WHEEL))
                valid_rows = {("sha256=" + expected[0], str(expected[1])),
                              ("sha256=" + expected_actual[0], str(expected_actual[1]))}
            else:
                expected_actual = expected
                valid_rows = {("sha256=" + expected[0], str(expected[1]))}
            if actual != expected_actual or (checksum, size) not in valid_rows:
                raise RuntimeError("cuSPARSELt vendor file/RECORD checksum mismatch: " + name)
        elif checksum:
            if (checksum, size) != ("sha256=" + actual[0], str(actual[1])):
                raise RuntimeError("cuSPARSELt installer metadata checksum mismatch: " + name)
        elif size not in ("", str(actual[1])):
            raise RuntimeError("Invalid installer metadata size")
    if not set(OFFICIAL_FILES).issubset(indexed) or not {DIST_INFO + "/RECORD", DIST_INFO + "/direct_url.json"}.issubset(indexed):
        raise RuntimeError("Incomplete cuSPARSELt vendor RECORD")
    with installed(BINARY).open("rb") as handle:
        elf = handle.read(20)
    if len(elf) != 20 or elf[:6] != b"\x7fELF\x02\x01" or int.from_bytes(elf[18:20], "little") != 183:
        raise RuntimeError("cuSPARSELt payload is not a Linux ARM64 ELF binary")
    if missing_hash:
        print("UV omitted the archive hash; independently verifying the pinned NVIDIA reference archive.", flush=True)
        # The reference archive hash and all installed vendor file hashes must
        # pass before touching metadata. No existing bad hash is waived.
        verify_reference_archive()
    repaired_row = indexed[DIST_INFO + "/WHEEL"]
    status = "repaired" if wheel == OLD_WHEEL else "already-corrected"
    if wheel != NEW_WHEEL:
        atomic(wheel_path, NEW_WHEEL)
    repaired_row[1:] = ["sha256=" + digest_bytes(NEW_WHEEL), str(len(NEW_WHEEL))]
    output = io.StringIO()
    csv.writer(output, lineterminator="\n").writerows(rows)
    new_record = output.getvalue().encode()
    if record_path.read_bytes() != new_record:
        atomic(record_path, new_record)
    receipt = {"package": NAME, "version": VERSION, "artifact_sha256": ARCHIVE_SHA256,
            "archive_verification": "reference-download" if missing_hash else "installer-record",
            "old_tag": "py3-none-manylinux2014_sbsa", "new_tag": "py3-none-manylinux2014_aarch64",
            "verified_vendor_files": len(OFFICIAL_FILES)}
    receipt_path = Path(sys.prefix) / "nlp-training-metadata-repair.json"
    encoded = json.dumps(receipt, sort_keys=True).encode()
    if not receipt_path.exists() or receipt_path.read_bytes() != encoded:
        atomic(receipt_path, encoded)
    return {"status": status, **receipt}


if __name__ == "__main__":
    print(json.dumps(repair(), sort_keys=True))
