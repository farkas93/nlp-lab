import csv
import io
import json
import os
import hashlib
from pathlib import Path

import pytest

from nlp_training import wheel_metadata as repair


class Distribution:
    version = repair.VERSION

    def __init__(self, root):
        self.root = root

    def locate_file(self, name):
        return self.root / name


@pytest.fixture
def environment(tmp_path, monkeypatch):
    root = tmp_path / "site-packages"
    dist = Distribution(root)
    elf = b"\x7fELF\x02\x01" + b"\0" * 12 + (183).to_bytes(2, "little") + b"test-binary"
    contents = {repair.BINARY: elf, repair.DIST_INFO + "/WHEEL": repair.OLD_WHEEL,
                repair.DIST_INFO + "/METADATA": b"Name: nvidia-cusparselt-cu13\nVersion: 0.8.0\n"}
    expected = {}
    for name, data in contents.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        expected[name] = (repair.digest_bytes(data), len(data))
    origin = root / repair.DIST_INFO / "direct_url.json"
    origin.write_text(json.dumps({"url": repair.URL, "archive_info": {"hashes": {"sha256": repair.ARCHIVE_SHA256}}}))
    record = root / repair.DIST_INFO / "RECORD"
    rows = [[name, "sha256=" + digest, str(size)] for name, (digest, size) in expected.items()]
    rows += [[repair.DIST_INFO + "/direct_url.json", "sha256=" + repair.file_digest(origin), str(origin.stat().st_size)],
             [repair.DIST_INFO + "/RECORD", "", ""]]
    with record.open("w", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerows(rows)
    monkeypatch.setattr(repair, "OFFICIAL_FILES", expected)
    monkeypatch.setattr(repair.importlib.metadata, "distribution", lambda _: dist)
    monkeypatch.setattr(repair.platform, "system", lambda: "Linux")
    monkeypatch.setattr(repair.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(repair.sys, "prefix", str(tmp_path))
    return dist


def test_vendor_wheel_metadata_matches_pinned_record():
    assert (repair.digest_bytes(repair.OLD_WHEEL), len(repair.OLD_WHEEL)) == repair.OFFICIAL_FILES[repair.DIST_INFO + "/WHEEL"]


def test_repairs_only_wheel_tag_and_record_and_is_idempotent(environment):
    root = environment.root
    binary = (root / repair.BINARY).read_bytes()
    metadata = (root / repair.DIST_INFO / "METADATA").read_bytes()
    assert repair.repair()["status"] == "repaired"
    wheel = root / repair.DIST_INFO / "WHEEL"
    record = root / repair.DIST_INFO / "RECORD"
    assert wheel.read_bytes() == repair.NEW_WHEEL
    rows = list(csv.reader(io.StringIO(record.read_text())))
    row = next(row for row in rows if row[0].endswith("/WHEEL"))
    assert row[1:] == ["sha256=" + repair.digest_bytes(repair.NEW_WHEEL), str(len(repair.NEW_WHEEL))]
    before = (wheel.read_bytes(), record.read_bytes(), wheel.stat().st_mtime_ns, record.stat().st_mtime_ns)
    assert repair.repair()["status"] == "already-corrected"
    assert before == (wheel.read_bytes(), record.read_bytes(), wheel.stat().st_mtime_ns, record.stat().st_mtime_ns)
    assert (root / repair.BINARY).read_bytes() == binary
    assert (root / repair.DIST_INFO / "METADATA").read_bytes() == metadata
    assert json.loads((Path(repair.sys.prefix) / "nlp-training-metadata-repair.json").read_text())["artifact_sha256"] == repair.ARCHIVE_SHA256


def test_completes_interrupted_wheel_then_record_update(environment):
    (environment.root / repair.DIST_INFO / "WHEEL").write_bytes(repair.NEW_WHEEL)
    assert repair.repair()["status"] == "already-corrected"
    assert repair.repair()["status"] == "already-corrected"


def test_repair_does_not_modify_a_hardlinked_installer_cache(environment, tmp_path):
    cached = tmp_path / "cached-wheel"
    os.link(environment.root / repair.DIST_INFO / "WHEEL", cached)
    repair.repair()
    assert cached.read_bytes() == repair.OLD_WHEEL


def test_uv_installer_metadata_is_checked_and_preserved(environment):
    cache = environment.root / repair.DIST_INFO / "uv_cache.json"
    cache.write_bytes(b'{"cache":"metadata"}')
    record = environment.root / repair.DIST_INFO / "RECORD"
    with record.open("a", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerow([repair.DIST_INFO + "/uv_cache.json", "sha256=" + repair.file_digest(cache), str(cache.stat().st_size)])
    repair.repair()
    assert cache.read_bytes() == b'{"cache":"metadata"}'
    cache.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="installer metadata checksum mismatch"):
        repair.repair()


def set_origin(dist, archive_info):
    origin = dist.root / repair.DIST_INFO / "direct_url.json"
    origin.write_text(json.dumps({"url": repair.URL, "archive_info": archive_info}))
    record = dist.root / repair.DIST_INFO / "RECORD"
    rows = list(csv.reader(io.StringIO(record.read_text())))
    next(row for row in rows if row[0] == repair.DIST_INFO + "/direct_url.json")[1:] = ["sha256=" + repair.file_digest(origin), str(origin.stat().st_size)]
    with record.open("w", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerows(rows)


def test_empty_uv_archive_info_requires_an_independent_reference_hash(environment, monkeypatch):
    set_origin(environment, {})
    archive = b"independently verified reference fixture"
    monkeypatch.setattr(repair, "ARCHIVE_SHA256", hashlib.sha256(archive).hexdigest())
    monkeypatch.setattr(repair, "urlopen", lambda url, timeout: io.BytesIO(archive))
    result = repair.repair()
    assert result["status"] == "repaired"
    assert result["archive_verification"] == "reference-download"
    assert (environment.root / repair.DIST_INFO / "WHEEL").read_bytes() == repair.NEW_WHEEL


@pytest.mark.parametrize("failure", ["checksum", "network"])
def test_reference_verification_failure_leaves_metadata_untouched(environment, monkeypatch, failure):
    set_origin(environment, {})
    if failure == "network":
        def unavailable(*args, **kwargs):
            raise OSError("network unavailable")
        monkeypatch.setattr(repair, "urlopen", unavailable)
    else:
        monkeypatch.setattr(repair, "urlopen", lambda *args, **kwargs: io.BytesIO(b"wrong archive"))
    wheel = environment.root / repair.DIST_INFO / "WHEEL"
    record = environment.root / repair.DIST_INFO / "RECORD"
    before = wheel.read_bytes(), record.read_bytes()
    with pytest.raises(RuntimeError, match="reference archive"):
        repair.repair()
    assert before == (wheel.read_bytes(), record.read_bytes())


def test_recorded_hash_mismatch_is_not_replaced_by_reference_verification(environment, monkeypatch):
    set_origin(environment, {"hashes": {"sha256": "0" * 64}})
    monkeypatch.setattr(repair, "verify_reference_archive", lambda: pytest.fail("A recorded mismatch must not be bypassed"))
    with pytest.raises(RuntimeError, match="Recorded cuSPARSELt SHA-256"):
        repair.repair()


def test_legacy_recorded_hash_is_accepted_without_downloading_reference(environment, monkeypatch):
    set_origin(environment, {"hash": "sha256=" + repair.ARCHIVE_SHA256})
    monkeypatch.setattr(repair, "verify_reference_archive", lambda: pytest.fail("Recorded hash already supplied"))
    assert repair.repair()["archive_verification"] == "installer-record"


@pytest.mark.parametrize("fault", ["host", "version", "origin-hash", "origin-url", "binary", "record", "wheel"])
def test_unverified_inputs_fail_before_metadata_changes(environment, monkeypatch, fault):
    root = environment.root
    if fault == "host":
        monkeypatch.setattr(repair.platform, "machine", lambda: "x86_64")
    elif fault == "version":
        environment.version = "0.8.1"
    elif fault.startswith("origin"):
        origin = root / repair.DIST_INFO / "direct_url.json"
        value = json.loads(origin.read_text())
        if fault == "origin-hash":
            value["archive_info"]["hashes"]["sha256"] = "0" * 64
        else:
            value["url"] = value["url"].replace("aarch64", "x86_64")
        origin.write_text(json.dumps(value))
    elif fault == "binary":
        (root / repair.BINARY).write_bytes(b"corrupted-binary")
    elif fault == "record":
        record = root / repair.DIST_INFO / "RECORD"
        record.write_text(record.read_text() + record.read_text().splitlines()[0] + "\n")
    else:
        (root / repair.DIST_INFO / "WHEEL").write_bytes(b"unknown wheel metadata")
    before = (root / repair.DIST_INFO / "WHEEL").read_bytes()
    with pytest.raises(RuntimeError):
        repair.repair()
    assert (root / repair.DIST_INFO / "WHEEL").read_bytes() == before


def test_even_checksum_valid_non_arm_binary_is_rejected(environment, monkeypatch):
    binary = environment.root / repair.BINARY
    data = bytearray(binary.read_bytes())
    data[18:20] = (62).to_bytes(2, "little")
    binary.write_bytes(data)
    expected = dict(repair.OFFICIAL_FILES)
    expected[repair.BINARY] = (repair.digest_bytes(data), len(data))
    monkeypatch.setattr(repair, "OFFICIAL_FILES", expected)
    record = environment.root / repair.DIST_INFO / "RECORD"
    rows = list(csv.reader(io.StringIO(record.read_text())))
    next(row for row in rows if row[0] == repair.BINARY)[1:] = ["sha256=" + repair.digest_bytes(data), str(len(data))]
    with record.open("w", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerows(rows)
    with pytest.raises(RuntimeError, match="ARM64 ELF"):
        repair.repair()
