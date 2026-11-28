"""Validate the workspace release version against the actual client wheel."""

import argparse
from email import policy
from email.parser import BytesParser
from pathlib import Path
import re
import tomllib
from zipfile import ZipFile


def check_release(root, wheel, expected=None):
    workspace = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    client = tomllib.loads((root / "packages/training-client/pyproject.toml").read_text())["project"]
    with ZipFile(wheel) as archive:
        [metadata_path] = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        metadata = BytesParser(policy=policy.default).parsebytes(archive.read(metadata_path))
    if client["name"] != "nlp-lab-training" or metadata["Name"] != client["name"]:
        raise ValueError("Expected the nlp-lab-training wheel")
    versions = {"workspace": workspace["version"], "client": client["version"], "wheel": metadata["Version"]}
    if expected is not None:
        versions["release"] = expected
    if len(set(versions.values())) != 1 or not re.fullmatch(r"\d+\.\d+\.\d+", workspace["version"]):
        raise ValueError(f"Training release version mismatch: {versions}")
    return workspace["version"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--expected")
    args = parser.parse_args()
    try:
        print(check_release(Path(__file__).resolve().parents[1], args.wheel, args.expected))
    except ValueError as exc:
        parser.exit(1, f"{exc}\n")
