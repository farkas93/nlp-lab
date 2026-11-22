"""Install the two operator notebooks and visible environment template, preserving user work."""
import argparse
import hashlib
from pathlib import Path


def install(repo, destination):
    repo, destination = Path(repo).resolve(), Path(destination).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "runs").mkdir(exist_ok=True)
    files = {"training.env": repo / "configs/training-workspace.env.example",
             **{name: repo / "notebooks" / name for name in ("hass-dataset.ipynb", "hass-sft.ipynb")}}
    result = {}
    for name, source in files.items():
        data = source.read_bytes()
        if name == "training.env":
            data = data.replace(b"/home/jovyan/eliza/nlp-lab", str(repo).encode())
        target = destination / name
        if target.exists() and target.read_bytes() != data:
            if name == "training.env":
                result[name] = "preserved existing configuration"
                continue
            target = destination / f"{target.stem}.updated-{hashlib.sha256(data).hexdigest()[:8]}{target.suffix}"
        if not target.exists():
            with target.open("xb") as handle:
                handle.write(data)
        result[name] = str(target)
    guide = repo / "docs/runs/hass-remote-operator.md"
    shutil_target = destination / "WORKSPACE-GUIDE.md"
    if not shutil_target.exists():
        shutil_target.write_bytes(guide.read_bytes())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path.home() / "eliza/training")
    args = parser.parse_args()
    for name, status in install(Path(__file__).resolve().parents[1], args.destination).items():
        print(f"{name}: {status}")
