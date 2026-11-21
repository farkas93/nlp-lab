"""Training-owned runtime setup. Host installers delegate here after provisioning."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def profile(name):
    profiles = json.loads((Path(__file__).parent / "profiles/runtime.json").read_text())
    if name not in profiles:
        raise ValueError("Unknown training runtime profile: " + name)
    return profiles[name]


def run(args, **kwargs):
    subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def provision(name, root: Path, source: Path, *, check=False):
    definition = profile(name)
    training = root / "venv"
    python = training / "bin/python"
    env = {**os.environ, "UV_PYTHON_INSTALL_DIR": str(root / "python")}
    uv = Path(sys.executable).parent / "uv"
    if not check:
        if not python.exists():
            run([uv, "venv", "--python", definition["python"], training], env=env)
        # Resolve client and trainer together: s3fs constrains botocore via
        # aiobotocore, while the client constrains the same package via boto3.
        install = [uv, "pip", "install", "--python", python, source]
        if definition["requirements"]:
            requirements = Path(__file__).parent / "profiles" / definition["requirements"]
            install += ["-r", requirements]
        run(install, env=env)
    if not python.exists():
        raise RuntimeError("Training interpreter is missing")
    if not check and definition.get("metadata_repair"):
        if definition["metadata_repair"] != "cusparselt-cu13-0.8.0-arm64":
            raise ValueError("Unsupported runtime metadata repair")
        run([python, "-m", "nlp_training.wheel_metadata"])
    run([uv, "pip", "check", "--python", python])
    # Import metadata only, never allocate CUDA while the host is in inference mode.
    if definition.get("cuda"):
        statement = ("import torch,importlib.util as u; "
                     f"assert torch.version.cuda == {definition['cuda']!r}, torch.version.cuda; "
                     f"assert all(u.find_spec(n) for n in {definition['modules']!r})")
        run([python, "-c", statement])
    return {"profile": name, "python": str(python), "definition": definition}


def main():
    parser = argparse.ArgumentParser(description="Prepare nlp-lab's isolated training runtime")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(json.dumps(provision(args.profile, args.root, args.source, check=args.check)))


if __name__ == "__main__":
    main()
