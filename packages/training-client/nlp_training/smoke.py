"""Submit a small CUDA forward/backward/artifact probe through standard Slurm REST."""
import argparse
import json
import os
import time
import uuid
from pathlib import Path
from .control import HostControl
from .client import TrainingClient


def job(probe_id, hold_seconds):
    root = "/var/lib/eliza-training/runs"
    # The probe uses the isolated training environment; no dataset or MinIO needed.
    script = f"""#!/bin/bash
set -euo pipefail
/opt/eliza-training/venv/bin/python - <<'PY'
import json,time,torch
from pathlib import Path
assert torch.cuda.is_available(), 'CUDA unavailable'
path = Path('{root}/smoke-{probe_id}')
path.mkdir()
x = torch.randn(32, 32, device='cuda', requires_grad=True)
loss = (x @ x.T).square().mean()
loss.backward()
assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
torch.cuda.synchronize()
torch.save({{'tensor': x.detach().cpu(), 'gradient': x.grad.cpu()}}, path / 'probe.pt')
print(json.dumps({{'torch': torch.__version__, 'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(), 'loss': loss.item(), 'artifact': str(path / 'probe.pt')}}), flush=True)
time.sleep({hold_seconds})
PY
"""
    return {"script": script, "job": {"name": "eliza-smoke-" + probe_id, "partition": "training", "nodes": "1", "tasks": 1,
                                      "cpus_per_task": 2, "tres_per_node": "gres/gpu:1", "requeue": False,
                                      "memory_per_node": {"set": True, "number": 8192, "infinite": False},
                                      "time_limit": {"set": True, "number": 5, "infinite": False},
                                      "current_working_directory": root,
                                      "standard_output": f"{root}/logs/smoke-{probe_id}-%j.out",
                                      "standard_error": f"{root}/logs/smoke-{probe_id}-%j.err",
                                      "environment": ["PATH=/usr/local/bin:/usr/bin:/bin", "PYTHONUNBUFFERED=1"]}}


def main():
    parser = argparse.ArgumentParser(description="GPU smoke job; switch the host to training mode first.")
    parser.add_argument("--hold-seconds", type=int, default=30, help="Keep CUDA active briefly to test a busy switch")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    args = parser.parse_args()
    if not 0 <= args.hold_seconds <= 120 or args.timeout_seconds < 1:
        parser.error("hold-seconds must be 0..120; timeout must be positive")
    control = os.environ.get("INFERENCE_CONTROL_URL", "http://127.0.0.1:" + os.environ.get("PORT", "8030")).rstrip("/")
    slurm = os.environ.get("SLURM_REST_URL", os.environ.get("ELIZA_SLURM_REST_URL", "http://127.0.0.1:6820")).rstrip("/")
    client = TrainingClient(control_url=control, endpoint_url=slurm)
    if client.environment().get("phase") != "training":
        raise RuntimeError("Switch to training with ./scripts/environment training first")
    probe = str(uuid.uuid4())
    print("Probe identity (save before submission): " + probe, flush=True)
    # No retry: an HTTP timeout may follow acceptance.
    submitted = client._slurm("/slurm/v0.0.45/job/submit", "POST", job(probe, args.hold_seconds))
    job_id = submitted["job_id"]
    print(f"Slurm job {job_id}; logs: /var/lib/eliza-training/runs/logs/smoke-{probe}-{job_id}.out", flush=True)
    deadline = time.monotonic() + args.timeout_seconds
    previous = None
    while True:
        jobs = client._slurm(f"/slurm/v0.0.45/job/{job_id}").get("jobs", [])
        if len(jobs) != 1 or jobs[0].get("name") != "eliza-smoke-" + probe:
            raise RuntimeError("Probe job identity missing/mismatched; inspect scheduler completion records")
        states = jobs[0].get("job_state", [])
        states = [states] if isinstance(states, str) else states
        if states != previous:
            print(json.dumps({"job_id": job_id, "states": states}), flush=True)
            previous = states
        if "COMPLETED" in states and "COMPLETING" not in states:
            print("GPU probe completed; inspect its logged device, finite loss and saved probe.pt. Return with ./scripts/environment inference.")
            return 0
        if any(state in states for state in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY")):
            raise RuntimeError("GPU probe failed; inspect the job logs: " + ",".join(states))
        if time.monotonic() >= deadline:
            raise TimeoutError("Probe watcher timed out; job may still be active. Inspect its ID before retrying")
        time.sleep(2)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        raise SystemExit(str(exc))
