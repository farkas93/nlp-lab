"""Submit a self-contained GPU probe; the submitted job owns its runtime."""
import argparse
import base64
import hashlib
import json
import os
import time
import uuid

from .client import TrainingClient, runtime_files, archive_files, batch_script


def job(probe_id, hold_seconds, run_root="/var/lib/slurm/jobs", runtime_profile="trl"):
    files = runtime_files(runtime_profile)
    files["probe.py"] = f"""import json,time,torch
from pathlib import Path
assert torch.cuda.is_available(), 'CUDA unavailable'
x = torch.randn(32,32,device='cuda',requires_grad=True)
loss = (x @ x.T).square().mean()
loss.backward()
assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
torch.cuda.synchronize()
torch.save({{'tensor':x.detach().cpu(),'gradient':x.grad.cpu()}}, 'probe.pt')
print(json.dumps({{'device':torch.cuda.get_device_name(),'loss':loss.item(),'artifact':str(Path('probe.pt').resolve())}}),flush=True)
time.sleep({hold_seconds})
""".encode()
    data, manifest = archive_files(files, {"version": 2, "run_id": probe_id, "kind": "probe", "runtime": {"profile": runtime_profile}})
    script = batch_script(["--inline-bundle", base64.b64encode(data).decode(), "--sha256", hashlib.sha256(data).hexdigest(),
                           "--run-id", probe_id, "--run-root", run_root])
    return {"script": script, "job": {"name": "eliza-smoke-" + probe_id, "partition": "training", "nodes": "1", "tasks": 1,
                                      "cpus_per_task": 2, "tres_per_node": "gres/gpu:1", "requeue": False,
                                      "memory_per_node": {"set": True, "number": 8192, "infinite": False},
                                      "time_limit": {"set": True, "number": 30, "infinite": False},
                                      "current_working_directory": run_root,
                                      "standard_output": f"{run_root}/logs/smoke-{probe_id}-%j.out",
                                      "standard_error": f"{run_root}/logs/smoke-{probe_id}-%j.err",
                                      "environment": ["PATH=/usr/local/bin:/usr/bin:/bin", "PYTHONUNBUFFERED=1"]}}


def main():
    parser = argparse.ArgumentParser(description="GPU probe with a job-owned runtime")
    parser.add_argument("--hold-seconds", type=int, default=30)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--runtime-profile", default="trl")
    args = parser.parse_args()
    if not 0 <= args.hold_seconds <= 120 or args.timeout_seconds < 1:
        parser.error("hold-seconds must be 0..120; timeout must be positive")
    control = os.environ.get("INFERENCE_CONTROL_URL", "http://127.0.0.1:" + os.environ.get("PORT", "8030"))
    client = TrainingClient(control_url=control)
    if client.environment().get("phase") != "training":
        raise RuntimeError("Switch the host to Slurm-enabled training mode first")
    advertised = client.discover()
    if not advertised["ready"]:
        raise RuntimeError("Slurm is not ready")
    probe = str(uuid.uuid4())
    print("Probe identity (save before submission): " + probe, flush=True)
    submitted = client._slurm("/slurm/v0.0.45/job/submit", "POST", job(probe, args.hold_seconds, client.run_root, args.runtime_profile))
    job_id = submitted["job_id"]
    print(f"Slurm job {job_id}; logs: {client.run_root}/logs/smoke-{probe}-{job_id}.out", flush=True)
    deadline, previous = time.monotonic() + args.timeout_seconds, None
    while True:
        jobs = client._slurm(f"/slurm/v0.0.45/job/{job_id}").get("jobs", [])
        if len(jobs) != 1 or jobs[0].get("name") != "eliza-smoke-" + probe:
            raise RuntimeError("Probe identity missing; inspect scheduler records")
        states = jobs[0].get("job_state", [])
        states = [states] if isinstance(states, str) else states
        if states != previous:
            print(json.dumps({"job_id": job_id, "states": states}), flush=True)
            previous = states
        if "COMPLETED" in states and "COMPLETING" not in states:
            print("GPU probe completed; inspect its artifact and return to inference when ready.")
            return 0
        if any(state in states for state in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY")):
            raise RuntimeError("GPU probe failed; inspect job logs")
        if time.monotonic() >= deadline:
            raise TimeoutError("Watcher timed out; inspect the job before retrying")
        time.sleep(2)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        raise SystemExit(str(exc))
