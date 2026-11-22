"""Small authenticated MLflow REST probe, safe for the tokenizer-only runtime."""
import base64
import json
import os
import time
from urllib.error import URLError
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def verify_tracking(uri, experiment_name):
    headers = {"Content-Type": "application/json"}
    if os.getenv("MLFLOW_TRACKING_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["MLFLOW_TRACKING_TOKEN"]
    elif os.getenv("MLFLOW_TRACKING_USERNAME"):
        pair = os.environ["MLFLOW_TRACKING_USERNAME"] + ":" + os.getenv("MLFLOW_TRACKING_PASSWORD", "")
        headers["Authorization"] = "Basic " + base64.b64encode(pair.encode()).decode()

    def call(route, body=None, query=None):
        url = uri.rstrip("/") + "/api/2.0/mlflow/" + route
        if query:
            url += "?" + urlencode(query)
        request = Request(url, headers=headers, data=json.dumps(body).encode() if body is not None else None)
        try:
            with urlopen(request, timeout=30) as response:
                return json.load(response)
        except HTTPError as exc:
            if route == "experiments/get-by-name":
                try:
                    error = json.loads(exc.read(65536))
                except (ValueError, TypeError):
                    error = {}
                if error.get("error_code") == "RESOURCE_DOES_NOT_EXIST":
                    return None
            # Surface REST errors without printing credentials or response HTML.
            raise RuntimeError(f"MLflow {route} returned HTTP {exc.code}; verify endpoint, authentication and permissions") from exc
        except (URLError, ValueError) as exc:
            raise RuntimeError(f"MLflow {route} failed: no valid MLflow REST response; verify server URL, routing and connectivity") from exc

    found = call("experiments/get-by-name", query={"experiment_name": experiment_name})
    if found is None:
        experiment_id = call("experiments/create", {"name": experiment_name})["experiment_id"]
    else:
        experiment_id = found["experiment"]["experiment_id"]
    run = call("runs/create", {"experiment_id": experiment_id, "start_time": int(time.time() * 1000),
                              "tags": [{"key": "mlflow.runName", "value": "remote-connectivity-check"},
                                       {"key": "eliza.run_kind", "value": "tracking-preflight"}]})["run"]["info"]
    run_id = run["run_id"]
    try:
        call("runs/log-metric", {"run_id": run_id, "key": "eliza_connectivity", "value": 1,
                                 "timestamp": int(time.time() * 1000), "step": 0})
    finally:
        call("runs/update", {"run_id": run_id, "status": "FINISHED", "end_time": int(time.time() * 1000)})
    readback = call("runs/get", query={"run_id": run_id})
    if not any(item["key"] == "eliza_connectivity" and item["value"] == 1 for item in readback["run"]["data"]["metrics"]):
        raise RuntimeError("MLflow probe metric could not be read back")
    return {"tracking_uri": uri, "experiment_id": experiment_id, "run_id": run_id,
            "url": f"{uri.rstrip('/')}/#/experiments/{experiment_id}/runs/{run_id}", "verified": True}
