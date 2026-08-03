# Jupyter Spark Training (DGX Single GPU)

This runbook launches `nlp-lab` training from a Jupyter notebook while using Spark to place the training task on the DGX GPU worker.

Workflow:

1. Stop `eliza-inference` stack via control daemon.
2. Build a fresh Spark session with GPU resource request.
3. Launch `start_sft.sh` or `start_dpo.sh` from a Spark executor.
4. Always restart `eliza-inference` stack in `finally`.

## Example notebook cell

```python
from eliza_trainer.orchestration import (
    build_training_spark_session,
    run_training_via_spark,
    start_stack,
    stop_stack,
)

CONTROL_DAEMON_URL = "http://10.42.47.8:8030"
SPARK_MASTER_URL = "spark://10.42.47.7:30077"
NLP_LAB_ROOT = "/home/zskalo/private/nlp-lab"

stop_stack(CONTROL_DAEMON_URL)

spark = None
try:
    spark = build_training_spark_session(
        master_url=SPARK_MASTER_URL,
        app_name="nlp-lab-sft-spark",
        num_gpus=1,
        executor_instances=1,
        executor_cores=1,
    )

    result = run_training_via_spark(
        spark_session=spark,
        nlp_lab_root=NLP_LAB_ROOT,
        mode="sft",
        config_path="configs/sft_general_qwen3_5_0_8b.yaml",
        extra_args=["--assume-yes"],
    )
    print(result.to_dict())
finally:
    if spark is not None:
        spark.stop()
    start_stack(CONTROL_DAEMON_URL)
```

## Notes

- Keep `num_gpus=1` for current single-GPU policy.
- Start each training run with a fresh Spark session so resource configs apply.
- The returned `TrainingRunResult` includes executor host and Spark GPU addresses for placement verification.
