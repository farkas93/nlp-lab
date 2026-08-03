from __future__ import annotations

from typing import Any


def build_training_spark_session(
    *,
    master_url: str,
    app_name: str = "nlp-lab-spark-training",
    num_gpus: int = 1,
    executor_instances: int = 1,
    executor_cores: int = 1,
    extra_conf: dict[str, str] | None = None,
) -> Any:
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder.appName(app_name)
        .master(master_url)
        .config("spark.dynamicAllocation.enabled", "false")
        .config("spark.executor.instances", str(executor_instances))
        .config("spark.executor.cores", str(executor_cores))
    )

    if num_gpus > 0:
        builder = builder.config("spark.executor.resource.gpu.amount", str(num_gpus)).config(
            "spark.task.resource.gpu.amount", "1"
        )

    for key, value in (extra_conf or {}).items():
        builder = builder.config(key, value)

    return builder.getOrCreate()
