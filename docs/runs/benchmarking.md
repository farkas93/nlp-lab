# Benchmarking Protocol

This protocol compares SFT backends (initially TRL vs Unsloth) fairly.

## Fairness rules

- Same dataset manifest URI and hash.
- Same train/eval split names.
- Same model repo and max sequence length.
- Same optimizer-relevant hyperparameters where supported.
- Same seed and epoch/step budget.

## Baseline order

1. TRL baseline run.
2. Unsloth run with matched settings.

JAX can be added later behind the same protocol.

## Metrics to compare

- wall-clock training time
- tokens/sec or examples/sec
- peak GPU memory
- final train/eval loss
- optional qualitative generation checks

## Reporting

Create a compact benchmark note per run pair:

- config paths used
- backend values used (`identity.backend`)
- MLflow run links/ids
- key metric table
- decision note (keep/tune/switch)
