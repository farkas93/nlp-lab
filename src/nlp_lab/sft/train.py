from __future__ import annotations

try:
    from src.eliza_trainer.sft.train import main
except ModuleNotFoundError:  # pragma: no cover - legacy entrypoint compatibility
    from eliza_trainer.sft.train import main


if __name__ == "__main__":
    main()
