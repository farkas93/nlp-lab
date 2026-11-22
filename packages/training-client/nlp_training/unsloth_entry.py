"""Import Unsloth's patches before Transformers/PEFT, then invoke the trainer."""
import runpy
import sys


def main():
    module = sys.argv[1]
    if module not in {"src.eliza_trainer.sft.train", "src.eliza_trainer.sft.remote_checks"}:
        raise ValueError("Unsupported Unsloth entry module")
    import unsloth  # noqa: F401 -- import order is required by upstream

    sys.argv = [module, *sys.argv[2:]]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
