import os
from pathlib import Path


def credential(explicit, name):
    if explicit is not None:
        return explicit
    value = os.environ.get(name)
    if value:
        return value
    filename = os.environ.get(name + "_FILE")
    if filename:
        return Path(filename).read_text().strip()
    return None
