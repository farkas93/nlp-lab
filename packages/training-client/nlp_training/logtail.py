"""Turn noisy carriage-return progress output into a short human-readable tail."""
import re

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
# tqdm/rich-style progress frames such as " 17%|###   | 2/12 [00:14<01:14]".
PROGRESS_FRAME = re.compile(r"^\s*\d{1,3}%\|")
MACHINE_EVENT = "ELIZA_PROGRESS "


def compact(text, *, max_lines=20, max_chars=4000):
    """Collapse in-place updates and drop transient progress frames.

    tqdm redraws its bar with a bare carriage return, so the raw stream contains
    thousands of frames. Keeping only the last frame per line and filtering the
    bar rows leaves the actual log messages, which is what a monitor should show.
    """
    lines = []
    for raw in ANSI.sub("", text).split("\n"):
        line = raw.split("\r")[-1].rstrip()
        if not line or line.lstrip().startswith(MACHINE_EVENT) or PROGRESS_FRAME.match(line):
            continue
        if lines and lines[-1] == line:
            continue
        lines.append(line)
    tail = "\n".join(lines[-max_lines:])
    if len(tail) > max_chars:
        tail = tail[-max_chars:]
    return tail
