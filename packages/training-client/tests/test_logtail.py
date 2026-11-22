from nlp_training.logtail import compact


def test_compact_collapses_progress_frames_and_keeps_messages():
    raw = (
        "\x1b[32mhello\x1b[0m\n"
        " 17%|█▋        | 2/12 [00:14<01:14,  7.43s/it]\r 25%|██▌       | 3/12 [00:22<01:06]\r100%|██████████| 12/12 [01:29<00:00]\n"
        "ELIZA_PROGRESS {\"stage\": \"optimizing\", \"step\": 3, \"total\": 12}\n"
        "loss=0.1736\n"
        "loss=0.1736\n"
        "final message\n"
    )
    tail = compact(raw, max_lines=10)
    assert tail.splitlines() == ["hello", "loss=0.1736", "final message"]


def test_compact_bounds_lines_and_characters():
    raw = "\n".join(f"line {index}" for index in range(100))
    tail = compact(raw, max_lines=5)
    assert tail.splitlines() == [f"line {index}" for index in range(95, 100)]
    assert len(compact("x" * 10000, max_lines=5, max_chars=100)) == 100


def test_compact_keeps_text_after_carriage_return():
    # A bare carriage return redraws one line; only the final frame is retained.
    assert compact("a\rb\nc") == "b\nc"
