import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.survey_tool_output import _percentile, _repetition, classify  # noqa: E402

_LONG = "\n".join(f"value {i} of the payload here" for i in range(12))


def test_classify_empty_and_short() -> None:
    assert classify("") == "empty"
    assert classify("   \n  ") == "empty"
    assert classify("Done.") == "short"


def test_classify_numbered_file_read() -> None:
    text = "\n".join(f"{i}\tdef f{i}():" for i in range(1, 15))
    assert classify(text) == "numbered_file"


def test_classify_diff_by_hunk_header() -> None:
    text = "diff --git a/x.py b/x.py\n@@ -1,3 +1,4 @@\n context\n+added\n-removed\n"
    assert classify(text) == "diff"


def test_classify_diff_needs_both_added_and_removed_lines() -> None:
    # The regression this guards: a PowerShell directory listing begins every entry with a dash
    # and was reported as a diff by a rule that counted + and - lines together.
    listing = "Directory: D:\\NeonGPT\n" + "\n".join(
        f"-a---  9/9/2026  12:0{i}  1234 file{i}.txt" for i in range(10)
    )
    assert classify(listing) != "diff"


def test_classify_traceback_beats_generic_error_text() -> None:
    text = 'Traceback (most recent call last):\n  File "x.py", line 4, in <module>\n' + _LONG
    assert classify(text) == "traceback"


def test_classify_grep_hits() -> None:
    text = "\n".join(f"src/mod{i}.py:{i}:    return None" for i in range(10))
    assert classify(text) == "grep_hits"


def test_classify_path_listing() -> None:
    text = "\n".join(f"assets/fonts/file{i}.woff2" for i in range(12))
    assert classify(text) == "path_listing"


def test_classify_json() -> None:
    body = ",\n".join(f'  "key{i}": {i}' for i in range(12))
    assert classify("{\n" + body + "\n}") == "json"


def test_classify_falls_through_to_other() -> None:
    text = "\n".join("banner art ##########" for _ in range(10))
    assert classify(text) == "other"


def test_repetition_counts_duplicate_lines_only() -> None:
    assert _repetition(["a", "b", "c"]) == 0.0
    assert _repetition(["a", "a", "a", "a"]) == 0.75
    assert _repetition([]) == 0.0
    assert _repetition(["only"]) == 0.0


def test_repetition_ignores_blank_lines() -> None:
    assert _repetition(["a", "", "", "b"]) == 0.0


def test_percentile_endpoints() -> None:
    values = list(range(1, 101))
    assert _percentile(values, 0.5) == 50
    assert _percentile(values, 1.0) == 100
    assert _percentile([], 0.5) == 0
