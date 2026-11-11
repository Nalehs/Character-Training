"""JSONL input and crash-safe JSONL output shared by the data stages.

`character-pipeline run` treats a step as finished when its output file exists
(run.Step.done). A stage that wrote straight to its final path and then died
half way (a server timeout, a full disk) would therefore leave a truncated file
that the next run accepts as complete. Stages write through `resumable_jsonl`
instead: rows go to `<out>.partial` and the file is renamed to `<out>` only
after the last row. The final path is still never overwritten.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path


def read_jsonl(path) -> list[dict]:
    """Parse a JSONL file, skipping blank lines and naming the line that fails."""
    rows = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{n}: not valid JSON ({e.msg})") from e
    return rows


def partial_path(out) -> Path:
    out = Path(out)
    return out.with_name(out.name + ".partial")


def read_partial(out) -> list[dict]:
    """Complete rows of an interrupted run. A torn last line (no newline) is cut off the file."""
    partial = partial_path(out)
    if not partial.exists():
        return []
    text = partial.read_text()
    complete = text[: text.rfind("\n") + 1]
    if complete != text:
        partial.write_text(complete)
    try:
        return [json.loads(line) for line in complete.splitlines() if line.strip()]
    except json.JSONDecodeError as e:
        raise SystemExit(f"{partial} is corrupt ({e.msg}); delete it to start this stage again") from e


@contextmanager
def resumable_jsonl(out, resume: bool = True):
    """Yield (rows already written by an interrupted run, write(row)).

    Rows are flushed one by one to `<out>.partial`; on a clean exit that file
    becomes `<out>`. With resume=False an old partial file is discarded. The
    caller must check that the returned rows are the prefix it expects.
    """
    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists and is never overwritten; move it away to run this stage again")
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = partial_path(out)
    if not resume and partial.exists():
        partial.unlink()
    done = read_partial(out)
    with partial.open("a") as f:
        def write(row: dict) -> None:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
        yield done, write
    os.replace(partial, out)
