"""Turn raw teacher output into chat rows, with mechanical filters only.

For recipes without a human-review gate (Open Character Training's stage 3-5
path), this converts `generate.teacher` output into `{"id", "messages"}` rows
that `train.sample` and `data.pairs` accept. A turn is dropped if it carries a
mechanical flag from generation (empty, incomplete or unknown finish, role
markers; the whole conversation is dropped, not just the turn) or, with
`--require-final-punct`, if it does not end in punctuation (OCT's truncation
filter, distillation/data.py:15-18).

Our own recipe never uses this: its chats go through `review-sheet` and
`data.clean`. Rows made here are marked `"reviewed": false`.

    python -m character_pipeline.data.chats runs/golem-15/oct/teacher.jsonl --out runs/golem-15/oct/teacher_chats.jsonl
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

from character_pipeline.data.io import read_jsonl

# generate.teacher writes "incomplete_or_unknown_finish" (it drops controller.hygiene's own
# "incomplete"); both spellings are listed so that a truncated reply is never kept.
DROP_FLAGS = {"empty", "incomplete", "incomplete_or_unknown_finish", "role_marker", "role_or_think_marker"}


def ends_in_punctuation(text: str) -> bool:
    text = text.rstrip()
    return bool(text) and unicodedata.category(text[-1]).startswith("P")


def convert(rows: list[dict], require_final_punct: bool = False) -> tuple[list[dict], dict]:
    out, dropped = [], {"flagged": 0, "no_final_punct": 0, "no_character_turn": 0}
    for row in rows:
        messages, ok = [], True
        for turn in row["history"]:
            if turn["speaker"] == "partner":
                messages.append({"role": "user", "content": turn["text"]})
                continue
            if DROP_FLAGS & set(turn.get("flags", [])):
                dropped["flagged"] += 1
                ok = False
                break
            if require_final_punct and not ends_in_punctuation(turn["text"]):
                dropped["no_final_punct"] += 1
                ok = False
                break
            messages.append({"role": "assistant", "content": turn["text"]})
        if ok and messages and messages[-1]["role"] == "assistant":
            out.append({"id": f'{row["scenario_id"]}:{row["seed"]}', "messages": messages, "reviewed": False})
        elif ok:
            dropped["no_character_turn"] += 1
    return out, dropped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("teacher_output", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--require-final-punct", action="store_true")
    args = ap.parse_args(argv)
    rows = read_jsonl(args.teacher_output)
    chats, dropped = convert(rows, args.require_final_punct)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chats))
    print(json.dumps({"chats": len(chats), "dropped": dropped}))


if __name__ == "__main__":
    main()
