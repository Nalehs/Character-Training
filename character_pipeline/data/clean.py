"""Stage 5: clean chats. Human-reviewed turns in, SFT/preference files out.

Rules carried over from earlier work:
- Keep only the intact prefix before the first defective character turn, never
  repair it (`retained`, golem-15 gen/engine/controller.py).
- Drop exact duplicate conversations by content hash and split by group, not
  by adjacent turns (aineko-character-training prepare_data.py; golem-15
  recipe/DATA.md asks for conversation and prompt-family splits).
- Embedding diversity: flag cosine near-duplicates at 0.92 for inspection
  and pick a diverse subset greedily. 0.92 is the archived inspection
  proposal in golem-15 recipe/DATA.md, not a validated boundary. This step
  is new code and has not been run on real Golem data ("try").
- Raw inputs are never modified. A manifest records input hashes and every
  exclusion with its reason.

A turn is kept only on an explicit "accept". Blank, "reject", an unknown word
or two rows that disagree about the same turn all count as not accepted, and
the last two are also listed in the manifest, as are decisions that match no
turn (usually a mistyped conversation id).

The split is by scenario id. Two scenario ids can still carry the same caller
turns (for example `data.constitution --repeats`); validation conversations
whose opening caller turn also opens a training conversation are listed in the
manifest as `validation_prompt_overlap` and reported, not moved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

from character_pipeline.data.io import read_jsonl
from character_pipeline.generate.controller import retained

DECISIONS = ("accept", "reject", "")


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def conversation_key(row: dict) -> str:
    return f'{row["scenario_id"]}:{row["seed"]}'


def apply_review(row: dict, decisions: dict[tuple[str, int], str]) -> dict:
    """Mark a character turn defective unless a human accepted it."""
    key = conversation_key(row)
    out, caller = [], -1
    for turn in row["history"]:
        turn = dict(turn)
        if turn["speaker"] == "partner":
            caller += 1
        else:
            decision = decisions.get((key, caller), "unreviewed")
            if decision != "accept":
                turn["flags"] = list(turn.get("flags", [])) + [f"review:{decision}"]
        out.append(turn)
    return dict(row, history=out)


def read_decisions(rows: list[dict]) -> tuple[dict[tuple[str, int], str], list[dict]]:
    """Review rows -> {(conversation, turn): decision}, plus problems for the manifest.

    Decisions are compared after strip/lower. Anything that is not exactly accept or reject
    becomes "invalid", and reviewers who disagree become "conflict"; neither is an accept.
    """
    decisions, problems = {}, []
    for d in rows:
        key = (d["conversation"], d["turn"])
        decision = str(d.get("decision") or "").strip().lower()
        if decision not in DECISIONS:
            problems.append({"conversation": key[0], "turn": key[1], "problem": "invalid decision",
                             "value": d.get("decision")})
            decision = "invalid"
        decision = decision or "unreviewed"
        if key in decisions and decisions[key] != decision:
            problems.append({"conversation": key[0], "turn": key[1], "problem": "conflicting decisions"})
            decision = "conflict"
        decisions[key] = decision
    return decisions, problems


def unmatched_decisions(rows: list[dict], decisions: dict) -> list[dict]:
    """Decisions that name no character turn in the input."""
    turns = set()
    for row in rows:
        n = sum(t["speaker"] == "partner" for t in row["history"])
        turns |= {(conversation_key(row), i) for i in range(n)}
    return [{"conversation": c, "turn": t, "problem": "no such turn"} for c, t in sorted(
        (k for k in decisions if k not in turns), key=str)]


def cosine(a, b) -> float:
    den = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b)) / den if den else 0.0


def near_duplicates(vectors: list[list[float]], threshold: float = 0.92) -> list[tuple[int, int, float]]:
    pairs = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            c = cosine(vectors[i], vectors[j])
            if c >= threshold:
                pairs.append((i, j, c))
    return pairs


def diverse_subset(vectors: list[list[float]], k: int, seed: int = 0) -> list[int]:
    """Greedy farthest-point selection on cosine distance; deterministic for a seed."""
    if k <= 0:
        return []
    if k >= len(vectors):
        return list(range(len(vectors)))
    chosen = [random.Random(seed).randrange(len(vectors))]
    best = [1 - cosine(v, vectors[chosen[0]]) for v in vectors]
    while len(chosen) < k:
        i = max((i for i in range(len(vectors)) if i not in chosen), key=lambda i: (best[i], -i))
        chosen.append(i)
        best = [min(b, 1 - cosine(v, vectors[i])) for b, v in zip(best, vectors)]
    return sorted(chosen)


def group_split(rows: list[dict], group_of, holdout_fraction: float = 0.125, seed: int = 42):
    """Whole groups go to one side. At least one group is held out and at least one is kept,
    so a single group yields an empty validation side, never an empty training side."""
    if not 0 <= holdout_fraction < 1:
        raise ValueError("holdout_fraction must be in [0, 1)")
    groups = sorted({group_of(r) for r in rows})
    random.Random(seed).shuffle(groups)
    n = min(len(groups) - 1, max(1, round(len(groups) * holdout_fraction))) if holdout_fraction else 0
    held = set(groups[: max(0, n)])
    return [r for r in rows if group_of(r) not in held], [r for r in rows if group_of(r) in held]


def clean(rows: list[dict], decisions: dict, family_of=lambda r: r["scenario_id"]):
    kept, excluded, seen = [], [], set()
    for row in rows:
        reviewed = apply_review(row, decisions)
        messages, cut = retained(reviewed["history"])
        if not any(m["role"] == "assistant" for m in messages):
            excluded.append({"conversation": conversation_key(row), "reason": "no accepted character turn", "cut": cut})
            continue
        digest = sha_text(json.dumps(messages, sort_keys=True))
        if digest in seen:
            excluded.append({"conversation": conversation_key(row), "reason": "duplicate_conversation"})
            continue
        seen.add(digest)
        kept.append({"id": conversation_key(row), "family": family_of(row), "messages": messages,
                     "cut_at": cut, "sha256": digest})
    return kept, excluded


def caller_turns(row: dict) -> tuple[str, ...]:
    return tuple(m["content"] for m in row["messages"] if m["role"] == "user")


def prompt_overlap(train: list[dict], held: list[dict]) -> list[str]:
    """Ids of validation conversations whose opening caller turn also opens a training conversation."""
    seen = {caller_turns(r)[:1] for r in train}
    return [r["id"] for r in held if caller_turns(r)[:1] in seen]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("judged", type=Path, help="teacher output (optionally with d1 scores)")
    ap.add_argument("review", type=Path, help="JSONL: conversation, turn, decision (accept/reject)")
    ap.add_argument("--system", type=Path,
                    help="file with a system prompt to put first in every chat (prompted-teacher distillation)")
    ap.add_argument("--out", type=Path, default=Path("runs/clean"))
    ap.add_argument("--holdout-fraction", type=float, default=0.125)
    ap.add_argument("--seed", type=int, default=42, help="seed of the family split")
    args = ap.parse_args(argv)
    rows = read_jsonl(args.judged)
    decisions, problems = read_decisions(read_jsonl(args.review))
    problems += unmatched_decisions(rows, decisions)
    kept, excluded = clean(rows, decisions)
    if args.system:
        system = {"role": "system", "content": args.system.read_text().strip()}
        kept = [dict(r, messages=[system] + r["messages"]) for r in kept]
    train, held = group_split(kept, lambda r: r["family"], args.holdout_fraction, args.seed)
    if not train or (args.holdout_fraction and not held):
        raise SystemExit(f"{len(kept)} conversations in {len({r['family'] for r in kept})} scenario families "
                         f"survive review ({len(excluded)} excluded); a train/validation split needs two families")
    overlap = prompt_overlap(train, held)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists and is never overwritten")
    tmp = args.out.with_name(args.out.name + ".partial")
    tmp.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("validation", held)):
        (tmp / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part))
    manifest = {"inputs": {str(p): sha_text(p.read_text()) for p in (args.judged, args.review)},
                "system_prompt_sha256": sha_text(args.system.read_text().strip()) if args.system else None,
                "split": {"by": "scenario_id", "seed": args.seed, "holdout_fraction": args.holdout_fraction},
                "kept": len(kept), "train": len(train), "validation": len(held),
                "validation_families": sorted({r["family"] for r in held}),
                "validation_prompt_overlap": overlap, "review_problems": problems, "excluded": excluded}
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(tmp, args.out)  # the directory appears only when all three files are written
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("excluded", "review_problems")}, indent=2))
    if overlap:
        print(f"warning: {len(overlap)} validation conversations open with a caller turn that is also in "
              "training (see validation_prompt_overlap)", file=sys.stderr)
    if problems:
        print(f"warning: {len(problems)} review rows were invalid, conflicting or matched no turn "
              "(see review_problems); none of them accepted a turn", file=sys.stderr)


if __name__ == "__main__":
    main()
