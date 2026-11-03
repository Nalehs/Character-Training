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
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from character_pipeline.generate.controller import retained


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
    groups = sorted({group_of(r) for r in rows})
    random.Random(seed).shuffle(groups)
    held = set(groups[: max(1, round(len(groups) * holdout_fraction))])
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("judged", type=Path, help="teacher output (optionally with d1 scores)")
    ap.add_argument("review", type=Path, help="JSONL: conversation, turn, decision (accept/reject)")
    ap.add_argument("--system", type=Path, help="system prompt to prepend (prompted-teacher distillation)")
    ap.add_argument("--out", type=Path, default=Path("runs/clean"))
    ap.add_argument("--holdout-fraction", type=float, default=0.125)
    args = ap.parse_args(argv)
    rows = [json.loads(x) for x in args.judged.read_text().splitlines() if x.strip()]
    decisions = {(d["conversation"], d["turn"]): d["decision"]
                 for d in map(json.loads, args.review.read_text().splitlines()) if d}
    kept, excluded = clean(rows, decisions)
    train, held = group_split(kept, lambda r: r["family"], args.holdout_fraction)
    args.out.mkdir(parents=True, exist_ok=False)
    for name, part in (("train", train), ("validation", held)):
        with (args.out / f"{name}.jsonl").open("x") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {"inputs": {str(p): sha_text(p.read_text()) for p in (args.judged, args.review)},
                "kept": len(kept), "train": len(train), "validation": len(held),
                "validation_families": sorted({r["family"] for r in held}), "excluded": excluded}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "excluded"}, indent=2))


if __name__ == "__main__":
    main()
