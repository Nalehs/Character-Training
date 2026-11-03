"""OCT-format constitutions: read them and write single-turn prompt rows.

A constitution in Open Character Training (Maiya et al. 2025; maiush/
OpenCharacterTraining@d1da9f0, constitutions/hand-written/*.txt) is a JSON list
of about 10 objects, each a first-person trait and 5 hand-written user prompts:

    [{"trait": "I ...", "questions": ["...", "...", "...", "...", "..."]}, ...]

OCT's prompt-expanded files (constitutions/few-shot/*.jsonl) add
"additional_questions" to each trait; those are read too when present.

`characters/<id>/constitution.json` is an optional interop file in this format.
It may be the bare OCT list, or an object whose "traits" key holds that list
next to metadata such as a "note" (a bare list has nowhere to say "draft").
Nothing else in the pipeline requires it; it exists so that OCT's stages
(prompt expansion, teacher distillation, introspection) can be run on our
characters and so that others can run OCT on them.

    python -m character_pipeline.data.constitution characters/golem-15/constitution.json \
        --out runs/golem-15/oct/prompts.jsonl [--repeats 5] [--oct-out constitution-oct.json]

Rows: {"id": "t<trait>:q<question>[:r<repeat>]", "trait": "...", "messages": [{"role": "user", ...}]}
(a prompt ending on the user turn, ready for a teacher or student to answer).
OCT repeats every prompt K = 5 times (character/distillation/teacher.py:110, 194);
the default here is 1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_constitution(data) -> list[dict]:
    """Validated list of {"trait", "questions"[, "additional_questions"]} from parsed JSON."""
    traits = data.get("traits") if isinstance(data, dict) else data
    if not isinstance(traits, list) or not traits:
        raise ValueError("constitution must be a non-empty list of traits, or an object with a 'traits' list")
    out = []
    for n, item in enumerate(traits):
        if not isinstance(item, dict):
            raise ValueError(f"trait {n}: must be an object")
        trait = item.get("trait")
        if not isinstance(trait, str) or not trait.strip():
            raise ValueError(f"trait {n}: 'trait' must be a non-empty string")
        clean = {"trait": trait.strip()}
        for key in ("questions", "additional_questions"):
            if key not in item:
                continue
            qs = item[key]
            if not isinstance(qs, list) or not all(isinstance(q, str) and q.strip() for q in qs):
                raise ValueError(f"trait {n}: '{key}' must be a list of non-empty strings")
            clean[key] = [q.strip() for q in qs]
        if not clean.get("questions"):
            raise ValueError(f"trait {n}: needs at least one question")
        out.append(clean)
    if len({t["trait"] for t in out}) != len(out):
        raise ValueError("duplicate trait text")
    return out


def read_constitution(path) -> list[dict]:
    """Read a .json constitution (list or {"traits": [...]}) or an OCT few-shot .jsonl."""
    path = Path(path)
    text = path.read_text()
    if path.suffix == ".jsonl":
        return parse_constitution([json.loads(line) for line in text.splitlines() if line.strip()])
    return parse_constitution(json.loads(text))


def trait_string(traits: list[dict]) -> str:
    """OCT's numbered trait list for system prompts ("1: ...\\n2: ..."; introspection/self_reflection.py)."""
    return "\n".join(f"{i + 1}: {t['trait']}" for i, t in enumerate(traits))


def prompt_rows(traits: list[dict], repeats: int = 1, additional: bool = True) -> list[dict]:
    """One single-turn prompt row per question (and per repeat)."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    rows = []
    for i, t in enumerate(traits):
        questions = t["questions"] + (t.get("additional_questions", []) if additional else [])
        for j, q in enumerate(questions):
            for r in range(repeats):
                rid = f"t{i}:q{j}" + (f":r{r}" if repeats > 1 else "")
                rows.append({"id": rid, "trait": t["trait"], "messages": [{"role": "user", "content": q}]})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("constitution", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=1, help="OCT uses K = 5")
    ap.add_argument("--no-additional", action="store_true", help="ignore additional_questions")
    ap.add_argument("--oct-out", type=Path, help="also write the bare OCT list (for OCT's own scripts)")
    ap.add_argument("--scenarios-out", type=Path,
                    help="also write one-turn scenarios for generate.teacher (OCT stage 3, teacher replies)")
    args = ap.parse_args(argv)
    traits = read_constitution(args.constitution)
    rows = prompt_rows(traits, args.repeats, not args.no_additional)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    if args.scenarios_out:
        args.scenarios_out.parent.mkdir(parents=True, exist_ok=True)
        args.scenarios_out.write_text("".join(json.dumps(
            {"id": f'{r["id"]}:k{n}', "caller_turns": [r["messages"][0]["content"]], "trait": r["trait"]},
            ensure_ascii=False) + "\n" for n, r in enumerate(rows)))
    if args.oct_out:
        bare = [{"trait": t["trait"], "questions": t["questions"]} for t in traits]
        args.oct_out.write_text(json.dumps(bare, indent=4, ensure_ascii=False) + "\n")
    print(json.dumps({"traits": len(traits), "rows": len(rows)}))


if __name__ == "__main__":
    main()
