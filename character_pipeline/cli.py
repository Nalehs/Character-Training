"""character-pipeline: one command for every stage, for any number of characters.

    character-pipeline new   <id> --name "NAME"     scaffold characters/<id> from the template
    character-pipeline status [<character> ...]     what is done, what is next, eval rates
    character-pipeline run   <character> [...]      run every pending step, resumably
                             --model qwen3.5-4b-base   --recipe recipes/<name>.json
    character-pipeline models | recipes             list model-scale presets and recipes
    character-pipeline review-sheet <character>     write the human review template
    character-pipeline ecology [--cross-judge]      all characters side by side
    character-pipeline <stage> [args]               run one stage directly:
        generate check judge clean pairs sft sample dpo kto serve eval
"""
from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path

from character_pipeline import run as pipeline

STAGES = {
    "generate": "character_pipeline.generate.teacher",
    "check": "character_pipeline.check.run",
    "judge": "character_pipeline.judge.score",
    "clean": "character_pipeline.data.clean",
    "pairs": "character_pipeline.data.pairs",
    "sft": "character_pipeline.train.sft",
    "sample": "character_pipeline.train.sample",
    "serve": "character_pipeline.serve",
    "eval": "character_pipeline.evaluate",
}


def new(args) -> Path:
    root = Path(args.characters)
    dest = root / args.id
    if dest.exists():
        raise SystemExit(f"{dest} already exists")
    shutil.copytree(root / "_template", dest)
    config = json.loads((dest / "config.json").read_text())
    config.update(id=args.id, display_name=args.name or args.id.upper(), status="draft")
    (dest / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"created {dest}. Next: write BIBLE.md and JUDGE_RULES.md, adapt scenarios.jsonl, "
          f"then `character-pipeline run {dest}`.")
    return dest


def status(args) -> None:
    from character_pipeline.ecology import registry
    rows = registry(Path(args.characters), Path(args.runs))
    if args.character:
        wanted = {Path(c).name for c in args.character}
        rows = [r for r in rows if r["id"] in wanted]
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    for r in rows:
        print(f'{r["id"]:<24}{r["status"] or "":<20}{r["done"]:>3}/{r["steps"]:<4} next: {r["next"] or "-"}')
        for arm, e in r["eval"].items():
            rates = "  ".join(f"{k} {v:.2f}" for k, v in e.items() if isinstance(v, float))
            print(f'    {arm:<22}{rates}  (eligible {e["eligible"]})')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in STAGES:
        return importlib.import_module(STAGES[argv[0]]).main(argv[1:])
    if argv and argv[0] in ("dpo", "kto"):
        return importlib.import_module("character_pipeline.train.preference").main(argv)

    ap = argparse.ArgumentParser(prog="character-pipeline", description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--characters", default="characters")
    ap.add_argument("--runs", default="runs")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("new")
    p.add_argument("id")
    p.add_argument("--name")
    p = sub.add_parser("status")
    p.add_argument("character", nargs="*")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("models", help="list model-scale presets")
    p = sub.add_parser("recipes", help="list recipe files")
    p = sub.add_parser("run")
    p.add_argument("character", type=Path)
    p.add_argument("--recipe", type=Path, help="recipes/<name>.json (default: character-pipeline-v0)")
    p.add_argument("--model", help="model preset from configs/models, e.g. qwen3.5-4b-base")
    p.add_argument("--allow-unpinned", action="store_true", help="train a preset with no pinned revision on 'main'")
    p.add_argument("--only", help="run only steps whose name starts with this, e.g. sft or eval")
    p.add_argument("--until", help="stop after the first step whose name starts with this")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--local-files-only", action="store_true")
    p = sub.add_parser("review-sheet")
    p.add_argument("character", type=Path)
    p = sub.add_parser("ecology")
    p.add_argument("--cross-judge", action="store_true")
    p.add_argument("--arm", default="dpo-seed0", help="which evaluated arm's replies to compare")
    p.add_argument("--judge-url", default="http://127.0.0.1:8090")
    args = ap.parse_args(argv)

    if args.cmd == "new":
        new(args)
    elif args.cmd == "status":
        status(args)
    elif args.cmd == "models":
        for f in sorted(pipeline.MODELS.glob("*.json")):
            m = json.loads(f.read_text())
            print(f'{m["id"]:<22}{m["hf"]:<26}{m["status"]:<10}{(m["revision"] or "UNPINNED")[:12]:<14}{m["hardware"]}')
    elif args.cmd == "recipes":
        for f in sorted((pipeline.REPO / "recipes").glob("*.json")):
            r = json.loads(f.read_text())
            print(f'{r["name"]:<26}{r.get("status", "")}\n    {r.get("summary", "")}\n    cite: {r.get("cite", "")}')
    elif args.cmd == "run":
        steps = pipeline.plan(args.character, Path(args.runs), args.device, args.seeds,
                              ["--local-files-only"] if args.local_files_only else None,
                              args.recipe, args.model, args.allow_unpinned)
        if not args.dry_run:
            print("settings:", pipeline.resolved(args.character, Path(args.runs), args.recipe, args.model,
                                                 args.allow_unpinned))
        result = pipeline.execute(steps, args.only, args.until, args.dry_run)
        print(result)
        if result.startswith("stopped"):
            raise SystemExit(2)
    elif args.cmd == "review-sheet":
        cid = pipeline.settings(args.character)["id"]
        R = Path(args.runs) / cid
        n = pipeline.review_sheet(R / "judged.jsonl", R / "review.todo.jsonl")
        print(f"{n} character turns in {R / 'review.todo.jsonl'}. Set each decision to accept or reject, "
              f"then save as {R / 'review.jsonl'}.")
    elif args.cmd == "ecology":
        from character_pipeline.ecology import cross_judge, registry, replies_from_eval
        rows = registry(Path(args.characters), Path(args.runs))
        if not args.cross_judge:
            return status(argparse.Namespace(characters=args.characters, runs=args.runs, character=[], json=False))
        from character_pipeline.judge.d1 import D1Client
        client = D1Client(url=args.judge_url)
        chars, replies = {}, {}
        for r in rows:
            folder = Path(args.characters) / r["id"]
            config = json.loads((folder / "config.json").read_text())
            summary = Path(args.runs) / r["id"] / "eval" / args.arm / "summary.json"
            if summary.exists():
                rules = (folder / config["judge"]["rules_file"]).read_text().strip()
                chars[r["id"]] = {"name": config.get("display_name", r["id"]), "rules": rules}
                replies[r["id"]] = replies_from_eval(summary)
        print(json.dumps(cross_judge(chars, replies, client.p_break), indent=2))


if __name__ == "__main__":
    main()
